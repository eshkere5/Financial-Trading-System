"""
NewsAgentClient — единственная точка входа торгового блока к новостному агенту.
"""
from __future__ import annotations
import datetime
import email.utils
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/news.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    item_id TEXT PRIMARY KEY,
    source TEXT,
    title TEXT,
    text TEXT,
    title_clean TEXT,
    text_clean TEXT,
    summary TEXT,
    topics TEXT,
    tickers TEXT,
    date TEXT,
    collected_at TEXT,
    url TEXT,
    ner_persons TEXT,
    ner_companies TEXT,
    ner_orgs TEXT,
    ner_locations TEXT,
    sentiment_label TEXT,
    sentiment_score REAL,
    sanction_risk REAL
);
CREATE INDEX IF NOT EXISTS idx_news_date ON news(date);
CREATE INDEX IF NOT EXISTS idx_news_tickers ON news(tickers);
"""

SANCTION_KEYWORDS: Dict[str, float] = {
    "sanction": 0.9, "sanctions": 0.9, "санкции": 0.9, "санкция": 0.85,
    "frozen assets": 0.85, "заморозка активов": 0.85,
    "sdn list": 1.0, "sdn-list": 1.0, "чёрный список": 0.8,
    "запрет": 0.6, "ban": 0.6, "запрещен": 0.55,
    "embargo": 0.9, "эмбарго": 0.9, "ofac": 1.0, "eu sanctions": 1.0,
    "ограничения": 0.5, "restriction": 0.5, "штраф": 0.45, "fine": 0.4,
    "расследование": 0.4, "investigation": 0.4,
}

SANCTION_BLOCK_THRESHOLD = 0.5

### FIXED 2026-08-01 (К-4) ###
# Было: `if keyword in text` — подстрочный поиск. "ban" срабатывал внутри
# "sberbank"/"urban", "fine" внутри "define"/"refinery", "штраф" внутри
# "штрафстоянка". Практически каждая новость про Сбербанк получала
# sanction_score >= 0.6 и глушилась риск-контуром.
# Стало: одна предкомпилированная регулярка с границами слова. Использованы
# lookaround (?<!\w)/(?!\w) вместо \b, потому что \b некорректно работает на
# краях многословных фраз ("sdn-list", "eu sanctions") и вокруг кириллицы
# в сочетании с дефисами. re.UNICODE обязателен для \w по русским буквам.
_SANCTION_PATTERN = re.compile(
    r"(?<!\w)(" + "|".join(
        re.escape(k) for k in sorted(SANCTION_KEYWORDS, key=len, reverse=True)
    ) + r")(?!\w)",
    re.IGNORECASE | re.UNICODE,
)


@dataclass
class NewsDoc:
    """Одна новость с обогащением (NER, sentiment, sanction score)."""
    item_id: str
    title: str
    body: str
    source: str
    published_at: str  # ISO datetime string UTC
    url: str
    topics: List[str] = field(default_factory=list)
    tickers: List[str] = field(default_factory=list)
    ner_companies: List[str] = field(default_factory=list)
    ner_orgs: List[str] = field(default_factory=list)
    sentiment_label: str = "neutral"
    sentiment: float = 0.0            # [-1, 1]
    sanction_score: float = 0.0       # [0, 1], hybrid(keyword, DeepSeek)
    summary: str = ""


@dataclass
class NewsSnapshot:
    """Агрегированный новостной контекст по тикеру за horizon hours_back."""
    ticker: str
    hours_back: int
    docs: List[NewsDoc] = field(default_factory=list)
    doc_count: int = 0
    avg_sentiment: float = 0.0
    sanction_risk: float = 0.0        # max(sanction_score) по docs
    positive_ratio: float = 0.0
    negative_ratio: float = 0.0
    signal_weight: float = 0.0        # [-1, 1] итоговый вес для стратегии


def _parse_topics(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _parse_ner(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


### FIXED 2026-08-01 (Н-4) ###
# sqlite3.Row не поддерживает .get(); обращение к отсутствующей колонке на
# старой схеме БД роняло чтение. Единый безопасный аксессор вместо россыпи
# try/except IndexError по коду.
def _row_get(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    try:
        if key not in row.keys():
            return default
    except Exception:
        return default
    value = row[key]
    return default if value is None else value


### FIXED 2026-08-01 (К-2, часть 1) ###
# Даты в таблице news приходят из разных коллекторов в разных форматах:
# RSS отдаёт RFC-822 ("Tue, 29 Jul 2026 10:15:00 +0300"), внутренние вставки —
# ISO. Единый парсер приводит оба к timezone-aware UTC datetime.
def _parse_dt(raw: Any) -> Optional[datetime.datetime]:
    if not raw:
        return None
    if isinstance(raw, datetime.datetime):
        dt = raw
    else:
        s = str(raw).strip()
        if not s:
            return None
        dt = None
        try:
            dt = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            try:
                dt = email.utils.parsedate_to_datetime(s)
            except (TypeError, ValueError):
                return None
        if dt is None:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def _compute_sanction_score(title: str, body: str) -> float:
    """Keyword-эвристика [0, 1] — fallback, если DeepSeek-анализ недоступен."""
    ### FIXED 2026-08-01 (К-4) ###
    # Матчинг по границам слова вместо `keyword in text`.
    text = f"{title} {body}"
    score = 0.0
    for match in _SANCTION_PATTERN.finditer(text):
        score = max(score, SANCTION_KEYWORDS.get(match.group(1).lower(), 0.0))
    return round(score, 3)


def _hybrid_sanction_score(heuristic: float, deepseek: float) -> float:
    """
    Комбинирует keyword-эвристику и DeepSeek-оценку.
    Если оба флагуют угрозу (>= threshold) — берём максимум (консервативно).
    Если только DeepSeek — доверяем DeepSeek (умнее ловит контекст).
    Если только эвристика — не игнорируем: 70% DeepSeek + 30% heuristic.
    """
    if deepseek <= 0.0:
        return heuristic
    if heuristic <= 0.0:
        return deepseek

    h_flag = heuristic >= SANCTION_BLOCK_THRESHOLD
    d_flag = deepseek >= SANCTION_BLOCK_THRESHOLD
    if h_flag and d_flag:
        return round(max(heuristic, deepseek), 3)
    return round(0.7 * deepseek + 0.3 * heuristic, 3)


def _ticker_matches_ner(ticker: str, ner_companies: List[str], ner_orgs: List[str]) -> bool:
    """Fallback-сопоставление тикера через NER, если явного списка tickers нет."""
    haystack = " ".join(ner_companies + ner_orgs).lower()
    return ticker.lower() in haystack


def _ticker_matches_column(ticker: str, tickers_raw: Optional[str]) -> bool:
    if not tickers_raw:
        return False
    return ticker.upper() in [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]


def _doc_matches_ticker(ticker: str, doc: NewsDoc) -> bool:
    return (
        _ticker_matches_column(ticker, ",".join(doc.tickers))
        or _ticker_matches_ner(ticker, doc.ner_companies, doc.ner_orgs)
        or ticker.lower() in doc.title.lower()
    )


def _row_to_doc(row: sqlite3.Row) -> NewsDoc:
    title = _row_get(row, "title", "") or ""
    body = _row_get(row, "text_clean", "") or _row_get(row, "text", "") or ""
    heuristic_sanction = _compute_sanction_score(title, body)

    try:
        deepseek_sanction = float(_row_get(row, "sanction_risk", 0.0) or 0.0)
    except (TypeError, ValueError):
        deepseek_sanction = 0.0

    tickers_raw = _row_get(row, "tickers")

    return NewsDoc(
        item_id=_row_get(row, "item_id", "") or "",
        title=title,
        body=body,
        source=_row_get(row, "source", "") or "",
        published_at=_row_get(row, "date", "") or _row_get(row, "collected_at", "") or "",
        url=_row_get(row, "url", "") or "",
        topics=_parse_topics(_row_get(row, "topics")),
        tickers=_parse_ner(tickers_raw) if tickers_raw else [],
        ner_companies=_parse_ner(_row_get(row, "ner_companies")),
        ner_orgs=_parse_ner(_row_get(row, "ner_orgs")),
        sentiment_label=_row_get(row, "sentiment_label", "neutral") or "neutral",
        sentiment=float(_row_get(row, "sentiment_score", 0.0) or 0.0),
        sanction_score=_hybrid_sanction_score(heuristic_sanction, deepseek_sanction),
        summary=_row_get(row, "summary", "") or "",
    )


def _aggregate(ticker: str, hours_back: int, docs: List[NewsDoc]) -> NewsSnapshot:
    snap = NewsSnapshot(ticker=ticker, hours_back=hours_back, docs=docs, doc_count=len(docs))
    if not docs:
        return snap

    sentiments = [d.sentiment for d in docs]
    snap.avg_sentiment = round(sum(sentiments) / len(sentiments), 4)
    snap.sanction_risk = round(max((d.sanction_score for d in docs), default=0.0), 3)
    snap.positive_ratio = round(sum(1 for d in docs if d.sentiment_label == "positive") / len(docs), 3)
    snap.negative_ratio = round(sum(1 for d in docs if d.sentiment_label == "negative") / len(docs), 3)

    base = snap.avg_sentiment * (snap.positive_ratio - snap.negative_ratio + 1) / 2
    penalty = snap.sanction_risk * 0.5
    snap.signal_weight = round(max(-1.0, min(1.0, base - penalty)), 4)
    return snap


class NewsAgentClient:
    """
    Клиент новостного модуля.

    Parameters
    ----------
    cfg : dict
        Секция configs/news_agent.yaml -> news_agent.

    Usage
    -----
    client = NewsAgentClient(cfg["news_agent"])
    snap = client.get_snapshot("SBER", hours_back=6)
    print(snap.avg_sentiment, snap.sanction_risk)
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.transport: str = cfg.get("transport", "local")
        self.lock = threading.Lock()

        if self.transport == "local":
            self.db_path = Path(cfg.get("local_db_path", DEFAULT_DB_PATH))
            self.conn: Optional[sqlite3.Connection] = None
            self._init_db()
        elif self.transport == "http":
            import httpx  # noqa: F401
            self.http = httpx.Client(
                base_url=cfg["http_base_url"],
                timeout=cfg.get("timeout_seconds", 10),
            )
        else:
            raise ValueError(f"Unknown transport {self.transport!r} (local|http)")

        logger.info(
            "NewsAgentClient | transport=%s | db=%s",
            self.transport,
            self.db_path if self.transport == "local" else cfg.get("http_base_url"),
        )

    # ── lifecycle ────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """
        SQLite WAL, read-friendly. Создаёт родительскую директорию и схему,
        если файла не существует — раньше клиент просто варнил и работал
        без данных, теперь база гарантированно создаётся на новом пути.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        is_new = not self.db_path.exists()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

        if is_new:
            logger.warning(
                "NewsAgentClient: создана новая пустая БД %s — истории новостей нет, "
                "нужен прогон коллектора с backfill_existing=true",
                self.db_path,
            )

    def close(self) -> None:
        if self.transport == "local" and self.conn:
            self.conn.close()
            self.conn = None
        elif self.transport == "http" and hasattr(self, "http"):
            self.http.close()

    def __enter__(self) -> "NewsAgentClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ── public API ───────────────────────────────────────────────────────

    def get_snapshot(self, ticker: str, hours_back: int = 24) -> NewsSnapshot:
        """Возвращает NewsSnapshot: avg_sentiment, sanction_risk, signal_weight."""
        docs = self.search_news(ticker, hours_back)
        return _aggregate(ticker, hours_back, docs)

    def search_news(self, ticker: str, hours_back: int = 24) -> List[NewsDoc]:
        """Новости по тикеру за hours_back часов."""
        if self.transport == "local":
            return self._local_search(ticker, hours_back)
        return self._http_search(ticker, hours_back)

    def get_sanction_risk(self, ticker: str, hours_back: int = 24) -> float:
        """Санкционный риск [0, 1] по тикеру."""
        snap = self.get_snapshot(ticker, hours_back)
        return snap.sanction_risk

    def summarize_events(self, ticker: str, hours_back: int = 24) -> str:
        """Текстовое саммари новостей за период. llm_summarize=true — через LLMBackend."""
        if self.transport == "http":
            resp = self.http.get("/summarize", params={"ticker": ticker, "hours_back": hours_back})
            resp.raise_for_status()
            return resp.json().get("summary", "")

        docs = self._local_search(ticker, hours_back)
        if not docs:
            return f"No news for {ticker} in last {hours_back}h."

        if self.cfg.get("llm_summarize", False):
            from src.news_agent_client.llm_backend import LLMBackend
            return LLMBackend(self.cfg).summarize(ticker, docs)

        return " / ".join(d.title for d in docs[:5])

    def get_multi_snapshot(self, tickers: List[str], hours_back: int = 24) -> Dict[str, NewsSnapshot]:
        """Снапшоты по нескольким тикерам одним проходом."""
        ### FIXED 2026-08-01 (С-15) ###
        # Было: {t: self.get_snapshot(t, ...) for t in tickers} — N полных
        # сканов таблицы (LIKE '%X%' не использует индекс). На 30 тикерах и
        # базе в сотни тысяч строк это десятки секунд внутри торгового тика.
        # Стало: один SQL-запрос с OR-цепочкой LIKE, разбор по тикерам в Python.
        if self.transport != "local":
            return {t: self.get_snapshot(t, hours_back) for t in tickers}

        unique = list(dict.fromkeys(t for t in tickers if t))
        if not unique:
            return {}

        docs = self._local_search_multi(unique, hours_back)
        return {
            t: _aggregate(t, hours_back, [d for d in docs if _doc_matches_ticker(t, d)])
            for t in unique
        }

    # ── internal: local (SQLite) ────────────────────────────────────────

    ### FIXED 2026-08-01 (К-2, часть 2) ###
    # Было: `WHERE date >= ?` со строковым ISO-cutoff. SQLite сравнивает TEXT
    # лексикографически, а в колонке date лежат в том числе RFC-822 строки
    # ("Tue, 29 Jul..."), для которых такое сравнение бессмысленно: новости
    # либо все проходили фильтр, либо все отсекались, и hours_back не работал.
    # Стало: SQL отбирает только по тикеру (без даты), а отсечка по времени,
    # сортировка и LIMIT делаются в Python на распарсенных datetime.
    # Документы с неразбираемой датой отбрасываются — попадание "новости без
    # даты" в свежий срез опаснее её потери.
    def _select_rows(self, tickers: List[str], hours_back: int) -> List[sqlite3.Row]:
        if self.conn is None:
            return []

        # верхняя граница строк, поднимаемых из БД до фильтрации по дате
        max_scan = int(self.cfg.get("max_scan_rows", 5000))

        clauses = []
        params: List[Any] = []
        for ticker in tickers:
            like = f"%{ticker}%"
            clauses.append("(tickers LIKE ? OR title LIKE ? OR text_clean LIKE ?)")
            params.extend([like, like, like])

        query = (
            "SELECT * FROM news WHERE "
            + " OR ".join(clauses)
            + " ORDER BY date DESC LIMIT ?"
        )
        params.append(max_scan)

        with self.lock:
            return self.conn.execute(query, params).fetchall()

    def _filter_by_time(self, rows: List[sqlite3.Row], hours_back: int) -> List[NewsDoc]:
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours_back)

        dated: List[tuple] = []
        skipped = 0
        for row in rows:
            dt = _parse_dt(_row_get(row, "date")) or _parse_dt(_row_get(row, "collected_at"))
            if dt is None:
                skipped += 1
                continue
            if dt < cutoff:
                continue
            dated.append((dt, row))

        if skipped:
            logger.debug("NewsAgentClient: %d строк с неразбираемой датой пропущено", skipped)

        dated.sort(key=lambda pair: pair[0], reverse=True)
        return [_row_to_doc(row) for _dt, row in dated]

    def _local_search(self, ticker: str, hours_back: int) -> List[NewsDoc]:
        if self.conn is None:
            return []

        limit = int(self.cfg.get("max_docs_per_ticker", 100))
        docs = self._filter_by_time(self._select_rows([ticker], hours_back), hours_back)

        ### FIXED 2026-08-01 (К-3) ###
        # Было: `return filtered or docs` — если строгая проверка по
        # tickers/NER/заголовку отсекала всё, возвращался исходный LIKE-шум:
        # новость про "Сбербанк" уходила в снапшот BER/ERB и любого тикера,
        # чьи буквы встретились в тексте. Стало: возвращаем только то, что
        # реально сматчилось с тикером.
        filtered = [d for d in docs if _doc_matches_ticker(ticker, d)]
        return filtered[:limit]

    def _local_search_multi(self, tickers: List[str], hours_back: int) -> List[NewsDoc]:
        """Один проход по БД для набора тикеров (см. get_multi_snapshot)."""
        if self.conn is None:
            return []
        limit = int(self.cfg.get("max_docs_per_ticker", 100)) * max(1, len(tickers))
        docs = self._filter_by_time(self._select_rows(tickers, hours_back), hours_back)
        return docs[:limit]

    # ── internal: http transport ────────────────────────────────────────

    def _http_search(self, ticker: str, hours_back: int) -> List[NewsDoc]:
        resp = self.http.get("/search", params={"ticker": ticker, "hours_back": hours_back})
        resp.raise_for_status()
        items = resp.json().get("items", [])
        ### FIXED 2026-08-01 (К-1) ###
        # Маппинг HTTP-ответа продублирован здесь и в mappers.map_http_response,
        # причём вторая копия строила NewsDoc с несуществующими полями. Теперь
        # единственная реализация живёт в mappers; импорт ленивый — модуль
        # mappers импортирует NewsDoc отсюда (циклический импорт на уровне
        # модуля).
        from src.news_agent_client.mappers import map_http_response
        return map_http_response(items)

    def insert_doc(self, doc: NewsDoc) -> None:
        """Пишет NewsDoc в локальную SQLite. Нужен для сценарных/интеграционных
        тестов и для реального news-коллектора — раньше клиент был read-only."""
        if self.transport != "local" or self.conn is None:
            logger.warning("insert_doc недоступен для transport=%s", self.transport)
            return

        ### FIXED 2026-08-01 (К-2, часть 3) ###
        # Нормализуем дату к ISO-UTC на вставке, чтобы в колонке date не
        # копились RFC-822 строки и ORDER BY date работал предсказуемо.
        parsed = _parse_dt(doc.published_at)
        stored_date = parsed.isoformat() if parsed else doc.published_at

        with self.lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO news
                (item_id, source, title, text, title_clean, text_clean, summary,
                 topics, tickers, date, collected_at, url, ner_persons, ner_companies,
                 ner_orgs, ner_locations, sentiment_label, sentiment_score, sanction_risk)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc.item_id, doc.source, doc.title, doc.body, doc.title, doc.body,
                    doc.summary, ",".join(doc.topics), ",".join(doc.tickers),
                    stored_date, stored_date, doc.url, "",
                    ",".join(doc.ner_companies), ",".join(doc.ner_orgs), "",
                    doc.sentiment_label, doc.sentiment, doc.sanction_score,
                ),
            )
            self.conn.commit()
        logger.info("NewsAgentClient: сохранён doc %s в %s", doc.item_id, self.db_path)
