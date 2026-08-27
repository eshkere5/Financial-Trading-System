"""
news_risk_pipeline.py — событийный конвейер обработки новостей.

"""
from __future__ import annotations

import dataclasses
import difflib
import logging
import os
import re
import sqlite3
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from src.news_agent_client.rss_collector import RawNewsItem, RSSCollector, RSSCollectorConfig
from src.news_agent_client.keyword_filter import WatchTermsStore, build_filter
from src.news_agent_client.news_deep_dive import NewsDeepDiveAnalyzer, NewsRiskAssessment

logger = logging.getLogger(__name__)

DEFAULT_TTL_DAYS = {"short": 3, "long": 14}   # fallback, если у оценки нет своего ttl_days
DEFAULT_MAX_LLM_CALLS_PER_CYCLE = 20
DEFAULT_DB_PATH = "data/news.db"
DEDUP_TITLE_HOURS = 72          # окно кросс-источниковой дедупликации
FUZZY_DUP_RATIO = 0.87          # порог похожести заголовков (difflib)


@dataclass
class PipelineStats:
    """Метрики для дашборда/логов."""
    total_seen: int = 0
    passed_filter: int = 0
    duplicates_skipped: int = 0
    cross_source_duplicates: int = 0   ### NEW 2026-08-15 ###
    sent_to_deepseek: int = 0
    dampened_by_risk: int = 0
    reused_across_tickers: int = 0
    skipped_by_llm_cap: int = 0
    analysis_failed: int = 0

    def as_dict(self) -> dict:
        rate = (self.sent_to_deepseek / self.total_seen * 100) if self.total_seen else 0.0
        return {
            "total_seen": self.total_seen,
            "passed_filter": self.passed_filter,
            "duplicates_skipped": self.duplicates_skipped,
            "cross_source_duplicates": self.cross_source_duplicates,
            "sent_to_deepseek": self.sent_to_deepseek,
            "deepseek_call_rate_pct": round(rate, 1),
            "dampened_by_risk": self.dampened_by_risk,
            "reused_across_tickers": self.reused_across_tickers,
            "skipped_by_llm_cap": self.skipped_by_llm_cap,
            "analysis_failed": self.analysis_failed,
        }


def _title_fingerprint(title: str) -> str:
    """
    Нормализованный отпечаток заголовка для кросс-источниковой дедупликации.
    "Провал BIP-110 ... - ForkLog" и тот же текст из google_btc дают
    одинаковый отпечаток: нижний регистр, без суффикса источника, без
    пунктуации.
    """
    t = (title or "").lower()
    t = re.sub(r"[\s]*[-—|][^-—|]{0,40}$", "", t)      # хвост " - Источник"
    t = re.sub(r"[^0-9a-zа-яё]+", " ", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()


class NewsMemoryStore:
    """
    Память оценок по тикеру + дедупликация + персистентность в SQLite.

    TTL ИНДИВИДУАЛЬНЫЙ: у каждой оценки свой ttl_days (его выставляет
    DeepSeek). Затухание линейное: свежая новость весит полностью, к концу
    своего ttl_days вес уходит в 0 — без ступенек на границе short/long.
    """

    def __init__(
        self,
        ttl_days: Optional[Dict[str, int]] = None,
        db_path: Optional[str] = DEFAULT_DB_PATH,
    ) -> None:
        self.ttl_days = ttl_days or dict(DEFAULT_TTL_DAYS)
        self._seen_ids: set[str] = set()
        self._by_ticker: Dict[str, List[NewsRiskAssessment]] = {}
        self._title_fps: deque[Tuple[str, float]] = deque(maxlen=500)
        self._lock = threading.Lock()

        self._db: Optional[sqlite3.Connection] = None
        if db_path:
            try:
                os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
                self._db = sqlite3.connect(db_path, check_same_thread=False)
                self._init_db()
                self._load_recent()
            except Exception as exc:
                logger.warning("NewsMemoryStore: SQLite отключён (%s) — работаем в памяти", exc)
                self._db = None

    # ── персистентность ──────────────────────────────────────────────

    def _init_db(self) -> None:
        assert self._db is not None
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS news_assessments (
                item_id        TEXT PRIMARY KEY,
                ticker         TEXT,
                analyzed_at    REAL,
                risk_category  TEXT,
                severity       REAL,
                materiality    REAL,
                effective_risk REAL,
                sentiment      REAL,
                horizon        TEXT,
                ttl_days       REAL,
                action         TEXT,
                source         TEXT,
                title          TEXT,
                rationale      TEXT
            )
            """
        )
        self._db.commit()

    def _load_recent(self) -> None:
        """Поднимает из БД записи, чей ttl ещё не истёк — дедуп и агрегат
        переживают рестарт процесса."""
        assert self._db is not None
        now = datetime.now(timezone.utc).timestamp()
        max_ttl = max(30.0, float(max(self.ttl_days.values(), default=3)))
        cutoff = now - max_ttl * 86400.0
        fields = {f.name for f in dataclasses.fields(NewsRiskAssessment)}

        cols = [c[1] for c in self._db.execute("PRAGMA table_info(news_assessments)").fetchall()]
        rows = self._db.execute(
            "SELECT * FROM news_assessments WHERE analyzed_at >= ?", (cutoff,)
        ).fetchall()

        loaded = 0
        for row in rows:
            raw = dict(zip(cols, row))
            kwargs = {k: v for k, v in raw.items() if k in fields}
            try:
                a = NewsRiskAssessment(**kwargs)
            except Exception:
                continue
            age_days = (now - float(raw.get("analyzed_at") or now)) / 86400.0
            if age_days > self._ttl_for(a):
                continue
            self._seen_ids.add(a.item_id)
            self._by_ticker.setdefault(a.ticker, []).append(a)
            fp = _title_fingerprint(str(raw.get("title") or ""))
            if fp:
                self._title_fps.append((fp, now))
            loaded += 1
        if loaded:
            logger.info("NewsMemoryStore: восстановлено %d живых оценок из news.db", loaded)

    def _persist(self, ticker: str, a: NewsRiskAssessment, title: str, source: str) -> None:
        if self._db is None:
            return
        try:
            with self._lock:
                self._db.execute(
                    """
                    INSERT OR REPLACE INTO news_assessments
                    (item_id, ticker, analyzed_at, risk_category, severity, materiality,
                     effective_risk, sentiment, horizon, ttl_days, action, source, title, rationale)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        a.item_id, ticker,
                        float(getattr(a, "analyzed_at", datetime.now(timezone.utc).timestamp())),
                        getattr(a, "risk_category", "none"),
                        float(getattr(a, "severity", 0.0)),
                        float(getattr(a, "materiality", 0.0)),
                        float(getattr(a, "effective_risk", 0.0)),
                        float(getattr(a, "sentiment", 0.0)),
                        getattr(a, "horizon", "short"),
                        float(self._ttl_for(a)),
                        getattr(a, "action", "pass"),
                        source, (title or "")[:300],
                        str(getattr(a, "rationale", ""))[:500],
                    ),
                )
                self._db.commit()
        except Exception as exc:
            logger.warning("NewsMemoryStore: запись в SQLite не удалась (некритично): %s", exc)

    # ── TTL: индивидуальный ttl_days оценки, fallback — карта по горизонту ──

    def _ttl_for(self, a: NewsRiskAssessment) -> float:
        # Новый датакласс: ttl_days поле; effective_ttl_days property делает
        # то же самое — берём его, если есть (обратная совместимость).
        eff = getattr(a, "effective_ttl_days", None)
        if eff is not None:
            try:
                return float(eff)
            except (TypeError, ValueError):
                pass
        ttl = float(getattr(a, "ttl_days", 0.0) or 0.0)
        if ttl > 0:
            return ttl
        return float(self.ttl_days.get(getattr(a, "horizon", "short"), 3))

    # ── дедупликация ─────────────────────────────────────────────────

    def is_duplicate(self, item_id: str) -> bool:
        with self._lock:
            return item_id in self._seen_ids

    def is_cross_source_duplicate(self, title: str) -> bool:
        """Та же новость из другого источника: точное совпадение отпечатка
        или нечёткое (difflib) в окне DEDUP_TITLE_HOURS."""
        fp = _title_fingerprint(title)
        if not fp:
            return False
        now = datetime.now(timezone.utc).timestamp()
        with self._lock:
            recent = [f for f, ts in self._title_fps if (now - ts) <= DEDUP_TITLE_HOURS * 3600]
        if fp in recent:
            return True
        return bool(difflib.get_close_matches(fp, recent, n=1, cutoff=FUZZY_DUP_RATIO))

    def add(self, ticker: str, assessment: NewsRiskAssessment, title: str = "", source: str = "") -> None:
        fp = _title_fingerprint(title)
        now = datetime.now(timezone.utc).timestamp()
        with self._lock:
            self._seen_ids.add(assessment.item_id)
            self._by_ticker.setdefault(ticker, []).append(assessment)
            if fp:
                self._title_fps.append((fp, now))
            self._prune_locked(ticker)
        self._persist(ticker, assessment, title, source)

    # ── забывание и агрегация ────────────────────────────────────────

    def _prune_locked(self, ticker: str) -> None:
        """Удаляет записи старше ИХ СОБСТВЕННОГО ttl_days. Под self._lock."""
        now = datetime.now(timezone.utc).timestamp()
        kept: List[NewsRiskAssessment] = []
        for a in self._by_ticker.get(ticker, []):
            age_days = (now - float(getattr(a, "analyzed_at", now))) / 86400.0
            if age_days <= self._ttl_for(a):
                kept.append(a)
        self._by_ticker[ticker] = kept

    def active_for(self, ticker: str) -> List[NewsRiskAssessment]:
        with self._lock:
            self._prune_locked(ticker)
            return list(self._by_ticker.get(ticker, []))

    def aggregate_effective_risk(self, ticker: str) -> float:
        """
        max(severity × materiality × decay) по живым новостям.
        decay = 1 - age / ttl_days — линейное затухание по ИНДИВИДУАЛЬНОМУ
        времени актуальности новости.
        """
        active = self.active_for(ticker)
        if not active:
            return 0.0
        now = datetime.now(timezone.utc).timestamp()
        best = 0.0
        for a in active:
            ttl = self._ttl_for(a)
            age_days = max(0.0, (now - float(getattr(a, "analyzed_at", now))) / 86400.0)
            decay = max(0.0, 1.0 - age_days / ttl) if ttl > 0 else 0.0
            best = max(best, float(a.severity) * float(a.materiality) * decay)
        return round(best, 4)

    def build_aggregate(self, ticker: str) -> Optional[NewsRiskAssessment]:
        """«Оценка для публикации»: доминирующее по риску живое событие +
        агрегированный (затухающий) риск всей TTL-памяти тикера."""
        active = self.active_for(ticker)
        if not active:
            return None

        aggregated = self.aggregate_effective_risk(ticker)
        dominant = max(active, key=lambda a: float(a.severity) * float(a.materiality))
        avg_sentiment = round(sum(float(a.sentiment) for a in active) / len(active), 4)

        return dataclasses.replace(
            dominant,
            ticker=ticker,
            sentiment=avg_sentiment,
            aggregated_effective_risk=aggregated,
            active_count=len(active),
        )


class NewsRiskPipeline:
    """
    Основной событийный конвейер. Подключается в LiveEngine вместо
    старого NewsSignalExtractor.refresh()-по-таймеру.
    """

    def __init__(
        self,
        rss_cfg: RSSCollectorConfig,
        analyzer: NewsDeepDiveAnalyzer,
        on_assessment: Callable[[str, NewsRiskAssessment], None],
        filter_mode: str = "keyword",
        ttl_days: Optional[Dict[str, int]] = None,
        max_llm_calls_per_cycle: int = DEFAULT_MAX_LLM_CALLS_PER_CYCLE,
        db_path: Optional[str] = DEFAULT_DB_PATH,   ### NEW 2026-08-15 ###
    ) -> None:
        self.collector = RSSCollector(rss_cfg)
        self.watch_terms = WatchTermsStore()
        self.cheap_filter = build_filter(self.watch_terms, mode=filter_mode)
        self.analyzer = analyzer
        self.on_assessment = on_assessment
        self.memory = NewsMemoryStore(ttl_days=ttl_days, db_path=db_path)
        self.stats = PipelineStats()
        self.max_llm_calls_per_cycle = int(max_llm_calls_per_cycle)
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def update_watch_terms(self, watch_terms: Dict[str, List[str]], timestamp: str) -> None:
        """Вызывается ОДНОКРАТНО при старте торгов."""
        with self._lock:
            self.watch_terms.update(watch_terms, timestamp)

    def publish_restored(self, tickers: List[str]) -> None:
        """После рестарта: публикует агрегаты из восстановленной из news.db
        памяти в state.news — иначе до первой СВЕЖЕЙ новости система
        торгует без новостного риск-фона."""
        published = 0
        for ticker in tickers:
            agg = self.memory.build_aggregate(ticker)
            if agg is not None:
                try:
                    self.on_assessment(ticker, agg)
                    published += 1
                except Exception as exc:
                    logger.warning("publish_restored(%s) упал: %s", ticker, exc)
        if published:
            logger.info("publish_restored: восстановлен риск-фон для %d тикеров из news.db", published)
            
    # ── отбор кандидатов: фильтр → дедуп → скор → сортировка ─────────

    def _score_item(self, item: RawNewsItem, matched_tickers: List[str]) -> float:
        """
        Скор релевантности для top-N в пределах LLM-лимита.
        EmbeddingFilter.score(ticker, headline, summary) — косинусная близость
        к watch_terms тикера; берём максимум по матчнувшимся тикерам.
        KeywordFilter скора не имеет — fallback: число матчнувшихся тикеров.
        """
        scorer = getattr(self.cheap_filter, "score", None)
        if callable(scorer):
            try:
                return max(
                    float(scorer(t, item.title, item.summary)) for t in matched_tickers
                )
            except Exception:
                pass
        return float(len(matched_tickers))

    def _prepare_candidates(
        self, items: List[RawNewsItem]
    ) -> List[Tuple[float, RawNewsItem, List[str]]]:
        """
        Первый проход БЕЗ DeepSeek: локальный фильтр, оба вида дедупликации,
        скоринг. Кандидаты отсортированы по убыванию релевантности — лимит
        LLM-вызовов забирают самые важные новости, а не первые по фиду.
        """
        prepared: List[Tuple[float, RawNewsItem, List[str]]] = []
        for item in items:
            self.stats.total_seen += 1
            with self._lock:
                matched_tickers = self.cheap_filter.match(item.title, item.summary)
            if not matched_tickers:
                continue
            self.stats.passed_filter += 1

            if self.memory.is_duplicate(item.item_id):
                self.stats.duplicates_skipped += 1
                continue

            if self.memory.is_cross_source_duplicate(item.title):
                self.stats.cross_source_duplicates += 1
                logger.debug("NewsRiskPipeline: кросс-источниковый дубль пропущен: %s", item.title[:80])
                continue

            prepared.append((self._score_item(item, matched_tickers), item, matched_tickers))

        prepared.sort(key=lambda c: c[0], reverse=True)
        return prepared

    # ── обработка одной новости (DeepSeek) ───────────────────────────

    def _process_item(
        self,
        item: RawNewsItem,
        calls_used: int,
        publish: bool,
        matched_tickers: List[str],
    ) -> Tuple[int, Dict[str, NewsRiskAssessment]]:
        produced: Dict[str, NewsRiskAssessment] = {}

        if calls_used >= self.max_llm_calls_per_cycle:
            self.stats.skipped_by_llm_cap += 1
            return calls_used, produced

        # Один вызов DeepSeek на новость (по первому тикеру), остальным
        # тикерам оценка размножается через dataclasses.replace.
        primary = matched_tickers[0]
        self.stats.sent_to_deepseek += 1
        calls_used += 1
        try:
            assessment = self.analyzer.analyze(ticker=primary, item=item)
        except Exception as exc:
            logger.warning(
                "NewsRiskPipeline: анализ %s для %s провалился: %s", item.item_id, primary, exc
            )
            self.stats.analysis_failed += 1
            return calls_used, produced

        # None (ошибка API / битый JSON / max_tokens) — item пропускаем,
        # предыдущий риск по тикеру не трогаем (fail-closed).
        if assessment is None:
            self.stats.analysis_failed += 1
            logger.warning(
                "NewsRiskPipeline: оценка по %s (%s) недоступна — предыдущий риск сохранён",
                item.item_id, primary,
            )
            return calls_used, produced

        for ticker in matched_tickers:
            per_ticker = (
                assessment if ticker == primary
                else dataclasses.replace(assessment, ticker=ticker)
            )
            if ticker != primary:
                self.stats.reused_across_tickers += 1

            self.memory.add(ticker, per_ticker, title=item.title, source=item.source)

            # Публикуем агрегат по всей живой TTL-памяти, а не последнюю новость.
            aggregated = self.memory.build_aggregate(ticker) or per_ticker
            produced[ticker] = aggregated

            if aggregated.action in ("dampen", "block"):
                self.stats.dampened_by_risk += 1

            if publish:
                try:
                    self.on_assessment(ticker, aggregated)
                except Exception as exc:
                    logger.warning("NewsRiskPipeline: on_assessment(%s) упал: %s", ticker, exc)

            logger.info(
                "NewsRiskPipeline | %s | %s | category=%s severity=%.2f materiality=%.2f "
                "effective=%.2f (по %d живым) horizon=%s(ttl=%.1fd) action=%s | %s",
                ticker, item.source, aggregated.risk_category, aggregated.severity,
                aggregated.materiality, aggregated.effective_risk, aggregated.active_count,
                aggregated.horizon, self.memory._ttl_for(aggregated),
                aggregated.action, item.title[:80],
            )

        return calls_used, produced

    # ── циклы ────────────────────────────────────────────────────────

    def _handle_new_items(self, items: List[RawNewsItem]) -> None:
        # top-N самых релевантных в пределах лимита LLM-вызовов
        candidates = self._prepare_candidates(items)
        calls_used = 0
        for _score, item, matched in candidates:
            calls_used, _ = self._process_item(item, calls_used, publish=True, matched_tickers=matched)

        if self.stats.skipped_by_llm_cap:
            logger.warning(
                "NewsRiskPipeline: достигнут лимит %d LLM-вызовов за цикл, "
                "часть новостей отложена (всего пропущено: %d)",
                self.max_llm_calls_per_cycle, self.stats.skipped_by_llm_cap,
            )

    def backfill(self, items: List[RawNewsItem]) -> Dict[str, NewsRiskAssessment]:
        """
        Разовый прогон исторических новостей ПЕРЕД первым price-тиком —
        тоже через top-N по релевантности, чтобы cold-start не выжигал
        лимит на мусор из старых фидов.
        """
        latest: Dict[str, NewsRiskAssessment] = {}
        candidates = self._prepare_candidates(items)
        calls_used = 0
        for _score, item, matched in candidates:
            calls_used, produced = self._process_item(item, calls_used, publish=False, matched_tickers=matched)
            for ticker, assessment in produced.items():
                latest[ticker] = assessment
                logger.info(
                    "NewsRiskPipeline[backfill] | %s | category=%s effective=%.2f horizon=%s | %s",
                    ticker, assessment.risk_category, assessment.effective_risk,
                    assessment.horizon, item.title[:80],
                )
        return latest

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Фоновый поток сборщика RSS — не блокирует основной event loop."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self.collector.run_forever,
            args=(self._handle_new_items,),
            daemon=True,
            name="news-risk-pipeline",
        )
        self._thread.start()
        logger.info("NewsRiskPipeline запущен (poll_interval=%ds)", self.collector.cfg.poll_interval_sec)

    def stop(self) -> None:
        self.collector.stop()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
