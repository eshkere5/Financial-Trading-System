"""
keyword_filter.py — дешёвый локальный фильтр релевантности новостей.

watch_terms (ключевые слова, персоны, синонимы компаний) генерируются ОДИН
РАЗ при старте торгов (KronosDefaultStrategy.seed_watch_terms / AliasGenerator),
а не по таймеру. Этот модуль применяет их к каждому новому заголовку локально
(0 токенов) и решает, стоит ли передавать новость в полный DeepSeek-анализ
(NewsDeepDiveAnalyzer).

Два режима:
  - keyword: совпадение по границам слова (мгновенно, без зависимостей)
  - embedding: семантическое сходство через маленькую sentence-transformer
    модель (ловит смысл даже без точного совпадения слов)

"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Tuple

logger = logging.getLogger(__name__)

# Термины не длиннее этого матчатся с учётом регистра: короткие тикеры
# ("VK", "T", "МТС") в нижнем регистре дают лавину ложных срабатываний.
_CASE_SENSITIVE_MAX_LEN = 3


@dataclass
class WatchTermsStore:
    """
    Живой словарь ticker -> [ключевые слова/персоны/синонимы].
    Заполняется ОДНОКРАТНО при старте торгов (см. LiveEngine._run_async ->
    KronosDefaultStrategy.seed_watch_terms -> WatchTermsStore.update()).
    Больше не обновляется периодически из strategist_loop.
    """
    terms: Dict[str, List[str]] = field(default_factory=dict)
    updated_at: str = ""
    ### FIXED 2026-08-01 (С-6) ###
    # Счётчик версий: KeywordFilter кэширует скомпилированные регулярки и
    # должен уметь понять, что словарь подменили.
    version: int = 0

    def update(self, new_terms: Dict[str, List[str]], timestamp: str) -> None:
        ### FIXED 2026-08-01 (С-7) ###
        # Было: `if new_terms:` — пустой словарь молча игнорировался. Если
        # seed_watch_terms падал или LLM возвращал пустоту, фильтр оставался
        # пустым и весь новостной контур тихо не работал: ни одной новости
        # не матчилось, ни одной ошибки в логах.
        if not new_terms:
            logger.warning(
                "WatchTermsStore.update: получен пустой набор watch_terms (timestamp=%s) — "
                "предыдущие термины сохранены; новостной фильтр работает на старом словаре "
                "(%d тикеров). Проверьте seed_watch_terms/AliasGenerator.",
                timestamp, len(self.terms),
            )
            return

        self.terms = new_terms
        self.updated_at = timestamp
        self.version += 1
        logger.info(
            "WatchTermsStore: обновлено %d тикеров, всего %d терминов",
            len(self.terms), sum(len(v) for v in self.terms.values()),
        )

    def all_tickers(self) -> List[str]:
        return list(self.terms.keys())


### FIXED 2026-08-01 (С-6) ###
# Было: `term.lower() in text` — подстрочное совпадение. Тикер "VK" матчил
# "вклад"/"Volkswagen", "T" матчил вообще всё, "ГАЗ" матчил "газопровод" и
# "Газпром", "ROSN" — любую латиницу с этой последовательностью. Новость
# уезжала в DeepSeek под чужим тикером, и её риск-оценка применялась к
# позиции, к которой новость не имеет отношения.
# Стало: совпадение по границам слова. Lookaround (?<!\w)/(?!\w) вместо \b —
# корректно для многословных терминов и для кириллицы. Короткие термины
# (<= 3 символов) матчатся с учётом регистра, чтобы "VK" не ловил "вклад".
def _compile_term(term: str) -> Optional[Pattern[str]]:
    term = (term or "").strip()
    if not term:
        return None
    flags = re.UNICODE
    if len(term) > _CASE_SENSITIVE_MAX_LEN:
        flags |= re.IGNORECASE
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", flags)


class KeywordFilter:
    """Совпадение по границам слова — самый дешёвый уровень фильтра (0 токенов)."""

    def __init__(self, store: WatchTermsStore) -> None:
        self.store = store
        self._patterns: Dict[str, List[Pattern[str]]] = {}
        self._patterns_version: int = -1
        self._warned_empty = False

    def _ensure_patterns(self) -> Dict[str, List[Pattern[str]]]:
        if self._patterns_version == self.store.version:
            return self._patterns
        compiled: Dict[str, List[Pattern[str]]] = {}
        for ticker, terms in self.store.terms.items():
            patterns = [p for p in (_compile_term(t) for t in terms or []) if p is not None]
            if patterns:
                compiled[ticker] = patterns
        self._patterns = compiled
        self._patterns_version = self.store.version
        return compiled

    def match(self, headline: str, summary: str = "") -> List[str]:
        """Возвращает список тикеров, релевантных данному заголовку/сводке."""
        patterns = self._ensure_patterns()

        ### FIXED 2026-08-01 (С-7) ###
        # Пустой словарь = фильтр отсекает вообще всё. Раньше это выглядело
        # как «новостей нет»; теперь предупреждаем один раз явно.
        if not patterns:
            if not self._warned_empty:
                logger.warning(
                    "KeywordFilter: watch_terms пуст — ни одна новость не пройдёт фильтр, "
                    "новостной риск-контур фактически отключён"
                )
                self._warned_empty = True
            return []
        self._warned_empty = False

        text = f"{headline} {summary}"
        return [
            ticker for ticker, term_patterns in patterns.items()
            if any(p.search(text) for p in term_patterns)
        ]


class EmbeddingFilter:
    """
    Семантический фильтр через маленькую локальную embedding-модель.
    Ловит случаи вроде "госбанк №1" -> SBER, даже без точного совпадения строк.
    Ленивая загрузка модели — не тратит память, если используется только KeywordFilter.
    """

    def __init__(self, store: WatchTermsStore, model_name: str = "intfloat/multilingual-e5-small",
                 threshold: float = 0.62) -> None:
        self.store = store
        self.model_name = model_name
        self.threshold = threshold
        self._model = None
        self._term_cache: Dict[str, "object"] = {}
        self._load_failed = False
        self._warned_empty = False

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        ### FIXED 2026-08-01 (М-8) ###
        # Было: ловился только ImportError. Реальная загрузка e5-модели
        # ходит в сеть за весами и может упасть чем угодно (OSError,
        # HTTPError, нехватка памяти) — исключение улетало в поток
        # RSS-коллектора и убивало весь новостной конвейер молча.
        if self._load_failed:
            return False
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            return True
        except ImportError:
            self._load_failed = True
            logger.warning(
                "EmbeddingFilter: sentence-transformers не установлен, "
                "используйте KeywordFilter или `pip install sentence-transformers`"
            )
            return False
        except Exception as exc:
            self._load_failed = True
            logger.error(
                "EmbeddingFilter: не удалось загрузить модель %s (%s) — "
                "семантический фильтр отключён, новости не будут матчиться",
                self.model_name, exc,
            )
            return False

    def _term_embedding(self, term: str):
        if term not in self._term_cache:
            self._term_cache[term] = self._model.encode(term, normalize_embeddings=True)
        return self._term_cache[term]

    def match(self, headline: str, summary: str = "") -> List[str]:
        if not self._ensure_model():
            return []
        import numpy as np

        ### FIXED 2026-08-01 (С-7) ###
        if not self.store.terms:
            if not self._warned_empty:
                logger.warning(
                    "EmbeddingFilter: watch_terms пуст — новостной риск-контур фактически отключён"
                )
                self._warned_empty = True
            return []
        self._warned_empty = False

        text = f"{headline} {summary}".strip()
        if not text:
            return []
        h_emb = self._model.encode(text, normalize_embeddings=True)

        matched: List[str] = []
        for ticker, terms in self.store.terms.items():
            if not terms:
                continue
            sims = [float(np.dot(h_emb, self._term_embedding(t))) for t in terms if t]
            if sims and max(sims) >= self.threshold:
                matched.append(ticker)
        return matched

    def score(self, ticker: str, headline: str, summary: str = "") -> float:
        """Возвращает максимальное косинусное сходство заголовка с термами тикера, без порога."""
        if not self._ensure_model():
            return 0.0
        import numpy as np

        terms = self.store.terms.get(ticker, [])
        if not terms:
            return 0.0
        text = f"{headline} {summary}".strip()
        if not text:
            return 0.0
        h_emb = self._model.encode(text, normalize_embeddings=True)
        sims = [float(np.dot(h_emb, self._term_embedding(t))) for t in terms if t]
        return max(sims) if sims else 0.0

    def rank_top_k(self, ticker: str, items: List, k: int = 5) -> List:
        """
        Ранжирует список RawNewsItem по релевантности тикеру и возвращает top-k.
        Используется на холодном старте: вместо прогонки всего RSS-бэклога за
        N дней через DeepSeek, малая e5-модель сама отбирает k самых релевантных
        новостей по тикеру, и только они уходят на полный анализ.
        """
        if not self._ensure_model() or not items:
            return []
        scored = [(self.score(ticker, it.title, it.summary), it) for it in items]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [it for score, it in scored[:k] if score > 0.0]


def build_filter(store: WatchTermsStore, mode: str = "keyword", **kwargs) -> "KeywordFilter | EmbeddingFilter":
    """Фабрика: mode='keyword' (дефолт, 0 токенов) | mode='embedding' (слабая модель)."""
    if mode == "embedding":
        return EmbeddingFilter(store, **kwargs)
    return KeywordFilter(store)
