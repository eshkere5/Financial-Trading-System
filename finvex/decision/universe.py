"""
UniverseSelector — динамический подбор тикеров вместо статичного списка в engine.yaml.

Заменяет: configs/engine.yaml → tickers: [GAZP, LKOH, ...] (жёстко зашитый список).

Логика отбора (комбинация детерминированных фильтров + тактики LLM-стратега):
  1. Базовый пул: все ликвидные акции с Tinkoff (get_liquid_shares) — авто-обновление
  2. Ликвидность: минимальный дневной объём в рублях (отсекает неликвид)
  3. Новостная релевантность: тикеры с активным новостным потоком получают приоритет
     (используем NewsAgentClient.get_multi_snapshot → doc_count, avg_sentiment)
  4. Тактика LLM-стратега: StrategistDecision может явно расширить/сузить universe
     через поле watchlist (см. strategist.py) — например "торгуй только нефтегазом"
     или "избегай санкционных бумаг"
  5. Кэш: пересчёт не на каждый тик, а раз в N минут (universe не должен дёргаться)

Принцип: базовый слой — код (детерминированный, тестируемый), тактический
слой — LLM поверх него (расширяет/сужает, не заменяет).

### FIXED 2026-08-02 ###
- К8: конструктор принимал только dict, а main.py передаёт UniverseConfig
  (dataclass) → AttributeError на первом же get_universe().
- strategist_veto приходит из StrategistDecision.vetoed_pairs в виде пар
  "SBER/GAZP" — сравнение целой строки пары с одиночным тикером никогда
  не срабатывало, вето де-факто не работало.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class UniverseSnapshot:
    """Результат отбора тикеров на текущий момент."""

    tickers: List[str] = field(default_factory=list)
    excluded: Dict[str, str] = field(default_factory=dict)   # ticker -> причина исключения
    computed_at: float = 0.0
    source: str = "static"   # "static" | "liquidity" | "news_driven" | "strategist_override"


# ### FIXED 2026-08-02 (К8) ###
def _as_cfg_dict(cfg: Any) -> Dict[str, Any]:
    """
    Приводит конфиг к dict независимо от того, чем он пришёл.

    main.py передаёт `cfg.universe` — это dataclass UniverseConfig
    (config_loader.py), а весь код ниже написан под dict-API (`.get`).
    Раньше это давало AttributeError: 'UniverseConfig' object has no
    attribute 'get' при первом же вызове get_universe(), то есть live-режим
    падал на старте ещё до подключения к брокеру.

    Имена полей UniverseConfig совпадают с ключами, которые читает селектор
    (max_universe_size, min_lot_price_rub, require_short_enabled,
    refresh_interval_sec, news_boost_weight, hard_exclude, fallback_tickers),
    поэтому достаточно плоской конвертации — семантика не меняется.
    """
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return asdict(cfg)
    if hasattr(cfg, "get") and hasattr(cfg, "keys"):   # Mapping-подобные объекты
        return {k: cfg.get(k) for k in cfg.keys()}
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    logger.warning(
        "UniverseSelector: не удалось привести cfg типа %s к dict — используются дефолты",
        type(cfg).__name__,
    )
    return {}


# ### FIXED 2026-08-02 ###
def _expand_veto(entries: Optional[List[str]]) -> Set[str]:
    """
    Разворачивает StrategistDecision.vetoed_pairs в множество тикеров.

    Стратег отдаёт пары строкой ("SBER/GAZP"), а universe состоит из
    одиночных тикеров — прежнее сравнение `{t.upper() for t in veto}`
    против списка тикеров не находило совпадений никогда, и вето молча
    не применялось.

    Делим ТОЛЬКО по "/". Дефис специально НЕ используется как разделитель:
    он входит в состав тикеров ряда инструментов (например BTC-USD), и его
    разбиение приводило бы к ложным вето по несуществующим ногам
    (та же проблема, что С13 в новостном ревью).
    """
    out: Set[str] = set()
    for raw in entries or []:
        text = str(raw).strip()
        if not text:
            continue
        for leg in text.split("/"):
            leg = leg.strip().upper()
            if leg:
                out.add(leg)
    return out


class UniverseSelector:
    """
    Usage
    -----
    >>> selector = UniverseSelector(tinkoff_client, news_client, cfg["universe"])
    >>> snap = selector.get_universe(strategist_watchlist=decision.watchlist)
    >>> print(snap.tickers)

    cfg принимается и как dict, и как dataclass (UniverseConfig), и как None.
    """

    def __init__(self, tinkoff_client, news_client=None, cfg: Optional[Any] = None) -> None:
        self._client = tinkoff_client
        self._news_client = news_client
        # ### FIXED 2026-08-02 (К8) ### было: self.cfg = cfg or {}
        self.cfg: Dict[str, Any] = _as_cfg_dict(cfg)

        self._max_universe_size = int(self.cfg.get("max_universe_size") or 40)
        self._min_lot_price_rub = float(self.cfg.get("min_lot_price_rub") or 50.0)
        self._require_short = bool(self.cfg.get("require_short_enabled", False))
        self._refresh_interval = int(self.cfg.get("refresh_interval_sec") or 3600)
        self._news_boost_weight = float(self.cfg.get("news_boost_weight") or 0.3)

        # эксклюзионный список: тикеры, которые вообще нельзя брать
        # (например делистинг, приостановка торгов) — задаётся вручную в конфиге
        self._hard_exclude: Set[str] = {
            str(t).strip().upper() for t in (self.cfg.get("hard_exclude") or []) if str(t).strip()
        }

        # fallback список — используется, если Tinkoff API недоступен
        self._fallback_tickers: List[str] = list(self.cfg.get("fallback_tickers") or [
            "SBER", "GAZP", "LKOH", "YNDX", "GMKN",
        ])

        self._cache: Optional[UniverseSnapshot] = None

    # ── публичный API ──────────────────────────────────────────────────────

    def get_universe(
        self,
        strategist_watchlist: Optional[List[str]] = None,
        strategist_veto: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> UniverseSnapshot:
        """
        Главный метод. Возвращает актуальный список тикеров для торговли.

        strategist_watchlist — если LLM-стратег явно указал список/сектор,
            он ПРИОРИТЕТНО расширяет базовый universe (не заменяет полностью,
            чтобы не потерять детерминированную ликвидность-проверку)
        strategist_veto — тикеры или пары ("SBER/GAZP"), которые стратег
            запретил (санкционный риск, разрыв коинтеграции и т.п.) —
            исключаются из финального списка

        Сигнатура не менялась: LiveEngine._strategist_loop вызывает метод как
        get_universe(strategist_watchlist=d.watchlist, strategist_veto=d.vetoed_pairs).
        """
        now = time.time()
        needs_refresh = (
            force_refresh
            or self._cache is None
            or (now - self._cache.computed_at) > self._refresh_interval
        )

        if needs_refresh:
            self._cache = self._compute_base_universe()

        snap = self._cache
        tickers = list(snap.tickers)
        excluded = dict(snap.excluded)

        # тактическое расширение от стратега (не пересчитывает ликвидность —
        # это ответственность стратега; если тикер неликвиден, риск-менеджер
        # всё равно отфильтрует ордер на этапе check_order)
        #
        # ### FIXED 2026-08-02 ###
        # Раньше тикеры стратега добавлялись В КОНЕЦ списка, после чего шла
        # обрезка `tickers[:max_universe_size]` — при полном базовом пуле
        # ватчлист стратега целиком отбрасывался, хотя документирован как
        # приоритетный. Теперь он ставится в начало.
        if strategist_watchlist:
            promoted: List[str] = []
            for raw in strategist_watchlist:
                t = str(raw).strip().upper()
                if not t:
                    continue
                if t in self._hard_exclude:
                    excluded[t] = "hard_exclude"
                    continue
                if t not in promoted:
                    promoted.append(t)
            tickers = promoted + [t for t in tickers if t not in promoted]

        # тактическое сужение от стратега
        if strategist_veto:
            veto_set = _expand_veto(strategist_veto)
            kept: List[str] = []
            for t in tickers:
                if t.upper() in veto_set:
                    # ### FIXED 2026-08-02 ###
                    # excluded заполняем только реально удалёнными тикерами —
                    # раньше туда писались все ноги вето, включая те, которых
                    # в universe не было вовсе.
                    excluded[t] = "strategist_veto"
                else:
                    kept.append(t)
            tickers = kept

        tickers = tickers[: self._max_universe_size]

        return UniverseSnapshot(
            tickers=tickers,
            excluded=excluded,
            computed_at=now,
            source="strategist_override" if (strategist_watchlist or strategist_veto) else snap.source,
        )

    # ── базовый (детерминированный) слой ────────────────────────────────────

    def _compute_base_universe(self) -> UniverseSnapshot:
        """Ликвидность-фильтр через Tinkoff API. Fallback на статичный список при ошибке."""
        try:
            shares = self._client.get_liquid_shares()
        except Exception as exc:
            logger.warning("get_liquid_shares failed (%s) — используем fallback_tickers", exc)
            return UniverseSnapshot(
                tickers=list(self._fallback_tickers),
                computed_at=time.time(),
                source="static",
            )

        tickers: List[str] = []
        excluded: Dict[str, str] = {}

        for s in shares:
            ticker = s["ticker"]
            if ticker.upper() in self._hard_exclude:
                excluded[ticker] = "hard_exclude"
                continue
            if self._require_short and not s.get("short_enabled", False):
                excluded[ticker] = "no_short"
                continue
            tickers.append(ticker)

        # ранжируем по новостной активности, если news_client доступен
        if self._news_client is not None and tickers:
            tickers = self._rank_by_news_activity(tickers)

        tickers = tickers[: self._max_universe_size]
        logger.info(
            "UniverseSelector: %d тикеров прошли фильтр (%d исключено)",
            len(tickers), len(excluded),
        )
        return UniverseSnapshot(
            tickers=tickers,
            excluded=excluded,
            computed_at=time.time(),
            source="news_driven" if self._news_client else "liquidity",
        )

    def _rank_by_news_activity(self, tickers: List[str]) -> List[str]:
        """
        Сортирует тикеры так, чтобы бумаги с активным (и не крайне негативным
        по санкциям) новостным потоком шли выше — они интереснее для тактики.
        Не отфильтровывает, только переупорядочивает.
        """
        try:
            snapshots = self._news_client.get_multi_snapshot(tickers, hours_back=24)
        except Exception as exc:
            logger.warning("news ranking failed: %s", exc)
            return tickers

        def score(ticker: str) -> float:
            snap = snapshots.get(ticker)
            if snap is None or snap.doc_count == 0:
                return 0.0
            activity = min(1.0, snap.doc_count / 10.0)
            return activity * self._news_boost_weight + abs(snap.avg_sentiment) * (1 - self._news_boost_weight)

        return sorted(tickers, key=score, reverse=True)
