"""Подающий процесс: настоящие модули Kronos и разбора новостей пишут оценки в хранилище аналитики шлюза.

Шлюз сам ничего не считает по рынку: он читает готовые оценки. Модуль превращает результаты проекта (`market_analysis.kronos_module`,
`market_analysis.news_module`) в записи хранилища с сроком жизни. Правило безопасности: оценки не выдумываются. Если Kronos в режиме
заглушки или вернул ошибку, а новость не оценена моделью, запись не создаётся (шлюз считает это нейтральным).

Срок жизни новости должен оценивать разбор новостей. Пока он не отдаёт `ttl_hours`, берётся срок по типу события из DEFAULT_TTL_H,
и запись помечается `[ttl: default]`. Оценка Kronos действует до закрытия следующей свечи его таймфрейма.
"""
from __future__ import annotations

import hashlib
import math
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from finvex.core.models import MSK

DEFAULT_TTL_H = {"rumor": 2.0, "company": 12.0, "earnings": 48.0, "macro": 24.0, "other": 6.0}
_KEYWORDS = (("earnings", ("earn", "отчёт", "отчет", "дивиден", "прибыл", "выручк")), ("macro", ("macro", "макро", "ставк", "санкц", "инфляц", "регулят")),
             ("rumor", ("rumor", "слух", "неподтвержд")), ("company", ("company", "компани", "менеджмент", "сделк", "авари", "иск")))
_TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}


def map_category(raw: Any) -> str:
    s = str(raw or "").lower()
    for cat, keys in _KEYWORDS:
        if any(k in s for k in keys):
            return cat
    return "other"


def kronos_score(kr: dict) -> Optional[float]:
    """Формула чат-агента: 0,7 * tanh(изменение / (2 * нормированная волатильность)). Статус не «ok» означает, что оценки нет."""
    if not isinstance(kr, dict) or kr.get("status") != "ok":
        return None
    try:
        change, voln = float(kr["change"]), float(kr["volatility_norm"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(change) and math.isfinite(voln)):
        return None
    return 0.7 * math.tanh(change / (2 * max(voln, 1e-4)))


def kronos_valid_until(now: datetime, tf: str) -> datetime:
    """Свеча, закрывшаяся последней, плюс один период: оценка живёт до закрытия следующей свечи."""
    if tf == "1d":   # дневные свечи привязаны к местной полуночи биржи
        return now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    step = _TF_SECONDS[tf]
    return datetime.fromtimestamp(math.floor(now.timestamp() / step) * step + step, tz=now.tzinfo)   # границы свечей по UTC


def _parse_time(v: Any, fallback: datetime) -> datetime:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=MSK)
    for fmt in (None, "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            t = datetime.fromisoformat(str(v)) if fmt is None else datetime.strptime(str(v), fmt)
            return t if t.tzinfo else t.replace(tzinfo=MSK)
        except (TypeError, ValueError):
            continue
    return fallback


class AnalyticsFeeder:
    def __init__(self, store, clock: Optional[Callable[[], datetime]] = None, kronos_source: Optional[Callable[[str, str], dict]] = None,
                 news_source: Optional[Callable[[str], List[dict]]] = None, tf: str = "1h") -> None:
        if tf not in _TF_SECONDS:
            raise ValueError(f"таймфрейм: одно из {sorted(_TF_SECONDS)}")
        self.store, self.clock, self.tf = store, clock or (lambda: datetime.now(MSK)), tf
        self.kronos_source, self.news_source = kronos_source, news_source

    def push_kronos(self, symbol: str, kr: dict) -> Optional[dict]:
        score = kronos_score(kr)
        if score is None:
            return None
        return self.store.set_kronos(symbol, round(score, 4), kronos_valid_until(self.clock(), self.tf).isoformat(), kr.get("volatility_norm"))

    def push_news(self, symbol: str, row: dict) -> Optional[dict]:
        if not isinstance(row, dict) or row.get("sentiment") is None:
            return None   # новость не оценена моделью: в решение не попадает
        now = self.clock()
        pub = _parse_time(row.get("time") or row.get("published_at"), now)
        cat = map_category(row.get("category") or row.get("risk_category"))
        ttl, source = row.get("ttl_hours"), "parser"
        if ttl is None:
            ttl, source = DEFAULT_TTL_H[cat], "default"
        if pub + timedelta(hours=float(ttl)) <= now:
            return None   # новость уже неактуальна
        nid = hashlib.sha1(f"{symbol}|{row.get('title', '')}|{pub.isoformat()}".encode("utf-8")).hexdigest()[:12]
        item = {"id": nid, "score": max(-1.0, min(1.0, float(row["sentiment"]))), "materiality": max(0.0, min(1.0, float(row.get("materiality", 1.0)))),
                "category": cat, "title": f"{row.get('title', '')} [ttl: {source}]"[:200], "published_at": pub.isoformat(), "ttl_hours": float(ttl)}
        return self.store.add_news(symbol, item)

    def tick(self, symbols: List[str]) -> Dict[str, Any]:
        """Один проход по бумагам. Ошибки источников собираются в отчёт и не останавливают остальные бумаги."""
        rep: Dict[str, Any] = {"kronos": 0, "news": 0, "skipped": [], "errors": []}
        for s in symbols:
            if self.kronos_source:
                try:
                    kr = self.kronos_source(s, self.tf)
                    if self.push_kronos(s, kr):
                        rep["kronos"] += 1
                    else:
                        rep["skipped"].append(f"{s}: Kronos без оценки ({(kr or {}).get('status')}: {str((kr or {}).get('error', ''))[:80]})")
                except Exception as exc:
                    rep["errors"].append(f"{s}: Kronos: {type(exc).__name__}: {str(exc)[:120]}")
            if self.news_source:
                try:
                    for row in self.news_source(s) or []:
                        if self.push_news(s, row):
                            rep["news"] += 1
                except Exception as exc:
                    rep["errors"].append(f"{s}: новости: {type(exc).__name__}: {str(exc)[:120]}")
        return rep


# ---- настоящие источники (нужны сеть, модель Kronos и ключ LLM; в закрытой среде не запускались) -----------------
def real_kronos_source() -> Callable[[str, str], dict]:
    def fn(symbol: str, tf: str) -> dict:
        from finvex import market_analysis as ma
        inst = ma.resolve_instrument({"symbol": symbol})
        if inst is None:
            return {"status": "error", "error": "инструмент не найден"}
        end = datetime.now(MSK)
        req = ma.Request(text=f"feeder {symbol}", instruments=[inst], start=end - timedelta(days=20 if tf != "1d" else 120), end=end, timeframe=tf)
        return ma.kronos_module(ma.add_indicators(ma.load_market_data(inst, req)), symbol, tf, 3)
    return fn


def real_news_source(hours_back: int = 12, max_llm: int = 8) -> Callable[[str], List[dict]]:
    def fn(symbol: str) -> List[dict]:
        from finvex import market_analysis as ma
        inst = ma.resolve_instrument({"symbol": symbol})
        if inst is None:
            return []
        end = datetime.now(MSK)
        req = ma.Request(text=f"feeder {symbol}", instruments=[inst], start=end - timedelta(hours=hours_back), end=end, timeframe="1h")
        out = ma.news_module(inst, req, True, max_llm)
        return out.get("items", []) if out.get("status") == "ok" else []
    return fn


def start_background(store, symbols: List[str], interval_s: float = 900.0, tf: str = "1h", say: Callable[[str], None] = print,
                     kronos_source=None, news_source=None) -> threading.Event:
    """Запускает подачу в потоке. Возвращает Event: set() останавливает поток."""
    stop = threading.Event()
    feeder = AnalyticsFeeder(store, kronos_source=kronos_source or real_kronos_source(), news_source=news_source or real_news_source(), tf=tf)

    def loop() -> None:
        while not stop.is_set():
            rep = feeder.tick(symbols)
            say(f"[feeder] Kronos {rep['kronos']}, новостей {rep['news']}, пропущено {len(rep['skipped'])}, ошибок {len(rep['errors'])}")
            for line in (rep["skipped"] + rep["errors"])[:5]:
                say(f"[feeder]   {line}")
            stop.wait(interval_s)

    threading.Thread(target=loop, name="analytics_feeder", daemon=True).start()
    return stop
