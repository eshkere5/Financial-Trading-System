"""Лаборатория для проверки эффективности шлюза: управляемый контекст, размеченные сценарии, режимы «только правила» и «аналитика».

Контекст рынка (цена, режим, возраст данных, позиция) задаётся сценарием, поэтому проверка не зависит от сети и случайных свечей.
Разметка «плохая/хорошая» делается при построении сценария (тип изъяна известен заранее), а не по ответу шлюза.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Optional

from finvex.core.models import MSK
from finvex.gateway.context import PaperPortfolio
from finvex.gateway.factory import DemoClock
from finvex.gateway.gateway import Gateway, GatewayError
from finvex.gateway.prices import StaticPriceSource

MONDAY_NOON = datetime(2026, 10, 26, 12, 0, tzinfo=MSK)       # понедельник, сессия MOEX идёт
SUNDAY_EVENING = datetime(2026, 10, 25, 21, 0, tzinfo=MSK)    # воскресенье, акции закрыты
REGIMES = ("CRASH", "RALLY_FRENZY", "RECOVERY", "BEAR_TREND", "BULL_TREND", "HIGH_VOL_CHOP", "RANGE_BOUND", "RANGE_SQUEEZE", "TRANSITION")
BLOCKED_REGIMES = {"CRASH", "RALLY_FRENZY"}
LIMITS = dict(max_order_value=200_000, max_position_pct=20, price_deviation=3, data_freshness=180, min_confidence=0.55, max_rationale=1500)
SEVERITY = {"allow": 0, "revise": 1, "deny": 2}

# (тикер, ориентир цены, тип)
ASSETS = [("SBER", 300.0, "stock"), ("GAZP", 120.0, "stock"), ("LKOH", 6000.0, "stock"), ("VTBR", 85.0, "stock"), ("MTSS", 230.0, "stock"),
          ("YNDX", 4200.0, "stock"), ("GMKN", 130.0, "stock"), ("BTCUSDT", 95000.0, "crypto"), ("ETHUSDT", 3400.0, "crypto"), ("SOLUSDT", 160.0, "crypto")]
GOOD_RATIONALE = "Режим подтверждает идею, риск ограничен стопом под ближайшим минимумом, объём небольшой."


@dataclass
class Scenario:
    symbol: str = "SBER"
    kind: str = "stock"
    side: str = "buy"
    qty: float = 10
    price: float = 300.0
    regime: str = "BULL_TREND"
    age_min: float = 5.0
    position: float = 0.0
    equity: float = 1_000_000.0
    when: datetime = MONDAY_NOON
    confidence: Optional[float] = 0.7
    rationale: str = GOOD_RATIONALE
    order_type: str = "market"
    limit_price: Optional[float] = None
    news: Optional[float] = None
    kronos: Optional[float] = None
    known: bool = True
    label: str = "good"           # good | bad
    flaw: str = "none"            # тип изъяна, известен при построении
    detector: str = "rules"       # rules | analytics: чем изъян должен ловиться
    expect_min: str = "allow"     # минимально допустимая строгость: allow | revise | deny


class Lab:
    def __init__(self, mode: str = "rules", rules: Optional[dict] = None) -> None:
        assert mode in ("rules", "analytics")
        self.mode, self.clock = mode, DemoClock(MONDAY_NOON)
        self.gw = Gateway(None, StaticPriceSource(), PaperPortfolio(), clock=self.clock)
        self.gw.demo = True
        self.sc: Optional[Scenario] = None
        self.gw.context.build = self._build
        r = {k: dict(v) for k, v in self.gw.rules.items()}
        if mode == "analytics":
            for k in ("news_sentiment", "kronos_view"):
                r[k]["value"] = {**r[k]["value"], "enforce": True}
        for k, v in (rules or {}).items():
            r[k] = {**r[k], **v}
        self.gw.set_rules(r)
        self._n = 0

    def _build(self, symbol: str) -> dict:
        s = self.sc
        if not s.known:
            raise RuntimeError(f"нет данных по {symbol}")
        return {"symbol": symbol, "price": s.price, "price_time": (s.when - timedelta(minutes=s.age_min + 60)).isoformat(timespec="minutes"),
                "data_age_min": s.age_min, "regime": s.regime, "adx": 25.0, "position_qty": s.position, "equity": s.equity,
                "cash": s.equity, "positions": {symbol: s.position}, "extras": {}}

    def decide(self, s: Scenario) -> dict:
        """Возвращает вердикт; ошибка формата заявки превращается в вердикт deny с кодом input."""
        self._n += 1
        self.sc, self.clock.t = s, s.when
        ag, _ = self.gw.agents.create(f"lab{self._n}", daily_quota=10**6)
        self.gw.analytics._d.clear()
        if s.news is not None:
            self.gw.analytics.add_news(s.symbol, {"id": f"n{self._n}", "score": s.news, "category": "company", "ttl_hours": 10})
        if s.kronos is not None:
            self.gw.analytics.set_kronos(s.symbol, s.kronos, (s.when + timedelta(hours=3)).isoformat())
        raw = dict(client_id=f"c{self._n}", symbol=s.symbol, side=s.side, quantity=s.qty, order_type=s.order_type, limit_price=s.limit_price,
                   confidence=s.confidence, rationale=s.rationale)
        try:
            return self.gw.propose(ag, raw)
        except GatewayError as exc:
            return {"decision": "deny", "codes": ["input"], "reasons": [str(exc)], "suggested_quantity": None, "intent": raw}


def sev(v: dict) -> int:
    return SEVERITY[v["decision"]]


# ── генерация размеченной популяции ────────────────────────────────────────

def _qty_for(price: float, notional: float, kind: str, rng: random.Random) -> float:
    q = notional / price
    if kind == "crypto":
        return max(round(q, 4), 0.0001)
    return float(max(int(q), 1))


def good(rng: random.Random, i: int) -> Scenario:
    sym, p0, kind = rng.choice(ASSETS)
    price = p0 * rng.uniform(0.7, 1.4)
    notional = rng.uniform(0.02, 0.5) * LIMITS["max_order_value"]
    qty = _qty_for(price, notional, kind, rng)
    while qty * price > 0.9 * LIMITS["max_order_value"]:
        qty = _qty_for(price, qty * price * 0.5, kind, rng)
    regime = rng.choice([r for r in REGIMES if r not in BLOCKED_REGIMES])
    return Scenario(symbol=sym, kind=kind, qty=qty, price=price, regime=regime, age_min=rng.uniform(0, 60), confidence=rng.uniform(0.6, 0.95),
                    news=rng.choice([None, None, 0.4, 0.1, -0.1]), kronos=rng.choice([None, None, 0.3, -0.1]))


FLAWS = {  # изъян: (кто ловит, минимальная строгость)
    "oversize": ("rules", "revise"), "concentration": ("rules", "revise"), "off_session": ("rules", "revise"), "crash_buy": ("rules", "deny"),
    "frenzy_buy": ("rules", "deny"), "stale_data": ("rules", "revise"), "price_typo": ("rules", "revise"), "short_sell": ("rules", "revise"),
    "hallucinated_symbol": ("rules", "deny"), "low_confidence": ("rules", "revise"), "garbage_quantity": ("rules", "deny"),
    "bad_news": ("analytics", "deny"), "bearish_kronos": ("analytics", "deny"), "mild_negative_news": ("analytics", "revise"),
}


def flawed(rng: random.Random, i: int, flaw: str) -> Scenario:
    s = good(rng, i)
    s.news = s.kronos = None
    s.label, s.flaw = "bad", flaw
    s.detector, s.expect_min = FLAWS[flaw]
    if flaw == "oversize":
        s.qty = _qty_for(s.price, rng.uniform(1.5, 20) * LIMITS["max_order_value"], s.kind, rng)
    elif flaw == "concentration":
        s.equity = 300_000.0
        s.qty = _qty_for(s.price, rng.uniform(0.25, 0.55) * s.equity, s.kind, rng)
        if s.qty * s.price > LIMITS["max_order_value"]:
            s.qty = _qty_for(s.price, 0.9 * LIMITS["max_order_value"], s.kind, rng)
            s.equity = s.qty * s.price * 3.0
    elif flaw == "off_session":
        s.kind, s.symbol, s.price = "stock", "SBER", 300.0
        s.qty = 10.0
        s.when = rng.choice([SUNDAY_EVENING, MONDAY_NOON.replace(hour=22), MONDAY_NOON.replace(hour=7), MONDAY_NOON + timedelta(days=5, hours=1)])
    elif flaw == "crash_buy":
        s.regime = "CRASH"
    elif flaw == "frenzy_buy":
        s.regime = "RALLY_FRENZY"
    elif flaw == "stale_data":
        s.age_min = rng.uniform(400, 3000)
    elif flaw == "price_typo":
        s.order_type, s.limit_price = "limit", s.price * rng.choice([10, 0.1, 1.2, 0.8])
    elif flaw == "short_sell":
        s.side, s.position = "sell", 0.0
    elif flaw == "hallucinated_symbol":
        s.known, s.symbol = False, rng.choice(["ZZZZ", "SBERX", "MOONUSDT", "ТЕСЛА"])
    elif flaw == "low_confidence":
        s.confidence = rng.uniform(0.05, 0.4)
    elif flaw == "garbage_quantity":
        s.qty = rng.choice([-5, 0, float("nan"), float("inf")])
    elif flaw == "bad_news":
        s.news = rng.uniform(-0.95, -0.6)
    elif flaw == "bearish_kronos":
        s.kronos = rng.uniform(-0.7, -0.55)
    elif flaw == "mild_negative_news":
        s.news = rng.uniform(-0.45, -0.25)
    return s


def population(n_good: int = 600, per_flaw: int = 60, seed: int = 7) -> list:
    rng = random.Random(seed)
    out = [good(rng, i) for i in range(n_good)]
    for f in FLAWS:
        out += [flawed(rng, i, f) for i in range(per_flaw)]
    rng.shuffle(out)
    return out


def run_population(pop: list, mode: str) -> list:
    lab = Lab(mode)
    return [(s, lab.decide(s)) for s in pop]


def score(results: list) -> dict:
    goods = [(s, v) for s, v in results if s.label == "good"]
    bads = [(s, v) for s, v in results if s.label == "bad"]
    by = {}
    for s, v in bads:
        d = by.setdefault(s.flaw, {"n": 0, "stopped": 0, "denied": 0})
        d["n"] += 1
        d["stopped"] += v["decision"] != "allow"
        d["denied"] += v["decision"] == "deny"
    fb = sum(v["decision"] != "allow" for _, v in goods)
    return {"good_n": len(goods), "false_blocks": fb, "false_block_rate": fb / max(len(goods), 1), "bad_n": len(bads),
            "bad_stopped": sum(v["decision"] != "allow" for _, v in bads), "catch_rate": sum(v["decision"] != "allow" for _, v in bads) / max(len(bads), 1),
            "by_flaw": {k: {**d, "rate": d["stopped"] / d["n"]} for k, d in by.items()}}
