"""Свойства шлюза на тысячах случайных заявок. Условия безопасности пересказаны в тесте независимо от кода правил."""
import math
import random
from datetime import timedelta

import pytest

from .harness import ASSETS, BLOCKED_REGIMES, LIMITS, MONDAY_NOON, REGIMES, SEVERITY, Lab, Scenario, sev

N = 2500


def rand_scenario(rng: random.Random) -> Scenario:
    sym, p0, kind = rng.choice(ASSETS + [("ZZZZ", 100.0, "stock")])
    price = math.exp(rng.uniform(math.log(0.01), math.log(2e6)))
    side = rng.choice(["buy", "buy", "sell"])
    pos = rng.choice([0.0, 0.0, rng.uniform(0, 5000)])
    lp = price * rng.choice([1.0, 1.01, 1.05, 0.9, 10]) if rng.random() < 0.3 else None
    return Scenario(symbol=sym, kind=kind, side=side, qty=math.exp(rng.uniform(math.log(1e-3), math.log(1e6))), price=price, regime=rng.choice(REGIMES),
                    age_min=rng.choice([0, 5, 100, 179, 181, 1000]), position=pos, equity=rng.choice([1e5, 1e6, 1e7]),
                    when=MONDAY_NOON + timedelta(hours=rng.randint(-14, 40), minutes=rng.choice([0, 30, 41])),
                    confidence=rng.choice([None, 0.1, 0.54, 0.55, 0.9]), order_type="limit" if lp else "market", limit_price=lp,
                    news=rng.choice([None, None, -0.9, -0.3, 0.5]), kronos=rng.choice([None, None, -0.6, 0.4]), known=sym != "ZZZZ" and rng.random() > 0.02)


def session_open(s: Scenario) -> bool:
    if s.kind == "crypto":
        return True
    t = s.when
    return t.weekday() < 5 and (t.hour, t.minute) >= (10, 0) and (t.hour, t.minute) <= (18, 40)


def violations(s: Scenario, mode: str) -> list:
    """Что нарушает разрешённая заявка, по условиям из описания правил. Пустой список значит безопасно."""
    bad, ref = [], (s.limit_price if s.order_type == "limit" and s.limit_price else s.price)
    notional = s.qty * ref
    if not s.known: bad.append("unknown_symbol")
    if notional > LIMITS["max_order_value"] * (1 + 1e-9): bad.append("order_value")
    if s.side == "buy" and (s.position + s.qty) * ref > s.equity * LIMITS["max_position_pct"] / 100 * (1 + 1e-9): bad.append("position_share")
    if s.side == "sell" and s.qty > s.position + 1e-9: bad.append("short")
    if not session_open(s): bad.append("session")
    if s.side == "buy" and s.regime in BLOCKED_REGIMES: bad.append("regime")
    if session_open(s) and s.age_min > LIMITS["data_freshness"]: bad.append("stale")
    if s.confidence is not None and s.confidence < LIMITS["min_confidence"]: bad.append("confidence")
    if s.order_type == "limit" and abs(s.limit_price / s.price - 1) * 100 > LIMITS["price_deviation"] * (1 + 1e-9): bad.append("price_deviation")
    if mode == "analytics" and s.side == "buy":
        if s.news is not None and s.news <= -0.2: bad.append("news")
        if s.kronos is not None and s.kronos <= -0.2: bad.append("kronos")
    return bad


@pytest.mark.parametrize("mode", ["rules", "analytics"])
def test_nothing_unsafe_is_ever_allowed(mode):
    rng, lab, allowed = random.Random(11), Lab(mode), 0
    for _ in range(N):
        s = rand_scenario(rng)
        v = lab.decide(s)
        if v["decision"] == "allow":
            allowed += 1
            assert not violations(s, mode), (violations(s, mode), s)
    assert allowed > 20   # набор не вырожден: часть заявок проходит


def test_analytics_mode_never_loosens_rules_only_decisions():
    rng, lr, la = random.Random(5), Lab("rules"), Lab("analytics")
    stricter = 0
    for _ in range(N):
        s = rand_scenario(rng)
        a, b = sev(lr.decide(s)), sev(la.decide(s))
        assert b >= a, s
        stricter += b > a
    assert stricter > 50   # аналитика реально добавляет остановки


@pytest.mark.parametrize("mode", ["rules", "analytics"])
def test_decisions_are_deterministic(mode):
    rng = random.Random(3)
    sc = [rand_scenario(rng) for _ in range(300)]
    assert [Lab(mode).decide(s)["decision"] for s in sc] == [Lab(mode).decide(s)["decision"] for s in sc]


def test_revise_suggestion_is_valid_and_maximal_for_stocks():
    lab, rng, checked = Lab("rules"), random.Random(9), 0
    for _ in range(600):
        sym, p0, kind = rng.choice([a for a in ASSETS if a[2] == "stock"])
        price = p0 * rng.uniform(0.5, 2)
        s = Scenario(symbol=sym, kind="stock", qty=math.floor(rng.uniform(1.2, 30) * LIMITS["max_order_value"] / price) + 1, price=price)
        v = lab.decide(s)
        if v["decision"] != "revise" or not v["suggested_quantity"]:
            continue
        q = v["suggested_quantity"]
        assert q * price <= LIMITS["max_order_value"] + 1e-6 and (q + 1) * price > LIMITS["max_order_value"] - 1e-6
        again = lab.decide(Scenario(**{**s.__dict__, "qty": q}))
        assert again["decision"] == "allow", (s, v["suggested_quantity"], again["codes"])
        checked += 1
    assert checked > 150


@pytest.mark.parametrize("mode", ["rules", "analytics"])
def test_severity_never_decreases_when_quantity_grows(mode):
    lab, rng = Lab(mode), random.Random(2)
    for _ in range(150):
        sym, p0, kind = rng.choice(ASSETS)
        base = Scenario(symbol=sym, kind=kind, price=p0 * rng.uniform(0.6, 1.6), regime=rng.choice([r for r in REGIMES if r not in BLOCKED_REGIMES]))
        prev = -1
        for q in [1e-4, 0.01, 1, 10, 100, 1e3, 1e4, 1e5, 1e6]:
            cur = sev(lab.decide(Scenario(**{**base.__dict__, "qty": q})))
            assert cur >= prev, (sym, q)
            prev = cur


def test_worse_news_never_loosens_the_decision_in_analytics_mode():
    lab = Lab("analytics")
    for sym, p0, kind in ASSETS:
        prev = -1
        for news in [0.9, 0.3, 0.0, -0.1, -0.19, -0.2, -0.35, -0.5, -0.8, -1.0]:
            cur = sev(lab.decide(Scenario(symbol=sym, kind=kind, price=p0, qty=1 if kind == "stock" else 0.01, news=news)))
            assert cur >= prev
            prev = cur
        assert prev == SEVERITY["deny"]
