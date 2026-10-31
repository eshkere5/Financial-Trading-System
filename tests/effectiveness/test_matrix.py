"""Матрицы: режимы рынка x активы x стороны, цены, сессии, границы порогов, аналитика. Оба режима."""
from datetime import timedelta

import pytest

from .harness import ASSETS, BLOCKED_REGIMES, LIMITS, MONDAY_NOON, REGIMES, SUNDAY_EVENING, Lab, Scenario, sev

MODES = ["rules", "analytics"]


def sc(sym="SBER", kind="stock", **kw):
    p0 = next(a[1] for a in ASSETS if a[0] == sym)
    base = dict(symbol=sym, kind=kind, price=p0, qty=1 if kind == "stock" else 0.01)
    base.update(kw)
    return Scenario(**base)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("sym,p0,kind", ASSETS)
def test_regime_blocks_only_buys_in_dangerous_regimes(mode, regime, sym, p0, kind):
    lab = Lab(mode)
    buy = lab.decide(sc(sym, kind, regime=regime))
    assert (buy["decision"] == "deny") == (regime in BLOCKED_REGIMES), (regime, buy["codes"])
    held = 5 if kind == "stock" else 1.0
    sell = lab.decide(sc(sym, kind, regime=regime, side="sell", position=held, qty=held / 2))
    assert sell["decision"] == "allow", "выход из позиции нельзя блокировать режимом"


@pytest.mark.parametrize("mode", MODES)
def test_block_list_is_configurable(mode):
    lab = Lab(mode, {"regime_block_buy": {"value": ["CRASH", "RALLY_FRENZY", "BEAR_TREND"]}})
    assert lab.decide(sc(regime="BEAR_TREND"))["decision"] == "deny"
    assert Lab(mode, {"regime_block_buy": {"enabled": False}}).decide(sc(regime="CRASH"))["decision"] == "allow"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("price", [0.0001, 0.5, 85.0, 6000.0, 95000.0, 1e6, 1e7])
def test_order_value_boundary_at_every_price_scale(mode, price):
    lab, limit = Lab(mode), LIMITS["max_order_value"]
    qty_ok = limit * 0.99 / price
    ok = lab.decide(sc("BTCUSDT", "crypto", price=price, qty=qty_ok, equity=1e12))
    assert ok["decision"] == "allow", ok["reasons"]
    over = lab.decide(sc("BTCUSDT", "crypto", price=price, qty=limit * 1.01 / price, equity=1e12))
    assert over["decision"] in ("revise", "deny") and "max_order_value" in over["codes"]
    if over["decision"] == "revise":
        assert over["suggested_quantity"] * price <= limit + 1e-6


@pytest.mark.parametrize("mode", MODES)
def test_exact_limit_is_allowed_and_one_kopeck_over_is_stopped(mode):
    lab = Lab(mode)
    assert lab.decide(sc(price=100.0, qty=2000, equity=1e9))["decision"] == "allow"
    assert lab.decide(sc(price=100.01, qty=2000, equity=1e9))["decision"] != "allow"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("weekday_shift", range(7))
@pytest.mark.parametrize("hour,minute,open_", [(9, 59, False), (10, 0, True), (12, 0, True), (18, 40, True), (18, 41, False), (23, 0, False), (3, 0, False)])
def test_stock_session_matrix_and_crypto_is_always_open(mode, weekday_shift, hour, minute, open_):
    lab = Lab(mode)
    when = (MONDAY_NOON + timedelta(days=weekday_shift)).replace(hour=hour, minute=minute)
    weekend = when.weekday() >= 5
    stock = lab.decide(sc(when=when))
    assert (stock["decision"] == "allow") == (open_ and not weekend), (when, stock["codes"])
    assert lab.decide(sc("BTCUSDT", "crypto", when=when))["decision"] == "allow"


@pytest.mark.parametrize("mode", MODES)
def test_confidence_boundary(mode):
    lab = Lab(mode)
    assert lab.decide(sc(confidence=0.55))["decision"] == "allow"
    v = lab.decide(sc(confidence=0.54))
    assert v["decision"] == "revise" and "min_confidence" in v["codes"]


@pytest.mark.parametrize("mode", MODES)
def test_data_freshness_boundary_gives_a_retry_hint(mode):
    lab = Lab(mode)
    assert lab.decide(sc(age_min=180.0))["decision"] == "allow"
    v = lab.decide(sc(age_min=181.0))
    assert v["decision"] == "revise" and "data_freshness" in v["codes"] and v["retry_after"] > 0


@pytest.mark.parametrize("mode", MODES)
def test_limit_price_deviation_boundary(mode):
    lab = Lab(mode)
    assert lab.decide(sc(order_type="limit", limit_price=300.0 * 1.029))["decision"] == "allow"
    assert lab.decide(sc(order_type="limit", limit_price=300.0 * 1.031))["decision"] == "revise"
    assert lab.decide(sc(order_type="limit", limit_price=300.0 * 0.969))["decision"] == "revise"


@pytest.mark.parametrize("mode", MODES)
def test_position_share_boundary_and_selling_exactly_the_position(mode):
    lab = Lab(mode)
    assert lab.decide(sc(price=100.0, qty=1500, position=0, equity=800_000))["decision"] == "allow"         # 18,75% портфеля
    over = lab.decide(sc(price=100.0, qty=1700, position=0, equity=800_000))                                  # 21,25%, сумма заявки в лимите
    assert over["decision"] == "revise" and "max_position_pct" in over["codes"] and over["suggested_quantity"] == 1600
    assert lab.decide(sc(side="sell", qty=7, position=7))["decision"] == "allow"
    assert lab.decide(sc(side="sell", qty=7.0001, position=7))["decision"] != "allow"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("sym,p0,kind", ASSETS)
def test_kill_switch_denies_everything_including_sells(mode, sym, p0, kind):
    lab = Lab(mode, {"kill_switch": {"enabled": True}})
    assert lab.decide(sc(sym, kind))["decision"] == "deny"
    assert lab.decide(sc(sym, kind, side="sell", position=5, qty=1))["decision"] == "deny"


@pytest.mark.parametrize("sym,p0,kind", ASSETS)
def test_analytics_thresholds_three_zones_only_in_analytics_mode(sym, p0, kind):
    rules, ana = Lab("rules"), Lab("analytics")
    zones = [(0.6, "allow"), (-0.19, "allow"), (-0.2, "revise"), (-0.49, "revise"), (-0.5, "deny"), (-1.0, "deny")]
    for score, expected in zones:
        assert ana.decide(sc(sym, kind, news=score))["decision"] == expected, score
        assert ana.decide(sc(sym, kind, kronos=score))["decision"] == expected, score
        assert rules.decide(sc(sym, kind, news=score, kronos=score))["decision"] == "allow"      # без аналитики шлюз этого не видит


def test_warning_mode_and_sells_are_not_affected_by_analytics():
    warn = Lab("rules", {"news_sentiment": {"value": {"revise": -0.2, "deny": -0.5, "enforce": False}}})
    v = warn.decide(sc(news=-1.0))
    assert v["decision"] == "allow" and "Замечания" in v["agent_message"]
    assert Lab("analytics").decide(sc(side="sell", position=5, qty=2, news=-1.0, kronos=-1.0))["decision"] == "allow"


def test_fixable_problems_go_to_revise_with_a_hint_and_hallucinations_are_denied():
    lab = Lab("rules")
    night = lab.decide(sc(when=SUNDAY_EVENING))
    assert night["decision"] == "revise" and night["retry_after"] > 3600 and "повторите через" in night["agent_message"]
    assert lab.decide(sc(known=False))["decision"] == "deny"
    assert lab.decide(sc(qty=float("nan")))["decision"] == "deny"


@pytest.mark.parametrize("mode", MODES)
def test_revision_cap_prevents_pushing_through_with_small_edits(mode):
    lab = Lab(mode)
    ag, _ = lab.gw.agents.create("pusher", daily_quota=10**6)
    out = []
    for k in range(6):
        lab.sc, lab.clock.t = sc(price=300.0, qty=5000 + k, equity=1e9), MONDAY_NOON
        out.append(lab.gw.propose(ag, dict(client_id=f"p{k}", symbol="SBER", side="buy", quantity=5000 + k, rationale="x" * 30, confidence=0.8))["decision"])
    assert out[:3] == ["revise"] * 3 and set(out[3:]) == {"deny"}


@pytest.mark.parametrize("mode", MODES)
def test_cooldown_and_daily_limit(mode):
    lab = Lab(mode, {"max_orders_per_day": {"value": 3}})
    ag, _ = lab.gw.agents.create("burst", daily_quota=10**6)
    res = []
    for k in range(5):
        lab.sc, lab.clock.t = sc("BTCUSDT", "crypto", price=95000.0, qty=0.001), MONDAY_NOON + timedelta(minutes=10 * k)
        res.append(lab.gw.propose(ag, dict(client_id=f"b{k}", symbol="BTCUSDT", side="buy", quantity=0.001, rationale="y" * 30, confidence=0.8)))
    assert [r["decision"] for r in res] == ["allow", "allow", "allow", "revise", "revise"] and res[3]["retry_after"] > 0
