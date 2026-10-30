"""Какие правила блокируют, какие отправляют на доработку, какие только предупреждают."""
import pytest

from finvex.gateway.analytics import AnalyticsError
from finvex.gateway.rules import EFFECTS, RULES_META, mandate_hash, normalize
from tests.test_gateway import intent, mk


@pytest.fixture
def env():
    gw, clock = mk()
    ag, _ = gw.agents.create("p")
    return gw, ag, clock


def rules_with(gw, **over):
    r = {k: dict(v) for k, v in gw.rules.items()}
    for k, v in over.items():
        r[k] = {**r[k], **v}
    gw.set_rules(r)


def test_every_rule_has_declared_effect():
    assert {m["id"] for m in RULES_META} == set(EFFECTS) and all(m["effect"] for m in RULES_META)


def test_soft_rule_does_not_block_when_everything_else_is_fine(env):
    gw, ag, _ = env
    v = gw.propose(ag, intent(rationale=""))
    assert v["decision"] == "allow" and not v["codes"]


def test_hard_rule_is_not_ignored_even_with_perfect_rationale(env):
    gw, ag, _ = env
    rules_with(gw, blocked_symbols={"value": ["SBER"]})
    v = gw.propose(ag, intent(rationale="Подробное и убедительное обоснование покупки с уровнями входа и стопа"))
    assert v["decision"] == "deny" and "blocked_symbols" in v["codes"] and "Отклонено" in v["agent_message"]
    assert "SBER" in v["agent_message"]


def test_revise_message_tells_agent_what_to_do(env):
    gw, ag, _ = env
    v = gw.propose(ag, intent(quantity=5000))
    assert v["decision"] in ("revise", "deny") and ("Допустимый объём" in v["agent_message"] or "Отклонено" in v["agent_message"])


def test_too_long_rationale_goes_to_revise(env):
    gw, ag, _ = env
    v = gw.propose(ag, intent(rationale="а" * 1600))
    assert v["decision"] == "revise" and "max_rationale" in v["codes"]


def test_kill_switch_denies_everything(env):
    gw, ag, _ = env
    rules_with(gw, kill_switch={"enabled": True})
    v = gw.propose(ag, intent())
    assert v["decision"] == "deny" and "kill_switch" in v["codes"]


@pytest.mark.parametrize("score,expected", [(0.3, "allow"), (-0.1, "allow"), (-0.2, "revise"), (-0.35, "revise"), (-0.5, "deny"), (-0.9, "deny")])
def test_news_bands_three_zones(env, score, expected):
    gw, ag, _ = env
    rules_with(gw, news_sentiment={"value": {"revise": -0.2, "deny": -0.5, "enforce": True}})
    gw.analytics.add_news("SBER", {"id": "n", "score": score, "category": "company", "ttl_hours": 10})
    assert gw.propose(ag, intent(client_id=f"n{score}"))["decision"] == expected


def test_warn_mode_never_changes_decision(env):
    gw, ag, _ = env
    gw.analytics.add_news("SBER", {"id": "n", "score": -0.9, "category": "company", "ttl_hours": 10})
    v = gw.propose(ag, intent())
    assert v["decision"] == "allow" and "Замечания" in v["agent_message"]


def test_analytics_only_restricts_buys_not_sells(env):
    gw, ag, _ = env
    rules_with(gw, kronos_view={"value": {"revise": -0.2, "deny": -0.5, "enforce": True}})
    gw.analytics.set_kronos("SBER", -0.7, iso(gw.clock, 3))
    assert gw.propose(ag, intent(side="sell", quantity=5, client_id="s"))["decision"] == "allow"
    assert gw.propose(ag, intent(client_id="b"))["decision"] == "deny"


def iso(clock, hours):
    from datetime import timedelta
    return (clock() + timedelta(hours=hours)).isoformat(timespec="seconds")


def test_only_valid_news_enter_the_decision(env):
    gw, ag, clock = env
    rules_with(gw, news_sentiment={"value": {"revise": -0.2, "deny": -0.5, "enforce": True}})
    gw.analytics.add_news("SBER", {"id": "old", "score": -0.9, "category": "company", "ttl_hours": 2})
    assert gw.propose(ag, intent(client_id="a"))["decision"] == "deny"
    clock.t = clock.t.replace(hour=15)          # прошло 3 часа: новость просрочена и должна исчезнуть
    v = gw.propose(ag, intent(client_id="b"))
    assert v["decision"] == "allow" and "old" not in gw.analytics.snapshot("SBER", clock()).get("news_ids", [])
    assert "SBER" not in gw.analytics.all(clock())   # удалена из активного набора


def test_verdict_records_which_news_were_used(env):
    gw, ag, _ = env
    gw.analytics.add_news("SBER", {"id": "n1", "score": 0.4, "category": "company", "ttl_hours": 5})
    assert gw.propose(ag, intent())["context"]["analytics_used"]["news_ids"] == ["n1"]


def test_ttl_from_parser_is_clamped_by_category(env):
    gw = env[0]
    assert gw.analytics.add_news("SBER", {"id": "r", "score": -0.5, "category": "rumor", "ttl_hours": 500})["ttl_hours"] == 4.0
    assert gw.analytics.add_news("SBER", {"id": "e", "score": 0.5, "category": "earnings", "ttl_hours": 0.01})["ttl_hours"] == 6.0
    with pytest.raises(AnalyticsError):
        gw.analytics.add_news("SBER", {"id": "x", "score": 0.5})     # без срока от парсера новость не принимается


def test_news_weighted_by_materiality(env):
    gw = env[0]
    gw.analytics.add_news("SBER", {"id": "a", "score": -1.0, "materiality": 0.1, "category": "company", "ttl_hours": 10})
    gw.analytics.add_news("SBER", {"id": "b", "score": 0.5, "materiality": 0.9, "category": "company", "ttl_hours": 10})
    assert gw.analytics.snapshot("SBER")["news_score"] == pytest.approx((-0.1 + 0.45) / 1.0)


def test_kronos_valid_until_next_candle(env):
    gw, ag, clock = env
    rules_with(gw, kronos_view={"value": {"revise": -0.2, "deny": -0.5, "enforce": True}})
    gw.analytics.set_kronos("SBER", -0.7, iso(clock, 1))
    assert gw.propose(ag, intent(client_id="k1"))["decision"] == "deny"
    clock.t = clock.t.replace(hour=14)
    assert gw.propose(ag, intent(client_id="k2"))["decision"] == "allow"    # прогноз устарел и убран
    with pytest.raises(AnalyticsError):
        gw.analytics.set_kronos("SBER", 0.1, iso(clock, -1))


def test_no_valid_analytics_is_neutral_not_a_stop(env):
    gw, ag, _ = env
    rules_with(gw, news_sentiment={"value": {"revise": -0.2, "deny": -0.5, "enforce": True}})
    assert gw.propose(ag, intent())["decision"] == "allow"


def test_not_an_error_goes_to_revise_with_retry_hint(env):
    gw, ag, clock = env
    clock.t = clock.t.replace(hour=22)
    v = gw.propose(ag, intent(symbol="MTSS"))
    assert v["decision"] == "revise" and "trading_hours" in v["codes"] and v["retry_after"] > 3600


def test_hallucinated_input_is_denied(env):
    gw, ag, _ = env
    assert gw.propose(ag, intent(symbol="NOSUCH"))["decision"] == "deny"


def test_revision_cap_stops_pushing_with_small_edits(env):
    gw, ag, _ = env
    decisions = [gw.propose(ag, intent(quantity=5000 + k, client_id=f"r{k}"))["decision"] for k in range(5)]
    assert decisions == ["revise", "revise", "revise", "deny", "deny"]
    last = gw.propose(ag, intent(quantity=5000, client_id="r9"))
    assert "max_revisions" in last["codes"] and "Пересмотрите" in last["agent_message"]
    assert gw.propose(ag, intent(symbol="GAZP", quantity=10, client_id="other"))["decision"] == "allow"   # лимит по связке бумага+сторона


def test_band_validation_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        normalize({"news_sentiment": {"enabled": True, "value": {"revise": -0.6, "deny": -0.2}}})
    with pytest.raises(ValueError):
        normalize({"kronos_view": {"enabled": True, "value": {"revise": "x", "deny": -0.5}}})


def test_analytics_input_is_validated(env):
    gw = env[0]
    for bad in (dict(score=2), dict(score=float("nan")), dict(score="a")):
        with pytest.raises(AnalyticsError):
            gw.analytics.add_news("SBER", {**bad, "ttl_hours": 3})
    with pytest.raises(AnalyticsError):
        gw.analytics.add_news("S B;", {"score": 0.1, "ttl_hours": 3})


def test_mandate_hash_changes_with_rules(env):
    gw, ag, _ = env
    a = gw.propose(ag, intent(client_id="1"))["mandate_hash"]
    rules_with(gw, max_order_value={"value": 1000})
    assert gw.propose(ag, intent(client_id="2", quantity=1))["mandate_hash"] != a == mandate_hash(normalize({}))
