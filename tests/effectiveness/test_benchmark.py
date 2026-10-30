"""Размеченный набор: 600 хороших заявок и 840 с известными изъянами (14 типов). Два режима: только правила и аналитика."""
import pytest

from .harness import FLAWS, SEVERITY, population, run_population, score

POP = population()


@pytest.fixture(scope="module")
def results():
    return {m: run_population(POP, m) for m in ("rules", "analytics")}


@pytest.fixture(scope="module")
def scores(results):
    return {m: score(r) for m, r in results.items()}


@pytest.mark.parametrize("mode", ["rules", "analytics"])
def test_no_good_intent_is_blocked(scores, mode):
    assert scores[mode]["good_n"] >= 500 and scores[mode]["false_blocks"] == 0


@pytest.mark.parametrize("mode", ["rules", "analytics"])
@pytest.mark.parametrize("flaw", [f for f, (d, _) in FLAWS.items() if d == "rules"])
def test_every_rule_detectable_flaw_is_stopped_in_both_modes(scores, mode, flaw):
    d = scores[mode]["by_flaw"][flaw]
    assert d["n"] >= 50 and d["stopped"] == d["n"], f"{flaw}: пропущено {d['n'] - d['stopped']} из {d['n']}"


@pytest.mark.parametrize("flaw", [f for f, (d, _) in FLAWS.items() if d == "analytics"])
def test_analytics_flaws_pass_rules_only_and_are_stopped_in_analytics_mode(scores, flaw):
    """Честная граница: без аналитики негативные новости и прогноз шлюз не видит; с аналитикой видит все."""
    assert scores["rules"]["by_flaw"][flaw]["stopped"] == 0
    assert scores["analytics"]["by_flaw"][flaw]["stopped"] == scores["analytics"]["by_flaw"][flaw]["n"]


def test_analytics_strictly_improves_catch_rate_without_new_false_blocks(scores):
    assert scores["analytics"]["catch_rate"] == 1.0 > scores["rules"]["catch_rate"] > 0.7
    assert scores["analytics"]["false_blocks"] <= scores["rules"]["false_blocks"]


@pytest.mark.parametrize("mode", ["rules", "analytics"])
def test_fatal_flaws_are_denied_and_fixable_ones_are_never_allowed(results, mode):
    for s, v in results[mode]:
        if s.label == "bad" and not (s.detector == "analytics" and mode == "rules"):
            assert SEVERITY[v["decision"]] >= SEVERITY[s.expect_min], (s.flaw, v["decision"], v.get("codes"))


def test_baseline_without_gateway_lets_everything_through(scores):
    """Для сравнения: без шлюза пропускаются все 840 изъянов, со шлюзом остаётся 0 (режим аналитики) или 180 (только правила)."""
    bad = scores["rules"]["bad_n"]
    assert bad == 840 and bad - scores["rules"]["bad_stopped"] == 180 and bad - scores["analytics"]["bad_stopped"] == 0
