"""Эффективность на реальных данных проекта (research/results). Проверяются только устойчивые выводы; остальное фиксируется как есть."""
import math

import pytest

from .realdata import kronos_long_study, load_daily, price_regime_study, regimes_for, simulate

NAMES = ["sber", "gazp", "lkoh"]


@pytest.fixture(scope="module")
def sims():
    out = {}
    for n in NAMES:
        d = load_daily(n)
        reg = regimes_for(d, n.upper())
        out[n] = {"none": simulate(d, reg, n.upper(), gateway=False), "full": simulate(d, reg, n.upper(), gateway=True),
                  "regime": simulate(d, reg, n.upper(), gateway=True, only_regime=True)}
    return out


@pytest.mark.parametrize("name", NAMES)
def test_gateway_cuts_drawdown_and_worst_day_of_a_naive_agent_on_real_prices(sims, name):
    a, b = sims[name]["none"], sims[name]["full"]
    assert b["max_dd"] > a["max_dd"] * 0.5, f"просадка с шлюзом {b['max_dd']:.1%}, без {a['max_dd']:.1%}"   # не больше половины прежней
    assert b["worst_day"] > a["worst_day"]


def test_gateway_does_not_starve_the_agent(sims):
    """Шлюз ограничивает размер, но агент продолжает торговать: число сделок сопоставимо."""
    for n in NAMES:
        assert sims[n]["full"]["trades"] >= 0.5 * sims[n]["none"]["trades"]


def test_kronos_long_signals_in_blocked_regimes_were_worse_on_stocks():
    r = kronos_long_study("stocks")
    assert r["n_blocked"] >= 10 and r["mean_blocked"] < r["mean_allowed"] and r["ci_blocked"][1] < 0


def test_kronos_crypto_sample_in_blocked_regimes_is_too_small_to_conclude():
    r = kronos_long_study("crypto")
    assert r["n_blocked"] < 5   # вывода нет: фиксируем, что данных мало


def test_price_only_regime_study_is_recorded_without_assuming_the_result():
    r = price_regime_study(5)
    assert r["n_blocked"] > 30 and all(math.isfinite(r[k]) for k in ("mean_blocked", "mean_allowed"))
