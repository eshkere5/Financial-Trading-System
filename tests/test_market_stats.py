import numpy as np
import pandas as pd
import pytest

from finvex import market_stats as st


def frame(n=300, drift=0.001, seed=1):
    c = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(drift, 0.01, n)))
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c})


def test_rank_orders_uptrend_first():
    r = st.rank_trends([st.trend_snapshot(frame(drift=0.003, seed=1), "UP"), st.trend_snapshot(frame(drift=-0.003, seed=2), "DOWN")])
    assert list(r["symbol"]) == ["UP", "DOWN"] and 0 <= r["bull_score"].max() <= 100


def test_short_history_is_error():
    with pytest.raises(ValueError):
        st.trend_snapshot(frame(30), "X")


def test_coint_finds_planted_pair():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(0, 0.01, 400))
    logp = pd.DataFrame({"A": x + 4, "B": 0.8 * x + rng.normal(0, 0.005, 400) + 4, "C": np.cumsum(rng.normal(0, 0.01, 400)) + 4})
    res = st.scan_cointegration(logp, {"g": ["A", "B", "C"]})
    assert res["pairs"][0]["pair"] == ("A", "B") and res["pairs"][0]["p_bonferroni"] < 0.05
    assert any({"A", "B"} <= set(g) for g in st.cluster_by_corr(logp.diff().dropna(), 0.5))
