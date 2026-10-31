"""Математические утилиты."""

from __future__ import annotations
import numpy as np


def sharpe(returns: np.ndarray, risk_free: float = 0.0, periods: int = 252) -> float:
    excess = returns - risk_free / periods
    std = excess.std()
    return float(excess.mean() / std * np.sqrt(periods)) if std > 0 else 0.0


def max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / peak
    return float(dd.max())


def cagr(equity_curve: np.ndarray, periods_per_year: int = 252) -> float:
    if len(equity_curve) < 2 or equity_curve[0] <= 0:
        return 0.0
    n_years = len(equity_curve) / periods_per_year
    return float((equity_curve[-1] / equity_curve[0]) ** (1 / n_years) - 1)
