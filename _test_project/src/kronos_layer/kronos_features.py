"""
Преобразование KronosState → KronosFeatures.
Теперь использует только property rich KronosState — никакого ручного расчёта здесь.
"""

from __future__ import annotations
from dataclasses import dataclass
from src.kronos_layer.kronos_adapter import KronosState


@dataclass
class KronosFeatures:
    ticker:             str
    expected_return:    float   # прогнозируемая доходность (от state.expected_return)
    volatility_estimate: float  # нормированная волатильность
    trend_direction:    int     # +1 / -1 / 0
    confidence:         float   # [0, 1]


def extract_features(state: KronosState, last_price: float = 0.0) -> KronosFeatures:
    """
    last_price передаётся для совместимости — теперь expected_return
    считается внутри KronosState относительно первого прогнозируемого close.
    Если передан last_price > 0 — пересчитываем expected_return от него.
    """
    if last_price > 0 and len(state.forecast_close):
        expected_return = (float(state.forecast_close[-1]) - last_price) / last_price
    else:
        expected_return = state.expected_return

    return KronosFeatures(
        ticker=state.ticker,
        expected_return=expected_return,
        volatility_estimate=state.volatility_norm,
        trend_direction=state.direction,
        confidence=state.confidence,
    )