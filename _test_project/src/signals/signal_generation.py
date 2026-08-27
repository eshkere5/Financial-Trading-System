"""
signal_generation.py — SignalGenerator с адаптивными порогами.

Изменения vs старый файл:
- убраны фиксированные buy_threshold / sell_threshold
- добавлен threshold_sigma * volatility (адаптивный порог)
- добавлен from_features(KronosFeatures) — прямой путь
- добавлен from_spread() — stat-arb сигнал
- добавлен min_confidence фильтр
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional

from src.signals.signal_types import TradingSignal
from src.kronos_layer.kronos_features import KronosFeatures


class SignalGenerator:

    def __init__(
        self,
        threshold_sigma: float = 1.0,   # |ER| > sigma * vol → сигнал
        min_confidence:  float = 0.3,
        min_strength:    float = 0.005, # минимальный |ER| независимо от vol
    ) -> None:
        self.threshold_sigma = threshold_sigma
        self.min_confidence  = min_confidence
        self.min_strength    = min_strength

    # ── из KronosFeatures (основной путь) ────────────────────────────────────

    def from_features(self, kf: KronosFeatures) -> Optional[TradingSignal]:
        if kf.confidence < self.min_confidence:
            return None

        er  = kf.expected_return
        vol = kf.volatility_estimate

        # адаптивный порог: сигнал только если движение значимо относительно волатильности
        adaptive_thr = self.threshold_sigma * vol
        if abs(er) < max(adaptive_thr, self.min_strength):
            return None

        direction = 1 if er > 0 else -1
        # strength нормирован [0, 1]: насколько сигнал превышает порог
        strength  = min(abs(er) / (adaptive_thr + 1e-9), 3.0) / 3.0

        return TradingSignal(
            ticker=kf.ticker,
            direction=direction,
            strength=strength,
            confidence=kf.confidence,
            expected_return=er,
            source="kronos",
            timestamp=datetime.utcnow(),
            reason=(
                f"er={er:.4f} vol={vol:.4f} "
                f"thr={adaptive_thr:.4f} dir={direction}"
            ),
        )

    # ── совместимость со старым кодом ────────────────────────────────────────

    def from_kronos(
        self,
        ticker: str,
        expected_return: float,
        confidence: float,
        volatility: float = 0.01,   # если не передан — используем min_strength
    ) -> Optional[TradingSignal]:
        """Обратная совместимость. Предпочитай from_features()."""
        kf = KronosFeatures(
            ticker=ticker,
            expected_return=expected_return,
            volatility_estimate=volatility,
            trend_direction=1 if expected_return > 0 else -1,
            confidence=confidence,
        )
        return self.from_features(kf)

    # ── stat-arb: из zscore спреда ───────────────────────────────────────────

    def from_spread(
        self,
        ticker_a:     str,
        ticker_b:     str,
        zscore:       float,
        hedge_ratio:  float,
        zscore_entry: float = 2.0,
    ) -> Optional[TradingSignal]:
        """
        Stat-arb сигнал по z-score спреда.
        direction=+1 → long A / short B (спред ниже нормы, ждём возврата вверх)
        direction=-1 → short A / long B (спред выше нормы, ждём возврата вниз)
        """
        if abs(zscore) < zscore_entry:
            return None

        direction = -1 if zscore > zscore_entry else 1
        strength  = min(abs(zscore) / (zscore_entry * 2), 1.0)

        return TradingSignal(
            ticker=f"{ticker_a}/{ticker_b}",
            direction=direction,
            strength=strength,
            confidence=strength,
            expected_return=0.0,
            source="stat_arb",
            timestamp=datetime.utcnow(),
            reason=f"zscore={zscore:.2f} hedge={hedge_ratio:.4f}",
        )
