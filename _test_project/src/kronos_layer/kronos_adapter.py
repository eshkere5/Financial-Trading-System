"""
KronosAdapter — обёртка над KronosTokenizer + Kronos + KronosPredictor.

### FIXED 2026-08-02 ###
- К6: контекст для предиктора брался из НАЧАЛА истории (`df.loc[:lookback-1]`),
  а не из последних баров. См. encode_prices() ниже.
- Диагностика загрузки модели переведена с print/traceback.print_exc на logger.
- Добавлен публичный флаг `is_mock` — раньше mock-режим был неотличим от
  реального прогноза на стороне вызывающего кода.
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, is_dataclass, asdict
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

KRONOS_REPO_PATH = Path(__file__).resolve().parents[3] / "Kronos"

# Минимум баров, при котором прогноз вообще имеет смысл: нужно хотя бы две
# точки, чтобы определить шаг таймфрейма (freq) для будущего индекса.
MIN_CONTEXT_BARS = 2


# ── Rich KronosState ──────────────────────────────────────────────────────────

@dataclass
class KronosState:
    ticker: str
    forecast_open:  np.ndarray   # (pred_len,)
    forecast_high:  np.ndarray
    forecast_low:   np.ndarray
    forecast_close: np.ndarray

    # ── вычисляемые ───────────────────────────────────────────────────────────

    @property
    def forecast_median(self) -> np.ndarray:
        """Средняя линия канала (hi+lo)/2."""
        return (self.forecast_high + self.forecast_low) / 2.0

    @property
    def expected_return(self) -> float:
        """Прогнозируемая доходность: (last_close - first_close) / first_close."""
        fc = self.forecast_close
        if len(fc) < 2 or fc[0] == 0:
            return 0.0
        return float((fc[-1] - fc[0]) / fc[0])

    @property
    def volatility(self) -> float:
        """Средняя абсолютная ширина канала hi-lo (ненормированная)."""
        return float(np.mean(self.forecast_high - self.forecast_low))

    @property
    def volatility_norm(self) -> float:
        """Волатильность, нормированная на среднюю цену прогноза."""
        mid = float(np.mean(self.forecast_median))
        return self.volatility / mid if mid > 0 else 0.0

    @property
    def confidence(self) -> float:
        """1 - clip(volatility_norm, 0, 1). Чем уже канал — тем выше уверенность."""
        return float(1.0 - min(self.volatility_norm, 1.0))

    @property
    def direction(self) -> int:
        """Знак тренда: +1 вверх, -1 вниз, 0 боковик."""
        med = self.forecast_median
        if len(med) < 2:
            return 0
        return int(np.sign(med[-1] - med[0]))


# ── KronosAdapter ─────────────────────────────────────────────────────────────

class KronosAdapter:
    """
    Обёртка над KronosTokenizer + Kronos + KronosPredictor.
    Если модель не установлена — работает в mock-режиме (держит последнюю свечу).

    ### FIXED 2026-08-02 ###
    Публичный атрибут `is_mock` позволяет вызывающему коду отличить синтетический
    прогноз от настоящего. Mock генерирует узкий канал (+-0.5%), из-за чего
    KronosState.confidence получается близким к 1.0 — без явного флага стратегия
    принимала бы шум за очень уверенный сигнал.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
        self.predictor = None
        self.is_mock = True   # ### FIXED 2026-08-02 ### снимается в _try_load()
        self._try_load()

    def _try_load(self) -> None:
        try:
            if str(KRONOS_REPO_PATH) not in sys.path:
                sys.path.insert(0, str(KRONOS_REPO_PATH))
            from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore
            import torch

            tokenizer = KronosTokenizer.from_pretrained(
                self.cfg.get("tokenizer_name", "NeoQuasar/Kronos-Tokenizer-base")
            )
            model = Kronos.from_pretrained(
                self.cfg.get("model_name", "NeoQuasar/Kronos-small")
            )
            device = self.cfg.get("device", "cpu")
            model.to(device)

            self.predictor = KronosPredictor(
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_context=int(self.cfg.get("max_context", 512)),
            )
            self.is_mock = False
            logger.info("Kronos loaded OK | model=%s device=%s",
                        self.cfg.get("model_name", "NeoQuasar/Kronos-small"), device)
        except Exception as exc:
            # ### FIXED 2026-08-02 ###
            # Было: print(...) + traceback.print_exc() в критическом пути —
            # диагностика уходила мимо логов и не попадала в журнал прод-запуска.
            logger.warning(
                "Kronos недоступен (%s: %s) — MOCK-режим, прогнозы синтетические",
                type(exc).__name__, exc, exc_info=True,
            )
            self.predictor = None
            self.is_mock = True

    # ── основной метод ────────────────────────────────────────────────────────

    def encode_prices(
        self,
        kline_df: "pd.DataFrame | pd.Series",
        ticker: str = "",
    ) -> KronosState:
        """
        Принимает DataFrame[open,high,low,close,volume] или pd.Series (close-only).
        Возвращает rich KronosState.

        ### FIXED 2026-08-02 (К6) ###
        Контекст для предиктора берётся из ПОСЛЕДНИХ `lookback` баров.
        Никакого lookahead здесь нет и не появляется: срез идёт по хвосту уже
        переданного датафрейма, поэтому ответственность за то, чтобы в
        `kline_df` не попали бары будущего относительно текущего бара
        бэктеста, остаётся на вызывающей стороне (см. К5 в review_core.md —
        BacktestEngine отдаёт стратегии всю историю целиком).
        """
        pred_len = int(self.cfg.get("prediction_length", 120))

        # ── нормализация входа ─────────────────────────────────────────────
        if isinstance(kline_df, pd.Series):
            # Series → минимальный OHLCV (все OHLC = close)
            s = kline_df.copy()
            df = pd.DataFrame({
                "open":   s.values,
                "high":   s.values,
                "low":    s.values,
                "close":  s.values,
                "volume": np.zeros(len(s), dtype=np.float32),
            })
            df.index = s.index
        else:
            df = kline_df.copy()

        # ── подготовка timestamps ──────────────────────────────────────────
        if "timestamps" in df.columns:
            df["timestamps"] = pd.to_datetime(df["timestamps"])
            df = df.sort_values("timestamps")
        else:
            df = df.sort_index()
            df = df.reset_index().rename(columns={df.index.name or "index": "timestamps"})
            df["timestamps"] = pd.to_datetime(df["timestamps"])

        # ### FIXED 2026-08-02 (К6) ###
        # Явный отказ от сигнала при нехватке истории. Раньше на пустом/одном
        # баре код падал глубже с невнятным IndexError на x_timestamp.iloc[-1];
        # оба вызывающих места (KronosDefaultStrategy._get_kronos_signal и
        # AITacticStrategy._get_kronos_signal) ловят Exception и корректно
        # трактуют это как "сигнала нет".
        if len(df) < MIN_CONTEXT_BARS:
            raise ValueError(
                f"Kronos: недостаточно истории для {ticker or '<unknown>'}: "
                f"{len(df)} баров < минимума {MIN_CONTEXT_BARS}"
            )

        lookback  = min(len(df), int(self.cfg.get("max_context", 512)))
        cols      = ["open", "high", "low", "close", "volume"]
        if "amount" in df.columns:
            cols.append("amount")

        # ### FIXED 2026-08-02 (К6) ###
        # Было:
        #     x_df        = df.loc[:lookback - 1, cols]
        #     x_timestamp = df.loc[:lookback - 1, "timestamps"]
        # `.loc[:N]` — это срез ПО МЕТКАМ индекса, а не по позициям, поэтому
        # брались САМЫЕ СТАРЫЕ `lookback` баров (а на неотсортированном/
        # нецелочисленном индексе — вообще падало или возвращало мусор).
        # Прогноз строился по началу истории и применялся к сегодняшней цене.
        # Стало: позиционный срез по хвосту — последние `lookback` баров.
        x_df        = df.iloc[-lookback:][cols].reset_index(drop=True).copy()
        x_timestamp = df["timestamps"].iloc[-lookback:].reset_index(drop=True)

        last_ts = x_timestamp.iloc[-1]
        if len(x_timestamp) >= 2:
            freq = x_timestamp.iloc[-1] - x_timestamp.iloc[-2]
        else:
            freq = pd.Timedelta(hours=1)  # fallback

        future_index = pd.date_range(start=last_ts + freq, periods=pred_len, freq=freq)
        y_timestamp = pd.Series(future_index, name="timestamps")

        # ── mock ───────────────────────────────────────────────────────────
        if self.predictor is None:
            last = x_df.iloc[-1]
            close_val = float(last["close"])
            noise = np.random.normal(0, close_val * 0.005, pred_len).astype(np.float32)
            close_ = (close_val + noise).astype(np.float32)
            return KronosState(
                ticker=ticker,
                forecast_open=close_.copy(),
                forecast_high=(close_ * 1.005).astype(np.float32),
                forecast_low=(close_ * 0.995).astype(np.float32),
                forecast_close=close_,
            )

        # ── реальный predict ───────────────────────────────────────────────
        pred_df = self.predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=float(self.cfg.get("temperature", 1.0)),
            top_p=float(self.cfg.get("top_p", 0.9)),
            sample_count=int(self.cfg.get("sample_count", 8)),
            verbose=False,
        )

        return KronosState(
            ticker=ticker,
            forecast_open=pred_df["open"].to_numpy(dtype=np.float32),
            forecast_high=pred_df["high"].to_numpy(dtype=np.float32),
            forecast_low=pred_df["low"].to_numpy(dtype=np.float32),
            forecast_close=pred_df["close"].to_numpy(dtype=np.float32),
        )

    def predict_next_close(self, state: KronosState) -> float:
        return float(state.forecast_close[-1]) if len(state.forecast_close) else 0.0
