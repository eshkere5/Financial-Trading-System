"""
regime_detector.py — классификатор рыночного режима (8 режимов).

### NEW 2026-08-23 ###
Идея: точность Kronos зеркалит макро-режим (крипта-ралли: лонги 62% / шорты 26%;
MOEX-медведь: шорты 62% / лонги 32%). Значит, сигналы надо взвешивать по режиму.

Принципы:
- чисто ценовые признаки (SMA, вола, просадка, ширина диапазона) — дёшево,
  считается каждый тик без LLM; LLM-стратегист получает режим как контекст
  и может переопределить по новостям (новостной краш до реакции цен);
- безразмерно: работает и на дневках MOEX, и на часовиках Bybit (всё в барах);
- множители REGIME_MULTIPLIERS — ПРИОРЫ, калибруются измеренными hit-rate
  через regime_report.py, руками не тюнить.

Использование:
    from regime_detector import compute_regime, signal_multiplier
    info = compute_regime(closes)                    # closes: pd.Series
    factor = signal_multiplier(info.regime, pred_dir)  # 0.0=veto ... 1.2=boost
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

REGIMES = ("CRASH", "RALLY_FRENZY", "RECOVERY", "BEAR_TREND", "BULL_TREND",
           "HIGH_VOL_CHOP", "RANGE_BOUND", "RANGE_SQUEEZE", "TRANSITION")

# ── пороги (спека 2026-08-24) ────────────────────────────────────────
SMA_FAST, SMA_SLOW = 50, 200
EMA_TREND = 20
ADX_WIN = 14
ADX_TREND_MIN = 20.0       # отделение trend от range
ADX_FRENZY_MIN = 25.0      # сильный тренд для RALLY_FRENZY
VOL_WIN, PCT_WIN = 20, 252
DD_LOCAL_WIN = 60          # локальная просадка — триггер CRASH
CRASH_VOL_P = 0.90
CRASH_DD = -0.10
CHOP_VOL_P = 0.70          # выше — HIGH_VOL_CHOP
SQUEEZE_VOL_P = 0.20       # ниже И сжатый BB — RANGE_SQUEEZE
BB_WIN = 20
SQUEEZE_BB_P = 0.25

# перегрев от SMA200 для RALLY_FRENZY: дневной порог 0.08 валидирован на
# дневках MOEX; для крипто-часовиков (SMA200 ≈ 8 дней) порог мягче и
# ПРЕДВАРИТЕЛЬНЫЙ — валидировать regime_report'ом, не переносить вслепую.
OVERHEAT_MIN = {"stocks": 0.08, "crypto": 0.06, "default": 0.08}

# ── множители (лонг, шорт) — ПРИОРЫ до перекалибровки под v2-ярлыки ──
REGIME_MULTIPLIERS_STOCKS = {     # калибровка v2, 2026-08-24, 144 точки MOEX
    "CRASH":         (0.0, 0.6),  # лонги −25.3% (n=13): вето
    "RALLY_FRENZY":  (0.8, 0.2),  # n=1 — приор
    "RECOVERY":      (0.5, 1.0),  # v1-ядро (+21.8%) под v2 размылось: шорты n=4 — мягко
    "BEAR_TREND":    (0.5, 1.0),  # лонги 29%/−18.8% (n=48) — самая большая n
    "BULL_TREND":    (1.0, 1.0),  # шорты +12.6% (n=14) — НЕ трогаем
    "HIGH_VOL_CHOP": (0.6, 0.8),  # n=5 — мало
    "RANGE_BOUND":   (0.4, 1.0),  # лонги −16.0% (n=13) — новая токсичная ячейка
    "RANGE_SQUEEZE": (0.5, 0.8),  # лонги −4.9% (n=6)
    "TRANSITION":    (0.8, 0.8),
}

REGIME_MULTIPLIERS_CRYPTO = {     # калибровка v2, 2026-08-24, 252 точки Bybit
    "CRASH":         (0.0, 0.6),  # не встретился — приор
    "RALLY_FRENZY":  (0.8, 0.1),  # шорты −31.2% (n=22): самый жёсткий срез проекта
    "RECOVERY":      (1.1, 1.0),  # лонги 83%/+10.1% (n=12)
    "BEAR_TREND":    (0.6, 1.1),  # шорты 67%/+21.0% (n=40) — ядро
    "BULL_TREND":    (1.1, 0.5),  # лонги 73%/+17.1%; шорты смягчены после выноса frenzy
    "HIGH_VOL_CHOP": (1.0, 0.9),  # лонги 81%/+9.1% (n=16)
    "RANGE_BOUND":   (1.0, 0.7),  # шорты −4.1% (n=18)
    "RANGE_SQUEEZE": (1.0, 0.9),  # обе стороны 78-89% hit, n=9
    "TRANSITION":    (0.8, 0.8),
}
REGIME_MULTIPLIERS_DEFAULT = {
    "CRASH":         (0.0, 0.6), "RALLY_FRENZY":  (0.8, 0.2),
    "RECOVERY":      (0.7, 1.0), "BEAR_TREND":    (0.6, 1.0),
    "BULL_TREND":    (1.0, 0.7), "HIGH_VOL_CHOP": (0.7, 0.7),
    "RANGE_BOUND":   (0.8, 0.8), "RANGE_SQUEEZE": (0.9, 0.8),
    "TRANSITION":    (0.8, 0.8),
}

_CRYPTO_SUFFIXES = ("USDT", "USDC")


@dataclass
class RegimeInfo:
    regime: str
    close_vs_sma50: float     # P/SMA50 − 1
    sma50_slope: float        # наклон SMA50 за 10 баров
    ema_slope: float          # наклон EMA20 за 5 баров (Tt)
    vol_pctile: float         # перцентиль 20-барной realized vol (N=252)
    adx: float                # ADX(14), NaN если мало данных
    overheat: float           # P/SMA200 − 1 (NaN если < SMA_SLOW баров)
    dd_local: float           # просадка от макс. 60 баров (триггер CRASH)
    dd_global: float          # просадка от исторического макс. окна (контекст)


def asset_class(ticker: str) -> str:
    return "crypto" if ticker.upper().endswith(_CRYPTO_SUFFIXES) else "stocks"


def signal_multiplier(regime: str, pred_dir: int, ticker: str = "") -> float:
    table = (REGIME_MULTIPLIERS_CRYPTO if asset_class(ticker) == "crypto"
             else REGIME_MULTIPLIERS_STOCKS if ticker else REGIME_MULTIPLIERS_DEFAULT)
    long_f, short_f = table.get(regime, table["TRANSITION"])
    if pred_dir > 0:
        return long_f
    if pred_dir < 0:
        return short_f
    return 0.0


# ── ADX по Wilder ────────────────────────────────────────────────────

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, win: int = ADX_WIN) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr = pd.concat([(high - low),
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / win, adjust=False).mean()
    p_di = 100 * plus_dm.ewm(alpha=1 / win, adjust=False).mean() / atr
    m_di = 100 * minus_dm.ewm(alpha=1 / win, adjust=False).mean() / atr
    dx = 100 * (p_di - m_di).abs() / (p_di + m_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / win, adjust=False).mean()


def _pctile(series: pd.Series, value: float, win: int) -> float:
    tail = series.dropna().iloc[-win:]
    if len(tail) < 20 or np.isnan(value):
        return 0.5
    return float((tail < value).mean())


def compute_regime(df: pd.DataFrame, ticker: str = "") -> RegimeInfo:
    """df: DataFrame с колонками high/low/close (open не обязателен)."""
    cols = {c.lower(): c for c in df.columns}
    c = df[cols["close"]].astype(float).reset_index(drop=True)
    h = df[cols["high"]].astype(float).reset_index(drop=True)
    l = df[cols["low"]].astype(float).reset_index(drop=True)

    if len(c) < SMA_FAST + ADX_WIN + 1:
        return RegimeInfo("TRANSITION", 0.0, 0.0, 0.0, 0.5, np.nan, np.nan, 0.0, 0.0)

    price = float(c.iloc[-1])
    sma50 = c.rolling(SMA_FAST).mean()
    sma200 = c.rolling(SMA_SLOW).mean()
    ema20 = c.ewm(span=EMA_TREND, adjust=False).mean()

    ma50 = float(sma50.iloc[-1])
    ma200 = sma200.iloc[-1]
    dist = price / ma50 - 1.0
    slope50 = ma50 / float(sma50.iloc[-11]) - 1.0 if not np.isnan(sma50.iloc[-11]) else 0.0
    ema_slope = float(ema20.iloc[-1] / ema20.iloc[-6] - 1.0)
    overheat = float(price / ma200 - 1.0) if not np.isnan(ma200) else np.nan

    rets = c.pct_change()
    vol20 = rets.rolling(VOL_WIN).std()
    vol_p = _pctile(vol20, float(vol20.iloc[-1]), PCT_WIN)

    dd_local = price / float(c.iloc[-DD_LOCAL_WIN:].max()) - 1.0
    dd_global = price / float(c.max()) - 1.0

    adx_val = float(_adx(h, l, c).iloc[-1])

    bb_width = (c.rolling(BB_WIN).std() * 4) / c   # (upper-lower)/mid при k=2
    bb_p = _pctile(bb_width, float(bb_width.iloc[-1]), PCT_WIN)

    oh_min = OVERHEAT_MIN.get(asset_class(ticker), OVERHEAT_MIN["default"]) if ticker else OVERHEAT_MIN["default"]
    oh = 0.0 if np.isnan(overheat) else overheat

    up50 = dist > 0 and slope50 > 0
    down50 = dist < 0 and slope50 < 0

    # дерево приоритетов (спека 2026-08-24, RECOVERY до BEAR/BULL)
    if vol_p >= CRASH_VOL_P and dd_local <= CRASH_DD:
        regime = "CRASH"
    elif ema_slope > 0 and oh >= oh_min and adx_val >= ADX_FRENZY_MIN:
        regime = "RALLY_FRENZY"
    elif dist > 0 and slope50 > 0 and not np.isnan(ma200) and price < ma200:
        regime = "RECOVERY"                     # отскок внутри широкого медведя
    elif down50:
        regime = "BEAR_TREND"
    elif up50 and not np.isnan(ma200) and price > ma200 and adx_val >= ADX_TREND_MIN:
        regime = "BULL_TREND"
    elif vol_p <= SQUEEZE_VOL_P and bb_p <= SQUEEZE_BB_P:
        regime = "RANGE_SQUEEZE"
    elif vol_p >= CHOP_VOL_P:
        regime = "HIGH_VOL_CHOP"
    else:
        regime = "RANGE_BOUND"

    return RegimeInfo(regime, dist, slope50, ema_slope, vol_p, adx_val,
                      overheat, dd_local, dd_global)