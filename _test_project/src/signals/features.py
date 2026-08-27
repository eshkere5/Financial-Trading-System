from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional

from src.kronos_layer.kronos_adapter import KronosAdapter, KronosState
from src.kronos_layer.kronos_features import KronosFeatures, extract_features


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mu    = series.rolling(window).mean()
    sigma = series.rolling(window).std()
    return (series - mu) / sigma.replace(0, np.nan)


def compute_spread(
    price_a: pd.Series,
    price_b: pd.Series,
    hedge_ratio: float,
) -> pd.Series:
    return price_a - hedge_ratio * price_b


def build_features(
    prices: pd.DataFrame,
    kronos: Optional[KronosAdapter] = None,
    ohlcv: Optional[dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    prices : DataFrame с тикерами в колонках, индекс datetime (close-цены).
    ohlcv  : опционально {ticker: DataFrame[open,high,low,close,volume]}.
             Если передан — Kronos получает полный OHLCV вместо Series.
    Возвращает DataFrame фич.
    """
    feats: dict[str, pd.Series] = {}

    for ticker in prices.columns:
        s = prices[ticker].dropna()
        if s.empty:
            continue

        feats[f"{ticker}_ret"]    = np.log(s / s.shift(1))
        feats[f"{ticker}_zscore"] = rolling_zscore(s)

        if kronos is not None:
            # передаём полный OHLCV если есть, иначе Series (close-only)
            source = ohlcv.get(ticker, s) if ohlcv else s
            last_price = float(s.iloc[-1])

            kstate: KronosState   = kronos.encode_prices(source, ticker=ticker)
            kf:     KronosFeatures = extract_features(kstate, last_price)

            # скалярные фичи прикладываем к последней дате ряда
            idx = s.index[-1:]
            feats[f"{ticker}_kr_ret"] = pd.Series([kf.expected_return],    index=idx)
            feats[f"{ticker}_kr_vol"] = pd.Series([kf.volatility_estimate], index=idx)
            feats[f"{ticker}_kr_dir"] = pd.Series([float(kf.trend_direction)], index=idx)
            feats[f"{ticker}_kr_conf"]= pd.Series([kf.confidence],          index=idx)

    return pd.DataFrame(feats).dropna(how="all")