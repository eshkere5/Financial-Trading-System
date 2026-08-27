"""Препроцессинг: ресемплинг, очистка, нормализация."""

from __future__ import annotations
import pandas as pd
import numpy as np


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(timeframe).agg(agg).dropna()


def remove_outliers(series: pd.Series, std_thr: float = 4.0) -> pd.Series:
    mu, sigma = series.mean(), series.std()
    return series.where(series.between(mu - std_thr * sigma, mu + std_thr * sigma))


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()
