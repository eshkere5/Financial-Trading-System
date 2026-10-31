"""
Мост «провайдер данных → Kronos». Kronos не собирает данные сам: KronosAdapter.encode_prices
принимает DataFrame[open, high, low, close, volume(, amount)] с колонкой `timestamps`.
Здесь ряд приводится к этому контракту; сама модель и её адаптер не меняются.
"""
from __future__ import annotations

from finvex.core.errors import DataFormatError
from finvex.core.models import CandleSeries, MSK


def to_kronos_frame(series: CandleSeries, *, require_quality_ok: bool = True):
    """
    timestamps — время открытия бара по часам биржи без tz (час и день недели
    для Kronos значимы в локальном времени сессии). amount — оборот, если он есть
    у всех баров.
    """
    import pandas as pd

    if require_quality_ok and series.quality is not None and not series.quality.ok:
        codes = ", ".join(i.code for i in series.quality.issues if i.severity == "error")
        raise DataFormatError(f"Ряд {series.instrument} не прошёл контроль качества: {codes}")
    if not series.candles:
        raise DataFormatError(f"Пустой ряд {series.instrument}")

    df = series.to_frame(MSK).reset_index()
    df["timestamps"] = df["ts_open"].dt.tz_localize(None)
    cols = ["timestamps", "open", "high", "low", "close", "volume"]
    if df["turnover"].notna().all():
        df["amount"] = df["turnover"]
        cols.append("amount")
    return df[cols].sort_values("timestamps").reset_index(drop=True)
