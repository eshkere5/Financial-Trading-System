"""
Локальные свечи из CSV: офлайн-режим, воспроизводимые бэктесты, данные, которые
пользователь выгрузил сам. Файл: <directory>/<SYMBOL>_<tf>.csv, например SBER_1h.csv.
Обязательные колонки: timestamp, open, high, low, close, volume (turnover — по желанию).
timestamp без пояса трактуется как время биржи (по умолчанию MSK).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence

from finvex.core.base import Capability, MarketDataProvider, ProviderInfo
from finvex.core.errors import DataFormatError, ProviderError
from finvex.core.models import Candle, CandleSeries, Instrument, InstrumentId, MSK, Timeframe, AssetClass
from finvex.core.quality import check_candles
from finvex.core.registry import register

_REQUIRED = ("timestamp", "open", "high", "low", "close", "volume")


@register("csv_file", kind="market_data")
class CsvFileProvider(MarketDataProvider):
    def __init__(self, directory: str, venue: str = "LOCAL", tz_offset_hours: int = 3) -> None:
        self.dir = Path(directory)
        self.venue = venue.upper()
        self._tz = timezone(timedelta(hours=tz_offset_hours))
        self._info = ProviderInfo(
            name=f"csv_file:{self.dir.name}", venue=self.venue,
            capabilities=Capability.INSTRUMENTS | Capability.CANDLES,
        )

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def _path(self, sym: str, tf: Timeframe) -> Path:
        return self.dir / f"{sym}_{tf.value}.csv"

    def get_instruments(self) -> Sequence[Instrument]:
        syms = sorted({p.stem.rsplit("_", 1)[0] for p in self.dir.glob("*_*.csv")})
        return [Instrument(InstrumentId(s, self.venue), AssetClass.OTHER) for s in syms]

    def get_candles(self, instrument, timeframe, start: datetime, end: Optional[datetime] = None,
                    *, only_closed: bool = True) -> CandleSeries:
        import pandas as pd

        tf = Timeframe.parse(timeframe)
        iid = InstrumentId.parse(instrument, default_venue=self.venue)
        path = self._path(iid.symbol, tf)
        if not path.exists():
            raise ProviderError(f"Нет файла {path}")
        df = pd.read_csv(path, encoding="utf-8")
        df.columns = [c.lower() for c in df.columns]
        missing = [c for c in _REQUIRED if c not in df.columns]
        if missing:
            raise DataFormatError(f"{path.name}: нет колонок {missing}")

        ts = pd.to_datetime(df["timestamp"])
        ts = ts.dt.tz_localize(self._tz) if ts.dt.tz is None else ts
        ts = ts.dt.tz_convert(timezone.utc)
        now = datetime.now(timezone.utc)
        end = end or now
        step = timedelta(seconds=tf.seconds)
        has_turn = "turnover" in df.columns
        candles = []
        for i in range(len(df)):
            t0 = ts.iloc[i].to_pydatetime()
            candles.append(Candle(
                ts_open=t0, ts_close=t0 + step,
                open=float(df["open"].iloc[i]), high=float(df["high"].iloc[i]),
                low=float(df["low"].iloc[i]), close=float(df["close"].iloc[i]),
                volume=float(df["volume"].iloc[i]),
                turnover=float(df["turnover"].iloc[i]) if has_turn else None,
            ))
        candles = [c for c in candles if start <= c.ts_open <= end and (not only_closed or c.ts_close <= now)]
        return CandleSeries(
            instrument=iid, timeframe=tf, candles=tuple(candles),
            source=self.info.name, fetched_at=now, quality=check_candles(candles, tf, now=now),
        )
