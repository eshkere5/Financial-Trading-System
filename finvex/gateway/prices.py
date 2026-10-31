"""Источники цен: живой (ProviderHub проекта) и детерминированный (демо, тесты)."""
from __future__ import annotations

import concurrent.futures as cf
import threading
import time as _time
import zlib
from datetime import datetime, timedelta
from typing import Optional, Protocol

import numpy as np
import pandas as pd

SPAN = {"1h": timedelta(days=45), "1d": timedelta(days=500)}


class PriceSource(Protocol):
    def candles(self, symbol: str, tf: str, bars: int, now: datetime) -> pd.DataFrame: ...


class HubPriceSource:
    """Данные через ProviderHub. Короткий кэш и жёсткий таймаут: шлюз не должен зависать на бирже."""

    def __init__(self, hub, name: Optional[str] = None, ttl: float = 30.0, timeout: float = 12.0) -> None:
        self.hub, self.name, self.ttl, self.timeout = hub, name, ttl, timeout
        self._pool = cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix="prices")
        self._cache: dict = {}
        self._lock = threading.Lock()

    def _provider(self, symbol: str):
        """Крипто-тикеры (…USDT, …USDC) идут к провайдеру Bybit, если он настроен; остальное к источнику по умолчанию."""
        if self.name is None and symbol.upper().endswith(("USDT", "USDC")):
            for n in self.hub.market_data_names:
                if getattr(self.hub.market_data(n).info, "venue", "") == "BYBIT":
                    return self.hub.market_data(n)
        return self.hub.market_data(self.name)

    def candles(self, symbol: str, tf: str, bars: int, now: datetime) -> pd.DataFrame:
        key = (symbol, tf)
        with self._lock:
            hit = self._cache.get(key)
            if hit and _time.time() - hit[0] < self.ttl:
                return hit[1].tail(bars)
        fut = self._pool.submit(self._provider(symbol).get_candles, symbol, tf, now - SPAN[tf], now, only_closed=True)
        try:
            series = fut.result(timeout=self.timeout)
        except cf.TimeoutError:
            raise RuntimeError(f"источник данных не ответил за {self.timeout:g} с")
        df = series.to_frame()
        if df.empty:
            raise RuntimeError(f"нет свечей {symbol} {tf}")
        with self._lock:
            self._cache[key] = (_time.time(), df)
        return df.tail(bars)


SYNTH = {
    "SBER": (295.0, 0.0004, 0.008, None), "GAZP": (130.0, 0.0002, 0.009, None), "LKOH": (5600.0, 0.0003, 0.008, None),
    "ROSN": (430.0, 0.0003, 0.009, None), "GMKN": (130.0, 0.0001, 0.011, None), "YNDX": (4200.0, 0.0005, 0.012, None),
    "VTBR": (85.0, 0.0001, 0.010, "crash"), "MTSS": (230.0, 0.0002, 0.007, None), "MOEX": (205.0, 0.0003, 0.009, None),
    "BTCUSDT": (95000.0, 0.0004, 0.014, None), "ETHUSDT": (3400.0, 0.0004, 0.016, None),
}


class StaticPriceSource:
    """Синтетические свечи с фиксированным зерном: для демонстрации и тестов без сети."""

    @staticmethod
    def _walk(symbol: str, tf: str, n: int = 320) -> np.ndarray:
        base, drift, vol, special = SYNTH[symbol]
        rng = np.random.default_rng(zlib.crc32(symbol.encode()) + (0 if tf == "1d" else 7))
        scale = 1.0 if tf == "1d" else 0.35
        px = base * np.exp(np.cumsum(rng.normal(drift * scale, vol * scale, n)))
        if special == "crash":
            px[-8:] *= np.cumprod(np.full(8, 0.95))
        return px

    def candles(self, symbol: str, tf: str, bars: int, now: datetime) -> pd.DataFrame:
        if symbol not in SYNTH:
            raise RuntimeError(f"нет данных по {symbol}")
        n = 320
        daily = self._walk(symbol, "1d", n)
        px = daily if tf == "1d" else self._walk(symbol, "1h", n)
        if tf != "1d":
            px = px * (daily[-1] / px[-1])
        vol = SYNTH[symbol][2] * (1.0 if tf == "1d" else 0.35)
        step = timedelta(hours=1) if tf == "1h" else timedelta(days=1)
        end = now.replace(minute=0, second=0, microsecond=0)
        end = end - timedelta(hours=1) if tf == "1h" else end.replace(hour=0)
        idx = pd.DatetimeIndex([end - step * (n - 1 - k) for k in range(n)], name="ts_open")
        rng = np.random.default_rng(zlib.crc32(symbol.encode()) + 99)
        return pd.DataFrame({"open": np.concatenate([[px[0]], px[:-1]]), "high": px * (1 + vol * 0.4),
                             "low": px * (1 - vol * 0.4), "close": px, "volume": rng.uniform(1e5, 5e5, n)},
                            index=idx).tail(bars)
