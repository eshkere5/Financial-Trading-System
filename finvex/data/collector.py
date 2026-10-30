"""
DataCollector — запрос котировок из Tinkoff и/или исторических файлов.
Оригинал из твоего data_collector.py, адаптирован под новую структуру.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


class DataCollector:
    def __init__(self, client=None, historical_dir: str = "data/raw/historical") -> None:
        self._client = client
        self._hist_dir = Path(historical_dir)

    def fetch_live(self, tickers: list[str]) -> pd.DataFrame:
        if self._client is None:
            raise RuntimeError("TinkoffClient not provided")
        prices = self._client.get_prices(tickers)
        return pd.DataFrame([prices])

    # ↓↓↓ добавим то, что ожидает старый тест ↓↓↓

    def get_last_prices(self, tickers: list[str]) -> dict[str, float]:
        """Совместимость со старым интерфейсом: просто обёртка над client.get_prices()."""
        if self._client is None:
            raise RuntimeError("TinkoffClient not provided")
        return self._client.get_prices(tickers)

    def get_last_prices_df(self, tickers: list[str]) -> pd.DataFrame:
        """Если тест ожидает датафрейм цен."""
        prices = self.get_last_prices(tickers)
        return pd.DataFrame([prices])

    def load_historical(self, ticker: str) -> pd.DataFrame:
        path = self._hist_dir / f"{ticker}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No historical data for {ticker}: {path}")
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        logger.debug("Loaded %d rows for %s", len(df), ticker)
        return df
