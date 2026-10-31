"""
Интерфейсы провайдеров.

Идеи из открытых систем:
- Nautilus Trader: адаптер = InstrumentProvider + DataClient + ExecutionClient;
  данные и исполнение — отдельные клиенты с разными правами.
- CCXT: словарь возможностей `has` — вызывающий код проверяет, что умеет биржа.
- OpenBB: Fetcher «Transform-Extract-Transform» — запрос приводится к формату
  источника, данные забираются, ответ приводится к стандартной модели.
Здесь это сведено к трём ABC и одному объекту ProviderInfo с набором Capability.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Flag, auto
from typing import Any, FrozenSet, Optional, Sequence

from finvex.core.errors import UnsupportedCapability, UnsupportedTimeframe
from finvex.core.models import (
    CandleSeries, Instrument, InstrumentId, OrderAck, OrderRequest,
    PortfolioSnapshot, Quote, Timeframe, Trade,
)


class Capability(Flag):
    NONE = 0
    INSTRUMENTS = auto()
    CANDLES = auto()
    QUOTES = auto()
    TRADES = auto()
    ORDERBOOK = auto()
    STREAMING = auto()
    PORTFOLIO = auto()
    EXECUTION = auto()


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    venue: str
    capabilities: Capability
    timeframes: FrozenSet[Timeframe] = frozenset()
    delayed: bool = False            # источник отдаёт данные с задержкой
    license_note: str = ""           # условия использования данных/API
    extra: Any = None


class BaseProvider(ABC):
    """Общий жизненный цикл. Контекстный менеджер закрывает сетевые ресурсы."""

    @property
    @abstractmethod
    def info(self) -> ProviderInfo: ...

    def supports(self, cap: Capability) -> bool:
        return cap != Capability.NONE and (self.info.capabilities & cap) == cap

    def require(self, cap: Capability) -> None:
        if not self.supports(cap):
            raise UnsupportedCapability(
                f"Провайдер {self.info.name!r} не поддерживает {cap.name}"
            )

    def require_timeframe(self, tf: Timeframe) -> None:
        if self.info.timeframes and tf not in self.info.timeframes:
            ok = ", ".join(sorted(t.value for t in self.info.timeframes))
            raise UnsupportedTimeframe(
                f"{self.info.name}: таймфрейм {tf.value} не поддерживается (доступно: {ok})"
            )

    def close(self) -> None:  # переопределяется при наличии ресурсов
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class MarketDataProvider(BaseProvider):
    """Только чтение рынка. У этого интерфейса нет метода отправки заявок."""

    @abstractmethod
    def get_instruments(self) -> Sequence[Instrument]: ...

    @abstractmethod
    def get_candles(
        self,
        instrument: "InstrumentId | str",
        timeframe: "Timeframe | str",
        start: datetime,
        end: Optional[datetime] = None,
        *,
        only_closed: bool = True,
    ) -> CandleSeries: ...

    def get_quotes(self, instruments: Optional[Sequence["InstrumentId | str"]] = None) -> Sequence[Quote]:
        raise UnsupportedCapability(f"{self.info.name}: котировки не поддерживаются")

    def get_trades(self, instrument: "InstrumentId | str", limit: int = 100) -> Sequence[Trade]:
        raise UnsupportedCapability(f"{self.info.name}: сделки не поддерживаются")


class PortfolioProvider(BaseProvider):
    @abstractmethod
    def get_portfolio(self) -> PortfolioSnapshot: ...


class ExecutionProvider(BaseProvider):
    """Исполнение. Принимает только OrderRequest с plan_id утверждённого плана."""

    @abstractmethod
    def submit(self, order: OrderRequest) -> OrderAck: ...

    @abstractmethod
    def cancel(self, client_order_id: str) -> OrderAck: ...


class AsyncMarketData:
    """Обёртка для async-движка (LiveEngine): синхронные вызовы идут в пул потоков."""

    def __init__(self, provider: MarketDataProvider) -> None:
        self._p = provider

    @property
    def info(self) -> ProviderInfo:
        return self._p.info

    async def get_instruments(self):
        return await asyncio.to_thread(self._p.get_instruments)

    async def get_candles(self, *a, **kw):
        return await asyncio.to_thread(self._p.get_candles, *a, **kw)

    async def get_quotes(self, *a, **kw):
        return await asyncio.to_thread(self._p.get_quotes, *a, **kw)

    async def get_trades(self, *a, **kw):
        return await asyncio.to_thread(self._p.get_trades, *a, **kw)
