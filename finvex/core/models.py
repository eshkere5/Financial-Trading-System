"""
Канонические модели данных Finvex.

Провайдеры переводят «сырые» ответы бирж и брокеров в эти типы (идея Nautilus
Trader: адаптер превращает протокол площадки в доменные объекты). Дальше по
системе — Kronos, тактика, риск, интерфейс — работают только с ними.

Время: все метки времени — tz-aware UTC. Цены/объёмы — float (аналитический
контур). Для исполнения денежные величины следует приводить к Decimal на
границе ExecutionProvider.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

# Москва — UTC+3 без перехода на летнее время с 2014 г.; фиксированный сдвиг
# не требует tzdata и одинаково работает в Windows и Linux.
MSK = timezone(timedelta(hours=3), "MSK")


class AssetClass(str, Enum):
    EQUITY = "equity"
    BOND = "bond"
    FUTURE = "future"
    CURRENCY = "currency"
    CRYPTO = "crypto"
    INDEX = "index"
    OTHER = "other"


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M10 = "10m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"

    @property
    def seconds(self) -> int:
        return _TF_SECONDS[self]

    @classmethod
    def parse(cls, value: "Timeframe | str") -> "Timeframe":
        if isinstance(value, Timeframe):
            return value
        for tf in cls:
            if value == tf.value or value.upper() == tf.name:
                return tf
        raise ValueError(f"Неизвестный таймфрейм: {value!r}")


_TF_SECONDS: Dict[Timeframe, int] = {
    Timeframe.M1: 60, Timeframe.M5: 300, Timeframe.M10: 600, Timeframe.M15: 900,
    Timeframe.M30: 1800, Timeframe.H1: 3600, Timeframe.H4: 14400,
    Timeframe.D1: 86400, Timeframe.W1: 604800, Timeframe.MN1: 2678400,
}


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass(frozen=True, slots=True)
class InstrumentId:
    """Идентификатор вида SBER.MOEX (как SYMBOL.VENUE в Nautilus)."""
    symbol: str
    venue: str

    def __str__(self) -> str:
        return f"{self.symbol}.{self.venue}"

    @classmethod
    def parse(cls, text: "InstrumentId | str", default_venue: Optional[str] = None) -> "InstrumentId":
        if isinstance(text, InstrumentId):
            return text
        sym, sep, venue = text.rpartition(".")
        if sep and sym and venue:
            return cls(sym.upper(), venue.upper())
        if default_venue:
            return cls(text.upper(), default_venue.upper())
        raise ValueError(f"Ожидался формат SYMBOL.VENUE, получено {text!r}")


@dataclass(frozen=True, slots=True)
class Instrument:
    id: InstrumentId
    asset_class: AssetClass
    name: str = ""
    isin: Optional[str] = None
    currency: Optional[str] = None
    lot_size: int = 1
    tick_size: Optional[float] = None
    price_precision: Optional[int] = None
    board: Optional[str] = None


@dataclass(frozen=True, slots=True)
class Candle:
    ts_open: datetime
    ts_close: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: Optional[float] = None  # оборот в валюте инструмента


@dataclass(frozen=True, slots=True)
class Quote:
    instrument: InstrumentId
    bid: Optional[float]
    ask: Optional[float]
    last: Optional[float]
    volume_today: Optional[float]
    trading_status: Optional[str]
    ts_snapshot: Optional[datetime]   # когда биржа сформировала снимок (НЕ время рынка)
    ts_received: datetime
    ts_last_trade: Optional[datetime] = None  # время последней сделки: по нему судим о свежести цены

    def age_seconds(self, now: datetime) -> Optional[float]:
        """Возраст цены last. None, если биржа не отдала время последней сделки."""
        if self.ts_last_trade is None:
            return None
        return (now - self.ts_last_trade).total_seconds()

    @property
    def spread(self) -> Optional[float]:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid

    @property
    def mid(self) -> Optional[float]:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True, slots=True)
class Trade:
    instrument: InstrumentId
    trade_id: str
    price: float
    quantity: float
    side: Optional[Side]
    ts_event: datetime


@dataclass(frozen=True, slots=True)
class QualityIssue:
    code: str
    severity: str  # "info" | "warning" | "error"
    message: str
    count: int = 1


@dataclass(frozen=True, slots=True)
class DataQuality:
    """Результат проверки ряда. Попадает в EvidencePackage и журнал решений."""
    n_bars: int
    issues: Tuple[QualityIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def worst(self) -> str:
        sev = {i.severity for i in self.issues}
        return "error" if "error" in sev else "warning" if "warning" in sev else "ok"


@dataclass(frozen=True)
class CandleSeries:
    """Свечи вместе с происхождением (источник, время получения, качество)."""
    instrument: InstrumentId
    timeframe: Timeframe
    candles: Tuple[Candle, ...]
    source: str
    fetched_at: datetime
    quality: Optional[DataQuality] = None

    def __len__(self) -> int:
        return len(self.candles)

    @property
    def last_close_time(self) -> Optional[datetime]:
        return self.candles[-1].ts_close if self.candles else None

    def to_frame(self, exchange_tz: timezone = MSK):
        """DataFrame с индексом времени открытия бара (tz-aware, в поясе биржи)."""
        import pandas as pd
        idx = pd.DatetimeIndex([c.ts_open.astimezone(exchange_tz) for c in self.candles], name="ts_open")
        return pd.DataFrame(
            {
                "open": [c.open for c in self.candles],
                "high": [c.high for c in self.candles],
                "low": [c.low for c in self.candles],
                "close": [c.close for c in self.candles],
                "volume": [c.volume for c in self.candles],
                "turnover": [c.turnover for c in self.candles],
            },
            index=idx,
        )


# ── контракты портфеля и исполнения ────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Position:
    instrument: InstrumentId
    quantity: float
    avg_price: Optional[float] = None
    market_price: Optional[float] = None


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    account_id: str
    cash: Dict[str, float]
    positions: Tuple[Position, ...]
    ts: datetime
    source: str


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """
    Заявка на исполнение. plan_id обязателен: исполнять можно только то, что
    прошло согласование как TradePlan (аналитика не вызывает исполнение напрямую).
    """
    plan_id: str
    client_order_id: str
    instrument: InstrumentId
    side: Side
    quantity: float
    order_type: OrderType = OrderType.LIMIT
    limit_price: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.plan_id:
            raise ValueError("OrderRequest требует plan_id утверждённого TradePlan")
        if self.quantity <= 0:
            raise ValueError("quantity должна быть > 0")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Для LIMIT-заявки нужен limit_price")


@dataclass(frozen=True, slots=True)
class OrderAck:
    client_order_id: str
    venue_order_id: Optional[str]
    status: str
    ts: datetime
