"""
Слой адаптеров веб-интерфейса.

Интерфейс видит рынок и исполнение только через этот реестр, поэтому подключение нового источника данных или
брокера - это новый класс здесь, а не правка интерфейса. Исполнение по умолчанию ОТКЛЮЧЕНО: ордера не уходят,
пока явно не зарегистрирован адаптер с enabled=True.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Protocol

import pandas as pd


@dataclass
class OrderRequest:
    symbol: str
    side: str            # buy | sell
    qty: float
    order_type: str = "market"   # market | limit
    price: Optional[float] = None
    client_id: str = ""


class MarketDataAdapter(Protocol):
    id: str
    name: str

    def candles(self, symbol: str, tf: str, start: datetime, end: datetime) -> pd.DataFrame: ...


class ExecutionAdapter(Protocol):
    id: str
    name: str
    enabled: bool

    def place_order(self, order: OrderRequest) -> dict[str, Any]: ...
    def positions(self) -> list[dict[str, Any]]: ...


class DisabledExecution:
    id, name, enabled = "none", "Исполнение не подключено", False

    def place_order(self, order: OrderRequest) -> dict[str, Any]:
        raise PermissionError("адаптер исполнения не подключён")

    def positions(self) -> list[dict[str, Any]]:
        return []


@dataclass
class Registry:
    market: list[Any] = field(default_factory=list)
    execution: Any = field(default_factory=DisabledExecution)

    def describe(self) -> dict[str, Any]:
        rows = [{"id": a.id, "name": a.name, "kind": "данные", "status": "read-only"} for a in self.market]
        rows.append({"id": self.execution.id, "name": self.execution.name, "kind": "исполнение",
                     "status": "enabled" if self.execution.enabled else "disabled"})
        return {"adapters": rows, "execution_enabled": bool(self.execution.enabled)}

    def submit(self, order: OrderRequest) -> dict[str, Any]:
        if not self.execution.enabled:
            raise PermissionError("исполнение отключено: подключите адаптер исполнения")
        if order.side not in ("buy", "sell") or order.qty <= 0 or order.order_type not in ("market", "limit"):
            raise ValueError("некорректный ордер")
        if order.order_type == "limit" and not order.price:
            raise ValueError("для лимитного ордера нужна цена")
        return self.execution.place_order(order)


class _Feed:
    def __init__(self, id_: str, name: str) -> None:
        self.id, self.name = id_, name


def default_registry() -> Registry:
    return Registry(market=[_Feed("moex", "Мосбиржа (ISS), TQBR"), _Feed("bybit", "Bybit, USDT-перпетуалы")])
