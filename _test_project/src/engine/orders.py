"""
Типы ордеров и их статусы.

Order хранит локальный жизненный цикл заявки. `order_id` — стабильный
client-side idempotency key, который передаётся брокеру на всех ретраях.
`broker_order_id` — идентификатор, возвращённый брокером.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from src.utils.time_utils import utcnow


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


_TERMINAL_STATUSES = {
    OrderStatus.FILLED,
    OrderStatus.CANCELLED,
    OrderStatus.REJECTED,
}


@dataclass
class Order:
    ticker: str
    side: OrderSide
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_qty: float = 0.0
    created_at: datetime = field(default_factory=utcnow)
    filled_at: Optional[datetime] = None

    # Стабильный UUID заявки: используется Tinkoff как idempotency key.
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Идентификатор, который вернул broker после принятия заявки.
    broker_order_id: Optional[str] = None

    # Закрытие позиции/stop-loss/take-profit всегда разрешено при strategist halt.
    closes_position: bool = False

    # Причина отклонения/отмены: диагностируется в логах и backtest JSONL.
    reject_reason: Optional[str] = None

    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper().strip()

        if not self.ticker:
            raise ValueError("Order.ticker must not be empty")

        if not isinstance(self.side, OrderSide):
            self.side = OrderSide(str(self.side).lower())

        if not isinstance(self.order_type, OrderType):
            self.order_type = OrderType(str(self.order_type).lower())

        self.qty = float(self.qty)
        if not math.isfinite(self.qty) or self.qty <= 0:
            raise ValueError(f"Order.qty must be finite and > 0, got {self.qty}")

        self.filled_qty = float(self.filled_qty)
        if not math.isfinite(self.filled_qty) or self.filled_qty < 0:
            raise ValueError(
                f"Order.filled_qty must be finite and >= 0, got {self.filled_qty}"
            )
        if self.filled_qty > self.qty + 1e-12:
            raise ValueError(
                f"Order.filled_qty={self.filled_qty} exceeds qty={self.qty}"
            )

        if self.order_type == OrderType.LIMIT:
            if self.limit_price is None or not math.isfinite(float(self.limit_price)):
                raise ValueError("Limit order requires finite limit_price")
            if float(self.limit_price) <= 0:
                raise ValueError("limit_price must be > 0")
            self.limit_price = float(self.limit_price)
        elif self.limit_price is not None:
            self.limit_price = float(self.limit_price)

        if self.filled_price is not None:
            self.filled_price = float(self.filled_price)
            if not math.isfinite(self.filled_price) or self.filled_price <= 0:
                raise ValueError("filled_price must be finite and > 0")

        if not self.order_id:
            self.order_id = str(uuid.uuid4())

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES

    @property
    def remaining_qty(self) -> float:
        return max(0.0, self.qty - self.filled_qty)

    @property
    def is_fully_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    def mark_accepted(self, broker_order_id: Optional[str] = None) -> None:
        if self.is_terminal:
            return
        self.status = OrderStatus.ACCEPTED
        if broker_order_id:
            self.broker_order_id = broker_order_id

    def mark_fill(
        self,
        filled_qty: float,
        filled_price: float,
        *,
        broker_order_id: Optional[str] = None,
    ) -> None:
        if self.is_terminal:
            return

        filled_qty = float(filled_qty)
        filled_price = float(filled_price)

        if not math.isfinite(filled_qty) or filled_qty <= 0:
            raise ValueError(f"filled_qty must be finite and > 0, got {filled_qty}")
        if not math.isfinite(filled_price) or filled_price <= 0:
            raise ValueError(
                f"filled_price must be finite and > 0, got {filled_price}"
            )

        old_filled = self.filled_qty
        new_filled = min(self.qty, old_filled + filled_qty)

        if old_filled <= 0 or self.filled_price is None:
            self.filled_price = filled_price
        else:
            self.filled_price = (
                self.filled_price * old_filled + filled_price * filled_qty
            ) / (old_filled + filled_qty)

        self.filled_qty = new_filled
        if broker_order_id:
            self.broker_order_id = broker_order_id

        if self.remaining_qty <= 1e-12:
            self.status = OrderStatus.FILLED
            self.filled_at = utcnow()
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def mark_rejected(self, reason: str) -> None:
        if self.is_terminal:
            return
        self.status = OrderStatus.REJECTED
        self.reject_reason = str(reason)[:1000]

    def mark_cancelled(self, reason: str = "") -> None:
        if self.is_terminal:
            return
        self.status = OrderStatus.CANCELLED
        self.reject_reason = str(reason)[:1000] or None