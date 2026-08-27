"""Управление позициями и маржой.

### NEW 2026-08-15 ###
Каждая подтверждённая сделка дополнительно пишется в data/trading.db
через TradeJournal — персистентность на уровне компонента, а не теста:
журнал ведут и live-цикл, и soak, и бэктест, и pending-подтверждения.
"""

from __future__ import annotations

import logging
from datetime import datetime

from src.engine.market_state import MarketState, Position, Trade
from src.engine.orders import Order, OrderSide, OrderStatus
from src.engine.trade_journal import TradeJournal

logger = logging.getLogger(__name__)


class Portfolio:
    def __init__(self, state: MarketState, commission: float) -> None:
        self._state = state
        self._commission = commission
        # data/trading.db; при ошибке инициализации журнал молча отключается
        self._journal = TradeJournal()

    def apply_order(self, order: Order) -> bool:
        """Применяет исполненный ордер к состоянию портфеля. Возвращает True при успехе."""
        price = order.filled_price
        if price is None:
            logger.warning("Order %s has no filled_price", order.order_id)
            return False

        cost = price * abs(order.qty)
        comm = cost * self._commission
        is_buy = order.side == OrderSide.BUY

        # ── проверка кэша ─────────────────────────────────────────────
        if is_buy and self._state.cash < cost + comm:
            logger.warning(
                "Insufficient cash for %s: need %.2f, have %.2f",
                order.ticker, cost + comm, self._state.cash,
            )
            order.status = OrderStatus.REJECTED
            return False

        # ── обновление кэша ───────────────────────────────────────────
        # BUY: cash уменьшается на cost + comm
        # SELL: cash увеличивается на cost - comm (комиссия из выручки)
        if is_buy:
            self._state.cash -= cost + comm
        else:
            self._state.cash += cost - comm

        # ── обновление позиции ────────────────────────────────────────
        sign = 1 if is_buy else -1
        pos = self._state.positions.get(order.ticker)

        if pos is None:
            self._state.positions[order.ticker] = Position(
                ticker=order.ticker,
                qty=sign * order.qty,
                avg_price=price,
                open_time=datetime.utcnow(),
            )
        else:
            new_qty = pos.qty + sign * order.qty
            if abs(new_qty) < 1e-9:
                del self._state.positions[order.ticker]
            else:
                # avg_price пересчитываем только при увеличении позиции
                if (pos.qty > 0 and is_buy) or (pos.qty < 0 and not is_buy):
                    pos.avg_price = (
                        pos.avg_price * abs(pos.qty) + price * order.qty
                    ) / abs(new_qty)
                pos.qty = new_qty

        # ── запись сделки (in-memory) ─────────────────────────────────
        self._state.trades.append(Trade(
            ticker=order.ticker,
            qty=sign * order.qty,
            price=price,
            side=order.side.value,
            ts=datetime.utcnow(),
            commission=comm,
        ))
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.utcnow()

        # ── запись сделки (persistent, data/trading.db) ───────────────
        self._journal.record(
            ticker=order.ticker,
            side=order.side.value,
            qty=order.qty,
            price=price,
            commission=comm,
            order_id=str(order.order_id or ""),
            source="live",
        )
        return True
