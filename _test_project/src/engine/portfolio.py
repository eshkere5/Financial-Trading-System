"""Управление позициями и маржой.

### NEW 2026-08-15 ###
Каждая подтверждённая сделка дополнительно пишется в data/trading.db
через TradeJournal — персистентность на уровне компонента, а не теста:
журнал ведут и live-цикл, и soak, и бэктест, и pending-подтверждения.
"""

from __future__ import annotations

import logging
from src.utils.time_utils import utcnow

from src.engine.market_state import MarketState, Position, Trade
from src.engine.orders import Order, OrderSide
from src.engine.trade_journal import TradeJournal

logger = logging.getLogger(__name__)


class Portfolio:
    def __init__(self, state: MarketState, commission: float) -> None:
        self._state = state
        self._commission = commission
        # data/trading.db; при ошибке инициализации журнал молча отключается
        self._journal = TradeJournal()
        # Защита от повторного локального применения одной заявки.
        # Broker reconciliation всё равно остаётся источником истины.
        self._applied_order_ids: set[str] = set()

    def apply_order(self, order: Order) -> bool:
        """Применяет исполненный ордер к состоянию портфеля. Возвращает True при успехе."""
        price = order.filled_price
        if price is None:
            logger.warning("Order %s has no filled_price", order.order_id)
            return False

        order_id = str(order.order_id or "")
        if order_id and order_id in self._applied_order_ids:
            logger.warning(
                "DUPLICATE FILL ignored | order_id=%s ticker=%s",
                order_id,
                order.ticker,
            )
            return False

        cost = price * abs(order.qty)
        comm = cost * self._commission
        is_buy = order.side == OrderSide.BUY

        accounting_mode = getattr(
            self._state,
            "accounting_mode",
            "spot",
        )

        # Spot: reserve full purchase cost locally.
        # Linear/inverse derivatives: broker equity is authoritative and
        # cash changes only on reconciliation; full notional is never spent.
        if accounting_mode == "spot":
            if is_buy and self._state.cash < cost + comm:
                logger.warning(
                    "Insufficient cash for %s: need %.2f, have %.2f",
                    order.ticker,
                    cost + comm,
                    self._state.cash,
                )
                order.mark_rejected("insufficient_cash")
                return False

            if is_buy:
                self._state.cash -= cost + comm
            else:
                self._state.cash += cost - comm
        else:
            logger.info(
                "DERIVATIVE FILL | ticker=%s side=%s qty=%.8f "
                "notional=%.8f — cash deferred to reconciliation",
                order.ticker,
                order.side.value,
                order.qty,
                cost,
            )

        # ── обновление позиции ────────────────────────────────────────
        sign = 1 if is_buy else -1
        pos = self._state.positions.get(order.ticker)

        if pos is None:
            self._state.positions[order.ticker] = Position(
                ticker=order.ticker,
                qty=sign * order.qty,
                avg_price=price,
                open_time=utcnow(),
            )
        else:
            new_qty = pos.qty + sign * order.qty
            if abs(new_qty) < 1e-9:
                del self._state.positions[order.ticker]
            else:
                old_qty = pos.qty

                # Если сделка перевернула позицию через ноль, остаток новой
                # позиции считается открытым по цене текущего исполнения.
                if old_qty * new_qty < 0:
                    pos.qty = new_qty
                    pos.avg_price = price
                    pos.open_time = utcnow()
                else:
                    # Средняя цена меняется только при увеличении позиции
                    # в прежнем направлении.
                    increasing = (
                        (old_qty > 0 and is_buy)
                        or (old_qty < 0 and not is_buy)
                    )
                    if increasing:
                        pos.avg_price = (
                            pos.avg_price * abs(old_qty)
                            + price * order.qty
                        ) / abs(new_qty)

                    pos.qty = new_qty

        # ── запись сделки (in-memory) ─────────────────────────────────
        self._state.trades.append(Trade(
            ticker=order.ticker,
            qty=sign * order.qty,
            price=price,
            side=order.side.value,
            ts=utcnow(),
            commission=comm,
        ))
        order.mark_fill(
            filled_qty=order.qty,
            filled_price=price,
            broker_order_id=order.broker_order_id,
        )

        if order_id:
            self._applied_order_ids.add(order_id)

        # Persistent journal не должен делать уже применённый fill
        # повторно применимым при временной ошибке SQLite.
        try:
            self._journal.record(
                ticker=order.ticker,
                side=order.side.value,
                qty=order.qty,
                price=price,
                commission=comm,
                order_id=order_id,
                source="live",
            )
        except Exception:
            logger.exception(
                "Trade journal write failed after fill | "
                "order_id=%s ticker=%s",
                order_id,
                order.ticker,
            )

        logger.info(
            "ORDER FILLED | ticker=%s side=%s qty=%.8f "
            "price=%.8f commission=%.8f order_id=%s",
            order.ticker,
            order.side.value,
            order.qty,
            price,
            comm,
            order_id,
        )
        return True
