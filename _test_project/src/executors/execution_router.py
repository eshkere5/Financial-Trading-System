"""
ExecutionRouter — синхронная маршрутизация заявок.

Контракт submit(order):
- True: заявка имеет подтверждённый немедленный fill, её можно передать в
  Portfolio.apply_order().
- False: заявка отклонена либо принята брокером и ожидает подтверждения fill.

LiveEngine не должен вызывать Portfolio.apply_order() для ACCEPTED заявок.
Их факт исполнения приходит через reconcile или отдельный order-status poller.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from src.engine.orders import Order, OrderSide, OrderStatus
from src.executors.lot_utils import round_qty_to_step

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0


def _backoff_delay(attempt: int) -> float:
    return RETRY_BASE_DELAY * (2 ** (attempt - 1))


class ExecutionRouter:
    def __init__(
        self,
        mode: str = "sandbox",
        broker: str = "tinkoff",
        token: str = "",
        instrument_type: str = "share",
        currency: Optional[str] = "rub",
        bybit_cfg: Any = None,
    ) -> None:
        self.mode = mode
        self.broker = broker.lower().strip()

        if self.broker == "tinkoff":
            from src.executors.tinkoff_client import TinkoffClient

            self._client = TinkoffClient(
                token=token,
                sandbox=(mode == "sandbox"),
                instrument_type=instrument_type,
                currency=currency,
            )
        elif self.broker == "bybit":
            from src.engine.config_loader import load_bybit_config
            from src.executors.bybit_client import BybitClient

            config = bybit_cfg or load_bybit_config(testnet=(mode != "live"))
            self._client = BybitClient(config)
        else:
            raise ValueError(
                f"Unknown broker: {broker!r}; expected 'tinkoff' or 'bybit'"
            )

    @classmethod
    def from_cfg(cls, cfg: Any, token: str = "") -> "ExecutionRouter":
        instrument_cfg = getattr(cfg, "instrument", None) or {}
        if hasattr(instrument_cfg, "__dict__"):
            instrument_cfg = instrument_cfg.__dict__

        return cls(
            mode=getattr(cfg, "mode", "sandbox"),
            broker=getattr(cfg, "broker", "tinkoff"),
            token=token,
            instrument_type=instrument_cfg.get("type", "share"),
            currency=instrument_cfg.get("currency", "rub"),
        )

    def fetch_prices(self, tickers: list[str]) -> dict[str, float]:
        return self._client.get_prices(tickers)

    def get_portfolio(self) -> dict[str, Any]:
        get_portfolio = getattr(self._client, "get_portfolio", None)
        if get_portfolio is None:
            raise RuntimeError(
                f"{self.broker} client has no get_portfolio() implementation"
            )
        return get_portfolio()

    def submit(self, order: Order) -> bool:
        """
        Синхронная отправка заявки.

        Возвращает True только для немедленно исполненной заявки; в normal
        live-пути Tinkoff/Bybit это обычно False с order.status=ACCEPTED.
        """
        if order.is_terminal:
            logger.warning(
                "Order ignored: terminal status=%s id=%s",
                order.status,
                order.order_id,
            )
            return False

        if self.broker == "bybit":
            return self._submit_bybit(order)

        return self._submit_tinkoff(order)

    def _submit_tinkoff(self, order: Order) -> bool:
        from src.executors._sdk import OrderDirection

        direction = (
            OrderDirection.ORDER_DIRECTION_BUY
            if order.side == OrderSide.BUY
            else OrderDirection.ORDER_DIRECTION_SELL
        )

        try:
            figi = self._client.get_figi(order.ticker)
            lot_size = self._client.get_lot_size(order.ticker)
        except Exception as exc:
            order.mark_rejected(f"Instrument resolution failed: {exc}")
            logger.warning(
                "Tinkoff instrument resolution rejected | ticker=%s | error=%s",
                order.ticker,
                exc,
            )
            return False

        if not figi:
            order.mark_rejected(f"FIGI not resolved for {order.ticker}")
            logger.warning("Tinkoff FIGI not resolved | ticker=%s", order.ticker)
            return False

        if lot_size <= 0:
            order.mark_rejected(f"Invalid lot_size={lot_size} for {order.ticker}")
            return False

        lots = int(order.qty // lot_size)
        if lots < 1:
            order.mark_rejected(
                f"qty={order.qty} below lot size={lot_size} for {order.ticker}"
            )
            logger.warning(
                "Tinkoff order below one lot | ticker=%s qty=%s lot=%s",
                order.ticker,
                order.qty,
                lot_size,
            )
            return False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                broker_order_id = self._client.submit_order(
                    figi=figi,
                    direction=direction,
                    quantity=lots,
                    order_type=order.order_type,
                    order_id=order.order_id,
                )

                order.mark_accepted(str(broker_order_id))
                logger.info(
                    "Tinkoff order accepted | ticker=%s side=%s shares=%.8f lots=%d client_id=%s broker_id=%s",
                    order.ticker,
                    order.side.value,
                    order.qty,
                    lots,
                    order.order_id,
                    broker_order_id,
                )

                # Принятие broker API не является фактом исполнения.
                return False

            except Exception as exc:
                if attempt == MAX_RETRIES:
                    order.mark_rejected(
                        f"Tinkoff submit failed after {MAX_RETRIES} attempts: {exc}"
                    )
                    logger.exception(
                        "Tinkoff order rejected | ticker=%s id=%s",
                        order.ticker,
                        order.order_id,
                    )
                    return False

                delay = _backoff_delay(attempt)
                logger.warning(
                    "Tinkoff submit retry | ticker=%s attempt=%d/%d delay=%.1fs error=%s",
                    order.ticker,
                    attempt,
                    MAX_RETRIES,
                    delay,
                    exc,
                )
                time.sleep(delay)

        return False

    def _submit_bybit(self, order: Order) -> bool:
        side = "Buy" if order.side == OrderSide.BUY else "Sell"

        try:
            spec = self._client.get_instrument_spec(order.ticker)
            qty = round_qty_to_step(
                abs(order.qty),
                spec["qty_step"],
                spec["min_order_qty"],
            )
        except Exception as exc:
            order.mark_rejected(f"Bybit instrument specification failed: {exc}")
            logger.warning(
                "Bybit specification rejected | ticker=%s error=%s",
                order.ticker,
                exc,
            )
            return False

        if qty <= 0:
            order.mark_rejected(
                f"qty={order.qty} below Bybit minimum/step for {order.ticker}"
            )
            logger.warning(
                "Bybit order below minimum | ticker=%s qty=%s spec=%s",
                order.ticker,
                order.qty,
                spec,
            )
            return False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                broker_order_id = self._client.submit_order_simple(
                    symbol=order.ticker,
                    side=side,
                    qty=qty,
                    order_link_id=order.order_id,
                    reduce_only=bool(getattr(order, "closes_position", False)),
                )

                if not broker_order_id:
                    raise RuntimeError("Bybit returned empty order id")

                order.mark_accepted(str(broker_order_id))
                logger.info(
                    "Bybit order accepted | ticker=%s side=%s qty=%s client_id=%s broker_id=%s",
                    order.ticker,
                    side,
                    qty,
                    order.order_id,
                    broker_order_id,
                )

                return False

            except Exception as exc:
                if attempt == MAX_RETRIES:
                    order.mark_rejected(
                        f"Bybit submit failed after {MAX_RETRIES} attempts: {exc}"
                    )
                    logger.exception(
                        "Bybit order rejected | ticker=%s id=%s",
                        order.ticker,
                        order.order_id,
                    )
                    return False

                delay = _backoff_delay(attempt)
                logger.warning(
                    "Bybit submit retry | ticker=%s attempt=%d/%d delay=%.1fs error=%s",
                    order.ticker,
                    attempt,
                    MAX_RETRIES,
                    delay,
                    exc,
                )
                time.sleep(delay)

        return False
    def get_order_state(self, order: Order) -> dict:
        if not order.order_id:
            raise ValueError("Cannot query an order without broker order_id")

        if self.broker == "bybit":
            return self._client.get_order_state(
                order_id=order.order_id,
                symbol=order.ticker,
            )

        return self._client.get_order_state(order_id=order.order_id)


    def cancel_order(self, order: Order) -> None:
        if not order.order_id:
            return

        if self.broker == "bybit":
            self._client.cancel_order(
                order_id=order.order_id,
                symbol=order.ticker,
            )
            return

        self._client.cancel_order(order.order_id)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()