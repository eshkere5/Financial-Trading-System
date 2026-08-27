"""
src/executors/bybit_execution_router.py
Async execution router для Bybit.
"""
from __future__ import annotations
import asyncio
import logging
import uuid
from typing import List, Tuple, Dict

from src.engine.orders import Order, OrderStatus, OrderSide
from src.engine.config_loader import load_bybit_config
from src.executors.bybit_client import BybitClient
from src.executors.lot_utils import round_qty_to_step

logger = logging.getLogger(__name__)
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0


def _backoff_delay(attempt: int) -> float:
    return RETRY_BASE_DELAY * (2 ** (attempt - 1))


class BybitExecutionRouter:
    def __init__(self, category: str = "spot", testnet: bool = True) -> None:
        cfg = load_bybit_config(testnet=testnet)
        # Фикс С7: category из yaml перезаписывается только если явно не передан
        if category:
            cfg.category = category
        self.category = cfg.category
        self.client = BybitClient(cfg)

    @classmethod
    async def create(cls, category: str = "spot", testnet: bool = True) -> "BybitExecutionRouter":
        return cls(category=category, testnet=testnet)

    async def fetch_prices(self, tickers: List[str]) -> Dict[str, float]:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, self.client.get_prices, tickers)
        except Exception as exc:
            logger.exception("fetch_prices failed: %s", exc)
            return {}

    async def get_portfolio(self) -> dict:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, self.client.get_portfolio)
        except Exception as exc:
            logger.exception("get_portfolio failed: %s", exc)
            return {"positions": [], "cash": 0.0}

    async def submit(self, order: Order) -> bool:
        if not order.order_id:
            order.order_id = str(uuid.uuid4())

        loop = asyncio.get_running_loop()

        # Фикс С5: округление вниз до qtyStep инструмента (Decimal)
        spec = await loop.run_in_executor(None, self.client.get_instrument_spec, order.ticker)
        qty = round_qty_to_step(abs(order.qty), spec["qty_step"], spec["min_order_qty"])
        if qty <= 0:
            logger.warning(
                "[BYBIT] Order below min_order_qty, skipped: %s qty=%s spec=%s",
                order.ticker, order.qty, spec,
            )
            order.status = OrderStatus.REJECTED
            return False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await loop.run_in_executor(
                    None,
                    lambda: self.client.place_order(
                        symbol=order.ticker,
                        side="Buy" if order.side == OrderSide.BUY else "Sell",
                        qty=str(qty),
                        order_type="Market",
                        market_unit="baseCoin" if self.category == "spot" else None,
                    ),
                )
                if resp.get("retCode") == 0:
                    order.status = OrderStatus.FILLED
                    order.order_id = resp.get("result", {}).get("orderId") or order.order_id
                    logger.info("[BYBIT] Order OK %s %s x%s id=%s", order.side, order.ticker, qty, order.order_id)
                    return True
                raise RuntimeError(resp.get("retMsg"))
            except Exception as exc:
                delay = _backoff_delay(attempt)
                logger.warning("[BYBIT] Submit attempt %d/%d failed %s, retry in %.1fs", attempt, MAX_RETRIES, exc, delay)
                await asyncio.sleep(delay)

        order.status = OrderStatus.REJECTED
        return False

    async def submit_pair(self, order_a: Order, order_b: Order) -> Tuple[bool, bool]:
        ok_a, ok_b = await asyncio.gather(
            self.submit(order_a), self.submit(order_b), return_exceptions=False,
        )
        if ok_a and not ok_b:
            logger.error(
                "[BYBIT] LEG RISK: leg A %s filled but leg B %s REJECTED — portfolio unbalanced!",
                order_a.ticker, order_b.ticker,
            )
        elif ok_b and not ok_a:
            logger.error(
                "[BYBIT] LEG RISK: leg B %s filled but leg A %s REJECTED — portfolio unbalanced!",
                order_b.ticker, order_a.ticker,
            )
        return ok_a, ok_b

    async def close(self) -> None:
        """Фикс С6: реально закрывает HTTP-сессию клиента вместо no-op."""
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self.client.close)
        except Exception as exc:
            logger.warning("[BYBIT] close() failed: %s", exc)