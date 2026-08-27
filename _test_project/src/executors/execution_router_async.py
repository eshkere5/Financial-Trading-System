from __future__ import annotations
import asyncio
import logging
import time
import uuid
from typing import Optional, List, Tuple

from src.engine.orders import Order, OrderStatus, OrderSide

from src.executors.lot_utils import shares_to_lots as _shares_to_lots_pure 

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0

def _backoff_delay(attempt: int) -> float:
    return RETRY_BASE_DELAY * (2 ** (attempt - 1))

class AsyncExecutionRouter:
    def __init__(self, client, mode: str = "sandbox", broker: str = "tinkoff") -> None:
        self._client = client
        self.mode = mode
        self.broker = broker
        self._lot_cache: dict[str, int] = {}  # ticker → lot_size (только Tinkoff)

    @classmethod
    async def create(
        cls,
        mode: str = "sandbox",
        broker: str = "tinkoff",
        token: str = "",
        instrument_type: str = "share",
        currency: str = "rub",
        bybit_cfg=None,  # BybitConfig | None
    ) -> "AsyncExecutionRouter":
        if broker == "tinkoff":
            try:
                from t_tech.invest import AsyncClient
            except ImportError:
                from tinkoff.invest import AsyncClient  # type: ignore
            client = AsyncClient(token=token)
            return cls(client=client, mode=mode, broker=broker)

        elif broker == "bybit":
            from src.executors.bybit_client import BybitClient
            from src.engine.config_loader import load_bybit_config
            cfg = bybit_cfg or load_bybit_config(testnet=(mode == "sandbox"))
            # BybitClient синхронный — создаём в executor, чтобы не блокировать loop
            loop = asyncio.get_running_loop()
            client = await loop.run_in_executor(None, BybitClient, cfg)
            return cls(client=client, mode=mode, broker=broker)

        else:
            raise ValueError(f"Unknown broker: {broker!r} (ожидается 'tinkoff' или 'bybit')")

    # ── fetch prices ──────────────────────────────────────────────────────────

    async def fetch_prices(self, tickers: list[str]) -> dict[str, float]:
        if self.broker == "bybit":
            return await self._fetch_prices_bybit(tickers)
        return await self._fetch_prices_tinkoff(tickers)

    async def _fetch_prices_bybit(self, tickers: list[str]) -> dict[str, float]:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, self._client.get_prices, tickers)
        except Exception as exc:
            logger.exception("Bybit fetch_prices failed: %s", exc)
            return {}

    async def _fetch_prices_tinkoff(self, tickers: list[str]) -> dict[str, float]:
        try:
            async with self._client as c:
                figi_map: dict[str, str] = {}
                for ticker in tickers:
                    try:
                        resp = await c.instruments.find_instrument(query=ticker)
                        if resp.instruments:
                            figi_map[ticker] = resp.instruments[0].figi
                    except Exception as exc:
                        logger.warning("find_instrument %s: %s", ticker, exc)

                if not figi_map:
                    return {}

                try:
                    from t_tech.invest.utils import quotation_to_decimal
                except ImportError:
                    from tinkoff.invest.utils import quotation_to_decimal  # type: ignore
                response = await c.market_data.get_last_prices(
                    figi=list(figi_map.values())
                )

                prices = {}
                for item in response.last_prices:
                    for ticker, figi in figi_map.items():
                        if item.figi == figi and item.price:
                            prices[ticker] = float(quotation_to_decimal(item.price))
                            break
                return prices
        except Exception as exc:
            logger.exception("Tinkoff fetch_prices failed: %s", exc)
            return {}

    # ── lot size (только Tinkoff — у Bybit qty в контрактах, лотов нет) ────────

    async def get_lot_size(self, ticker: str) -> int:
        if self.broker == "bybit":
            return 1
        if ticker in self._lot_cache:
            return self._lot_cache[ticker]
        try:
            async with self._client as c:
                resp = await c.instruments.find_instrument(query=ticker)
                for inst in resp.instruments:
                    lot = getattr(inst, "lot", None)
                    if lot and lot > 0:
                        self._lot_cache[ticker] = int(lot)
                        logger.debug("lot_size[%s] = %d", ticker, lot)
                        return int(lot)
        except Exception as exc:
            logger.warning("get_lot_size failed for %s: %s — using 1", ticker, exc)
        self._lot_cache[ticker] = 1
        return 1

    async def shares_to_lots(self, ticker: str, qty_shares: float) -> int:
        lot = await self.get_lot_size(ticker)
        lots = int(qty_shares // lot)
        return max(1, lots)

    # ── submit одного ордера ────────────────────────────────────────────────

    async def submit(self, order: Order) -> bool:
        if self.broker == "bybit":
            return await self._submit_bybit(order)
        return await self._submit_tinkoff(order)

    async def _submit_bybit(self, order: Order) -> bool:
        loop = asyncio.get_running_loop()
        side = "Buy" if order.side == OrderSide.BUY else "Sell"
        qty = abs(order.qty)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                order_id = await loop.run_in_executor(
                    None, self._client.submit_order_simple, order.ticker, side, qty
                )
                if order_id:
                    order.order_id = order_id
                    order.status = OrderStatus.FILLED
                    logger.info("Bybit order OK | %s %s x%s | id=%s", side, order.ticker, qty, order_id)
                    return True
                logger.warning("Bybit submit attempt %d/%d returned no order_id", attempt, MAX_RETRIES)
            except Exception as exc:
                delay = _backoff_delay(attempt)
                logger.warning("Bybit submit attempt %d/%d failed (%s), retry in %.1fs", attempt, MAX_RETRIES, exc, delay)
                await asyncio.sleep(delay)

        order.status = OrderStatus.REJECTED
        return False

    async def _submit_tinkoff(self, order: Order) -> bool:
        if not order.order_id:
            order.order_id = str(uuid.uuid4())

        try:
            async with self._client as c:
                figi = await self._resolve_figi(c, order.ticker)
                if not figi:
                    order.status = OrderStatus.REJECTED
                    return False

                lots = await self.shares_to_lots(order.ticker, order.qty)

                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        result = await self._post_order(c, order, figi, lots)
                        if result:
                            order.status = OrderStatus.FILLED
                            logger.info(
                                "Order OK | %s %s x%d lots | id=%s",
                                order.side, order.ticker, lots, order.order_id,
                            )
                            return True
                    except Exception as exc:
                        delay = _backoff_delay(attempt)
                        logger.warning(
                            "Submit attempt %d/%d failed (%s), retry in %.1fs",
                            attempt, MAX_RETRIES, exc, delay,
                        )
                        await asyncio.sleep(delay)
        except Exception as exc:
            logger.error("submit failed for %s: %s", order.ticker, exc)

        order.status = OrderStatus.REJECTED
        return False

    # ── атомарный submit пары (только Tinkoff — Bybit-ветка не поддерживает) ──

    async def submit_pair(self, order_a: Order, order_b: Order) -> Tuple[bool, bool]:
        """
        Одновременно (asyncio.gather) исполняет обе ноги пары.
        ПРИМЕЧАНИЕ: для broker="bybit" leg risk тот же (gather всё равно
        параллелит два run_in_executor вызова), но БЕЗ server-side
        idempotency key (order.order_id у Bybit не используется как dedup) —
        в отличие от Tinkoff, где order_id передаётся в PostOrder.
        """
        ok_a, ok_b = await asyncio.gather(
            self.submit(order_a),
            self.submit(order_b),
            return_exceptions=False,
        )
        if ok_a and not ok_b:
            logger.error(
                "LEG RISK: leg A (%s) filled but leg B (%s) REJECTED — portfolio unbalanced!",
                order_a.ticker, order_b.ticker,
            )
        elif ok_b and not ok_a:
            logger.error(
                "LEG RISK: leg B (%s) filled but leg A (%s) REJECTED — portfolio unbalanced!",
                order_b.ticker, order_a.ticker,
            )
        return ok_a, ok_b

    # ── внутренние helpers (Tinkoff only) ──────────────────────────────────────

    async def _resolve_figi(self, client, ticker: str) -> Optional[str]:
        try:
            resp = await client.instruments.find_instrument(query=ticker)
            if resp.instruments:
                return resp.instruments[0].figi
        except Exception as exc:
            logger.warning("FIGI resolve failed for %s: %s", ticker, exc)
        return None

    async def _post_order(self, client, order: Order, figi: str, lots: int) -> str:
        try:
            from t_tech.invest.schemas import OrderDirection, OrderType as TinkoffOrderType
        except ImportError:
            from tinkoff.invest.schemas import OrderDirection, OrderType as TinkoffOrderType  # type: ignore

        direction = (
            OrderDirection.ORDER_DIRECTION_BUY
            if order.side == OrderSide.BUY
            else OrderDirection.ORDER_DIRECTION_SELL
        )

        kwargs = dict(
            figi=figi,
            quantity=lots,
            direction=direction,
            order_id=order.order_id,  # idempotency key
        )

        if self.mode == "sandbox":
            account = await self._get_sandbox_account(client)
            kwargs["account_id"] = account
            resp = await client.sandbox.post_sandbox_order(**kwargs)
        else:
            account = await self._get_account(client)
            kwargs["account_id"] = account
            resp = await client.orders.post_order(**kwargs)

        return resp.order_id

    async def _get_sandbox_account(self, client) -> str:
        try:
            resp = await client.sandbox.get_sandbox_accounts()
            if resp.accounts:
                return resp.accounts[0].id
        except Exception:
            pass
        resp = await client.sandbox.open_sandbox_account()
        return resp.account_id

    async def _get_account(self, client) -> str:
        resp = await client.users.get_accounts()
        if not resp.accounts:
            raise RuntimeError("No trading accounts")
        return resp.accounts[0].id

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        if self.broker == "bybit":
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._client.close)
