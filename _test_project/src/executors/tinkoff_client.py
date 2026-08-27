"""
TinkoffClient — синхронная обёртка над T-Tech Invest SDK.

Контракт:
- get_prices(tickers) -> {ticker: last_price}
- get_lot_size(ticker) -> lot size в штуках
- submit_order(...) -> broker order_id, с client order_id для idempotency
- get_portfolio() -> {cash, total_amount, positions[{ticker, qty, avg_price}]}

Всё сетевое I/O синхронное. В asyncio-коде вызовы должны идти через
run_in_executor либо использоваться через AsyncExecutionRouter.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from t_tech.invest import Client
from t_tech.invest.constants import INVEST_GRPC_API, INVEST_GRPC_API_SANDBOX
from t_tech.invest.schemas import CandleInterval, CandleSource
from t_tech.invest.utils import quotation_to_decimal

logger = logging.getLogger(__name__)

GRPC_OPTIONS = [
    ("grpc.ssl_target_name_override", "invest-public-api.tinkoff.ru"),
    ("grpc.max_receive_message_length", 100 * 1024 * 1024),
    ("grpc.max_send_message_length", 100 * 1024 * 1024),
]

_DEFAULT_INSTRUMENT_TYPE = "share"


class TinkoffClient:
    def __init__(
        self,
        token: str,
        sandbox: bool = True,
        instrument_type: str = _DEFAULT_INSTRUMENT_TYPE,
        currency: Optional[str] = "rub",
    ) -> None:
        if not token:
            raise ValueError("Tinkoff API token is required")

        self.token = token
        self.sandbox = sandbox
        self.instrument_type = instrument_type.lower().strip()
        self.currency = currency.lower().strip() if currency else None

        self._uid_cache: Dict[str, str] = {}
        self._figi_cache: Dict[str, str] = {}
        self._ticker_by_figi: Dict[str, str] = {}
        self._lot_cache: Dict[str, int] = {}
        self._account_id: Optional[str] = None

        logger.info(
            "TinkoffClient initialized | sandbox=%s | instrument_type=%s | currency=%s",
            self.sandbox,
            self.instrument_type,
            self.currency,
        )

    def _make_client(self) -> Client:
        target = INVEST_GRPC_API_SANDBOX if self.sandbox else INVEST_GRPC_API
        return Client(token=self.token, target=target, options=GRPC_OPTIONS)

    # ── account ───────────────────────────────────────────────────────

    def _get_account_id(self, client: Client) -> str:
        if self._account_id:
            return self._account_id

        if self.sandbox:
            response = client.sandbox.get_sandbox_accounts()
            if response.accounts:
                self._account_id = response.accounts[0].id
            else:
                opened = client.sandbox.open_sandbox_account()
                self._account_id = opened.account_id
                logger.info("Sandbox account created: %s", self._account_id)
        else:
            response = client.users.get_accounts()
            if not response.accounts:
                raise RuntimeError("No trading accounts available")
            self._account_id = response.accounts[0].id

        return self._account_id

    def get_order_state(self, order_id: str) -> Dict[str, Any]:
        """
        Нормализует состояние заявки Tinkoff под контракт LiveEngine.

        Returns:
            {
                "status": "accepted|partially_filled|filled|cancelled|rejected",
                "filled_qty": float,       # В ШТУКАХ, не в лотах
                "filled_price": float,
                "broker_order_id": str,
            }
        """
        from src.executors._sdk import OrderExecutionReportStatus

        try:
            with self._make_client() as client:
                account_id = self._get_account_id(client)
                response = (
                    client.sandbox.get_sandbox_order_state(
                        account_id=account_id,
                        order_id=order_id,
                    )
                    if self.sandbox
                    else client.orders.get_order_state(
                        account_id=account_id,
                        order_id=order_id,
                    )
                )

                execution_status = response.execution_report_status
                status_map = {
                    OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_NEW: "accepted",
                    OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_PARTIALLYFILL: "partially_filled",
                    OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_FILL: "filled",
                    OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_CANCELLED: "cancelled",
                    OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_REJECTED: "rejected",
                    OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_EXPIRED: "cancelled",
                }

                lots_executed = int(getattr(response, "lots_executed", 0) or 0)
                figi = str(getattr(response, "figi", "") or "")
                ticker = self._ticker_for_figi(figi)
                lot_size = self.get_lot_size(ticker) if ticker else 1

                executed_price = getattr(response, "executed_order_price", None)
                filled_price = (
                    float(quotation_to_decimal(executed_price))
                    if executed_price is not None
                    else 0.0
                )

                return {
                    "status": status_map.get(execution_status, "accepted"),
                    "filled_qty": float(lots_executed * lot_size),
                    "filled_price": filled_price,
                    "broker_order_id": str(getattr(response, "order_id", order_id)),
                }

        except Exception as exc:
            logger.exception("get_order_state failed | order_id=%s", order_id)
            raise RuntimeError(f"Tinkoff get_order_state failed: {exc}") from exc


    def cancel_order(self, order_id: str) -> None:
        try:
            with self._make_client() as client:
                account_id = self._get_account_id(client)

                if self.sandbox:
                    client.sandbox.cancel_sandbox_order(
                        account_id=account_id,
                        order_id=order_id,
                    )
                else:
                    client.orders.cancel_order(
                        account_id=account_id,
                        order_id=order_id,
                    )
                    

            logger.info("Tinkoff order cancelled | id=%s", order_id)
        except Exception as exc:
            logger.exception("cancel_order failed | order_id=%s", order_id)
            raise RuntimeError(f"Tinkoff cancel_order failed: {exc}") from exc
        
    # ── instrument resolution ─────────────────────────────────────────

    @staticmethod
    def _instrument_type_name(instrument: Any) -> str:
        kind = (
            getattr(instrument, "instrument_kind", None)
            or getattr(instrument, "instrument_type", None)
        )
        return str(getattr(kind, "name", kind)).lower()

    def _filter_instrument(self, instruments: list[Any]) -> Optional[Any]:
        """
        Выбирает только доступный для API инструмент подходящего типа/валюты.

        Нельзя fallback-ить на первый search result: query может вернуть
        депозитарную расписку, не торгуемый класс или другой рынок.
        """
        candidates = [
            inst
            for inst in instruments
            if getattr(inst, "api_trade_available_flag", False)
        ]

        if self.instrument_type:
            candidates = [
                inst
                for inst in candidates
                if self.instrument_type in self._instrument_type_name(inst)
            ]

        if self.currency:
            candidates = [
                inst
                for inst in candidates
                if str(getattr(inst, "currency", "")).lower() == self.currency
            ]

        if not candidates:
            return None

        tqbr = [
            inst for inst in candidates
            if getattr(inst, "class_code", "") == "TQBR"
        ]
        return tqbr[0] if tqbr else candidates[0]

    def _cache_instrument(self, ticker: str, instrument: Any) -> None:
        ticker = ticker.upper()
        figi = str(instrument.figi)
        lot = max(1, int(getattr(instrument, "lot", 1) or 1))
        uid = str(getattr(instrument, "uid", "") or "")

        self._figi_cache[ticker] = figi
        self._ticker_by_figi[figi] = ticker
        self._lot_cache[ticker] = lot
        if uid:
            self._uid_cache[ticker] = uid

    def _resolve_instrument(self, ticker: str) -> Optional[Any]:
        ticker = ticker.upper().strip()

        if ticker in self._figi_cache:
            return None

        try:
            with self._make_client() as client:
                response = client.instruments.find_instrument(query=ticker)
                instrument = self._filter_instrument(list(response.instruments))
                if instrument is None:
                    logger.warning(
                        "No tradable instrument: ticker=%s type=%s currency=%s",
                        ticker,
                        self.instrument_type,
                        self.currency,
                    )
                    return None

                self._cache_instrument(ticker, instrument)
                return instrument
        except Exception as exc:
            logger.error("Instrument resolve failed for %s: %s", ticker, exc)
            return None

    def get_figi(self, ticker: str) -> Optional[str]:
        ticker = ticker.upper().strip()

        figi = self._figi_cache.get(ticker)
        if figi:
            return figi

        instrument = self._resolve_instrument(ticker)
        if instrument is None:
            return None

        return self._figi_cache.get(ticker)

    def get_lot_size(self, ticker: str) -> int:
        ticker = ticker.upper().strip()

        cached = self._lot_cache.get(ticker)
        if cached:
            return cached

        instrument = self._resolve_instrument(ticker)
        if instrument is None:
            raise RuntimeError(f"Cannot resolve lot size for {ticker}")

        return self._lot_cache.get(ticker, 1)

    def get_instrument_uid(self, ticker: str) -> Optional[str]:
        ticker = ticker.upper().strip()

        cached = self._uid_cache.get(ticker)
        if cached:
            return cached

        instrument = self._resolve_instrument(ticker)
        if instrument is None:
            return None

        return self._uid_cache.get(ticker)

    def _ticker_for_figi(self, figi: str) -> Optional[str]:
        cached = self._ticker_by_figi.get(figi)
        if cached:
            return cached

        try:
            with self._make_client() as client:
                response = client.instruments.find_instrument(query=figi)
                instrument = self._filter_instrument(list(response.instruments))
                if instrument is None:
                    return None

                ticker = str(instrument.ticker).upper()
                self._cache_instrument(ticker, instrument)
                return ticker
        except Exception as exc:
            logger.warning("Ticker resolve failed for FIGI=%s: %s", figi, exc)
            return None

    def resolve_figis(self, tickers: List[str]) -> Dict[str, str]:
        output: Dict[str, str] = {}
        for ticker in tickers:
            figi = self.get_figi(ticker)
            if figi:
                output[ticker.upper()] = figi
        return output

    # ── market data ───────────────────────────────────────────────────

    def get_prices(self, tickers: List[str]) -> Dict[str, float]:
        if not tickers:
            return {}

        figi_map = self.resolve_figis(tickers)
        if not figi_map:
            return {}

        try:
            with self._make_client() as client:
                response = client.market_data.get_last_prices(
                    figi=list(figi_map.values())
                )

            ticker_by_figi = {figi: ticker for ticker, figi in figi_map.items()}
            prices: Dict[str, float] = {}

            for item in response.last_prices:
                if item.price is None:
                    continue
                ticker = ticker_by_figi.get(item.figi)
                if ticker is None:
                    continue

                price = float(quotation_to_decimal(item.price))
                if price > 0:
                    prices[ticker] = price

            return prices
        except Exception as exc:
            logger.exception("get_prices failed: %s", exc)
            return {}

    def get_candles(
        self,
        ticker: str,
        from_date: datetime,
        to_date: Optional[datetime] = None,
        interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_DAY,
    ) -> pd.DataFrame:
        if isinstance(interval, str):
            raise TypeError(
                "interval must be CandleInterval enum, not a string"
            )

        if to_date is None:
            to_date = datetime.now(timezone.utc)

        uid = self.get_instrument_uid(ticker)
        if not uid:
            logger.error("Cannot resolve instrument UID for %s", ticker)
            return pd.DataFrame()

        try:
            with self._make_client() as client:
                candles = list(
                    client.get_all_candles(
                        instrument_id=uid,
                        from_=from_date,
                        to=to_date,
                        interval=interval,
                        candle_source_type=CandleSource.CANDLE_SOURCE_EXCHANGE,
                    )
                )

            if not candles:
                return pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume"]
                )

            data = [
                {
                    "time": candle.time,
                    "open": float(quotation_to_decimal(candle.open)),
                    "high": float(quotation_to_decimal(candle.high)),
                    "low": float(quotation_to_decimal(candle.low)),
                    "close": float(quotation_to_decimal(candle.close)),
                    "volume": int(candle.volume),
                }
                for candle in candles
            ]

            return pd.DataFrame(data).set_index("time").sort_index()
        except Exception as exc:
            logger.exception("get_candles failed for %s: %s", ticker, exc)
            return pd.DataFrame()

    # ── orders ────────────────────────────────────────────────────────

    def submit_order(
        self,
        figi: str,
        direction: Any,
        quantity: int,
        price: Any = None,
        order_type: Any = None,
        order_id: Optional[str] = None,
    ) -> str:
        """
        Отправляет заявку. Возвращает broker order_id.

        `order_id` — idempotency key Tinkoff. Его нельзя менять между
        retry одной и той же заявки.
        """
        if int(quantity) < 1:
            raise ValueError(f"quantity must be >= 1 lot, got {quantity}")

        client_order_id = order_id or str(uuid.uuid4())

        try:
            with self._make_client() as client:
                kwargs: Dict[str, Any] = {
                    "figi": figi,
                    "quantity": int(quantity),
                    "direction": direction,
                    "account_id": self._get_account_id(client),
                    "order_id": client_order_id,
                }

                if order_type is not None:
                    kwargs["order_type"] = order_type
                if price is not None:
                    kwargs["price"] = price

                response = (
                    client.sandbox.post_sandbox_order(**kwargs)
                    if self.sandbox
                    else client.orders.post_order(**kwargs)
                )

                broker_order_id = (
                    getattr(response, "order_id", None)
                    or client_order_id
                )
                logger.info(
                    "Order accepted | client_id=%s broker_id=%s qty=%d lots",
                    client_order_id,
                    broker_order_id,
                    quantity,
                )
                return str(broker_order_id)
        except Exception as exc:
            logger.error(
                "submit_order failed | figi=%s client_id=%s error=%s",
                figi,
                client_order_id,
                exc,
            )
            raise

    # ── portfolio / reconciliation ────────────────────────────────────

    def get_portfolio(self) -> Dict[str, Any]:
        """
        Единый контракт для LiveEngine reconciliation.

        Позиции возвращаются в штуках, не в лотах:
        [{"ticker": "SBER", "qty": 10.0, "avg_price": 270.0}].
        """
        try:
            with self._make_client() as client:
                account_id = self._get_account_id(client)

                portfolio = (
                    client.sandbox.get_sandbox_portfolio(account_id=account_id)
                    if self.sandbox
                    else client.operations.get_portfolio(account_id=account_id)
                )

                position_response = (
                    client.sandbox.get_sandbox_positions(account_id=account_id)
                    if self.sandbox
                    else client.operations.get_positions(account_id=account_id)
                )

            positions: list[Dict[str, Any]] = []
            for position in portfolio.positions:
                qty = float(quotation_to_decimal(position.quantity))
                if abs(qty) < 1e-12:
                    continue

                ticker = self._ticker_for_figi(position.figi)
                if not ticker:
                    logger.warning(
                        "Skipping portfolio position: FIGI=%s cannot map to ticker",
                        position.figi,
                    )
                    continue

                avg_price = (
                    float(quotation_to_decimal(position.average_position_price))
                    if position.average_position_price
                    else 0.0
                )

                positions.append(
                    {
                        "ticker": ticker,
                        "qty": qty,
                        "avg_price": avg_price,
                    }
                )

            expected_currency = self.currency or "rub"
            cash = sum(
                float(quotation_to_decimal(money))
                for money in getattr(position_response, "money", [])
                if str(getattr(money, "currency", "")).lower()
                == expected_currency
            )

            total_amount = float(
                quotation_to_decimal(portfolio.total_amount_portfolio)
            )

            return {
                "cash": cash,
                "total_amount": total_amount,
                "positions_count": len(positions),
                "positions": positions,
            }

        except Exception as exc:
            logger.exception("get_portfolio failed: %s", exc)
            raise

    # ── sandbox helpers ───────────────────────────────────────────────

    def sandbox_pay_in(self, amount: float = 1_000_000.0) -> None:
        if not self.sandbox:
            raise RuntimeError("sandbox_pay_in is available only in sandbox mode")
        if amount <= 0:
            raise ValueError("amount must be > 0")

        try:
            from t_tech.invest.schemas import MoneyValue

            units = int(amount)
            nano = int(round((amount - units) * 1_000_000_000))

            with self._make_client() as client:
                client.sandbox.sandbox_pay_in(
                    account_id=self._get_account_id(client),
                    amount=MoneyValue(currency=self.currency or "rub", units=units, nano=nano),
                )

            logger.info("Sandbox credited: %.2f %s", amount, self.currency or "rub")
        except Exception as exc:
            logger.error("sandbox_pay_in failed: %s", exc)
            raise

    # ── universe ──────────────────────────────────────────────────────

    def get_liquid_shares(
        self,
        currency: str = "rub",
        min_price_rub: float = 10.0,
        exclude_illiquid: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Возвращает API-доступные акции для UniverseSelector.

        `min_price_rub` оставлен в сигнатуре для обратной совместимости:
        фактический фильтр цены требует отдельного price query и должен
        выполняться в UniverseSelector батчево, не по одной акции здесь.
        """
        try:
            with self._make_client() as client:
                response = client.instruments.shares()

            output: List[Dict[str, Any]] = []
            for instrument in response.instruments:
                if exclude_illiquid and not getattr(
                    instrument,
                    "api_trade_available_flag",
                    False,
                ):
                    continue
                if getattr(instrument, "for_qual_investor_flag", False):
                    continue
                if str(getattr(instrument, "currency", "")).lower() != currency.lower():
                    continue

                output.append(
                    {
                        "ticker": instrument.ticker,
                        "figi": instrument.figi,
                        "uid": instrument.uid,
                        "lot": max(1, int(getattr(instrument, "lot", 1) or 1)),
                        "currency": instrument.currency,
                        "short_enabled": bool(
                            getattr(instrument, "short_enabled_flag", False)
                        ),
                        "name": instrument.name,
                    }
                )

            logger.info("get_liquid_shares: %d instruments", len(output))
            return output
        except Exception as exc:
            logger.exception("get_liquid_shares failed: %s", exc)
            return []

    def close(self) -> None:
        logger.debug("TinkoffClient.close()")