from __future__ import annotations
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from pybit.unified_trading import HTTP
from src.engine.config_loader import BybitConfig


logger = logging.getLogger(__name__)


class BybitClient:
    def __init__(self, cfg: BybitConfig):
        self.cfg = cfg
        self.category = cfg.category
        self.session = HTTP(
            testnet=cfg.testnet,
            demo=False,  # demo требует KYC на mainnet — не используем
            api_key=cfg.api_key,
            api_secret=cfg.api_secret,
        )
        self._last_call_ts = 0.0
        self._instrument_cache: Dict[str, Dict[str, float]] = {}

    # ── throttle (см. Bybit rate-limit backoff: read 100ms, write 300ms) ──

    def _throttle(self, min_interval: float = 0.1) -> None:
        elapsed = time.time() - self._last_call_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call_ts = time.time()

    def _call(self, fn, *, write: bool, **kwargs) -> Dict[str, Any]:
        """Единая точка вызова API с throttle + retCode-проверкой."""
        self._throttle(0.3 if write else 0.1)
        resp = fn(**kwargs)
        ret_code = resp.get("retCode", -1)
        if ret_code != 0:
            logger.warning(
                "Bybit API error retCode=%s retMsg=%s kwargs=%s",
                ret_code, resp.get("retMsg"), kwargs,
            )
        return resp

    # ── баланс / портфель ──────────────────────────────────────────────

    def get_wallet_balance(self) -> Dict[str, Any]:
        return self._call(
            self.session.get_wallet_balance, write=False, accountType="UNIFIED"
        )

    
    # ── цены ──────────────────────────────────────────────────────────

    def get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Аналог TinkoffClient.get_prices() — {symbol: last_price}."""
        if not symbols:
            return {}
        prices: Dict[str, float] = {}
        for symbol in symbols:
            try:
                resp = self._call(
                    self.session.get_tickers, write=False,
                    category=self.category, symbol=symbol,
                )
                items = resp.get("result", {}).get("list", [])
                if items:
                    last_price = items[0].get("lastPrice")
                    if last_price is not None:
                        prices[symbol] = float(last_price)
            except Exception as exc:
                logger.warning("get_tickers failed for %s: %s", symbol, exc)
        return prices

    # ── инструменты (аналог get_liquid_shares) ───────────────────────────

    def get_liquid_symbols(self, min_turnover_usdt: float = 1_000_000.0) -> List[Dict[str, Any]]:
        """
        Возвращает список ликвидных перпетуалов/спот-пар по 24h turnover.
        Аналог TinkoffClient.get_liquid_shares() для UniverseSelector.
        """
        resp = self._call(self.session.get_tickers, write=False, category=self.category)
        items = resp.get("result", {}).get("list", [])
        out = []
        for it in items:
            turnover = float(it.get("turnover24h", 0.0) or 0.0)
            if turnover < min_turnover_usdt:
                continue
            out.append({
                "ticker": it.get("symbol"),
                "figi": it.get("symbol"),  # у Bybit нет FIGI, symbol используется как id
                "uid": it.get("symbol"),
                "lot": 1,
                "currency": "usdt",
                "short_enabled": True,  # перпетуалы всегда шортабельны
                "name": it.get("symbol"),
            })
        return out


    _INTERVAL_MAP = {
        "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
        "1d": "D", "1w": "W", "1M": "M",
    }

    def get_candles(
        self,
        symbol: str,
        interval: str = "1h",
        from_date: Optional["datetime"] = None,
        limit: int = 200,
    ) -> "pd.DataFrame":
        """Аналог TinkoffClient.get_candles() — DataFrame timestamp/open/high/low/close/volume."""
        import pandas as pd
        from datetime import timezone as _tz

        bybit_interval = self._INTERVAL_MAP.get(interval, interval)
        params: Dict[str, Any] = {
            "category": self.category, "symbol": symbol,
            "interval": bybit_interval, "limit": limit,
        }
        if from_date is not None:
            if from_date.tzinfo is None:
                from_date = from_date.replace(tzinfo=_tz.utc)
            params["start"] = int(from_date.timestamp() * 1000)

        resp = self._call(self.session.get_kline, write=False, **params)
        rows = resp.get("result", {}).get("list", [])
        if not rows:
            logger.warning("get_kline вернул 0 свечей для %s", symbol)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df = df.drop(columns=["turnover"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        return df.sort_values("timestamp").reset_index(drop=True)
    # ── ордера ────────────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: str,
        order_type: str = "Market",
        category: Optional[str] = None,
        reduce_only: bool = False,
        market_unit: Optional[str] = None,
        order_link_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "category": category or self.category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": qty,
            "reduceOnly": reduce_only,
        }

        if market_unit is not None:
            kwargs["marketUnit"] = market_unit
        if order_link_id:
            kwargs["orderLinkId"] = order_link_id

        return self._call(self.session.place_order, write=True, **kwargs)


    def get_portfolio(self) -> Dict[str, Any]:
        """
        Returns a normalized broker portfolio.

        For spot accounts, `cash` is locally usable cash and MarketState
        calculates equity as cash plus marked spot positions.

        For linear/inverse derivatives, Bybit `totalEquity` is authoritative:
        it already includes wallet balances, collateral valuation and
        unrealized PnL. Therefore MarketState must not add position notional
        to it again.
        """
        response = self.get_wallet_balance()
        accounts = response.get("result", {}).get("list", [])

        if not accounts:
            return {
                "cash": 0.0,
                "total_amount": 0.0,
                "broker_equity": None,
                "accounting_mode": "broker_equity",
                "broker": "bybit",
                "positions_count": 0,
                "positions": [],
            }

        account = accounts[0]
        total_equity = float(
            account.get("totalEquity", 0.0) or 0.0
        )

        if self.category not in ("linear", "inverse"):
            return {
                "cash": total_equity,
                "total_amount": total_equity,
                "broker_equity": None,
                "accounting_mode": "spot",
                "broker": "bybit",
                "positions_count": 0,
                "positions": [],
            }

        positions_response = self._call(
            self.session.get_positions,
            write=False,
            category=self.category,
            settleCoin="USDT",
        )

        raw_positions = (
            positions_response.get("result", {}).get("list", [])
        )

        positions: List[Dict[str, Any]] = []

        for raw_position in raw_positions:
            size = float(raw_position.get("size", 0.0) or 0.0)

            if size <= 0:
                continue

            side = str(raw_position.get("side", "")).lower()
            signed_qty = -size if side == "sell" else size

            symbol = str(raw_position.get("symbol", "")).upper()
            avg_price = float(
                raw_position.get("avgPrice", 0.0) or 0.0
            )

            if not symbol or avg_price <= 0:
                logger.warning(
                    "Skipping invalid Bybit position: symbol=%s "
                    "size=%s avgPrice=%s",
                    symbol,
                    size,
                    avg_price,
                )
                continue

            positions.append(
                {
                    "symbol": symbol,
                    "qty": signed_qty,
                    "avg_price": avg_price,
                    "unrealised_pnl": float(
                        raw_position.get(
                            "unrealisedPnl",
                            0.0,
                        ) or 0.0
                    ),
                }
            )

        return {
            # Compatibility fields for old callers.
            "cash": total_equity,
            "total_amount": total_equity,

            # Derivatives accounting contract.
            "broker_equity": total_equity,
            "accounting_mode": "broker_equity",
            "broker": "bybit",

            # Diagnostics only. Do not rebuild totalEquity from them.
            "wallet_balance": float(
                account.get("totalWalletBalance", 0.0) or 0.0
            ),
            "unrealized_pnl": float(
                account.get("totalPerpUPL", 0.0) or 0.0
            ),
            "available_balance": float(
                account.get(
                    "totalAvailableBalance",
                    0.0,
                ) or 0.0
            ),
            "initial_margin": float(
                account.get(
                    "totalInitialMargin",
                    0.0,
                ) or 0.0
            ),

            "positions_count": len(positions),
            "positions": positions,
        }

    def submit_order_simple(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_link_id: Optional[str] = None,
        reduce_only: bool = False,
    ) -> Optional[str]:
        response = self.place_order(
            symbol=symbol,
            side=side,
            qty=str(qty),
            order_link_id=order_link_id,
            reduce_only=reduce_only,
        )

        if response.get("retCode") != 0:
            raise RuntimeError(
                f"Bybit rejected order: {response.get('retCode')} "
                f"{response.get('retMsg')}"
            )

        result = response.get("result", {})
        order_id = result.get("orderId")
        if not order_id:
            raise RuntimeError(f"Bybit response has no orderId: {response}")

        return str(order_id)
    def get_order_state(
        self,
        order_id: str,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "category": self.category,
            "orderId": order_id,
        }
        if symbol:
            params["symbol"] = symbol

        response = self._call(
            self.session.get_open_orders,
            write=False,
            **params,
        )

        if response.get("retCode") != 0:
            raise RuntimeError(
                f"Bybit get_order_state failed: {response.get('retMsg')}"
            )

        items = response.get("result", {}).get("list", [])
        if not items:
            history = self._call(
                self.session.get_order_history,
                write=False,
                **params,
            )
            if history.get("retCode") != 0:
                raise RuntimeError(
                    f"Bybit get_order_history failed: {history.get('retMsg')}"
                )
            items = history.get("result", {}).get("list", [])

        if not items:
            raise RuntimeError(f"Bybit order not found: {order_id}")

        item = items[0]
        raw_status = str(item.get("orderStatus", "")).lower()

        status_map = {
            "new": "accepted",
            "partiallyfilled": "partially_filled",
            "filled": "filled",
            "cancelled": "cancelled",
            "rejected": "rejected",
            "deactivated": "cancelled",
        }

        filled_qty = float(item.get("cumExecQty", 0.0) or 0.0)
        filled_value = float(item.get("cumExecValue", 0.0) or 0.0)
        filled_price = filled_value / filled_qty if filled_qty > 0 else 0.0

        return {
            "status": status_map.get(raw_status, "accepted"),
            "filled_qty": filled_qty,
            "filled_price": filled_price,
            "broker_order_id": str(item.get("orderId", order_id)),
        }


    def cancel_order(self, order_id: str, symbol: str) -> None:
        response = self._call(
            self.session.cancel_order,
            write=True,
            category=self.category,
            symbol=symbol,
            orderId=order_id,
        )

        if response.get("retCode") != 0:
            raise RuntimeError(
                f"Bybit cancel_order failed: {response.get('retMsg')}"
            )
    def get_instrument_spec(self, symbol: str) -> Dict[str, float]:
        """Возвращает {qty_step, min_order_qty} для symbol, с кэшированием."""
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]
        try:
            resp = self._call(
                self.session.get_instruments_info, write=False,
                category=self.category, symbol=symbol,
            )
            items = resp.get("result", {}).get("list", [])
            if not items:
                logger.warning("get_instruments_info empty for %s — using qty_step=1", symbol)
                spec = {"qty_step": 1.0, "min_order_qty": 0.0}
            else:
                lot_filter = items[0].get("lotSizeFilter", {})
                step = lot_filter.get("qtyStep") or lot_filter.get("basePrecision") or "1.0"
                spec = {
                    "qty_step": float(step),
                    "min_order_qty": float(lot_filter.get("minOrderQty", 0.0) or 0.0),
                }
            self._instrument_cache[symbol] = spec
            return spec
        except Exception as exc:
            logger.warning("get_instrument_spec failed for %s: %s — using qty_step=1", symbol, exc)
            spec = {"qty_step": 1.0, "min_order_qty": 0.0}
            self._instrument_cache[symbol] = spec
            return spec
    # ── lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        logger.debug("BybitClient.close (no-op, pybit не требует явного закрытия)")
