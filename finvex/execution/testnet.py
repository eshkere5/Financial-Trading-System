"""Тестовое исполнение: Bybit Testnet и T-Invest Sandbox.

Оба адаптера работают ТОЛЬКО с тестовым контуром, это зашито в конструктор: клиент с боевым режимом они не принимают.
Деньги виртуальные. SDK (pybit, t_tech/tinkoff) подгружаются лениво: без них модуль импортируется, а ошибка появляется при первом использовании.
"""
from __future__ import annotations

import math
import threading
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Dict, Optional

from finvex.core.base import Capability, ExecutionProvider, ProviderInfo
from finvex.core.errors import ProviderError
from finvex.core.models import InstrumentId, OrderAck, OrderRequest, OrderType, Side
from finvex.core.registry import register

CRYPTO_SUFFIXES = ("USDT", "USDC")
HOWTO_BYBIT = ("Нужен ключ тестовой сети: testnet.bybit.com, ключ создаётся там (не в Demo Trading), файл configs/bybit_testnet_key.yaml: "
               "bybit: {api_key: ..., api_secret: ..., category: spot}")
HOWTO_TINKOFF = "Нужен токен песочницы T-Invest в configs/sandbox_key.yaml: tinkoff: {token: \"t....\"}"


class TestVenueError(RuntimeError):
    """Тестовый контур недоступен: нет ключа, SDK или связи. Текст содержит инструкцию."""

    __test__ = False


def _ack(order: OrderRequest, venue_id: Optional[str], status: str) -> OrderAck:
    return OrderAck(client_order_id=order.client_order_id, venue_order_id=venue_id, status=status, ts=datetime.now(timezone.utc))


@register("bybit_testnet", kind="execution")
class BybitTestnetExecution(ExecutionProvider):
    def __init__(self, category: str = "spot", client: Any = None) -> None:
        if client is None:
            try:
                from finvex.engine.config_loader import load_bybit_config
                from finvex.venues.bybit.client import BybitClient
                cfg = load_bybit_config(testnet=True)   # только файл тестовой сети
                cfg.category = category
                client = BybitClient(cfg)
            except ImportError as exc:
                raise TestVenueError(f"Не установлен pybit ({exc}). Установите: pip install pybit") from exc
            except (FileNotFoundError, ValueError) as exc:
                raise TestVenueError(f"{exc}. {HOWTO_BYBIT}") from exc
        if getattr(getattr(client, "cfg", None), "testnet", False) is not True:
            raise ValueError("bybit_testnet принимает только клиента тестовой сети")
        self.client, self.category = client, getattr(client, "category", category)
        self._sent: Dict[str, tuple] = {}
        self._lock = threading.Lock()
        self._info = ProviderInfo(name="bybit_testnet", venue="BYBIT", capabilities=Capability.EXECUTION | Capability.PORTFOLIO,
                                  license_note="Только тестовая сеть Bybit, виртуальные деньги.")

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def submit(self, order: OrderRequest) -> OrderAck:
        if order.instrument.venue != "BYBIT":
            raise ProviderError(f"bybit_testnet принимает только BYBIT, получено {order.instrument}")
        from finvex.venues.lot_utils import round_qty_to_step
        sym = order.instrument.symbol
        spec = self.client.get_instrument_spec(sym)
        qty = round_qty_to_step(abs(order.quantity), spec["qty_step"], spec["min_order_qty"])
        if qty <= 0:
            return _ack(order, None, f"rejected: объём {order.quantity:g} ниже минимального шага {spec['min_order_qty']:g}")
        kw: Dict[str, Any] = {"category": self.category, "symbol": sym, "side": "Buy" if order.side is Side.BUY else "Sell",
                              "qty": str(qty), "orderLinkId": order.client_order_id[:36]}
        if order.order_type is OrderType.LIMIT:
            kw.update(orderType="Limit", price=str(order.limit_price))
        else:
            kw.update(orderType="Market")
            if self.category == "spot":
                kw["marketUnit"] = "baseCoin"
        resp = self.client._call(self.client.session.place_order, write=True, **kw)
        if resp.get("retCode") != 0:
            return _ack(order, None, f"rejected: {resp.get('retMsg')}")
        oid = (resp.get("result") or {}).get("orderId")
        with self._lock:
            self._sent[order.client_order_id] = (sym, oid)
        return _ack(order, oid, "accepted")

    def cancel(self, client_order_id: str) -> OrderAck:
        with self._lock:
            sym, oid = self._sent.get(client_order_id, (None, None))
        if not sym:
            raise ProviderError("заявка не найдена среди отправленных в этой сессии")
        resp = self.client._call(self.client.session.cancel_order, write=True, category=self.category, symbol=sym, orderId=oid)
        return OrderAck(client_order_id, oid, "cancelled" if resp.get("retCode") == 0 else f"rejected: {resp.get('retMsg')}", datetime.now(timezone.utc))

    def portfolio(self) -> dict:
        return self.client.get_portfolio()


@register("tinkoff_sandbox", kind="execution")
class TinkoffSandboxExecution(ExecutionProvider):
    def __init__(self, client: Any = None, sdk: Any = None) -> None:
        if client is None:
            try:
                from finvex.engine.config_loader import load_tinkoff_config
                from finvex.venues.tinkoff.client import TinkoffClient
                client = TinkoffClient(load_tinkoff_config(sandbox=True).token, sandbox=True)   # только sandbox_key.yaml
            except ImportError as exc:
                raise TestVenueError(f"Не установлен SDK T-Invest ({exc}). Установите t-tech-investments или tinkoff-investments") from exc
            except (FileNotFoundError, ValueError, TypeError) as exc:
                raise TestVenueError(f"{exc}. {HOWTO_TINKOFF}") from exc
        if getattr(client, "sandbox", False) is not True:
            raise ValueError("tinkoff_sandbox принимает только клиента песочницы")
        self.client, self._sdk = client, sdk
        self._info = ProviderInfo(name="tinkoff_sandbox", venue="MOEX", capabilities=Capability.EXECUTION | Capability.PORTFOLIO,
                                  license_note="Песочница T-Invest: виртуальные деньги, заявки не выходят на биржу.")

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def _s(self) -> Any:
        if self._sdk is None:
            try:
                from finvex.venues.tinkoff._sdk import OrderDirection, OrderType as SdkOrderType
                try:
                    from t_tech.invest.utils import decimal_to_quotation
                except ImportError:
                    from tinkoff.invest.utils import decimal_to_quotation  # type: ignore
            except ImportError as exc:
                raise TestVenueError(f"Не установлен SDK T-Invest ({exc})") from exc
            self._sdk = SimpleNamespace(OrderDirection=OrderDirection, OrderType=SdkOrderType, decimal_to_quotation=decimal_to_quotation)
        return self._sdk

    def submit(self, order: OrderRequest) -> OrderAck:
        if order.instrument.venue != "MOEX":
            raise ProviderError(f"tinkoff_sandbox принимает только MOEX, получено {order.instrument}")
        sym = order.instrument.symbol
        lot = int(self.client.get_lot_size(sym) or 1)
        lots = math.floor(abs(order.quantity) / lot)
        if lots < 1:
            return _ack(order, None, f"rejected: {order.quantity:g} шт. меньше одного лота ({lot} шт.)")
        s = self._s()
        figi = self.client.get_figi(sym)
        direction = s.OrderDirection.ORDER_DIRECTION_BUY if order.side is Side.BUY else s.OrderDirection.ORDER_DIRECTION_SELL
        price = None
        if order.order_type is OrderType.LIMIT:
            price, otype = s.decimal_to_quotation(Decimal(str(order.limit_price))), s.OrderType.ORDER_TYPE_LIMIT
        else:
            otype = s.OrderType.ORDER_TYPE_MARKET
        oid = self.client.submit_order(figi, direction, lots, price=price, order_type=otype, order_id=order.client_order_id)
        return _ack(order, str(oid), "accepted")

    def cancel(self, client_order_id: str) -> OrderAck:
        self.client.cancel_order(client_order_id)
        return _ack(OrderRequest("cancel", client_order_id, InstrumentId("-", "MOEX"), Side.BUY, 1, OrderType.MARKET), client_order_id, "cancelled")

    def portfolio(self) -> dict:
        return self.client.get_portfolio()

    def top_up(self, amount: float = 1_000_000.0) -> None:
        self.client.sandbox_pay_in(amount)


class TestVenues:
    """Тикеры …USDT/USDC идут на Bybit Testnet, остальное в песочницу T-Invest."""

    __test__ = False

    def __init__(self, factories: Optional[Dict[str, Any]] = None) -> None:
        self._factories = factories or {"BYBIT": BybitTestnetExecution, "MOEX": TinkoffSandboxExecution}
        self._made: Dict[str, Any] = {}
        self._lock = threading.Lock()

    @staticmethod
    def venue_for(symbol: str) -> str:
        return "BYBIT" if symbol.upper().endswith(CRYPTO_SUFFIXES) else "MOEX"

    def get(self, venue: str):
        with self._lock:
            if venue not in self._made:
                self._made[venue] = self._factories[venue]()
            return self._made[venue]

    def for_symbol(self, symbol: str):
        return self.get(self.venue_for(symbol))

    def status(self) -> list:
        out = []
        for venue, title in (("BYBIT", "Bybit Testnet"), ("MOEX", "T-Invest Sandbox")):
            try:
                self.get(venue)
                out.append({"venue": venue, "title": title, "ready": True, "message": "ключ найден, контур готов"})
            except Exception as exc:
                out.append({"venue": venue, "title": title, "ready": False, "message": str(exc)[:300]})
        return out
