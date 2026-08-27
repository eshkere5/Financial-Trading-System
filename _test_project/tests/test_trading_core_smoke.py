from __future__ import annotations

import asyncio
import importlib
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Минимальные подмены для запуска LiveEngine без реальных SDK ────────────

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    ticker: str
    side: OrderSide
    qty: float
    filled_price: float | None = None
    status: Any = OrderStatus.PENDING
    order_id: str = ""
    closes_position: bool = False


@dataclass
class Position:
    ticker: str
    qty: float
    avg_price: float
    open_time: datetime


@dataclass
class MarketState:
    cash: float
    timestamp: datetime
    positions: dict[str, Position] = field(default_factory=dict)
    prices: dict[str, float] = field(default_factory=dict)
    pnl_history: list = field(default_factory=list)
    news: dict = field(default_factory=dict)

    @property
    def portfolio_value(self) -> float:
        return self.cash + sum(
            position.qty * self.prices.get(position.ticker, position.avg_price)
            for position in self.positions.values()
        )

    @property
    def total_pnl(self) -> float:
        return self.portfolio_value - 100_000.0


class Portfolio:
    def __init__(self, state: MarketState, commission: float) -> None:
        self.state = state
        self.commission = commission
        self.applied_orders: list[Order] = []

    def apply_order(self, order: Order) -> None:
        assert order.filled_price is not None
        assert order.filled_price > 0

        self.applied_orders.append(order)
        sign = 1.0 if order.side == OrderSide.BUY else -1.0
        delta = sign * order.qty

        current = self.state.positions.get(order.ticker)
        current_qty = current.qty if current else 0.0
        new_qty = current_qty + delta

        self.state.cash -= delta * order.filled_price

        if abs(new_qty) < 1e-12:
            self.state.positions.pop(order.ticker, None)
            return

        self.state.positions[order.ticker] = Position(
            ticker=order.ticker,
            qty=new_qty,
            avg_price=order.filled_price,
            open_time=self.state.timestamp,
        )


class RiskManager:
    def __init__(self, config: Any) -> None:
        self.is_halted = False

    def reset_peak(self, value: float) -> None:
        pass

    def check_positions(self, state: MarketState) -> list[Order]:
        return []

    def check_order(self, order: Order, state: MarketState) -> bool:
        return True

    def check_drawdown(self, state: MarketState) -> None:
        pass


@dataclass
class RiskConfig:
    pass


@dataclass
class EngineConfig:
    initial_capital: float = 100_000.0
    commission: float = 0.0
    risk: RiskConfig = field(default_factory=RiskConfig)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def install_src_stubs() -> None:
    src_dir = ROOT / "src"
    engine_dir = src_dir / "engine"
    utils_dir = src_dir / "utils"

    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [str(src_dir)]

    engine_pkg = types.ModuleType("src.engine")
    engine_pkg.__path__ = [str(engine_dir)]

    utils_pkg = types.ModuleType("src.utils")
    utils_pkg.__path__ = [str(utils_dir)]

    sys.modules["src"] = src_pkg
    sys.modules["src.engine"] = engine_pkg
    sys.modules["src.utils"] = utils_pkg

    modules = {
        "src.engine.config_loader": types.ModuleType(
            "src.engine.config_loader"
        ),
        "src.engine.market_state": types.ModuleType(
            "src.engine.market_state"
        ),
        "src.engine.orders": types.ModuleType("src.engine.orders"),
        "src.engine.portfolio": types.ModuleType(
            "src.engine.portfolio"
        ),
        "src.engine.risk_manager": types.ModuleType(
            "src.engine.risk_manager"
        ),
        "src.utils.time_utils": types.ModuleType(
            "src.utils.time_utils"
        ),
    }

    modules["src.engine.config_loader"].EngineConfig = EngineConfig

    modules["src.engine.market_state"].MarketState = MarketState
    modules["src.engine.market_state"].Position = Position

    modules["src.engine.orders"].Order = Order
    modules["src.engine.orders"].OrderSide = OrderSide
    modules["src.engine.orders"].OrderStatus = OrderStatus

    modules["src.engine.portfolio"].Portfolio = Portfolio

    modules["src.engine.risk_manager"].RiskManager = RiskManager
    modules["src.engine.risk_manager"].RiskConfig = RiskConfig

    modules["src.utils.time_utils"].utcnow = utcnow

    sys.modules.update(modules)


class FakeAcceptedThenFilledRouter:
    """
    Имитирует новый контракт:
    submit() -> False, но order.status='accepted'.
    Следующий get_order_state() -> фактическое исполнение.
    """

    def __init__(self) -> None:
        self.submit_calls = 0
        self.poll_calls = 0

    def submit(self, order: Order) -> bool:
        self.submit_calls += 1
        order.order_id = "broker-order-42"
        order.status = "accepted"
        return False

    def get_order_state(self, order: Order) -> dict[str, Any]:
        assert order.order_id == "broker-order-42"

        self.poll_calls += 1
        return {
            "status": "filled",
            "filled_qty": 7.0,
            "filled_price": 101.5,
        }

    def get_portfolio(self) -> dict[str, Any]:
        return {
            "cash": 99_289.5,
            "positions": [
                {
                    "ticker": "SBER",
                    "qty": 7.0,
                    "avg_price": 101.5,
                }
            ],
        }

    def fetch_prices(self, tickers: list[str]) -> dict[str, float]:
        return {"SBER": 101.5}

    def close(self) -> None:
        pass


class FakeBrokenReconcileRouter:
    """Имитирует роутер, у которого портфель временно недоступен."""

    def get_portfolio(self) -> dict[str, Any]:
        raise RuntimeError("Broker API temporarily unavailable")


async def run_smoke_test() -> None:
    install_src_stubs()

    sys.modules.pop("src.engine.live_engine", None)
    live_engine = importlib.import_module("src.engine.live_engine")

    engine = live_engine.LiveEngine(
        cfg=EngineConfig(),
        mode="sandbox",
        price_poll_interval=1,
    )

    router = FakeAcceptedThenFilledRouter()
    engine._routers = {"tinkoff": router}

    order = Order(
        ticker="SBER",
        side=OrderSide.BUY,
        qty=7.0,
        filled_price=100.0,
    )

    # 1. Принятие заявки не должно создавать позицию сразу.
    submitted = await engine._submit_and_track(order)

    assert submitted is False
    assert order.status == "accepted"
    assert order.order_id == "broker-order-42"
    assert "broker-order-42" in engine._pending_orders
    assert "SBER" not in engine._state.positions
    assert len(engine._portfolio.applied_orders) == 0

    # 2. Poll должен применить ровно подтверждённый fill.
    await engine._poll_pending_orders()

    assert router.poll_calls == 1
    assert "broker-order-42" not in engine._pending_orders
    assert len(engine._portfolio.applied_orders) == 1

    applied = engine._portfolio.applied_orders[0]
    assert applied.ticker == "SBER"
    assert applied.qty == 7.0
    assert applied.filled_price == 101.5

    position = engine._state.positions["SBER"]
    assert position.qty == 7.0
    assert position.avg_price == 101.5

    # 3. Ошибка reconcile не имеет права стереть реальную локальную позицию.
    engine._routers = {"tinkoff": FakeBrokenReconcileRouter()}
    cash_before = engine._state.cash

    await engine._reconcile_state()

    assert "SBER" in engine._state.positions
    assert engine._state.positions["SBER"].qty == 7.0
    assert engine._state.cash == cash_before


def test_trading_core_smoke() -> None:
    asyncio.run(run_smoke_test())