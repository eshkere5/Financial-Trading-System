from types import SimpleNamespace

import pytest

from src.engine.orders import Order, OrderSide, OrderStatus
from src.engine.portfolio import Portfolio


class JournalStub:
    def __init__(self):
        self.rows = []

    def record(self, **kwargs):
        self.rows.append(kwargs)


def make_portfolio(
    cash: float = 1000.0,
    commission: float = 0.001,
):
    state = SimpleNamespace(
        cash=cash,
        positions={},
        trades=[],
    )

    portfolio = object.__new__(Portfolio)
    portfolio._state = state
    portfolio._commission = commission
    portfolio._journal = JournalStub()
    portfolio._applied_order_ids = set()

    return portfolio, state


def filled_order(
    ticker: str,
    side: OrderSide,
    qty: float,
    price: float,
    order_id: str,
):
    return Order(
        ticker=ticker,
        side=side,
        qty=qty,
        filled_price=price,
        order_id=order_id,
    )


def test_buy_updates_cash_position_and_fill_fields():
    portfolio, state = make_portfolio()

    order = filled_order(
        "BTCUSDT",
        OrderSide.BUY,
        1.5,
        100.0,
        "order-buy-1",
    )

    assert portfolio.apply_order(order) is True

    assert state.cash == pytest.approx(849.85)
    assert state.positions["BTCUSDT"].qty == pytest.approx(1.5)
    assert state.positions["BTCUSDT"].avg_price == pytest.approx(100.0)

    assert order.status == OrderStatus.FILLED
    assert order.filled_qty == pytest.approx(1.5)
    assert order.remaining_qty == pytest.approx(0.0)
    assert order.is_fully_filled is True


def test_spot_like_equity_is_cash_plus_market_value():
    portfolio, state = make_portfolio()

    order = filled_order(
        "BTCUSDT",
        OrderSide.BUY,
        1.5,
        100.0,
        "order-equity-1",
    )

    assert portfolio.apply_order(order) is True

    market_price = 110.0
    equity = (
        state.cash
        + state.positions["BTCUSDT"].qty * market_price
    )

    # 1000 initial + 15 unrealized - 0.15 commission.
    assert equity == pytest.approx(1014.85)


def test_duplicate_order_is_not_applied_twice():
    portfolio, state = make_portfolio()

    order = filled_order(
        "ETHUSDT",
        OrderSide.BUY,
        2.0,
        50.0,
        "duplicate-order",
    )

    assert portfolio.apply_order(order) is True

    cash_after_first = state.cash
    trades_after_first = len(state.trades)

    assert portfolio.apply_order(order) is False
    assert state.cash == pytest.approx(cash_after_first)
    assert len(state.trades) == trades_after_first
    assert len(portfolio._journal.rows) == 1


def test_position_reversal_resets_average_price():
    portfolio, state = make_portfolio(
        cash=1000.0,
        commission=0.0,
    )

    buy = filled_order(
        "BTCUSDT",
        OrderSide.BUY,
        1.0,
        100.0,
        "open-long",
    )
    assert portfolio.apply_order(buy) is True

    sell = filled_order(
        "BTCUSDT",
        OrderSide.SELL,
        2.0,
        110.0,
        "reverse-short",
    )
    assert portfolio.apply_order(sell) is True

    position = state.positions["BTCUSDT"]
    assert position.qty == pytest.approx(-1.0)
    assert position.avg_price == pytest.approx(110.0)


def test_reduction_keeps_original_average_price():
    portfolio, state = make_portfolio(
        cash=1000.0,
        commission=0.0,
    )

    buy = filled_order(
        "BTCUSDT",
        OrderSide.BUY,
        2.0,
        100.0,
        "open-two",
    )
    assert portfolio.apply_order(buy) is True

    sell = filled_order(
        "BTCUSDT",
        OrderSide.SELL,
        1.0,
        120.0,
        "reduce-one",
    )
    assert portfolio.apply_order(sell) is True

    position = state.positions["BTCUSDT"]
    assert position.qty == pytest.approx(1.0)
    assert position.avg_price == pytest.approx(100.0)


def test_sell_to_close_restores_cash_and_removes_position():
    portfolio, state = make_portfolio(
        cash=1000.0,
        commission=0.0,
    )

    buy = filled_order(
        "BTCUSDT",
        OrderSide.BUY,
        2.0,
        100.0,
        "buy-close-test",
    )
    assert portfolio.apply_order(buy) is True

    sell = filled_order(
        "BTCUSDT",
        OrderSide.SELL,
        2.0,
        110.0,
        "sell-close-test",
    )
    assert portfolio.apply_order(sell) is True

    assert "BTCUSDT" not in state.positions
    assert state.cash == pytest.approx(1020.0)


def test_derivative_fill_does_not_change_cash():
    portfolio, state = make_portfolio(
        cash=1000.0,
        commission=0.001,
    )
    state.accounting_mode = "broker_equity"
    state.broker_equity = 1000.0

    order = filled_order(
        "BTCUSDT",
        OrderSide.BUY,
        0.001,
        70000.0,
        "linear-buy-1",
    )

    assert portfolio.apply_order(order) is True
    assert state.cash == pytest.approx(1000.0)
    assert state.positions["BTCUSDT"].qty == pytest.approx(0.001)
    assert order.is_fully_filled is True
