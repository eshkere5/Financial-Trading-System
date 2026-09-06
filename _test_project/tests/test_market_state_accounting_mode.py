from src.engine.market_state import MarketState, Position
from src.utils.time_utils import utcnow


def test_spot_portfolio_value_includes_position_market_value():
    state = MarketState(
        cash=900.0,
        accounting_mode="spot",
        prices={"BTCUSDT": 110.0},
    )
    state.positions["BTCUSDT"] = Position(
        ticker="BTCUSDT",
        qty=1.0,
        avg_price=100.0,
        open_time=utcnow(),
    )

    assert state.portfolio_value == 1010.0


def test_broker_equity_mode_does_not_add_position_notional():
    state = MarketState(
        cash=1018.20,
        broker_equity=1018.20,
        broker_equity_source="bybit",
        accounting_mode="broker_equity",
        prices={"BTCUSDT": 77398.6},
    )
    state.positions["BTCUSDT"] = Position(
        ticker="BTCUSDT",
        qty=0.00160255,
        avg_price=76162.1,
        open_time=utcnow(),
    )

    assert state.portfolio_value == 1018.20
    assert state.total_pnl == 0.0
