from datetime import timedelta

from src.engine.market_state import MarketState, Position
from src.signals.ai_tactic_strategy import (
    AITacticConfig,
    AITacticStrategy,
    TacticalPosition,
)
from src.utils.time_utils import utcnow


def make_strategy():
    cfg = AITacticConfig(initial_tickers=("BTCUSDT",))
    return AITacticStrategy(cfg=cfg, kronos=None, ohlcv={})


def test_phantom_dropped_after_grace():
    s = make_strategy()
    s._positions["BTCUSDT"] = TacticalPosition(
        ticker="BTCUSDT", direction=1, entry_conviction=0.5)
    s._position_intent_at["BTCUSDT"] = utcnow() - timedelta(seconds=901)

    s.on_bar(MarketState(cash=1000.0, positions={}))

    assert "BTCUSDT" not in s._positions


def test_recent_intent_is_kept():
    s = make_strategy()
    s._positions["BTCUSDT"] = TacticalPosition(
        ticker="BTCUSDT", direction=1, entry_conviction=0.5)
    s._position_intent_at["BTCUSDT"] = utcnow()

    s.on_bar(MarketState(cash=1000.0, positions={}))

    assert "BTCUSDT" in s._positions


def test_real_position_blocks_reentry():
    s = make_strategy()
    s.tactical_bias["ETHUSDT"] = {
        "direction": 1, "conviction": 0.9, "horizon": "short"}

    state = MarketState(cash=1000.0, positions={})
    state.positions["ETHUSDT"] = Position(
        ticker="ETHUSDT", qty=0.03,
        avg_price=2400.0, open_time=utcnow())

    orders = s.on_bar(state)

    # реальная позиция уже есть — новых ордеров быть не должно
    assert orders == []


def test_exchange_min_bump():
    cfg = AITacticConfig(
        initial_tickers=("BTCUSDT",),
        base_risk_frac=0.005,
        min_conviction=0.35,
        full_conviction=0.80,
        conviction_gamma=1.5,
        stop_distance_pct=0.02,
        fee_pct=0.0011,
        slippage_pct=0.0005,
        max_notional_frac=0.5,
        max_gross_frac=0.85,
        min_notional=5.0,
    )
    s = AITacticStrategy(cfg=cfg, kronos=None, ohlcv={})
    s.tactical_bias["BTCUSDT"] = {
        "direction": 1, "conviction": 0.5, "horizon": "medium"}

    state = MarketState(cash=1000.0, positions={})
    state.prices["BTCUSDT"] = 80000.0

    orders = s.on_bar(state)

    assert len(orders) == 1
    # risk-based расчёт дал бы ~44 USD; bump поднимает до 85
    assert orders[0].qty * 80000.0 >= 85.0
