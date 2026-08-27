from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN_REAL_E2E = os.getenv("RUN_REAL_E2E") == "1"
RUN_BYBIT_TRADE = os.getenv("RUN_BYBIT_TRADE") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_REAL_E2E,
    reason=(
        "Реальный E2E выключен. Запуск: "
        "$env:RUN_REAL_E2E='1'; pytest -q -s tests/test_e2e_trading_real.py"
    ),
)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    return {
        name: getattr(value, name)
        for name in dir(value)
        if not name.startswith("_") and not callable(getattr(value, name))
    }


def _status_str(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).strip().lower()


def _load_configs():
    import yaml

    from src.engine.config_loader import (
        load_bybit_config,
        load_engine_config,
        load_kronos_config,
        load_llm_shared,
        merge_llm_section,
    )

    engine_cfg = load_engine_config("configs/engine.yaml")
    bybit_cfg = load_bybit_config(testnet=True)
    kronos_cfg = load_kronos_config("configs/kronos.yaml")

    ls_raw = yaml.safe_load(
        Path("configs/llm_strategist.yaml").read_text(encoding="utf-8")
    ) or {} 
    strategist_cfg = merge_llm_section(
        ls_raw.get("llm_strategist", {}),
        shared=load_llm_shared("configs/llm_shared.yaml"),
    )

    return engine_cfg, bybit_cfg, kronos_cfg, strategist_cfg


def _print_kronos(symbol: str, candles: pd.DataFrame, state: Any) -> None:
    print("\n" + "=" * 88)
    print("[E2E][MARKET -> KRONOS]")
    print(f"symbol              : {symbol}")
    print(f"bars                : {len(candles)}")
    print(f"first_timestamp     : {candles['timestamp'].iloc[0]}")
    print(f"last_timestamp      : {candles['timestamp'].iloc[-1]}")
    print(f"last_close          : {float(candles['close'].iloc[-1]):.8f}")
    print(f"forecast_close      : {getattr(state, 'forecast_close', None)}")
    print(f"expected_return     : {getattr(state, 'expected_return', None)}")
    print(f"volatility          : {getattr(state, 'volatility', None)}")
    print(f"volatility_norm     : {getattr(state, 'volatility_norm', None)}")
    print(f"confidence          : {getattr(state, 'confidence', None)}")
    print(f"direction           : {getattr(state, 'direction', None)}")
    print("=" * 88)


def _print_strategist(decision: Any) -> None:
    d = _as_dict(decision)

    print("\n" + "=" * 88)
    print("[E2E][STRATEGIST]")
    print(f"is_fallback         : {d.get('is_fallback')}")
    print(f"halted              : {d.get('halted')}")
    print(f"size_multiplier     : {d.get('size_multiplier')}")
    print(f"watchlist           : {d.get('watchlist')}")
    print(f"vetoed_pairs        : {d.get('vetoed_pairs')}")
    print(f"tactical_bias       : {d.get('tactical_bias')}")
    print(f"news_alpha_override : {d.get('news_alpha_override')}")
    print(f"rationale           : {str(d.get('rationale', ''))[:1200]}")
    print("=" * 88)


def test_real_bybit_kronos_strategist_pipeline() -> None:
    """
    Реальная цепочка:
      Bybit testnet -> OHLCV -> Kronos weights -> KronosState ->
      LLMStrategist(DeepSeek) -> AITacticStrategy -> RiskManager.

    Без RUN_BYBIT_TRADE=1 заявки не создаются.
    """
    from src.engine.market_state import MarketState
    from src.engine.orders import Order, OrderSide, OrderType
    from src.engine.risk_manager import RiskManager
    from src.executors.bybit_client import BybitClient
    from src.executors.execution_router import ExecutionRouter
    from src.kronos_layer.kronos_adapter import KronosAdapter
    from src.llm_strategist.strategist import LLMStrategist, build_context
    from src.signals.ai_tactic_strategy import AITacticConfig, AITacticStrategy
    from src.utils.time_utils import utcnow

    engine_cfg, bybit_cfg, kronos_cfg, strategist_cfg = _load_configs()

    symbol = os.getenv("E2E_SYMBOL", "BTCUSDT").upper()
    interval = os.getenv("E2E_INTERVAL", "1h")
    bars = int(os.getenv("E2E_BARS", "240"))

    # 1. Реальный Bybit testnet: котировки, баланс, OHLCV.
    client = BybitClient(bybit_cfg)

    prices = client.get_prices([symbol])
    assert symbol in prices, f"Bybit не вернул цену для {symbol}: {prices}"

    last_price = float(prices[symbol])
    assert last_price > 0.0

    portfolio = client.get_portfolio()
    candles = client.get_candles(
        symbol=symbol,
        interval=interval,
        limit=bars,
    )

    assert isinstance(candles, pd.DataFrame)
    assert len(candles) >= 32, f"Недостаточно свечей: {len(candles)}"
    assert {"timestamp", "open", "high", "low", "close", "volume"} <= set(
        candles.columns
    )

    print("\n" + "=" * 88)
    print("[E2E][BYBIT TESTNET]")
    print(f"category            : {bybit_cfg.category}")
    print(f"symbol              : {symbol}")
    print(f"last_price          : {last_price:.8f}")
    print(f"portfolio           : {portfolio}")
    print("=" * 88)

    # 2. Реальный Kronos. MOCK запрещён: иначе не проверяются реальные веса.
    kronos = KronosAdapter(kronos_cfg)
    assert not getattr(kronos, "is_mock", True), (
        "Kronos запущен в MOCK-режиме. "
        "Этот E2E-тест должен проверять реальные веса модели."
    )

    kronos_state = kronos.encode_prices(candles, ticker=symbol)
    next_close = kronos.predict_next_close(kronos_state)

    assert kronos_state is not None
    assert next_close is not None
    assert getattr(kronos_state, "confidence", None) is not None
    assert getattr(kronos_state, "direction", None) is not None

    _print_kronos(symbol, candles, kronos_state)
    
    # 3. Реальный контекст, который получит strategist.
    market_state = MarketState(
        cash=float(getattr(engine_cfg, "initial_capital", 100_000.0)),
        timestamp=utcnow(),
    )
    market_state.prices[symbol] = last_price
    market_state.pnl_history.append(
        (market_state.timestamp, market_state.portfolio_value)
    )

    # Временная стратегия нужна только для формирования реального Kronos signal
    # в build_context через стандартный public API стратегии.
    tactic_cfg = AITacticConfig(
        min_conviction=0.35,
        qty_per_trade=float(os.getenv("E2E_QTY", "0.001")),
        max_positions=1,
        kronos_confirmation_required=False,
        initial_tickers=(symbol,),
    )
    tactic = AITacticStrategy(
        cfg=tactic_cfg,
        kronos=kronos,
        ohlcv={symbol: candles},
    )

    tactic.last_kronos_states = {
        symbol: kronos_state,
    }

    strategist = LLMStrategist(strategist_cfg)
    context = build_context(
        market_state,
        strategies=[tactic],
        recent_pnl=list(market_state.pnl_history),
    )
    decision = strategist.decide(context)

    print("\n[E2E][CONTEXT TO STRATEGIST]")
    print(context)
    assert symbol in context["kronos"], (
        f"KronosState для {symbol} не попал в strategist context: {context}"
    )

    assert not getattr(decision, "is_fallback", True), (
        "Strategist вернул fallback. Проверь DeepSeek key, баланс, API/timeout. "
        f"Decision={decision!r}"
    )

    _print_strategist(decision)

    # 5. Реальное применение tactical_bias в стратегии.
    tactic.set_tactical_bias(getattr(decision, "tactical_bias", {}))
    orders = tactic.on_bar(market_state)

    print("\n" + "=" * 88)
    print("[E2E][STRATEGY -> RISK]")
    print(f"generated_orders    : {len(orders)}")
    for index, order in enumerate(orders, start=1):
        print(
            f"order[{index}] "
            f"ticker={order.ticker} side={order.side} qty={order.qty} "
            f"price={order.filled_price} closes={order.closes_position}"
        )
    print("=" * 88)

    risk = RiskManager(engine_cfg.risk)
    approved_orders: list[Order] = []

    for order in orders:
        approved = risk.check_order(order, market_state)
        print(
            "[E2E][RISK] "
            f"ticker={order.ticker} side={order.side} qty={order.qty} "
            f"approved={approved}"
        )
        if approved:
            approved_orders.append(order)

    # Нормально, если strategist не дал bias или дал HOLD:
    # тест проверяет живую полную цепочку без требования искусственно открыть сделку.
    if not RUN_BYBIT_TRADE:
        return

    # 6. РЕАЛЬНАЯ testnet-сделка: выполняй только с явным флагом.
    if not approved_orders:
        pytest.skip(
            "Торговый сигнал не сформирован: реальную testnet-заявку не создаём."
        )

    router = ExecutionRouter(
        mode="sandbox",
        broker="bybit",
        bybit_cfg=bybit_cfg,
    )

    order = approved_orders[0]
    order.order_type = OrderType.MARKET
    order.filled_price = market_state.prices[order.ticker]

    accepted = router.submit(order)

    print("\n" + "=" * 88)
    print("[E2E][BYBIT TESTNET ORDER]")
    print(f"submit_return       : {accepted}")
    print(f"order_id            : {order.order_id}")
    print(f"status              : {order.status}")
    print(f"ticker              : {order.ticker}")
    print(f"side                : {order.side}")
    print(f"qty                 : {order.qty}")
    print("=" * 88)

    assert order.order_id, "Роутер не вернул order_id."
    assert accepted or _status_str(order.status) in {"accepted", "filled"}, (
        f"Bybit testnet не подтвердил заявку: {order!r}"
    )

    time.sleep(2.0)

    after = client.get_portfolio()
    print("\n[E2E][BYBIT TESTNET PORTFOLIO AFTER ORDER]")
    print(after)