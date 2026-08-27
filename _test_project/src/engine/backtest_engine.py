"""
BacktestEngine — цикл по историческим барам.
Интегрирует: стратегии, RiskManager, Kronos-фичи, новостной риск-слой.

"""

from __future__ import annotations
import json
import logging
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.engine.config_loader import EngineConfig
from src.engine.market_state import MarketState
from src.engine.portfolio import Portfolio
from src.engine.orders import Order
from src.engine.risk_manager import RiskManager, RiskConfig
from src.utils.time_utils import utcnow

logger = logging.getLogger(__name__)


def _safe_asdict(obj: Any) -> Any:
    """Конвертирует dataclass/объект в JSON-совместимый словарь, не падая на незнакомых типах."""
    if obj is None:
        return None
    if is_dataclass(obj):
        return {k: _safe_asdict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _safe_asdict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_asdict(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class BacktestLogger:
    """
    Пишет по одной JSON-строке на бар в data/backtest_logs/{run_id}.jsonl.
    Отдельно логирует каждую сделку с trade_id для последующего "replay".
    """

    def __init__(self, run_id: Optional[str] = None, log_dir: str = "data/backtest_logs") -> None:
        self.run_id = run_id or utcnow().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.bars_path = self.log_dir / f"{self.run_id}_bars.jsonl"
        self.trades_path = self.log_dir / f"{self.run_id}_trades.jsonl"
        self._bars_f = open(self.bars_path, "a", encoding="utf-8")
        self._trades_f = open(self.trades_path, "a", encoding="utf-8")
        logger.info("BacktestLogger: run_id=%s -> %s / %s", self.run_id, self.bars_path, self.trades_path)

    def log_bar(
        self,
        ts,
        state: MarketState,
        kronos_states: Dict[str, Any],
        news_by_ticker: Dict[str, Any],
        decision: Any,
        orders_raw: List[Order],
        orders_final: List[Order],
    ) -> None:
        record = {
            "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "equity": round(state.portfolio_value, 2),
            "cash": round(state.cash, 2),
            "n_positions": len(state.positions),
            "total_pnl": round(state.total_pnl, 2),
            "kronos": {
                ticker: _safe_asdict(k) for ticker, k in kronos_states.items()
            },
            "news": {
                ticker: _safe_asdict(snap) for ticker, snap in news_by_ticker.items()
            },
            "strategist_decision": _safe_asdict(decision) if decision is not None else None,
            "orders_raw": [_safe_asdict(o) for o in orders_raw],
            "orders_final": [_safe_asdict(o) for o in orders_final],
        }
        self._bars_f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def log_trade(self, trade_id: str, order: Order, state: MarketState, reason: str = "") -> None:
        record = {
            "trade_id": trade_id,
            "ts": state.timestamp.isoformat() if hasattr(state.timestamp, "isoformat") else str(state.timestamp),
            "ticker": order.ticker,
            "side": getattr(order.side, "value", str(order.side)),
            "qty": order.qty,
            "filled_price": order.filled_price,
            "reason": reason,
            "equity_after": round(state.portfolio_value, 2),
        }
        self._trades_f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def close(self) -> None:
        self._bars_f.close()
        self._trades_f.close()


class BacktestEngine:
    def __init__(self, cfg: EngineConfig, strategist=None, news_by_ticker_provider=None) -> None:
        """
        strategist: опциональный LLMStrategist — если передан, на каждом баре
            строится context (как в LiveEngine.build_context) и вызывается
            .decide(), чтобы бэктест воспроизводил ту же логику size/veto/bias,
            что и live-режим, а не только "чистый" Kronos.
        news_by_ticker_provider: опциональная функция (ts) -> Dict[ticker, NewsSnapshot],
            позволяющая прогнать заранее посчитанный историчный новостной риск
            (например, из сохранённых NewsRiskAssessment) синхронно с ts бара.
        """
        self.cfg = cfg
        self._state = MarketState(
            cash=cfg.initial_capital,
            timestamp=utcnow(),
        )

        self._portfolio = Portfolio(self._state, cfg.commission)

        risk_cfg: RiskConfig = getattr(cfg, "risk", RiskConfig())
        self._risk = RiskManager(risk_cfg)
        self._risk.reset_peak(cfg.initial_capital)

        self._strategies: list = []
        self._strategist = strategist
        self._news_by_ticker_provider = news_by_ticker_provider
        self._decision = None
        self._log = BacktestLogger()

    def register_strategy(self, strategy) -> None:
        self._strategies.append(strategy)

    # ── загрузка данных ────────────────────────────────────────────────────

    def _load_data(self) -> pd.DataFrame:
        """
        Загружает исторические данные из data/raw/historical/{ticker}.csv.
        Возвращает wide DataFrame: индекс datetime, колонки = тикеры (close).
        """
        from src.signals.data_collector import DataCollector

        tickers: List[str] = getattr(self.cfg, "tickers", [])
        if not tickers:
            raise RuntimeError(
                "EngineConfig.tickers пуст — добавь список тикеров в engine.yaml"
            )

        collector = DataCollector(historical_dir="data/raw/historical")
        frames: Dict[str, pd.Series] = {}

        for ticker in tickers:
            try:
                df = collector.load_historical(ticker)
                col = "close" if "close" in df.columns else df.columns[0]
                frames[ticker] = df[col]
                logger.info("Loaded %d bars for %s", len(df), ticker)
            except FileNotFoundError:
                logger.warning("No historical data for %s — skipping", ticker)

        if not frames:
            raise RuntimeError("No historical data loaded for any ticker")

        prices = pd.DataFrame(frames).sort_index()

        if hasattr(self.cfg, "start_date") and self.cfg.start_date:
            prices = prices[prices.index >= pd.Timestamp(self.cfg.start_date)]
        if hasattr(self.cfg, "end_date") and self.cfg.end_date:
            prices = prices[prices.index <= pd.Timestamp(self.cfg.end_date)]

        return prices

    # ── strategist filter (то же самое, что в LiveEngine._apply_strategist_filter) ──

    def _apply_strategist_filter(self, orders: List[Order]) -> List[Order]:
        d = self._decision
        if d is None or getattr(d, "is_fallback", False):
            return orders

        veto = d.veto_set() if hasattr(d, "veto_set") else set()
        out: List[Order] = []
        for order in orders:
            is_closing = getattr(order, "closes_position", False) or self._is_reducing(order)

            if getattr(d, "halted", False) and not is_closing:
                logger.info("[backtest] Strategist HALT — блокирован открывающий ордер %s", order.ticker)
                continue
            if order.ticker in veto and not is_closing:
                logger.info("[backtest] Strategist VETO — блокирован ордер %s", order.ticker)
                continue

            size_mult = getattr(d, "size_multiplier", 1.0)
            if not is_closing and size_mult != 1.0:
                order.qty = max(1, int(order.qty * size_mult))
            out.append(order)
        return out

    def _is_reducing(self, order: Order) -> bool:
        pos = self._state.positions.get(order.ticker)
        if pos is None:
            return False
        pos_qty = getattr(pos, "qty", 0)
        side = getattr(order, "side", "")
        side_val = side.value if hasattr(side, "value") else str(side)
        return (pos_qty > 0 and side_val.upper() == "SELL") or (pos_qty < 0 and side_val.upper() == "BUY")

    # ── основной цикл ──────────────────────────────────────────────────────

    def run(self) -> MarketState:
        logger.info(
            "Backtest started | %s → %s | run_id=%s",
            getattr(self.cfg, "start_date", "?"),
            getattr(self.cfg, "end_date", "?"),
            self._log.run_id,
        )

        data = self._load_data()

        try:
            for ts, bar in data.iterrows():
                if self._risk.is_halted:
                    logger.warning("Trading halted by RiskManager at %s", ts)
                    break

                self._state.timestamp = ts
                self._state.prices.update(bar.dropna().to_dict())
                self._state.pnl_history.append((ts, self._state.portfolio_value))

                # ── новостной контекст на этот бар (если провайдер передан) ──
                if self._news_by_ticker_provider is not None:
                    news_snap = self._news_by_ticker_provider(ts) or {}
                    self._state.news.update(news_snap)

                # ── strategist decision (опционально, для parity с live-режимом) ──
                if self._strategist is not None:
                    try:
                        from src.llm_strategist.strategist import build_context
                        context = build_context(self._state, self._strategies, recent_pnl=list(self._state.pnl_history))
                        self._decision = self._strategist.decide(context)
                    except Exception as exc:
                        logger.warning("[backtest] Strategist error at %s: %s", ts, exc)

                # ── stop-loss / take-profit по открытым позициям ──────────────
                close_orders = self._risk.check_positions(self._state)
                for order in close_orders:
                    trade_id = str(uuid.uuid4())[:8]
                    self._portfolio.apply_order(order)
                    self._log.log_trade(trade_id, order, self._state, reason="stop_loss_take_profit")

                # ── сигналы стратегий ─────────────────────────────────────────
                kronos_states_snapshot: Dict[str, Any] = {}
                orders_raw: List[Order] = []
                orders_final: List[Order] = []

                for strategy in self._strategies:
                    ks = getattr(strategy, "last_kronos_states", None)
                    if ks:
                        kronos_states_snapshot.update(ks)

                    orders: List[Order] = strategy.on_bar(self._state)
                    orders_raw.extend(orders)

                    filtered = self._apply_strategist_filter(orders)
                    for order in filtered:
                        order.filled_price = self._state.prices.get(order.ticker)
                        if self._risk.check_order(order, self._state):
                            self._portfolio.apply_order(order)
                            orders_final.append(order)
                            trade_id = str(uuid.uuid4())[:8]
                            self._log.log_trade(trade_id, order, self._state, reason="strategy_signal")

                # ── drawdown check ────────────────────────────────────────────
                self._risk.check_drawdown(self._state)
                if hasattr(self._risk, "check_daily_loss"):
                    self._risk.check_daily_loss(self._state)

                # ── лог бара (для последующего разбора "почти всего") ─────────
                self._log.log_bar(
                    ts=ts,
                    state=self._state,
                    kronos_states=kronos_states_snapshot,
                    news_by_ticker=dict(self._state.news),
                    decision=self._decision,
                    orders_raw=orders_raw,
                    orders_final=orders_final,
                )

            logger.info(
                "Backtest done | final PnL=%.2f | trades=%d | run_id=%s",
                self._state.total_pnl,
                len(self._state.trades),
                self._log.run_id,
            )
        finally:
            self._log.close()

        return self._state
