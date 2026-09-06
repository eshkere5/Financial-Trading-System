"""
AITacticStrategy — одиночные позиции по tactical_bias от LLM-стратега.

Факт позиции берётся только из MarketState.positions, который обновляется
после подтверждённого broker fill / reconciliation. Локального shadow-book
позиций здесь нет: это исключает фантомные позиции после reject/partial fill.
"""

from __future__ import annotations
from src.utils.time_utils import utcnow

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.engine.market_state import MarketState, combined_confidence
from src.engine.orders import Order, OrderSide, OrderType
from src.news_agent_client.news_deep_dive import RISK_POLICY, evaluate_news_risk
from src.signals.signal_types import TradingSignal
from src.kronos_layer.kronos_adapter import KronosAdapter
from src.kronos_layer.kronos_features import extract_features

logger = logging.getLogger(__name__)

# TEMP до risk-policy v2: минимумы notional ордеров Bybit linear
# (BTCUSDT 0.001 BTC, ETHUSDT 0.01 ETH) для exchange-min bump.
# В v2 значение будет браться из instrument spec роутера.
BYBIT_LINEAR_MIN_NOTIONAL = {
    "BTCUSDT": 85.0,
    "ETHUSDT": 26.0,
}

# ── режимный слой (мягкий импорт: нет модуля — работаем как раньше) ──
try:
    from src.regime.regime_detector import compute_regime, signal_multiplier
    _REGIME_ENABLED = True
except ImportError:
    logger.warning("regime_detector не найден (src/regime/) — режимный слой ОТКЛЮЧЁН")
    _REGIME_ENABLED = False

# informational-порог для журнала (сам override живёт на уровне стратегиста)
NEWS_OVERRIDE_RISK = 0.7



def calculate_order_size(
    equity, entry, stop, conviction, regime_mult,
    base_risk_frac=0.005, min_conviction=0.35, full_conviction=0.80,
    conviction_gamma=1.5, drawdown_mult=1.0, portfolio_mult=1.0,
    fee_pct=0.0011, slippage_pct=0.0005, max_notional_frac=0.50,
):
    """Сначала риск в деньгах R, затем notional = R / d_eff."""
    if equity <= 0 or entry <= 0 or stop <= 0:
        return 0.0, 0.0
    if conviction < min_conviction or regime_mult <= 0:
        return 0.0, 0.0
    x = (conviction - min_conviction) / (full_conviction - min_conviction)
    x = max(0.0, min(x, 1.0))
    conviction_mult = x ** conviction_gamma
    regime_size = max(0.0, min(regime_mult, 1.0))  # спека: только понижающие множители
    drawdown_mult = max(0.0, min(drawdown_mult, 1.0))
    portfolio_mult = max(0.0, min(portfolio_mult, 1.0))
    risk_usd = equity * base_risk_frac * conviction_mult * regime_size * drawdown_mult * portfolio_mult
    stop_pct = abs(entry - stop) / entry
    d_eff = stop_pct + fee_pct + slippage_pct
    if d_eff <= 0:
        return 0.0, 0.0
    raw_notional = risk_usd / d_eff
    notional = min(raw_notional, equity * max_notional_frac)
    return notional, notional / entry

@dataclass
class TacticalPosition:
    """Открытая одиночная позиция под тактику LLM (не пара)."""
    ticker: str
    direction: int  # +1 long, -1 short
    entry_conviction: float
    horizon: str = "short"  # "short" | "medium" | "long" — влияет на exit-условия


@dataclass
class AITacticConfig:
    min_conviction: float = 0.35        # ниже — сигнал игнорируется
    qty_per_trade: float = 100.0        # legacy, не используется в сайзинге
    risk_frac: float = 0.20             # доля equity на сделку при conviction=1.0
    max_position_frac: float = 0.50     # жёсткий потолок доли equity на позицию
    min_notional: float = 6.0           # ниже мин. ордера Bybit — пропуск
    base_risk_frac: float = 0.005       # базовый допустимый убыток на сделку (0.5% equity)
    full_conviction: float = 0.80       # выше — размер не растёт
    conviction_gamma: float = 1.5
    stop_distance_pct: float = 0.02     # волатильностный прокси стопа (= engine stop_pct)
    fee_pct: float = 0.0011
    slippage_pct: float = 0.0005
    max_notional_frac: float = 0.50     # потолок нотационала одной позиции
    max_gross_frac: float = 0.90        # потолок суммарной экспозиции портфеля
    max_positions: int = 10
    kronos_confirmation_required: bool = False  # true = Kronos должен подтвердить направление
    exit_on_bias_removed: bool = True   # закрывать, если тикер выпал из tactical_bias
    exit_on_direction_flip: bool = True # закрывать при смене направления тактики
    negative_asymmetry_factor: float = 1.5  # см. combined_confidence()
    initial_tickers: tuple = field(default_factory=tuple)  # стартовый watchlist
    regime_sizing_enabled: bool = True  # NEW: множители режима на размер


class AITacticStrategy:
    """
    Торгует одиночными тикерами напрямую по тактике LLM-стратега.

    Жизненный цикл:
    1. LiveEngine на каждом strategist-цикле обновляет tactical_bias через
       set_tactical_bias(decision.tactical_bias)
    2. UniverseSelector расширяет self.tickers через set_universe(...)
    3. on_bar(state): для каждого тикера в tactical_bias —
       a. проверить риск новостей через severity x materiality
       b. объединить Kronos-прогноз и новостной сигнал через combined_confidence()
       c. NEW: применить режимный множитель направления
       d. открыть/удержать/закрыть позицию по итоговой conviction и direction
    """

    def __init__(
        self,
        cfg: AITacticConfig,
        kronos: Optional[KronosAdapter] = None,
        ohlcv: Optional[Dict[str, object]] = None,
    ) -> None:
        self.cfg = cfg
        self._kronos = kronos
        self._ohlcv = ohlcv or {}

        self.tactical_bias: Dict[str, Dict[str, float]] = {}
        self._positions: Dict[str, TacticalPosition] = {}
        # intent-таймстемпы: grace для pending-подтверждения
        self._position_intent_at: Dict[str, object] = {}
        self.tickers: list = list(cfg.initial_tickers)

        # NEW 2026-08-24: журнал решений + кэш режимов + дедуп сигнатур
        self._journal = None
        self._regime_cache: Dict[str, object] = {}   # ticker -> (len_closes, RegimeInfo)
        self._last_decision_sig: Dict[str, tuple] = {}

        logger.info(
            "AITacticStrategy init | kronos=%s | min_conviction=%.2f | regime_layer=%s",
            "yes" if kronos else "no", cfg.min_conviction,
            "on" if (_REGIME_ENABLED and cfg.regime_sizing_enabled) else "off",
        )

    # ── NEW: подключение журнала (вызывается из main/live_engine) ────

    def set_journal(self, journal) -> None:
        """TradeJournal для записи решений. None — журнал отключён."""
        self._journal = journal

    # ── NEW: режим по тикеру (кэш по длине ряда — пересчёт на новом баре) ──

    def _regime_for(self, ticker: str):
        if not _REGIME_ENABLED or not self.cfg.regime_sizing_enabled:
            return None
        source = self._ohlcv.get(ticker)
        if source is None:
            return None
        try:
            closes = source["close"] if hasattr(source, "__getitem__") else None
            if closes is None:
                return None
            n = len(closes)
            cached = self._regime_cache.get(ticker)
            if cached and cached[0] == n:
                return cached[1]
            info = compute_regime(source, ticker=ticker)
            self._regime_cache[ticker] = (n, info)
            return info
        except Exception as exc:
            logger.debug("regime compute failed for %s: %s", ticker, exc)
            return None

    # ── NEW: журнал решений с дедупликацией по сигнатуре ─────────────

    def _log_decision(
        self,
        *,
        ticker: str,
        direction: int,
        action: str,
        skip_reason: str = "",
        qty: float = 0.0,
        entry_price: float = 0.0,
        kronos_signal: Optional[TradingSignal] = None,
        news=None,
        regime_info=None,
        direction_multiplier: float = 1.0,
    ) -> None:
        if self._journal is None:
            return
        regime_name = getattr(regime_info, "regime", "") if regime_info else ""
        sig = (ticker, action, skip_reason, regime_name, direction)
        if self._last_decision_sig.get(ticker) == sig:
            return  # то же решение в том же состоянии — не дублируем
        self._last_decision_sig[ticker] = sig
        try:
            self._journal.record_decision(
                ticker=ticker,
                pred_dir=direction,
                action=action,
                skip_reason=skip_reason,
                qty=qty,
                entry_price=entry_price,
                expected_return=getattr(kronos_signal, "expected_return", 0.0) or 0.0,
                kronos_conf=getattr(kronos_signal, "confidence", 0.0) or 0.0,
                regime=regime_name,
                regime_dist=getattr(regime_info, "close_vs_sma50", 0.0) if regime_info else 0.0,
                regime_vol_pctile=getattr(regime_info, "vol_pctile", 0.0) if regime_info else 0.0,
                news_effective_risk=getattr(news, "effective_risk", 0.0) if news else 0.0,
                news_category=getattr(news, "risk_category", "") if news else "",
                news_override=bool(news and getattr(news, "effective_risk", 0.0) >= NEWS_OVERRIDE_RISK),
                direction_multiplier=direction_multiplier,
            )
        except Exception as exc:
            logger.debug("journal record_decision failed (некритично): %s", exc)

    # ── обновление тактики от LLM-стратега ───────────────────────────

    def set_tactical_bias(self, tactical_bias: Dict[str, Dict[str, float]]) -> None:
        """Вызывается LiveEngine после каждого strategist-цикла (~15 мин)."""
        self.tactical_bias = dict(tactical_bias or {})
        self.tickers = sorted(set(self.tickers) | set(self.tactical_bias.keys()))
        logger.info(
            "AITacticStrategy | tactical_bias обновлён | tickers=%d bias=%d",
            len(self.tickers), len(self.tactical_bias),
        )

    def set_universe(self, tickers: list) -> None:
        """Вызывается из UniverseSelector-цикла — держит watchlist в живых тикерах."""
        self.tickers = sorted(set(self.tickers) | set(tickers))

    # ── основной хук (тот же контракт, что у StatArbStrategy) ────────

    def on_bar(self, state: MarketState) -> List[Order]:
        orders: List[Order] = []

        # ── sync shadow book with broker truth ──────────────
        # Реальная позиция подтверждена только через
        # MarketState.positions (broker fill / reconcile).
        # Rejected/ниже минимума ордер не должен навсегда
        # блокировать повторный вход (grace = PENDING_ORDER_TTL).
        now_ts = utcnow()
        for held in list(self._positions.keys()):
            if held in getattr(state, "positions", {}):
                continue
            intent_ts = self._position_intent_at.get(held)
            age = (
                None if intent_ts is None
                else (now_ts - intent_ts).total_seconds()
            )
            if intent_ts is None or age > 900:
                logger.info(
                    "AITacticStrategy: DROP phantom tactical "
                    "position %s (no broker position, age=%s)",
                    held, age,
                )
                del self._positions[held]
                self._position_intent_at.pop(held, None)
        logger.info(
            "AITacticStrategy: on_bar | internal_positions=%s bias=%s prices=%s",
            list(self._positions.keys()), list(self.tactical_bias.keys()),
            list(state.prices.keys()),
        )

        # 1. закрыть позиции, чья тактика исчезла/развернулась
        for ticker in list(self._positions.keys()):
            pos = self._positions[ticker]
            bias = self.tactical_bias.get(ticker)

            should_close = False
            if bias is None and self.cfg.exit_on_bias_removed:
                should_close = True
            elif bias is not None and self.cfg.exit_on_direction_flip:
                new_direction = int(bias.get("direction", 0))
                if new_direction != 0 and new_direction != pos.direction:
                    should_close = True

            # ── принудительное закрытие при блокирующем новостном риске ──
            news = state.get_news(ticker) if hasattr(state, "get_news") else None
            if news is not None and self._risk_action(news) == "block":
                logger.warning(
                    "AITacticStrategy: %s принудительно закрыт — риск-блок (%s, effective=%.2f)",
                    ticker, news.risk_category, news.effective_risk,
                )
                should_close = True

            if should_close and ticker in state.prices:
                orders.extend(self._close_position(state, pos))
                del self._positions[ticker]

        # 2. открыть/удержать позиции по актуальной тактике
        if len(self._positions) < self.cfg.max_positions:
            for ticker, bias in self.tactical_bias.items():
                if ticker in self._positions:
                    continue
                # Реальная позиция от брокера (reconcile/fill) —
                # не докупаем поверх неё после рестарта.
                if ticker in getattr(state, "positions", {}):
                    continue
                if ticker not in state.prices:
                    continue
                if len(self._positions) >= self.cfg.max_positions:
                    break

                order = self._try_open(state, ticker, bias)
                if order:
                    orders.append(order)

        return orders

    # ── риск-политика (severity x materiality) ───────────────────────

    @staticmethod
    def _risk_action(news) -> str:
        action, _, _ = evaluate_news_risk(news)
        return action

    @staticmethod
    def _risk_dampen_factor(news) -> float:
        action, multiplier, _ = evaluate_news_risk(news)
        return 0.0 if action == "block" else multiplier

    # ── попытка открыть позицию по тактике ───────────────────────────

    def _try_open(self, state: MarketState, ticker: str, bias: Dict[str, float]) -> Optional[Order]:
        direction = int(bias.get("direction", 0))
        conviction = float(bias.get("conviction", 0.0))
        horizon = str(bias.get("horizon", "short"))

        regime_info = self._regime_for(ticker)
        news = state.get_news(ticker) if hasattr(state, "get_news") else None

        if direction == 0 or conviction < self.cfg.min_conviction:
            logger.info(
                "AITacticStrategy: SKIP %s — conviction=%.3f < min=%.2f (dir=%+d)",
                ticker, conviction, self.cfg.min_conviction, direction,
            )
            self._log_decision(ticker=ticker, direction=direction, action="skipped",
                               skip_reason="bias_flat_or_low_conviction", news=news,
                               regime_info=regime_info)
            return None

        # ── жёсткий блок только если severity x materiality реально существенны ──
        if news is not None:
            action = self._risk_action(news)
            if action == "block":
                logger.warning(
                    "AITacticStrategy: %s блокирован риском | category=%s severity=%.2f materiality=%.2f effective=%.2f | %s",
                    ticker, news.risk_category, news.severity, news.materiality,
                    news.effective_risk, news.rationale,
                )
                self._log_decision(ticker=ticker, direction=direction, action="skipped",
                                   skip_reason="news_block", news=news, regime_info=regime_info)
                return None

        effective_conviction = conviction
        kronos_signal = self._get_kronos_signal(ticker, state) if self._kronos is not None else None

        if kronos_signal is not None:
            if news is not None and news.risk_category != "none":
                # ── единая метрика уверенности с учётом асимметрии негатива ──
                effective_conviction = combined_confidence(
                    kronos_confidence=kronos_signal.confidence,
                    kronos_direction=kronos_signal.direction,
                    news_sentiment=news.sentiment,
                    news_confidence=news.effective_risk,
                    news_direction=(-1 if news.sentiment < 0 else 1 if news.sentiment > 0 else 0),
                    negative_asymmetry_factor=self.cfg.negative_asymmetry_factor,
                ) * conviction
            elif kronos_signal.direction != 0 and kronos_signal.direction != direction:
                if self.cfg.kronos_confirmation_required:
                    logger.info(
                        "AITacticStrategy: %s тактика=%+d Kronos=%+d — конфликт, пропуск",
                        ticker, direction, kronos_signal.direction,
                    )
                    self._log_decision(ticker=ticker, direction=direction, action="skipped",
                                       skip_reason="kronos_conflict", kronos_signal=kronos_signal,
                                       news=news, regime_info=regime_info)
                    return None
                effective_conviction *= 0.5
            elif kronos_signal.direction == direction:
                effective_conviction = min(1.0, effective_conviction * 1.1)

        # ── дополнительный дэмпинг за риск-категорию (legal/reputational/etc) ──
        if news is not None:
            pass  # dampen перенесён в сайзинг

        if effective_conviction < self.cfg.min_conviction:
            logger.info(
                "AITacticStrategy: SKIP %s — effective_conviction=%.3f < min=%.2f (raw=%.3f, dampen=%.2f)",
                ticker, effective_conviction, self.cfg.min_conviction, conviction,
                self._risk_dampen_factor(news),
            )
            self._log_decision(ticker=ticker, direction=direction, action="skipped",
                               skip_reason="conviction_floor", kronos_signal=kronos_signal,
                               news=news, regime_info=regime_info)
            return None

        direction_mult = 1.0
        if regime_info is not None:
            direction_mult = signal_multiplier(regime_info.regime, direction, ticker=ticker)

        price = float(state.prices[ticker])
        equity = float(getattr(state, "portfolio_value", 0.0) or getattr(state, "equity", 0.0) or 0.0)
        if equity <= 0:
            equity = float(getattr(state, "cash", 0.0) or 1000.0)

        if direction_mult <= 0:
            logger.info("AITacticStrategy: SKIP %s — regime_mult %.2f <= 0 (режим против сделки)",
                        ticker, direction_mult)
            return None

        # drawdown-мультипликатор: линейное снижение от 0% до 5% просадки, пол 0.25
        dd_mult = 1.0
        hist = getattr(state, "pnl_history", None)
        if hist:
            try:
                peak = max(v for _, v in hist)
                if peak > 0:
                    dd = max(0.0, (peak - equity) / peak)
                    dd_mult = max(0.25, min(1.0, 1.0 - dd / 0.05))
            except (TypeError, ValueError):
                pass

        # портфельная экспозиция: остаток gross-лимита как кэп нотационала
        gross = 0.0
        for pos_t, pos in getattr(state, "positions", {}).items():
            p_price = float(getattr(state, "prices", {}).get(pos_t, 0.0) or 0.0)
            gross += abs(float(getattr(pos, "qty", 0.0) or 0.0) * p_price)
        gross_remaining = max(0.0, equity * self.cfg.max_gross_frac - gross)

        notional, _ = calculate_order_size(
            equity=equity, entry=price,
            stop=price * (1.0 - self.cfg.stop_distance_pct),
            conviction=effective_conviction,
            regime_mult=direction_mult,
            base_risk_frac=self.cfg.base_risk_frac,
            min_conviction=self.cfg.min_conviction,
            full_conviction=self.cfg.full_conviction,
            conviction_gamma=self.cfg.conviction_gamma,
            drawdown_mult=dd_mult,
            fee_pct=self.cfg.fee_pct,
            slippage_pct=self.cfg.slippage_pct,
            max_notional_frac=self.cfg.max_notional_frac,
        )
        # exchange-min bump: risk-budget-aware
        exchange_min = BYBIT_LINEAR_MIN_NOTIONAL.get(ticker, 85.0)
        if notional < exchange_min:
            eff_loss = (self.cfg.stop_distance_pct
                        + self.cfg.fee_pct
                        + self.cfg.slippage_pct)
            if exchange_min * eff_loss <= equity * self.cfg.base_risk_frac:
                logger.info(
                    "AITacticStrategy: BUMP %s notional %.2f -> %.2f",
                    ticker, notional, exchange_min)
                notional = exchange_min

        dampen = self._risk_dampen_factor(news)
        notional = min(notional * dampen, gross_remaining)

        floor = max(self.cfg.min_notional, exchange_min)
        if notional < floor:
            logger.info(
                "AITacticStrategy: SKIP %s — notional %.2f < floor %.2f",
                ticker, notional, floor)
            return None

        qty = notional / price
        logger.info("AITacticStrategy: SIZE %s notional=%.2f qty=%.6f dd_mult=%.2f gross_rem=%.2f",
                    ticker, notional, qty, dd_mult, gross_remaining)
        side = OrderSide.BUY if direction > 0 else OrderSide.SELL

        self._position_intent_at[ticker] = utcnow()
        self._positions[ticker] = TacticalPosition(
            ticker=ticker, direction=direction,
            entry_conviction=effective_conviction, horizon=horizon,
        )

        logger.info(
            "AITacticStrategy: OPEN %s dir=%+d conviction=%.2f horizon=%s regime=%s(x%.2f)",
            ticker, direction, effective_conviction, horizon,
            getattr(regime_info, "regime", "?"), direction_mult,
        )
        self._log_decision(ticker=ticker, direction=direction, action="taken",
                           qty=qty, entry_price=state.prices[ticker],
                           kronos_signal=kronos_signal, news=news,
                           regime_info=regime_info, direction_multiplier=direction_mult)

        return Order(
            order_id=str(uuid.uuid4()),
            ticker=ticker,
            side=side,
            qty=qty,
            order_type=OrderType.MARKET,
            filled_price=state.prices[ticker],
        )

    def _close_position(self, state: MarketState, pos: TacticalPosition) -> List[Order]:
        open_pos = state.positions.get(pos.ticker) if hasattr(state, "positions") else None
        if not open_pos:
            return []
        side = OrderSide.SELL if open_pos.qty > 0 else OrderSide.BUY
        logger.info("AITacticStrategy: CLOSE %s (bias removed/flipped/risk)", pos.ticker)
        return [Order(
            order_id=str(uuid.uuid4()),
            ticker=pos.ticker,
            side=side,
            qty=abs(open_pos.qty),
            order_type=OrderType.MARKET,
            filled_price=state.prices[pos.ticker],
        )]

    def _get_kronos_signal(self, ticker: str, state: MarketState) -> Optional[TradingSignal]:
        if not hasattr(self, "_ksig_cache"):
            self._ksig_cache = {}
            self._ksig_ttl = 900.0  # прогноз Kronos максимум раз в 15 мин на тикер
        now = time.monotonic()
        cached = self._ksig_cache.get(ticker)
        if cached is not None and now - cached[0] < self._ksig_ttl:
            return cached[1]
        try:
            source = self._ohlcv.get(ticker)
            if source is None:
                return None
            kstate = self._kronos.encode_prices(source, ticker=ticker)
            kf = extract_features(kstate, last_price=state.prices.get(ticker, 0.0))
            sig = TradingSignal(
                ticker=ticker,
                direction=kf.trend_direction,
                strength=abs(kf.expected_return),
                confidence=kf.confidence,
                expected_return=kf.expected_return,
                source="kronos",
                kronos_state=kstate,
            )
            self._ksig_cache[ticker] = (now, sig)
            return sig
        except Exception as exc:
            logger.debug("AITacticStrategy: Kronos signal for %s failed: %s", ticker, exc)
            return None