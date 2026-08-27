"""
StatArbStrategy — основная торговая стратегия.

Интегрирует:
  1. Cointegration pair selection (PairSelector + cointegration.py)
  2. Kronos прогноз на каждый тикер в паре
  3. News sentiment через NewsSignalExtractor
  4. Spread z-score для точки входа/выхода
  5. Адаптивные пороги: entry если |ER| > sigma * volatility (из signal_generation.py)

Архитектурный принцип: стратегия только генерирует Order-ы.
RiskManager и Portfolio — в движке, не здесь.
"""

from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

from src.engine.market_state import MarketState, NewsSnapshot
from src.engine.orders import Order, OrderSide, OrderType
from src.signals.signal_types import TradingSignal
from src.signals.news_signal import NewsSignalExtractor
from src.kronos_layer.kronos_adapter import KronosAdapter, KronosState
from src.kronos_layer.kronos_features import extract_features

logger = logging.getLogger(__name__)


@dataclass
class PairPosition:
    """Открытая позиция пары (stat-arb)."""
    ticker_a: str
    ticker_b: str
    hedge_ratio: float
    entry_zscore: float
    direction: int         # +1 = long A / short B, -1 = short A / long B
    entry_time: "pd.Timestamp | None" = None


@dataclass
class StrategyConfig:
    """
    Параметры стратегии. Читается из engine.yaml → signal:
    """
    entry_threshold_sigma: float = 2.0    # |zscore| > этого → открываем
    exit_threshold_sigma: float = 0.5     # |zscore| < этого → закрываем
    min_confidence: float = 0.55          # минимальный Kronos confidence
    news_alpha: float = 0.30              # вес новостного сигнала [0, 1]
    news_conflict_filter: bool = True     # блокировать сигналы при конфликте с новостями
    max_pair_positions: int = 5           # макс. открытых пар одновременно
    qty_per_trade: float = 100.0          # лоты на одну ногу (позже заменить position sizing)
    zscore_window: int = 20              # окно для z-score спреда
    news_hours_back: int = 6             # горизонт новостей


class StatArbStrategy:
    """
    Stat-arb стратегия с Kronos-фильтром и новостным слоем.

    Жизненный цикл:
      1. Инициализация: pair_selector.select(prices) → список пар
      2. Каждый бар: on_bar(state) → список Order
         a. Обновить z-score спреда
         b. Запросить Kronos прогноз (если есть KronosAdapter)
         c. Смотреть news_signal_weight из state.news
         d. Комбинировать → TradingSignal
         e. Генерировать Order
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str, float]],
        prices_history: pd.DataFrame,
        cfg: StrategyConfig,
        kronos: Optional[KronosAdapter] = None,
        news_extractor: Optional[NewsSignalExtractor] = None,
        ohlcv: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        """
        Args:
            pairs:          список (ticker_a, ticker_b, hedge_ratio) из PairSelector
            prices_history: исторические close-цены для z-score расчёта
            cfg:            параметры стратегии
            kronos:         KronosAdapter (None → только spread сигнал)
            news_extractor: NewsSignalExtractor (None → без новостного слоя)
            ohlcv:          {ticker: OHLCV DataFrame} для Kronos (None → close-only)
        """
        self.pairs = pairs
        self.cfg = cfg
        self._kronos = kronos
        self._news = news_extractor
        self._ohlcv = ohlcv or {}

        # тикеры, за которыми следим
        self.tickers: List[str] = list(
            {t for a, b, _ in pairs for t in (a, b)}
        )

        # история цен для z-score (мутируется on_bar)
        self._prices_history = prices_history.copy()

        # открытые позиции по парам
        self._pair_positions: Dict[Tuple[str, str], PairPosition] = {}

        logger.info(
            "StatArbStrategy init | pairs=%d | tickers=%d | kronos=%s | news=%s",
            len(pairs), len(self.tickers),
            "yes" if kronos else "no",
            "yes" if news_extractor else "no",
        )

    # ── основной хук ──────────────────────────────────────────────────────

    def on_bar(self, state: MarketState) -> List[Order]:
        """Вызывается движком на каждом баре. Возвращает список ордеров."""
        # добавляем текущие цены в историю
        bar = pd.DataFrame(
            {t: [state.prices[t]] for t in self.tickers if t in state.prices},
            index=[state.timestamp],
        )
        self._prices_history = pd.concat(
            [self._prices_history, bar]
        ).tail(max(self.cfg.zscore_window * 3, 100))

        orders: List[Order] = []

        for ticker_a, ticker_b, hedge_ratio in self.pairs:
            try:
                pair_orders = self._process_pair(
                    state, ticker_a, ticker_b, hedge_ratio
                )
                orders.extend(pair_orders)
            except Exception as exc:
                logger.warning(
                    "Pair %s/%s error: %s", ticker_a, ticker_b, exc
                )

        return orders

    # ── обработка одной пары ──────────────────────────────────────────────

    def _process_pair(
        self,
        state: MarketState,
        ticker_a: str,
        ticker_b: str,
        hedge_ratio: float,
    ) -> List[Order]:
        price_a = state.prices.get(ticker_a)
        price_b = state.prices.get(ticker_b)
        if price_a is None or price_b is None:
            return []

        # ── z-score спреда ────────────────────────────────────────────────
        zscore = self._compute_zscore(ticker_a, ticker_b, hedge_ratio)
        if zscore is None:
            return []

        pair_key = (ticker_a, ticker_b)
        open_pos = self._pair_positions.get(pair_key)

        # ── закрытие позиции ──────────────────────────────────────────────
        if open_pos is not None:
            if abs(zscore) < self.cfg.exit_threshold_sigma:
                logger.info(
                    "EXIT pair %s/%s | zscore=%.2f",
                    ticker_a, ticker_b, zscore,
                )
                orders = self._close_pair(state, open_pos, price_a, price_b)
                del self._pair_positions[pair_key]
                return orders
            return []

        # ── уже слишком много открытых пар ────────────────────────────────
        if len(self._pair_positions) >= self.cfg.max_pair_positions:
            return []

        # ── Kronos сигналы ────────────────────────────────────────────────
        signal_a = self._get_kronos_signal(ticker_a, state)
        signal_b = self._get_kronos_signal(ticker_b, state)

        # ── Kronos confidence check ───────────────────────────────────────
        conf_a = signal_a.confidence if signal_a else 1.0
        conf_b = signal_b.confidence if signal_b else 1.0
        if min(conf_a, conf_b) < self.cfg.min_confidence:
            logger.debug(
                "Pair %s/%s: low confidence %.2f < %.2f — skip",
                ticker_a, ticker_b, min(conf_a, conf_b), self.cfg.min_confidence,
            )
            return []

        # ── адаптивный порог входа ────────────────────────────────────────
        # Используем волатильность Kronos если есть, иначе rolling std z-score
        vol_norm = 0.0
        if signal_a and signal_a.kronos_state:
            vol_norm = signal_a.kronos_state.volatility_norm

        # entry если |zscore| > entry_threshold (адаптивный: threshold * (1 + vol_norm))
        adaptive_threshold = self.cfg.entry_threshold_sigma * (1.0 + vol_norm * 0.5)
        if abs(zscore) < adaptive_threshold:
            return []

        # ── направление ──────────────────────────────────────────────────
        # zscore > 0 → спред выше нормы → short A, long B
        # zscore < 0 → спред ниже нормы → long A, short B
        direction = -1 if zscore > 0 else 1

        # ── новостной фильтр ──────────────────────────────────────────────
        if self._news is not None:
            news_a = state.get_news(ticker_a)
            news_b = state.get_news(ticker_b)
            direction = self._apply_news_filter(
                direction, ticker_a, ticker_b, news_a, news_b, signal_a, signal_b
            )
            if direction == 0:
                return []

        # ── генерация сигнала и ордеров ───────────────────────────────────
        signal = self._build_combined_signal(
            ticker_a, ticker_b, hedge_ratio, zscore, direction,
            signal_a, signal_b, state,
        )

        logger.info(
            "ENTER pair %s/%s | zscore=%.2f | dir=%+d | %s",
            ticker_a, ticker_b, zscore, direction, signal,
        )

        orders = self._open_pair(state, ticker_a, ticker_b, hedge_ratio, direction, price_a, price_b)
        if orders:
            self._pair_positions[pair_key] = PairPosition(
                ticker_a=ticker_a,
                ticker_b=ticker_b,
                hedge_ratio=hedge_ratio,
                entry_zscore=zscore,
                direction=direction,
                entry_time=state.timestamp,
            )
        return orders

    # ── z-score спреда ────────────────────────────────────────────────────

    def _compute_zscore(
        self,
        ticker_a: str,
        ticker_b: str,
        hedge_ratio: float,
    ) -> Optional[float]:
        if ticker_a not in self._prices_history.columns or \
           ticker_b not in self._prices_history.columns:
            return None

        s_a = self._prices_history[ticker_a].dropna()
        s_b = self._prices_history[ticker_b].dropna()
        common = s_a.index.intersection(s_b.index)

        if len(common) < self.cfg.zscore_window:
            return None

        spread = s_a.loc[common] - hedge_ratio * s_b.loc[common]
        window = spread.iloc[-self.cfg.zscore_window:]
        mu, sigma = window.mean(), window.std()
        if sigma < 1e-10:
            return None
        return float((spread.iloc[-1] - mu) / sigma)

    # ── Kronos сигнал для тикера ──────────────────────────────────────────

    def _get_kronos_signal(
        self, ticker: str, state: MarketState
    ) -> Optional[TradingSignal]:
        if self._kronos is None:
            return None
        try:
            source = self._ohlcv.get(ticker) or self._prices_history.get(ticker)
            if source is None:
                return None
            kstate: KronosState = self._kronos.encode_prices(source, ticker=ticker)
            kf = extract_features(kstate, last_price=state.prices.get(ticker, 0.0))
            return TradingSignal(
                ticker=ticker,
                direction=kf.trend_direction,
                strength=abs(kf.expected_return),
                confidence=kf.confidence,
                expected_return=kf.expected_return,
                source="kronos",
                kronos_state=kstate,
                news_snapshot=state.get_news(ticker),
            )
        except Exception as exc:
            logger.debug("Kronos signal for %s failed: %s", ticker, exc)
            return None

    # ── применение новостного фильтра ─────────────────────────────────────

    def _apply_news_filter(
        self,
        direction: int,
        ticker_a: str,
        ticker_b: str,
        news_a: Optional[NewsSnapshot],
        news_b: Optional[NewsSnapshot],
        signal_a: Optional[TradingSignal],
        signal_b: Optional[TradingSignal],
    ) -> int:
        """
        Проверяет конфликты между торговым направлением и новостным сентиментом.
        Возвращает скорректированный direction или 0 если сигнал заблокирован.
        """
        # Санкционный риск — жёсткий стоп
        for ticker, news in [(ticker_a, news_a), (ticker_b, news_b)]:
            if news and news.sanction_risk > 0.6:
                logger.warning(
                    "Pair %s/%s: high sanction risk %.2f for %s — BLOCKED",
                    ticker_a, ticker_b, news.sanction_risk, ticker,
                )
                return 0

        if not self.cfg.news_conflict_filter:
            return direction

        # Конфликт сентимента
        weight_a = news_a.signal_weight if news_a else 0.0
        weight_b = news_b.signal_weight if news_b else 0.0

        # для пары: direction=+1 значит long A / short B
        # конфликт если новости A очень негативные при long A
        # или новости B очень позитивные при short B
        conflict_a = direction > 0 and weight_a < -0.5
        conflict_b = direction < 0 and weight_b > 0.5

        if conflict_a or conflict_b:
            logger.info(
                "Pair %s/%s: news conflict dir=%+d wa=%.2f wb=%.2f — BLOCKED",
                ticker_a, ticker_b, direction, weight_a, weight_b,
            )
            return 0

        # Новости усиливают/ослабляют сигнал, но не блокируют
        # (количественный эффект обрабатывается в _build_combined_signal)
        return direction

    # ── построение финального сигнала ────────────────────────────────────

    def _build_combined_signal(
        self,
        ticker_a: str,
        ticker_b: str,
        hedge_ratio: float,
        zscore: float,
        direction: int,
        signal_a: Optional[TradingSignal],
        signal_b: Optional[TradingSignal],
        state: MarketState,
    ) -> TradingSignal:
        news_a = state.get_news(ticker_a)
        news_b = state.get_news(ticker_b)

        # базовый сигнал от спреда
        spread_strength = min(abs(zscore) / self.cfg.entry_threshold_sigma, 1.0)

        # Kronos strength (если есть)
        kronos_strength = 0.0
        if signal_a and signal_b:
            kronos_strength = (signal_a.strength + signal_b.strength) / 2

        # новостной вес
        news_weight = 0.0
        if news_a or news_b:
            wa = news_a.signal_weight if news_a else 0.0
            wb = news_b.signal_weight if news_b else 0.0
            news_weight = (wa - wb) * direction  # нормализуем к направлению пары

        # комбинируем через weights
        alpha_news = self.cfg.news_alpha
        alpha_kronos = 0.3 if signal_a and signal_b else 0.0
        alpha_spread = 1.0 - alpha_news - alpha_kronos

        combined_strength = (
            alpha_spread * spread_strength
            + alpha_kronos * kronos_strength
            + alpha_news * max(0.0, news_weight)
        )

        conf = min(
            signal_a.confidence if signal_a else 1.0,
            signal_b.confidence if signal_b else 1.0,
        )

        return TradingSignal(
            ticker=ticker_a,
            direction=direction,
            strength=float(min(combined_strength, 1.0)),
            confidence=float(conf),
            expected_return=float(direction * spread_strength * 0.02),
            source="combined",
            spread_zscore=zscore,
            hedge_ratio=hedge_ratio,
            pair_ticker=ticker_b,
            kronos_state=signal_a.kronos_state if signal_a else None,
            news_snapshot=news_a,
        )

    # ── генерация ордеров ─────────────────────────────────────────────────

    def _open_pair(
        self,
        state: MarketState,
        ticker_a: str,
        ticker_b: str,
        hedge_ratio: float,
        direction: int,
        price_a: float,
        price_b: float,
    ) -> List[Order]:
        """
        direction=+1 → long A, short B
        direction=-1 → short A, long B
        qty_b подбирается пропорционально hedge_ratio
        """
        qty_a = self.cfg.qty_per_trade
        qty_b = round(qty_a * hedge_ratio)
        if qty_b < 1:
            qty_b = 1

        side_a = OrderSide.BUY if direction > 0 else OrderSide.SELL
        side_b = OrderSide.SELL if direction > 0 else OrderSide.BUY

        return [
            Order(
                order_id=str(uuid.uuid4()),
                ticker=ticker_a,
                side=side_a,
                qty=qty_a,
                order_type=OrderType.MARKET,
                filled_price=price_a,
            ),
            Order(
                order_id=str(uuid.uuid4()),
                ticker=ticker_b,
                side=side_b,
                qty=qty_b,
                order_type=OrderType.MARKET,
                filled_price=price_b,
            ),
        ]

    def _close_pair(
        self,
        state: MarketState,
        pos: PairPosition,
        price_a: float,
        price_b: float,
    ) -> List[Order]:
        open_a = state.positions.get(pos.ticker_a)
        open_b = state.positions.get(pos.ticker_b)

        orders = []
        if open_a:
            side = OrderSide.SELL if open_a.qty > 0 else OrderSide.BUY
            orders.append(Order(
                order_id=str(uuid.uuid4()),
                ticker=pos.ticker_a,
                side=side,
                qty=abs(open_a.qty),
                order_type=OrderType.MARKET,
                filled_price=price_a,
            ))
        if open_b:
            side = OrderSide.SELL if open_b.qty > 0 else OrderSide.BUY
            orders.append(Order(
                order_id=str(uuid.uuid4()),
                ticker=pos.ticker_b,
                side=side,
                qty=abs(open_b.qty),
                order_type=OrderType.MARKET,
                filled_price=price_b,
            ))
        return orders
