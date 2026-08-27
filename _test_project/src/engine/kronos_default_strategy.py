"""
KronosDefaultStrategy — дефолтный режим "чистый Kronos" на фиксированном
списке акций/крипты + сразу подключённый новостной парсер по этим же тикерам.

Идея: не всегда нужен полный цикл LLMStrategist (15-минутный tactical_bias).
Для быстрого старта/бэктеста/простого автономного режима достаточно:
  1. Взять статичный watchlist (акции + крипта), заданный пользователем в конфиге.
  2. Kronos сам прогнозирует direction/confidence по каждому тикеру на price-тике.
  3. NewsRiskPipeline сразу знает про эти тикеры — watch_terms генерирует
     DeepSeek (AliasGenerator) ОДНИМ разовым вызовом по названиям тикеров,
     НЕ дожидаясь первого strategist-цикла и без статичного справочника.
  4. Никакого tactical_bias от LLM не требуется — это самостоятельный режим.

LLMStrategist можно включить позже как надстройку (strategy_mode="hybrid" в
strategist.py) — тогда он будет ДОПОЛНЯТЬ этот дефолтный режим, а не заменять.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.engine.market_state import MarketState, combined_confidence
from src.engine.orders import Order, OrderSide, OrderType
from src.news_agent_client.news_deep_dive import RISK_POLICY
from src.signals.signal_types import TradingSignal
from src.kronos_layer.kronos_adapter import KronosAdapter
from src.kronos_layer.kronos_features import extract_features
from src.news_agent_client.ticker_intelligence import AliasGenerator, TickerResolver, ResolveResult

logger = logging.getLogger(__name__)

@dataclass
class KronosDefaultConfig:
    tickers: tuple = field(default_factory=tuple)   # фиксированный watchlist, e.g. ("SBER","BTCUSD")
    qty_per_trade: float = 100.0
    max_positions: int = 10
    min_confidence: float = 0.4            # ниже — Kronos-сигнал игнорируется
    flat_confidence: float = 0.15          # ниже этого — считаем сигнал "flat", закрываем позицию
    negative_asymmetry_factor: float = 1.5
    alias_generator_cfg: dict = field(default_factory=dict)  # конфиг для AliasGenerator (DeepSeek api_key/model/...)
    ticker_resolver_cfg: dict = field(default_factory=dict)  # конфиг для TickerResolver (тот же DeepSeek-ключ)


@dataclass
class _OpenPos:
    ticker: str
    direction: int
    confidence: float


class KronosDefaultStrategy:
    """
    Простейший самостоятельный режим: торгует напрямую по прогнозу Kronos
    для заданного списка тикеров, без участия LLM-стратега.

    on_bar(state) — тот же контракт, что у AITacticStrategy/StatArbStrategy,
    поэтому режимы можно комбинировать в LiveEngine (list из strategies).
    """

    def __init__(
        self,
        cfg: KronosDefaultConfig,
        kronos: KronosAdapter,
        ohlcv: Dict[str, object],
    ) -> None:
        self.cfg = cfg
        self._kronos = kronos
        self._ohlcv = ohlcv
        self.tickers: List[str] = list(cfg.tickers)
        self._positions: Dict[str, _OpenPos] = {}
        self._alias_generator = AliasGenerator(cfg.alias_generator_cfg)
        self._ticker_resolver = TickerResolver(cfg.ticker_resolver_cfg or cfg.alias_generator_cfg)
        # user_bias: {ticker: direction} — стартовое направление из текстового запроса пользователя
        # (например "шорт по Газпрому"). Используется как override для первого сигнала Kronos,
        # затем Kronos продолжает управлять позицией сам.
        self.user_bias: Dict[str, int] = {}

        logger.info(
            "KronosDefaultStrategy init | tickers=%d min_confidence=%.2f",
            len(self.tickers), cfg.min_confidence,
        )

    # ── интеграция с NewsRiskPipeline: сидирование watch_terms без LLM ──────

    def seed_watch_terms(self) -> Dict[str, List[str]]:
        """
        Отдаёт готовые watch_terms для этого watchlist — вызывается ОДИН РАЗ
        при старте LiveEngine (до первого strategist-цикла). DeepSeek САМ
        определяет полное название компании/монеты, синонимы и ключевые
        фигуры по тикеру — никакого статичного справочника не требуется.
        Это разовый (не событийный) вызов — не влияет на скорость парсера.
        """
        if not self.tickers:
            return {}
        generated = self._alias_generator.generate(self.tickers)
        return {ticker: list({ticker, *terms})[:8] for ticker, terms in generated.items()}

    def set_watchlist_from_text(self, user_text: str) -> ResolveResult:
        """
        Принимает свободный текст пользователя ("играть на Сбербанк шорт",
        "ai сектор лонг", "биткоин и эфир") и:
          1. Резолвит его в конкретные тикеры через DeepSeek (TickerResolver).
          2. Прямые совпадения (названные компании/монеты) сразу добавляются
             в watchlist с указанным направлением как user_bias.
          3. Предложения по сектору возвращаются В ЭТОМ ЖЕ ResolveResult —
             вызывающий код (UI/CLI) решает, показывать подтверждение
             пользователю перед добавлением в watchlist.
        """
        result = self._ticker_resolver.resolve(user_text)

        for r in result.direct_matches():
            if not r.ticker:
                continue
            if r.ticker not in self.tickers:
                self.tickers.append(r.ticker)
            if r.direction != 0:
                self.user_bias[r.ticker] = r.direction

        logger.info(
            "KronosDefault: watchlist из текста | direct=%d sector_suggestions=%d ambiguous=%d",
            len(result.direct_matches()), len(result.sector_suggestions()), len(result.ambiguous),
        )
        return result

    def confirm_sector_tickers(self, tickers_with_direction: Dict[str, int]) -> None:
        """Вызывается UI/CLI после того, как пользователь подтвердил предложенные по сектору тикеры."""
        for ticker, direction in tickers_with_direction.items():
            ticker = ticker.upper()
            if ticker not in self.tickers:
                self.tickers.append(ticker)
            if direction != 0:
                self.user_bias[ticker] = direction

    def set_universe(self, tickers: list) -> None:
        """Позволяет расширять watchlist динамически (UniverseSelector), как у AITacticStrategy."""
        self.tickers = sorted(set(self.tickers) | set(tickers))

    # ── основной хук ──────────────────────────────────────────────────────

    def on_bar(self, state: MarketState) -> List[Order]:
        orders: List[Order] = []

        for ticker in self.tickers:
            if ticker not in state.prices:
                continue

            signal = self._get_kronos_signal(ticker, state)
            if signal is None:
                continue

            news = state.get_news(ticker) if hasattr(state, "get_news") else None

            # ── жёсткий блок при существенном риске (severity x materiality) ──
            if news is not None and self._risk_action(news) == "block":
                if ticker in self._positions:
                    orders.extend(self._close(state, ticker))
                logger.warning(
                    "KronosDefault: %s блокирован риском | category=%s effective=%.2f",
                    ticker, news.risk_category, news.effective_risk,
                )
                continue

            effective_confidence = signal.confidence
            direction = signal.direction

            # ── user_bias: пользователь явно указал направление текстом ──
            # (например "шорт по Газпрому") — форсируем direction, если
            # позиция ещё не открыта. После открытия Kronos управляет сам
            # (переворот/закрытие по обычной логике сигнала).
            forced_direction = self.user_bias.get(ticker)
            if forced_direction is not None and ticker not in self._positions:
                direction = forced_direction
                effective_confidence = max(effective_confidence, self.cfg.min_confidence)

            if news is not None and news.risk_category != "none":
                news_direction = -1 if news.sentiment < 0 else 1 if news.sentiment > 0 else 0
                effective_confidence = combined_confidence(
                    kronos_confidence=signal.confidence,
                    kronos_direction=direction,
                    news_sentiment=news.sentiment,
                    news_confidence=news.effective_risk,
                    news_direction=news_direction,
                    negative_asymmetry_factor=self.cfg.negative_asymmetry_factor,
                )
                effective_confidence *= self._risk_dampen_factor(news)

            # ── закрыть, если сигнал ослаб до "flat" или развернулся ──
            if ticker in self._positions:
                pos = self._positions[ticker]
                if effective_confidence < self.cfg.flat_confidence or direction != pos.direction:
                    orders.extend(self._close(state, ticker))
                    del self._positions[ticker]
                    continue

            # ── открыть новую позицию ──
            if ticker not in self._positions and direction != 0 and effective_confidence >= self.cfg.min_confidence:
                if len(self._positions) >= self.cfg.max_positions:
                    continue
                order = self._open(state, ticker, direction, effective_confidence)
                if order:
                    orders.append(order)

        return orders

    # ── риск-политика (та же RISK_POLICY, что и у AITacticStrategy) ──────────

    @staticmethod
    def _risk_action(news) -> str:
        if news is None or getattr(news, "risk_category", "none") == "none":
            return "pass"
        policy = RISK_POLICY.get(news.risk_category)
        if not policy:
            return "pass"
        return policy["action"] if news.effective_risk >= policy["threshold"] else "pass"

    @staticmethod
    def _risk_dampen_factor(news) -> float:
        policy = RISK_POLICY.get(getattr(news, "risk_category", "none"))
        if policy and policy.get("action") == "dampen":
            return float(policy.get("factor", 1.0))
        return 1.0

    def _open(self, state: MarketState, ticker: str, direction: int, confidence: float) -> Optional[Order]:
        qty = self.cfg.qty_per_trade * confidence
        side = OrderSide.BUY if direction > 0 else OrderSide.SELL
        self._positions[ticker] = _OpenPos(ticker=ticker, direction=direction, confidence=confidence)
        logger.info("KronosDefault: OPEN %s dir=%+d confidence=%.2f", ticker, direction, confidence)
        return Order(
            order_id=str(uuid.uuid4()), ticker=ticker, side=side, qty=qty,
            order_type=OrderType.MARKET, filled_price=state.prices[ticker],
        )

    def _close(self, state: MarketState, ticker: str) -> List[Order]:
        open_pos = state.positions.get(ticker) if hasattr(state, "positions") else None
        if not open_pos:
            return []
        side = OrderSide.SELL if open_pos.qty > 0 else OrderSide.BUY
        logger.info("KronosDefault: CLOSE %s", ticker)
        return [Order(
            order_id=str(uuid.uuid4()), ticker=ticker, side=side, qty=abs(open_pos.qty),
            order_type=OrderType.MARKET, filled_price=state.prices[ticker],
        )]

    def _get_kronos_signal(self, ticker: str, state: MarketState) -> Optional[TradingSignal]:
        try:
            source = self._ohlcv.get(ticker)
            if source is None:
                return None
            kstate = self._kronos.encode_prices(source, ticker=ticker)
            kf = extract_features(kstate, last_price=state.prices.get(ticker, 0.0))
            return TradingSignal(
                ticker=ticker, direction=kf.trend_direction, strength=abs(kf.expected_return),
                confidence=kf.confidence, expected_return=kf.expected_return,
                source="kronos", kronos_state=kstate,
            )
        except Exception as exc:
            logger.debug("KronosDefault: Kronos signal for %s failed: %s", ticker, exc)
            return None
