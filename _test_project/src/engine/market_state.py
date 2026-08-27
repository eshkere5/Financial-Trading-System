"""
MarketState — каноническая локальная проекция портфеля и рынка.

Broker остаётся источником истины для фактически исполненных заявок и
состояния счёта. MarketState обновляется только в asyncio event loop:
- prices: price_loop;
- positions/cash/trades: Portfolio.apply_order() либо reconcile;
- news: LiveEngine._apply_news_assessment().
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from src.utils.time_utils import utcnow
from typing import Dict, List, Optional

from src.utils.time_utils import utcnow


def _clamp(value: float, low: float, high: float) -> float:
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


@dataclass
class Position:
    ticker: str
    qty: float  # > 0 long, < 0 short
    avg_price: float
    open_time: datetime

    def __post_init__(self) -> None:
        if not self.ticker:
            raise ValueError("Position.ticker must not be empty")
        if not math.isfinite(self.qty) or abs(self.qty) < 1e-12:
            raise ValueError(f"Invalid position qty for {self.ticker}: {self.qty}")
        if not math.isfinite(self.avg_price) or self.avg_price <= 0:
            raise ValueError(f"Invalid avg_price for {self.ticker}: {self.avg_price}")


@dataclass
class Trade:
    ticker: str
    qty: float
    price: float
    side: str  # buy | sell
    ts: datetime
    commission: float

    def __post_init__(self) -> None:
        if not self.ticker:
            raise ValueError("Trade.ticker must not be empty")
        if not math.isfinite(self.qty) or abs(self.qty) < 1e-12:
            raise ValueError(f"Invalid trade qty: {self.qty}")
        if not math.isfinite(self.price) or self.price <= 0:
            raise ValueError(f"Invalid trade price: {self.price}")
        if not math.isfinite(self.commission) or self.commission < 0:
            raise ValueError(f"Invalid commission: {self.commission}")


@dataclass
class NewsSnapshot:
    """
    Канонический снимок news-risk для стратегии, RiskManager и LLM.

    effective_risk — агрегат активных новостей из NewsMemoryStore с TTL,
    а не пересчёт риска последней новости.
    """

    ticker: str
    risk_category: str = "none"
    severity: float = 0.0
    materiality: float = 0.0
    effective_risk: float = 0.0
    sentiment: float = 0.0
    horizon: str = "short"
    headline_count: int = 0
    updated_at: datetime = field(default_factory=utcnow)
    rationale: str = ""
    source_title: str = ""
    source_link: str = ""

    def __post_init__(self) -> None:
        self.risk_category = str(self.risk_category or "none").lower()
        self.severity = max(0.0, min(1.0, float(self.severity)))
        self.materiality = max(0.0, min(1.0, float(self.materiality)))
        self.effective_risk = max(0.0, min(1.0, float(self.effective_risk)))
        self.sentiment = max(-1.0, min(1.0, float(self.sentiment)))
        self.horizon = self.horizon if self.horizon in {"short", "long"} else "short"
        self.headline_count = max(0, int(self.headline_count))

    @property
    def signal_weight(self) -> float:
        return float(
            max(-1.0, min(1.0, self.sentiment - self.effective_risk * 0.5))
        )

    # Временная обратная совместимость для старых потребителей.
    @property
    def sanction_risk(self) -> float:
        return self.effective_risk

    @property
    def relevance(self) -> float:
        return self.materiality


def combined_confidence(
    kronos_confidence: float,
    kronos_direction: int,
    news_sentiment: float,
    news_confidence: float,
    news_direction: int,
    negative_asymmetry_factor: float = 1.5,
    negative_sentiment_threshold: float = -0.3,
    negative_confidence_threshold: float = 0.3,
) -> float:
    k_conf = _clamp(float(kronos_confidence), 0.0, 1.0)
    n_conf = _clamp(float(news_confidence), 0.0, 1.0)
    news_sentiment = _clamp(float(news_sentiment), -1.0, 1.0)

    if news_direction == 0 or n_conf == 0.0 or abs(news_sentiment) < 0.05:
        return k_conf

    w_pos = 0.30
    w_neg = w_pos * (
        negative_asymmetry_factor
        if (
            news_sentiment <= negative_sentiment_threshold
            and n_conf >= negative_confidence_threshold
        )
        else negative_asymmetry_factor / 2.0
    )
    w_news = w_pos if news_sentiment >= 0 else w_neg

    agree = (
        kronos_direction != 0
        and news_direction != 0
        and kronos_direction == news_direction
    )
    conflict = (
        kronos_direction != 0
        and news_direction != 0
        and kronos_direction != news_direction
    )
    sign_delta = 1.0 if agree else -1.0  # conflict и flat — консервативно вниз

    return _clamp(
        k_conf + sign_delta * w_news * n_conf * abs(news_sentiment),
        0.0,
        1.0,
    )


@dataclass
class MarketState:
    cash: float
    timestamp: datetime = field(default_factory=utcnow)
    prices: Dict[str, float] = field(default_factory=dict)
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    pnl_history: List[tuple[datetime, float]] = field(default_factory=list)
    news: Dict[str, NewsSnapshot] = field(default_factory=dict)
    watch_terms: Dict[str, List[str]] = field(default_factory=dict)

    # Заполняется в Engine при старте и НЕ меняется от pnl_history/reconcile.
    initial_equity: Optional[float] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.cash):
            raise ValueError(f"Invalid cash: {self.cash}")
        if self.initial_equity is None:
            self.initial_equity = self.cash

    @property
    def portfolio_value(self) -> float:
        return self.cash + sum(
            pos.qty * self.prices.get(ticker, pos.avg_price)
            for ticker, pos in self.positions.items()
        )

    @property
    def total_pnl(self) -> float:
        return self.portfolio_value - float(self.initial_equity)

    def get_news(self, ticker: str) -> Optional[NewsSnapshot]:
        return self.news.get(ticker)