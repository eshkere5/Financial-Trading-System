"""
TradingSignal — обогащённый тип торгового сигнала.

Несёт в себе:
  - базовые поля (direction, strength, confidence)
  - kronos_state: rich KronosState с прогнозом и волатильностью
  - news_sentiment: агрегированный сентимент из новостного блока
  - spread_zscore: z-score спреда для stat-arb пар
  - source: откуда пришёл сигнал (kronos | spread | news | combined)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.kronos_layer.kronos_adapter import KronosState
    from src.engine.market_state import NewsSnapshot


@dataclass
class TradingSignal:
    ticker: str
    direction: int         # -1 short, 0 neutral, +1 long
    strength: float        # абсолютная сила сигнала [0, 1]
    confidence: float      # уверенность модели [0, 1]
    expected_return: float # прогнозируемая доходность
    source: str            # "kronos" | "spread" | "news" | "combined"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    reason: str = ""

    # ── расширенный контекст ──────────────────────────────────────────────
    kronos_state: Optional["KronosState"] = field(default=None, repr=False)
    news_snapshot: Optional["NewsSnapshot"] = field(default=None, repr=False)
    spread_zscore: Optional[float] = None   # для stat-arb пар
    hedge_ratio: Optional[float] = None     # β для пары (a - β*b)
    pair_ticker: Optional[str] = None       # второй тикер в паре

    # ── удобные свойства ─────────────────────────────────────────────────

    @property
    def is_actionable(self) -> bool:
        """Сигнал достаточно сильный и уверенный для торговли."""
        return abs(self.direction) > 0 and self.strength > 0.1 and self.confidence > 0.0

    @property
    def news_sentiment(self) -> float:
        """Сентимент из прикреплённого NewsSnapshot или 0."""
        return self.news_snapshot.sentiment if self.news_snapshot else 0.0

    @property
    def has_news_conflict(self) -> bool:
        """
        True если Kronos/spread говорит лонг, а новости негативные — или наоборот.
        Полезно для фильтрации рискованных сигналов.
        """
        if self.news_snapshot is None or abs(self.news_snapshot.sentiment) < 0.2:
            return False
        news_dir = 1 if self.news_snapshot.sentiment > 0 else -1
        return news_dir != self.direction

    def __repr__(self) -> str:
        return (
            f"Signal({self.ticker} dir={self.direction:+d} "
            f"str={self.strength:.2f} conf={self.confidence:.2f} "
            f"er={self.expected_return:+.3f} src={self.source})"
        )
