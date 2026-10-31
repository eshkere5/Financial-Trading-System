"""Проверенный пакет контекста: единственный источник правды для агента."""
from __future__ import annotations

import math
from datetime import datetime
from typing import Callable, Optional

from finvex.core.models import MSK

from .prices import PriceSource

try:
    from finvex.analysis.regime import compute_regime
except Exception:  # pragma: no cover
    compute_regime = None

PROFILE_KEYS = {
    "minimal": ("symbol", "price", "price_time", "regime"),
    "standard": ("symbol", "price", "price_time", "data_age_min", "regime", "change_1d_pct", "volatility_20d_pct",
                 "rsi14", "adx", "drawdown_60d_pct", "position_qty", "limits", "extras"),
    "full": ("symbol", "price", "price_time", "data_age_min", "regime", "change_1d_pct", "volatility_20d_pct",
             "rsi14", "adx", "drawdown_60d_pct", "position_qty", "limits", "extras", "equity", "cash", "positions"),
}


def _rsi(close, n=14) -> Optional[float]:
    d = close.diff().dropna()
    if len(d) < n:
        return None
    up, dn = d.clip(lower=0).tail(n).mean(), (-d.clip(upper=0)).tail(n).mean()
    return 100.0 if dn == 0 else float(100 - 100 / (1 + up / dn))


class PaperPortfolio:
    """Портфель для проверки лимитов. Источник заменяется адаптером брокера (PortfolioProvider)."""

    def __init__(self, cash: float = 1_000_000.0, positions: Optional[dict] = None) -> None:
        self.cash, self.positions = float(cash), dict(positions or {})

    def snapshot(self, prices: Callable[[str], Optional[float]]) -> dict:
        val = {s: q * (prices(s) or 0.0) for s, q in self.positions.items()}
        return {"cash": self.cash, "positions": dict(self.positions), "equity": self.cash + sum(val.values())}


class ContextBuilder:
    def __init__(self, prices: PriceSource, portfolio: PaperPortfolio, clock: Callable[[], datetime]) -> None:
        self.prices, self.portfolio, self.clock = prices, portfolio, clock
        self.extras: "dict[str, Callable[[str], dict]]" = {}

    def register_extra(self, name: str, fn: Callable[[str], dict]) -> None:
        """Точка подключения остальной аналитики: новостной риск, Kronos и так далее."""
        self.extras[name] = fn

    def last_price(self, symbol: str) -> Optional[float]:
        try:
            return float(self.prices.candles(symbol, "1h", 5, self.clock())["close"].iloc[-1])
        except Exception:
            return None

    def build(self, symbol: str) -> dict:
        now = self.clock()
        h = self.prices.candles(symbol, "1h", 400, now)
        d = self.prices.candles(symbol, "1d", 320, now)
        price = float(h["close"].iloc[-1])
        last_open = h.index[-1].to_pydatetime()
        if last_open.tzinfo is None:
            last_open = last_open.replace(tzinfo=MSK)
        age = max((now - last_open).total_seconds() / 60.0 - 60.0, 0.0)
        close = d["close"].astype(float)
        rets = close.pct_change().dropna()
        regime = adx = None
        if compute_regime is not None:
            try:
                ri = compute_regime(d, symbol)
                regime, adx = ri.regime, (None if math.isnan(ri.adx) else round(float(ri.adx), 1))
            except Exception:
                pass
        snap = self.portfolio.snapshot(self.last_price)
        rsi = _rsi(close)
        ctx = {
            "symbol": symbol, "price": round(price, 4), "price_time": last_open.isoformat(timespec="minutes"),
            "data_age_min": round(age, 1), "regime": regime, "adx": adx,
            "change_1d_pct": round(float(close.iloc[-1] / close.iloc[-2] - 1) * 100, 2) if len(close) > 1 else None,
            "volatility_20d_pct": round(float(rets.tail(20).std()) * 100, 2) if len(rets) > 20 else None,
            "rsi14": None if rsi is None else round(rsi, 1),
            "drawdown_60d_pct": round(float(close.iloc[-1] / close.tail(60).max() - 1) * 100, 2),
            "position_qty": float(snap["positions"].get(symbol, 0.0)), "equity": round(snap["equity"], 2),
            "cash": round(snap["cash"], 2), "positions": snap["positions"], "extras": {},
        }
        for name, fn in self.extras.items():
            try:
                ctx["extras"][name] = fn(symbol)
            except Exception as exc:
                ctx["extras"][name] = {"error": str(exc)[:120]}
        return ctx

    @staticmethod
    def for_agent(ctx: dict, profile: str, limits: Optional[dict] = None) -> dict:
        keys = PROFILE_KEYS.get(profile, PROFILE_KEYS["standard"])
        out = {k: ctx[k] for k in keys if k in ctx}
        if "limits" in keys:
            out["limits"] = limits or {}
        return out
