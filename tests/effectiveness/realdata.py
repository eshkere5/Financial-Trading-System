"""Проверки на реальных данных из research/results: дневные свечи SBER, GAZP, LKOH и записи прогнозов Kronos с режимами."""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from finvex.analysis.regime import compute_regime
from finvex.core.models import MSK

from .harness import BLOCKED_REGIMES, Lab, Scenario

RES = Path(__file__).resolve().parents[2] / "research" / "results"
OTHER_RULES_OFF = ["require_rationale", "max_rationale", "allowed_symbols", "blocked_symbols", "no_short", "max_order_value", "max_position_pct",
                   "max_orders_per_day", "trading_hours", "price_deviation", "data_freshness", "min_confidence", "cooldown", "max_revisions"]


def load_daily(name: str) -> pd.DataFrame:
    d = pd.read_csv(RES / f"{name}_daily.csv")
    d["time"] = pd.to_datetime(d["time"])
    return d.reset_index(drop=True)


def regimes_for(d: pd.DataFrame, sym: str, start: int = 260) -> dict:
    out = {}
    for t in range(start, len(d)):
        try:
            out[t] = compute_regime(d.iloc[: t + 1][["high", "low", "close"]], sym).regime
        except Exception:
            out[t] = "TRANSITION"
    return out


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 1) -> tuple:
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def kronos_long_study(kind: str) -> dict:
    """Лонг-сигналы Kronos: средняя реализованная доходность в режимах, которые блокирует шлюз, и в остальных."""
    d = pd.read_csv(RES / f"kronos_{kind}_v2_with_regimes.csv")
    longs = d[d.pred_dir > 0]
    blk = longs[longs.regime.isin(BLOCKED_REGIMES)].actual_ret.to_numpy()
    ok = longs[~longs.regime.isin(BLOCKED_REGIMES)].actual_ret.to_numpy()
    return {"kind": kind, "n_long": len(longs), "n_blocked": len(blk), "n_allowed": len(ok),
            "mean_blocked": float(blk.mean()) if len(blk) else float("nan"), "mean_allowed": float(ok.mean()) if len(ok) else float("nan"),
            "ci_blocked": boot_ci(blk), "ci_allowed": boot_ci(ok)}


def price_regime_study(horizon: int = 5) -> dict:
    """Покупка в каждый день на дневках: доходность через horizon дней, когда шлюз блокирует и когда пропускает."""
    rows = []
    for name in ("sber", "gazp", "lkoh"):
        d = load_daily(name)
        reg = regimes_for(d, name.upper())
        for t, r in reg.items():
            if t + horizon < len(d):
                rows.append((name, r, d.close.iloc[t + horizon] / d.close.iloc[t] - 1))
    df = pd.DataFrame(rows, columns=["sym", "regime", "fwd"])
    blk = df[df.regime.isin(BLOCKED_REGIMES)].fwd.to_numpy()
    ok = df[~df.regime.isin(BLOCKED_REGIMES)].fwd.to_numpy()
    return {"horizon": horizon, "n": len(df), "n_blocked": len(blk), "n_allowed": len(ok),
            "mean_blocked": float(blk.mean()) if len(blk) else float("nan"), "mean_allowed": float(ok.mean()),
            "ci_blocked": boot_ci(blk), "ci_allowed": boot_ci(ok), "by_regime": df.groupby("regime").fwd.agg(["size", "mean"]).round(4).to_dict("index")}


def simulate(d: pd.DataFrame, regimes: dict, sym: str, gateway: bool, only_regime: bool = False, cash0: float = 1_000_000.0) -> dict:
    """Наивный агент «покупаю просадку»: вход на 95% свободных денег при цене ниже SMA20 на 3%, выход выше SMA20 на 1%."""
    lab = Lab("rules", {k: {"enabled": False} for k in OTHER_RULES_OFF} if only_regime else None) if gateway else None
    sma = d.close.rolling(20).mean()
    cash, pos, eq, trades, blocked = cash0, 0.0, [], 0, 0
    for t in range(260, len(d)):
        px = float(d.close.iloc[t])
        eqt = cash + pos * px
        eq.append(eqt)
        when = d.time.iloc[t].to_pydatetime().replace(hour=12, tzinfo=MSK)
        want = None
        if pos == 0 and px < sma.iloc[t] * 0.97:
            want = ("buy", math.floor(0.95 * cash / px))
        elif pos > 0 and px > sma.iloc[t] * 1.01:
            want = ("sell", pos)
        if not want or want[1] < 1:
            continue
        side, qty = want
        if gateway:
            v = lab.decide(Scenario(symbol=sym, side=side, qty=qty, price=px, regime=regimes[t], position=pos, equity=eqt, when=when))
            if v["decision"] == "deny":
                blocked += 1
                continue
            if v["decision"] == "revise":
                qty = v.get("suggested_quantity")
                if not qty:
                    blocked += 1
                    continue
        cash += (qty * px) * (-1 if side == "buy" else 1)
        pos += qty if side == "buy" else -qty
        trades += 1
    eq = np.array(eq + [cash + pos * float(d.close.iloc[-1])])
    peak = np.maximum.accumulate(eq)
    daily = np.diff(eq) / eq[:-1]
    return {"final": float(eq[-1]), "ret": float(eq[-1] / cash0 - 1), "max_dd": float(((eq - peak) / peak).min()), "worst_day": float(daily.min()),
            "trades": trades, "blocked": blocked}
