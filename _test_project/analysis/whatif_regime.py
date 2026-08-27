"""
whatif_regime.py — тест «Kronos + режимные множители» на ГОТОВЫХ данных.

"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
ROOT = _HERE if (_HERE / "src").is_dir() else _HERE.parent
sys.path.insert(0, str(ROOT))

from src.regime.regime_detector import signal_multiplier  # noqa: E402

FILES = [s.strip() for s in os.environ.get(
    "WI_FILES",
    "analysis/results/kronos_stocks_v2_with_regimes.csv,"
    "analysis/results/kronos_crypto_v2_with_regimes.csv",
).split(",") if s.strip()]

SKIP_BELOW = 0.5


def run(df: pd.DataFrame, label: str) -> None:
    df = df[df.hit.notna()].copy()
    if df.empty:
        print(f"\n{label}: нет точек"); return
    df["ret_signed"] = df["pred_dir"] * df["actual_ret"]
    df["mult"] = df.apply(
        lambda r: signal_multiplier(r["regime"], int(r["pred_dir"]), ticker=str(r["symbol"])),
        axis=1)

    raw = df["ret_signed"].sum() * 100
    weighted = (df["ret_signed"] * df["mult"]).sum() * 100
    taken = df[df["mult"] >= SKIP_BELOW]
    skip = taken["ret_signed"].sum() * 100
    hr = lambda x: f"{x.hit.mean()*100:.1f}% (n={len(x)})" if len(x) else "—"

    print(f"\n{'='*58}\n{label}\n{'='*58}")
    print(f"  A) сырой сигнал            : PnL {raw:+7.2f}% | hit {hr(df)}")
    print(f"  B) множители на размер     : PnL {weighted:+7.2f}% | эффект {weighted-raw:+.2f} п.п.")
    print(f"  C) пропуск если mult<{SKIP_BELOW}   : PnL {skip:+7.2f}% | hit {hr(taken)} | взято {len(taken)}/{len(df)}")
    by = df.groupby("regime").apply(
        lambda g: (g["ret_signed"] * g["mult"]).sum() * 100, include_groups=False)
    raw_by = df.groupby("regime")["ret_signed"].sum() * 100
    for reg in by.index:
        print(f"    {reg:<14} {raw_by[reg]:+7.2f}% -> {by[reg]:+7.2f}%  (n={len(df[df.regime==reg])})")


def main() -> None:
    for f in FILES:
        path = f if os.path.exists(f) else str(ROOT / f)
        if not os.path.exists(path):
            print(f"НЕТ ФАЙЛА: {f} — сначала прогони regime_report.py"); continue
        df = pd.read_csv(path)
        run(df, f)
        for sym in df.symbol.unique():
            sub = df[df.symbol == sym].sort_values("i").reset_index(drop=True)
            half = len(sub) // 2
            if half < 10:
                continue
            run(sub.iloc[:half], f"{f} :: {sym} ПЕРВАЯ половина")
            run(sub.iloc[half:], f"{f} :: {sym} ВТОРАЯ половина")


if __name__ == "__main__":
    main()