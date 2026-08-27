"""
regime_report.py — hit-rate Kronos по режимам рынка на готовом бэктесте.

### NEW 2026-08-23 ###
Берёт kronos_stocks_v2.csv (или любой BT_OUT) + исходные ценовые CSV,
вычисляет режим на каждой точке бэктеста и строит таблицу:
hit-rate и PnL по (режим × направление). Из неё видно, какие множители
в REGIME_MULTIPLIERS (regime_detector.py) надо поднять/опустить.

Новых инференсов Kronos НЕ нужно — работает на уже собранных данных.

Запуск (из корня проекта):
    $env:BT_OUT = "kronos_stocks_v2.csv"
    $env:BT_CSV = "sber_daily.csv,gazp_daily.csv,lkoh_daily.csv"
    python regime_report.py

Правило чтения таблицы:
- hit-rate > 55% и n >= 8  -> множитель поднять (1.0-1.2)
- hit-rate 45-55%          -> оставить (0.7-0.9)
- hit-rate < 45% и n >= 8  -> опустить (0.2-0.4)
- n < 8                    -> данных мало, не трогаем (иначе переобучимся)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# самонахождение корня: если рядом нет src\ — значит, мы в подпапке (analysis\)
_HERE = Path(__file__).resolve().parent
ROOT = _HERE if (_HERE / "src").is_dir() else _HERE.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.regime.regime_detector import compute_regime


BT_OUT = os.environ.get("BT_OUT", "kronos_crypto_v2.csv")
BT_CSV = [s.strip() for s in os.environ.get("BT_CSV", "").split(",") if s.strip()]
CRYPTO_SUFFIXES = ("USDT", "USDC")


def load_prices(symbol: str) -> pd.Series:
    if symbol.endswith(CRYPTO_SUFFIXES):
        from src.engine.config_loader import load_bybit_config
        from src.executors.bybit_client import BybitClient
        client = BybitClient(load_bybit_config(testnet=True))
        df = client.get_candles(symbol, interval="1h", limit=1000)
        print(f"{symbol}: подтянуто {len(df)} свечей с Bybit")
        return df
    matches = [p for p in BT_CSV if Path(p).stem.upper() == symbol]
    if not matches:
        raise KeyError(f"{symbol}: нет ценового CSV в BT_CSV")
    df = pd.read_csv(matches[0])
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def main() -> None:
    bt = pd.read_csv(BT_OUT)
    price_map = {s: load_prices(s) for s in bt.symbol.unique()}

    regimes = []
    for _, row in bt.iterrows():
        closes = price_map[row["symbol"]]
        info = compute_regime(price_map[row["symbol"]].iloc[: int(row["i"])], ticker=row["symbol"])
        regimes.append(info.regime)
    bt["regime"] = regimes
    bt["ret_signed"] = bt["pred_dir"] * bt["actual_ret"]

    print(f"\nРаспределение точек по режимам:\n{bt['regime'].value_counts().to_string()}")

    tab = (bt[bt.hit.notna()]
           .groupby(["regime", "pred_dir"])
           .agg(n=("hit", "size"), hit_rate=("hit", "mean"), pnl=("ret_signed", "sum"))
           .reset_index())
    tab["side"] = tab["pred_dir"].map({1: "LONG", -1: "SHORT", 0: "FLAT"})
    tab["hit_rate"] = (tab["hit_rate"] * 100).round(1)
    tab["pnl"] = (tab["pnl"] * 100).round(2)
    print("\nHIT-RATE ПО РЕЖИМАМ (все символы вместе):\n")
    print(tab[["regime", "side", "n", "hit_rate", "pnl"]].to_string(index=False))

    print("\nТо же по каждому символу:")
    for sym in bt.symbol.unique():
        sub = bt[(bt.symbol == sym) & bt.hit.notna()]
        t = (sub.groupby(["regime", "pred_dir"])
             .agg(n=("hit", "size"), hit_rate=("hit", "mean"))
             .reset_index())
        t["side"] = t["pred_dir"].map({1: "LONG", -1: "SHORT"})
        t["hit_rate"] = (t["hit_rate"] * 100).round(1)
        print(f"\n  {sym}:")
        print(t[["regime", "side", "n", "hit_rate"]].to_string(index=False))

    out = BT_OUT.replace(".csv", "_with_regimes.csv")
    bt.to_csv(out, index=False)
    print(f"\nСохранено с колонкой regime: {out}")


if __name__ == "__main__":
    main()