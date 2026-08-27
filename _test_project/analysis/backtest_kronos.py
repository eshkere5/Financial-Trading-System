"""
backtest_kronos.py — walk-forward валидация Kronos v2: режимы, тренд-фильтр, акции.

### NEW 2026-08-22 ###
Что нового против v1:
- SMA50 на каждой точке: шорт помечается «в аптренде» / «в даунтренде» —
  прямо измеряем, сколько стоит тренд-фильтр, ДО правок боевой стратегии;
- источники: Bybit (крипта, 1h) + CSV (акции/ETF, дневки) — проверка теории
  «плохие шорты — свойство крипты или модели» на другом классе активов;
- сводка: hit-rate лонг/шорт, шорты по тренду, PnL «сырой» vs «с фильтром».

Запуск (из корня проекта):
    # акции (CSV от download_stocks_moex.py, колонки time/open/high/low/close):
    $env:BT_SYMBOLS = ""
    $env:BT_CSV     = "sber_daily.csv,gazp_daily.csv,lkoh_daily.csv"
    $env:BT_OUT     = "kronos_stocks_v2.csv"
    python backtest_kronos.py

    # крипта, максимум истории за один запрос (1h, limit=1000 ≈ 41 день):
    $env:BT_SYMBOLS = "BTCUSDT,ETHUSDT"
    $env:BT_DAYS    = "38"
    $env:BT_STRIDE  = "6"
    $env:BT_CSV     = ""
    python backtest_kronos.py

ВНИМАНИЕ: ~4.5 мин/точка на CPU (Kronos-base). ~45 точек/тикер ≈ 3.5 часа.
Для быстрого прогона можно временно сменить модель в configs/kronos.yaml
на NeoQuasar/Kronos-small (в ~4 раза быстрее) и подтвердить выводы на base.
"""
from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent 
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("backtest_kronos.log", encoding="utf-8")])
log = logging.getLogger("backtest_kronos")

SYMBOLS = [s.strip().upper() for s in os.environ.get("BT_SYMBOLS", "BTCUSDT,ETHUSDT").split(",") if s.strip()]
CSV_PATHS = [s.strip() for s in os.environ.get("BT_CSV", "").split(",") if s.strip()]
DAYS = int(os.environ.get("BT_DAYS", "38"))
STRIDE = int(os.environ.get("BT_STRIDE", "6"))
CONTEXT = int(os.environ.get("BT_CONTEXT", "240"))
FWD = int(os.environ.get("BT_FWD", "4"))
SMA_WIN = int(os.environ.get("BT_SMA", "50"))
OUT_CSV = os.environ.get("BT_OUT", "kronos_backtest_v2.csv")


def load_bybit(client, sym: str) -> pd.DataFrame:
    need = CONTEXT + DAYS * 24 + FWD + 10
    df = client.get_candles(sym, interval="1h", limit=min(need, 1000))
    log.info("%s: загружено %d свечей (bybit 1h)", sym, len(df))
    return df


def load_csv(path: str) -> tuple[str, pd.DataFrame]:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    assert "close" in df.columns, f"{path}: нет колонки close"
    name = Path(path).stem.upper()
    log.info("%s: загружено %d баров из %s", name, len(df), path)
    return name, df


def summarize(name: str, rows: list[dict]) -> None:
    sr = [r for r in rows if r["symbol"] == name and r["hit"] is not None]
    if not sr:
        return
    def hr(rs): return f"{sum(r['hit'] for r in rs)}/{len(rs)} = {sum(r['hit'] for r in rs)/len(rs)*100:.1f}%" if rs else "—"
    def ret(rs): return sum(r["pred_dir"] * r["actual_ret"] for r in rs)

    longs = [r for r in sr if r["pred_dir"] > 0]
    shorts = [r for r in sr if r["pred_dir"] < 0]
    sh_up = [r for r in shorts if r["uptrend"] is True]
    sh_down = [r for r in shorts if r["uptrend"] is False]
    filtered = longs + sh_down   # фильтр: шорт только в даунтренде

    print(f"\n{'='*64}\n{name}  (точек: {len(sr)})\n{'='*64}")
    print(f"  hit-rate общий            : {hr(sr)}")
    print(f"  лонги                     : {hr(longs)}")
    print(f"  шорты ВСЕ                 : {hr(shorts)}")
    print(f"    шорты В АПТРЕНДЕ        : {hr(sh_up)}  → фильтр их ОТСЕЧЁТ")
    print(f"    шорты В ДАУНТРЕНДЕ      : {hr(sh_down)}  → фильтр их оставит")
    print(f"  PnL 'следуй слепо'        : {ret(sr):+.2f}%")
    print(f"  PnL 'с тренд-фильтром'    : {ret(filtered):+.2f}%")
    print(f"  эффект фильтра            : {ret(filtered)-ret(sr):+.2f} п.п.")


def main() -> None:
    from src.engine.config_loader import load_bybit_config, load_kronos_config
    from src.kronos_layer.kronos_adapter import KronosAdapter
    from src.kronos_layer.kronos_features import extract_features

    kronos = KronosAdapter(load_kronos_config("configs/kronos.yaml"))
    assert not kronos.is_mock, "Kronos в MOCK-режиме — бэктест бессмысленен"

    sources: list[tuple[str, pd.DataFrame]] = []
    if SYMBOLS:
        from src.executors.bybit_client import BybitClient
        client = BybitClient(load_bybit_config(testnet=True))
        sources += [(s, load_bybit(client, s)) for s in SYMBOLS]
    for p in CSV_PATHS:
        sources.append(load_csv(p))
    if not sources:
        sys.exit("Нет источников: задай BT_SYMBOLS и/или BT_CSV")

    rows: list[dict] = []
    for name, df in sources:
        closes = df["close"].astype(float).reset_index(drop=True)
        sma = closes.rolling(SMA_WIN).mean()
        if len(df) < CONTEXT + FWD + 5:
            log.warning("%s: мало истории (%d), пропуск", name, len(df))
            continue
        n = 0
        for i in range(CONTEXT, len(df) - FWD, STRIDE):
            window = df.iloc[i - CONTEXT:i]
            last_price = float(closes.iloc[i - 1])
            future_price = float(closes.iloc[i + FWD - 1])
            actual_ret = (future_price - last_price) / last_price
            ma = sma.iloc[i - 1]
            uptrend = None if pd.isna(ma) else bool(last_price > ma)
            try:
                kstate = kronos.encode_prices(window, ticker=name)
                kf = extract_features(kstate, last_price=last_price)
            except Exception as exc:
                log.warning("%s i=%d: инференс упал (%s)", name, i, exc)
                continue
            pred_dir = int(kf.trend_direction)
            hit = None if pred_dir == 0 else int((pred_dir > 0) == (actual_ret > 0))
            rows.append({"symbol": name, "i": i, "pred_dir": pred_dir,
                         "confidence": float(kf.confidence),
                         "expected_return": float(kf.expected_return),
                         "actual_ret": actual_ret, "hit": hit, "uptrend": uptrend})
            n += 1
            log.info("%s точка %d | dir=%+d conf=%.2f | факт %+.3f%% | hit=%s | %s",
                     name, n, pred_dir, kf.confidence, actual_ret * 100, hit,
                     "UP" if uptrend else "DOWN" if uptrend is False else "?")

    if not rows:
        sys.exit("Ни одной точки — проверь данные/инференс")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nKRONOS WALK-FORWARD v2 — тренд-фильтр SMA50, режимы, классы активов")
    for name, _ in sources:
        summarize(name, rows)
    print(f"\nДетали: {OUT_CSV}")


if __name__ == "__main__":
    main()