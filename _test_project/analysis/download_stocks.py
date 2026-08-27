"""
download_stocks_moex.py — дневные свечи MOEX напрямую с ISS API биржи.

### NEW 2026-08-22 ###
Зачем: обход проблем с Tinkoff-токеном ("No tradable instrument").
MOEX ISS — официальное API Московской биржи: бесплатно, без авторизации,
отдаёт настоящие биржевые свечи (те же данные, что идут в Тинькофф).

Что делает: тянет дневки (interval=24) за DL_DAYS дней по DL_TICKERS,
пишет {ticker}_daily.csv (time,open,high,low,close,volume) — формат
уже понимает backtest_kronos.py через BT_CSV.

Запуск (из корня проекта, нужен только интернет и requests из venv):
    python download_stocks_moex.py
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

DL_TICKERS = [s.strip().upper() for s in os.environ.get("DL_TICKERS", "SBER,GAZP,LKOH").split(",") if s.strip()]
DL_DAYS = int(os.environ.get("DL_DAYS", "730"))

ISS_URL = "https://iss.moex.com/iss/engines/stock/markets/shares/securities/{t}/candles.json"


def fetch_candles(ticker: str, date_from: str, date_till: str) -> pd.DataFrame:
    rows, start = [], 0
    while True:
        resp = requests.get(
            ISS_URL.format(t=ticker),
            params={"from": date_from, "till": date_till, "interval": 24, "start": start},
            timeout=20,
        )
        resp.raise_for_status()
        block = resp.json()["candles"]
        data = block["data"]
        if not data:
            break
        cols = block["columns"]
        rows.extend(data)
        start += len(data)
        if len(data) < 100:  # последняя страница
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=cols)
    df = df.rename(columns={"begin": "time"})
    df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")
    return df[["time", "open", "high", "low", "close", "volume"]].drop_duplicates("time")


def main() -> None:
    date_till = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    date_from = (datetime.now(timezone.utc) - timedelta(days=DL_DAYS)).strftime("%Y-%m-%d")
    for ticker in DL_TICKERS:
        try:
            df = fetch_candles(ticker, date_from, date_till)
        except Exception as exc:
            print(f"{ticker}: ОШИБКА сети/API — {exc}")
            continue
        if df.empty:
            print(f"{ticker}: ПУСТО — проверь тикер")
            continue
        df.to_csv(f"{ticker.lower()}_daily.csv", index=False)
        print(f"{ticker}: {len(df)} баров, {df['time'].iloc[0]} .. {df['time'].iloc[-1]}"
              f"  ->  {ticker.lower()}_daily.csv")


if __name__ == "__main__":
    main()
