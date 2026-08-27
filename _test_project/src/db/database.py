from __future__ import annotations
import json
import logging
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    ts TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (ticker, ts)
);
CREATE INDEX IF NOT EXISTS idx_prices_ticker_ts ON prices(ticker, ts);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    qty REAL NOT NULL,
    price REAL NOT NULL,
    side TEXT NOT NULL,
    ts TEXT NOT NULL,
    commission REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trades_run ON trades(run_id);

CREATE TABLE IF NOT EXISTS equity_curve (
    run_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    equity REAL NOT NULL,
    cash REAL,
    n_positions INTEGER,
    PRIMARY KEY (run_id, ts)
);
CREATE INDEX IF NOT EXISTS idx_equity_run ON equity_curve(run_id);

CREATE TABLE IF NOT EXISTS strategist_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    model TEXT,
    regime TEXT,
    risk_posture TEXT,
    size_multiplier REAL,
    news_alpha_override REAL,
    vetoed_pairs TEXT,
    strategy_mode TEXT,
    watchlist TEXT,
    tactical_bias TEXT,
    rationale TEXT,
    latency_sec REAL,
    tokens_in INTEGER,
    tokens_out INTEGER,
    is_fallback INTEGER
);
CREATE INDEX IF NOT EXISTS idx_decisions_run ON strategist_decisions(run_id);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    initial_capital REAL,
    final_equity REAL,
    total_trades INTEGER,
    llm_calls_made INTEGER,
    tokens_in_total INTEGER,
    tokens_out_total INTEGER,
    notes TEXT
);
"""


class TradingDB:
    def __init__(self, db_path: str = "data/trading.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        logger.info("TradingDB initialized | path=%s", self.db_path)

    def __enter__(self) -> "TradingDB":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    # ── runs ──────────────────────────────────────────────────────────

    def start_run(self, run_id: str, mode: str, initial_capital: float, notes: str = "") -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, mode, started_at, initial_capital, notes) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, mode, datetime.utcnow().isoformat(), initial_capital, notes),
            )
            self.conn.commit()

    def finish_run(
        self, run_id: str, final_equity: float, total_trades: int,
        llm_calls_made: int = 0, tokens_in_total: int = 0, tokens_out_total: int = 0,
    ) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE runs SET finished_at=?, final_equity=?, total_trades=?, "
                "llm_calls_made=?, tokens_in_total=?, tokens_out_total=? WHERE run_id=?",
                (
                    datetime.utcnow().isoformat(), final_equity, total_trades,
                    llm_calls_made, tokens_in_total, tokens_out_total, run_id,
                ),
            )
            self.conn.commit()

    # ── prices ────────────────────────────────────────────────────────

    def upsert_prices(self, ticker: str, df: pd.DataFrame) -> None:
        """df: индекс datetime, колонки open/high/low/close/volume (частично допустимо)."""
        rows = []
        for ts, row in df.iterrows():
            rows.append((
                ticker, pd.Timestamp(ts).isoformat(),
                float(row.get("open", row.get("close", 0.0))),
                float(row.get("high", row.get("close", 0.0))),
                float(row.get("low", row.get("close", 0.0))),
                float(row.get("close", 0.0)),
                float(row.get("volume", 0.0)) if "volume" in row else None,
            ))
        with self._lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO prices (ticker, ts, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self.conn.commit()
        logger.info("TradingDB: upserted %d bars for %s", len(rows), ticker)

    def load_prices(self, ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        query = "SELECT ts, open, high, low, close, volume FROM prices WHERE ticker=?"
        params: List[Any] = [ticker]
        if start:
            query += " AND ts >= ?"
            params.append(start)
        if end:
            query += " AND ts <= ?"
            params.append(end)
        query += " ORDER BY ts"
        df = pd.read_sql_query(query, self.conn, params=params, parse_dates=["ts"])
        return df.set_index("ts")

    def load_prices_wide(self, tickers: List[str], start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """Возвращает wide DataFrame close-цен: индекс ts, колонки = тикеры (замена _load_data в BacktestEngine)."""
        frames = {}
        for t in tickers:
            df = self.load_prices(t, start, end)
            if not df.empty:
                frames[t] = df["close"]
        return pd.DataFrame(frames).sort_index()

    # ── trades / equity ──────────────────────────────────────────────

    def log_trade(self, run_id: str, ticker: str, qty: float, price: float, side: str, ts, commission: float) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO trades (run_id, ticker, qty, price, side, ts, commission) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, ticker, qty, price, side, pd.Timestamp(ts).isoformat(), commission),
            )
            self.conn.commit()

    def log_equity_point(self, run_id: str, ts, equity: float, cash: float, n_positions: int) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO equity_curve (run_id, ts, equity, cash, n_positions) VALUES (?, ?, ?, ?, ?)",
                (run_id, pd.Timestamp(ts).isoformat(), equity, cash, n_positions),
            )
            self.conn.commit()

    def load_equity_curve(self, run_id: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT ts, equity, cash, n_positions FROM equity_curve WHERE run_id=? ORDER BY ts",
            self.conn, params=[run_id], parse_dates=["ts"],
        )

    def load_trades(self, run_id: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM trades WHERE run_id=? ORDER BY ts", self.conn, params=[run_id],
        )

    # ── strategist decisions ─────────────────────────────────────────

    def log_decision(self, run_id: str, decision) -> None:
        """decision: StrategistDecision dataclass instance."""
        d = asdict(decision)
        with self._lock:
            self.conn.execute(
                """INSERT INTO strategist_decisions
                (run_id, timestamp, model, regime, risk_posture, size_multiplier,
                 news_alpha_override, vetoed_pairs, strategy_mode, watchlist,
                 tactical_bias, rationale, latency_sec, tokens_in, tokens_out, is_fallback)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, d["timestamp"], d["model"], d["regime"], d["risk_posture"],
                    d["size_multiplier"], d["news_alpha_override"],
                    json.dumps(d["vetoed_pairs"], ensure_ascii=False),
                    d["strategy_mode"],
                    json.dumps(d["watchlist"], ensure_ascii=False),
                    json.dumps(d["tactical_bias"], ensure_ascii=False),
                    d["rationale"], d["latency_sec"], d["tokens_in"], d["tokens_out"],
                    int(d["is_fallback"]),
                ),
            )
            self.conn.commit()

    def load_decisions(self, run_id: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM strategist_decisions WHERE run_id=? ORDER BY timestamp",
            self.conn, params=[run_id],
        )