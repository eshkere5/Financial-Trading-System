"""
trade_journal.py — персистентный журнал сделок и решений (SQLite, data/trading.db).
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/trading.db"
MIN_CELL_N = 30  # минимальный размер ячейки (regime × side) для продовых решений

# колонка -> DDL-тип, для миграции trades
_TRADES_COLUMNS = {
    "ts": "REAL", "ticker": "TEXT", "side": "TEXT", "qty": "REAL",
    "price": "REAL", "commission": "REAL", "order_id": "TEXT", "source": "TEXT",
}

_DECISIONS_DDL = """
CREATE TABLE IF NOT EXISTS decisions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  REAL,
    ticker              TEXT,
    pred_dir            INTEGER,
    expected_return     REAL,
    kronos_conf         REAL,
    regime              TEXT,
    regime_dist         REAL,
    regime_vol_pctile   REAL,
    news_effective_risk REAL,
    news_category       TEXT,
    news_override       INTEGER,
    posture             TEXT,
    size_multiplier     REAL,
    direction_multiplier REAL,
    action              TEXT,     -- 'taken' | 'skipped'
    skip_reason         TEXT,
    qty                 REAL,
    entry_price         REAL,
    ret_4h              REAL,
    ret_16h             REAL,
    ret_24h             REAL,
    hit_16h             INTEGER
)
"""


class TradeJournal:
    """Потокобезопасная запись в SQLite. Ошибки записи не роняют торговлю."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self._lock = threading.Lock()
        self._db: Optional[sqlite3.Connection] = None
        try:
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
            self._db = sqlite3.connect(db_path, check_same_thread=False)
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          REAL,
                    ticker      TEXT,
                    side        TEXT,
                    qty         REAL,
                    price       REAL,
                    commission  REAL,
                    order_id    TEXT,
                    source      TEXT
                )
                """
            )
            self._migrate_trades()
            self._db.execute(_DECISIONS_DDL)
            self._db.commit()
        except Exception as exc:
            logger.warning("TradeJournal: SQLite отключён (%s) — сделки только в памяти", exc)
            self._db = None

    def _migrate_trades(self) -> None:
        """Добавляет недостающие колонки в trades, созданную старой схемой."""
        assert self._db is not None
        existing = {row[1] for row in self._db.execute("PRAGMA table_info(trades)")}
        for col, ddl_type in _TRADES_COLUMNS.items():
            if col not in existing:
                self._db.execute(f"ALTER TABLE trades ADD COLUMN {col} {ddl_type}")
                logger.info("TradeJournal: миграция — добавлена колонка trades.%s", col)

    def record(
        self,
        *,
        ticker: str,
        side: str,
        qty: float,
        price: float,
        commission: float = 0.0,
        order_id: str = "",
        source: str = "live",
        ts: Optional[float] = None,
    ) -> None:
        if self._db is None:
            return
        try:
            with self._lock:
                self._db.execute(
                    "INSERT INTO trades (ts, ticker, side, qty, price, commission, order_id, source)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        ts if ts is not None else datetime.now(timezone.utc).timestamp(),
                        ticker, side, float(qty), float(price),
                        float(commission), order_id, source,
                    ),
                )
                self._db.commit()
        except Exception as exc:
            logger.warning("TradeJournal: запись сделки %s не удалась (некритично): %s", ticker, exc)

    # ── журнал решений (с фильтр-состояниями) ────────────────────────

    def record_decision(
        self,
        *,
        ticker: str,
        pred_dir: int,
        action: str,                      # 'taken' | 'skipped'
        skip_reason: str = "",
        qty: float = 0.0,
        entry_price: float = 0.0,
        expected_return: float = 0.0,
        kronos_conf: float = 0.0,
        regime: str = "",
        regime_dist: float = 0.0,
        regime_vol_pctile: float = 0.0,
        news_effective_risk: float = 0.0,
        news_category: str = "",
        news_override: bool = False,
        posture: str = "",
        size_multiplier: float = 1.0,
        direction_multiplier: float = 1.0,
        ts: Optional[float] = None,
    ) -> None:
        if self._db is None:
            return
        try:
            with self._lock:
                self._db.execute(
                    "INSERT INTO decisions (ts, ticker, pred_dir, expected_return, kronos_conf,"
                    " regime, regime_dist, regime_vol_pctile, news_effective_risk, news_category,"
                    " news_override, posture, size_multiplier, direction_multiplier,"
                    " action, skip_reason, qty, entry_price)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        ts if ts is not None else datetime.now(timezone.utc).timestamp(),
                        ticker, int(pred_dir), float(expected_return), float(kronos_conf),
                        regime, float(regime_dist), float(regime_vol_pctile),
                        float(news_effective_risk), news_category, int(news_override),
                        posture, float(size_multiplier), float(direction_multiplier),
                        action, skip_reason, float(qty), float(entry_price),
                    ),
                )
                self._db.commit()
        except Exception as exc:
            logger.warning("TradeJournal: запись решения %s не удалась (некритично): %s", ticker, exc)

    # ── отложенные исходы (заполняет отдельный проход по свечам) ─────

    def pending_outcomes(self, horizon_hours: int) -> List[tuple]:
        """Решения старше horizon_hours, у которых ещё нет исхода за этот горизонт."""
        if self._db is None:
            return []
        col = f"ret_{horizon_hours}h"
        cutoff = datetime.now(timezone.utc).timestamp() - horizon_hours * 3600
        with self._lock:
            return list(self._db.execute(
                f"SELECT id, ticker, entry_price, ts FROM decisions"
                f" WHERE action='taken' AND {col} IS NULL AND ts < ? AND entry_price > 0",
                (cutoff,),
            ))

    def set_outcome(self, decision_id: int, horizon_hours: int, ret: float, hit: Optional[int] = None) -> None:
        if self._db is None:
            return
        col = f"ret_{horizon_hours}h"
        try:
            with self._lock:
                self._db.execute(f"UPDATE decisions SET {col} = ? WHERE id = ?", (float(ret), decision_id))
                if hit is not None and horizon_hours == 16:
                    self._db.execute("UPDATE decisions SET hit_16h = ? WHERE id = ?", (int(hit), decision_id))
                self._db.commit()
        except Exception as exc:
            logger.warning("TradeJournal: исход решения #%d не записан: %s", decision_id, exc)

    def close(self) -> None:
        if self._db is not None:
            try:
                self._db.close()
            except Exception:
                pass
            self._db = None