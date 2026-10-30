"""
experiment_logger.py — журнал длительного эксперимента (SQLite, data/experiment.db).

### NEW 2026-08-19 ###
Единая точка записи для длительного теста системы: в любой момент можно
заглянуть внутрь (watch_experiment.py) и увидеть бюджет, решения каждого
контура и их вклад.

Таблицы:
- equity_snapshots — equity/cash/pnl на каждом тике (кривая бюджета);
- decision_events  — решение любого контура: kronos | strategist | news | risk;
- attribution      — на тике с сигналом: что сказал КАЖДЫЙ контур и что вышло
                     в итоге. Из неё считается «вклад» контуров пост-фактум.

run_id позволяет писать несколько конфигураций абляции (A_hold, B_kronos,
C_kronos_news, D_full) в одну базу и сравнивать их одним GROUP BY.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/experiment.db"


class ExperimentLogger:
    """Потокобезопасный журнал эксперимента. Ошибки записи не роняют торговлю."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH, run_id: str = "full") -> None:
        self.run_id = run_id
        self._lock = threading.Lock()
        self._db: Optional[sqlite3.Connection] = None
        try:
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
            self._db = sqlite3.connect(db_path, check_same_thread=False)
            self._init_schema()
        except Exception as exc:
            logger.warning("ExperimentLogger: SQLite отключён (%s) — эксперимент без журнала", exc)
            self._db = None

    def _init_schema(self) -> None:
        assert self._db is not None
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                ts REAL, run_id TEXT, equity REAL, cash REAL,
                positions INTEGER, pnl REAL
            );
            CREATE INDEX IF NOT EXISTS idx_equity_run ON equity_snapshots(run_id, ts);

            CREATE TABLE IF NOT EXISTS decision_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, run_id TEXT, actor TEXT, ticker TEXT,
                action TEXT, payload TEXT, rationale TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_events_run ON decision_events(run_id, ts);

            CREATE TABLE IF NOT EXISTS attribution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, run_id TEXT, ticker TEXT,
                kronos_dir INTEGER, kronos_conf REAL,
                news_risk REAL, news_action TEXT,
                strat_dir INTEGER, strat_conv REAL, size_mult REAL,
                final_action TEXT, final_qty REAL
            );
            CREATE INDEX IF NOT EXISTS idx_attr_run ON attribution(run_id, ts);
            """
        )
        self._db.commit()

    @staticmethod
    def _now() -> float:
        return datetime.now(timezone.utc).timestamp()

    def _exec(self, sql: str, params: tuple) -> None:
        if self._db is None:
            return
        try:
            with self._lock:
                self._db.execute(sql, params)
                self._db.commit()
        except Exception as exc:
            logger.warning("ExperimentLogger: запись не удалась (некритично): %s", exc)

    # ── equity (каждый тик) ──────────────────────────────────────────

    def snapshot_equity(self, equity: float, cash: float, n_positions: int, pnl: float) -> None:
        self._exec(
            "INSERT INTO equity_snapshots (ts, run_id, equity, cash, positions, pnl)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (self._now(), self.run_id, equity, cash, n_positions, pnl),
        )

    # ── события решений ──────────────────────────────────────────────

    def log_event(
        self,
        actor: str,                     # "kronos" | "strategist" | "news" | "risk"
        ticker: str,
        action: str,                    # "signal" | "bias" | "dampen" | "block" | "reject" | ...
        payload: Any = None,
        rationale: str = "",
    ) -> None:
        self._exec(
            "INSERT INTO decision_events (ts, run_id, actor, ticker, action, payload, rationale)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                self._now(), self.run_id, actor, ticker, action,
                json.dumps(payload, ensure_ascii=False, default=str) if payload is not None else "",
                rationale[:500],
            ),
        )

    # ── вклад контуров (тик с сигналом) ──────────────────────────────

    def log_attribution(
        self,
        ticker: str,
        *,
        kronos_dir: int = 0,
        kronos_conf: float = 0.0,
        news_risk: float = 0.0,
        news_action: str = "pass",
        strat_dir: int = 0,
        strat_conv: float = 0.0,
        size_mult: float = 1.0,
        final_action: str = "none",     # "open" | "close" | "rejected" | "skipped"
        final_qty: float = 0.0,
    ) -> None:
        self._exec(
            "INSERT INTO attribution (ts, run_id, ticker, kronos_dir, kronos_conf,"
            " news_risk, news_action, strat_dir, strat_conv, size_mult, final_action, final_qty)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self._now(), self.run_id, ticker, kronos_dir, kronos_conf,
                news_risk, news_action, strat_dir, strat_conv, size_mult,
                final_action, final_qty,
            ),
        )

    def close(self) -> None:
        if self._db is not None:
            try:
                self._db.close()
            except Exception:
                pass
            self._db = None
