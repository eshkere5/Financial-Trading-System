"""
news_db_migration.py — ALTER TABLE для добавления колонок tickers и
sanction_risk в существующую БД (при переходе на DeepSeekNewsAnalyzer).

Безопасно: ADD COLUMN не трогает существующие данные, старые строки
получат NULL в новых колонках (обрабатывается как fallback в client.py
торгового блока).

Использование:
    from src.services.news_db_migration import migrate_add_deepseek_columns
    migrate_add_deepseek_columns(news_db.conn)

### FIXED 2026-08-01 (К-7) ###
Миграция падала на пустой/новой БД: PRAGMA table_info для несуществующей
таблицы возвращает пустой список БЕЗ исключения, поэтому все колонки
считались отсутствующими и первый же ALTER TABLE давал
sqlite3.OperationalError: no such table: news.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

_NEW_COLUMNS = {
    "tickers": "TEXT",          # CSV MOEX-тикеров от DeepSeek, напр. "SBER,GAZP"
    "sanction_risk": "REAL",    # 0.0-1.0 от DeepSeek
    "relevance_score": "REAL",  # 0.0-1.0 от DeepSeek
}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """### FIXED 2026-08-01 (К-7) ### Явная проверка наличия таблицы."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def migrate_add_deepseek_columns(conn: sqlite3.Connection) -> None:
    """Добавляет недостающие колонки в таблицу news. Идемпотентно."""
    # ### FIXED 2026-08-01 (К-7) ###
    # Было: сразу PRAGMA table_info(news) + ALTER TABLE. На БД, где таблица
    # news ещё не создана, PRAGMA молча отдаёт пустой результат, а ALTER
    # роняет весь старт приложения. Теперь миграция на отсутствующей таблице
    # — это no-op с предупреждением: таблицу создаёт NewsDB при инициализации,
    # уже с новыми колонками, повторный прогон миграции ей не нужен.
    if not _table_exists(conn, "news"):
        logger.warning(
            "news_db_migration: таблица 'news' отсутствует — миграция пропущена "
            "(таблица будет создана схемой NewsDB уже с нужными колонками)"
        )
        return

    existing = {row[1] for row in conn.execute("PRAGMA table_info(news)").fetchall()}

    added = 0
    for col, col_type in _NEW_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE news ADD COLUMN {col} {col_type}")
            logger.info("news_db_migration: добавлена колонка %s %s", col, col_type)
            added += 1

    if added:
        conn.commit()
