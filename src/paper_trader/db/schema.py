from __future__ import annotations

import aiosqlite

TABLES = [
    """
    CREATE TABLE IF NOT EXISTS settings (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio (
        id          INTEGER PRIMARY KEY CHECK (id = 1),
        cash        REAL    NOT NULL,
        currency    TEXT    NOT NULL DEFAULT 'CHF',
        created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
        updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS positions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol      TEXT    NOT NULL,
        shares      REAL    NOT NULL CHECK (shares > 0),
        avg_cost    REAL    NOT NULL CHECK (avg_cost > 0),
        opened_at   TEXT    NOT NULL DEFAULT (datetime('now')),
        closed_at   TEXT,
        close_price REAL,
        is_dry_run  INTEGER NOT NULL DEFAULT 0,
        UNIQUE(symbol, is_dry_run, closed_at)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trades (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol      TEXT    NOT NULL,
        action      TEXT    NOT NULL CHECK (action IN ('BUY', 'SELL')),
        shares      REAL    NOT NULL CHECK (shares > 0),
        price       REAL    NOT NULL CHECK (price > 0),
        total       REAL    NOT NULL,
        decision_id INTEGER REFERENCES decisions(id),
        is_dry_run  INTEGER NOT NULL DEFAULT 0,
        executed_at TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS decisions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol      TEXT    NOT NULL,
        action      TEXT    NOT NULL CHECK (action IN ('BUY', 'SELL', 'HOLD')),
        confidence  REAL    NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
        reasoning   TEXT    NOT NULL,
        model       TEXT    NOT NULL,
        is_dry_run  INTEGER NOT NULL DEFAULT 0,
        created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS api_calls (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        model         TEXT    NOT NULL,
        purpose       TEXT    NOT NULL,
        input_tokens  INTEGER NOT NULL,
        output_tokens INTEGER NOT NULL,
        cost_usd      REAL    NOT NULL,
        is_dry_run    INTEGER NOT NULL DEFAULT 0,
        called_at     TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        cash        REAL    NOT NULL,
        positions_value REAL NOT NULL,
        total_value REAL    NOT NULL,
        is_dry_run  INTEGER NOT NULL DEFAULT 0,
        snapshot_at TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS strategy_journal (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_type    TEXT    NOT NULL CHECK (entry_type IN ('observation', 'pattern', 'rule', 'revision')),
        content       TEXT    NOT NULL,
        model         TEXT    NOT NULL,
        supersedes_id INTEGER REFERENCES strategy_journal(id),
        is_active     INTEGER NOT NULL DEFAULT 1,
        is_dry_run    INTEGER NOT NULL DEFAULT 0,
        created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS watchlist (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol     TEXT    NOT NULL UNIQUE,
        reason     TEXT,
        added_at   TEXT    NOT NULL DEFAULT (datetime('now')),
        removed_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dry_run_sessions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at  TEXT    NOT NULL DEFAULT (datetime('now')),
        ended_at    TEXT,
        initial_cash REAL   NOT NULL,
        current_cash REAL,
        final_value  REAL,
        pnl          REAL,
        total_trades INTEGER DEFAULT 0,
        api_cost_usd REAL   DEFAULT 0,
        summary      TEXT
    )
    """,
]


MIGRATIONS = [
    # (table, column, type_default) — idempotent ALTER TABLE ADD COLUMN
    ("dry_run_sessions", "current_cash", "REAL"),
    ("portfolio_snapshots", "spy_price", "REAL"),
    ("portfolio_snapshots", "benchmark_value", "REAL"),
    ("positions", "stop_loss_price", "REAL"),
    ("watchlist", "risk_tier", "TEXT DEFAULT 'growth'"),
    ("positions", "risk_tier", "TEXT DEFAULT 'growth'"),
]


async def create_tables(db: aiosqlite.Connection) -> None:
    """Create all tables if they don't exist, then apply migrations."""
    for ddl in TABLES:
        await db.execute(ddl)
    await db.commit()
    await _apply_migrations(db)


async def _apply_migrations(db: aiosqlite.Connection) -> None:
    """Add missing columns to existing tables (idempotent)."""
    for table, column, col_type in MIGRATIONS:
        existing = await db.execute_fetchall(f"PRAGMA table_info({table})")
        if not existing:
            continue  # Table doesn't exist yet, skip
        col_names = {row["name"] for row in existing}
        if column not in col_names:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    await db.commit()
