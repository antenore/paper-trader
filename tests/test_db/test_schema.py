import pytest
import aiosqlite

from paper_trader.db.schema import create_tables, _apply_migrations


@pytest.mark.asyncio
async def test_tables_created(db):
    """All expected tables should exist after schema creation."""
    rows = await db.execute_fetchall(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = {r["name"] for r in rows}
    expected = {
        "settings", "portfolio", "positions", "trades", "decisions",
        "api_calls", "portfolio_snapshots", "strategy_journal",
        "watchlist", "dry_run_sessions",
    }
    assert expected.issubset(tables)


@pytest.mark.asyncio
async def test_schema_idempotent(db):
    """Running create_tables twice should not fail."""
    from paper_trader.db.schema import create_tables
    await create_tables(db)  # Second time
    rows = await db.execute_fetchall(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    assert len(rows) >= 10


@pytest.mark.asyncio
async def test_portfolio_singleton_constraint(db):
    """Portfolio table should only allow id=1."""
    await db.execute("INSERT INTO portfolio (id, cash) VALUES (1, 800)")
    await db.commit()
    with pytest.raises(Exception):
        await db.execute("INSERT INTO portfolio (id, cash) VALUES (2, 800)")
        await db.commit()


@pytest.mark.asyncio
async def test_trade_action_constraint(db):
    """Trade action must be BUY or SELL."""
    with pytest.raises(Exception):
        await db.execute(
            "INSERT INTO trades (symbol, action, shares, price, total) VALUES ('SPY', 'INVALID', 1, 100, 100)"
        )
        await db.commit()


@pytest.mark.asyncio
async def test_position_positive_shares_constraint(db):
    """Position shares must be positive."""
    with pytest.raises(Exception):
        await db.execute(
            "INSERT INTO positions (symbol, shares, avg_cost) VALUES ('SPY', -1, 100)"
        )
        await db.commit()


@pytest.mark.asyncio
async def test_migration_adds_current_cash_to_existing_db():
    """Migration should add current_cash column to an existing dry_run_sessions table."""
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row

    # Create the OLD schema without current_cash
    await db.execute("""
        CREATE TABLE dry_run_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL DEFAULT (datetime('now')),
            ended_at TEXT,
            initial_cash REAL NOT NULL,
            final_value REAL,
            pnl REAL,
            total_trades INTEGER DEFAULT 0,
            api_cost_usd REAL DEFAULT 0,
            summary TEXT
        )
    """)
    # Also need portfolio_snapshots for the benchmark migrations
    await db.execute("""
        CREATE TABLE portfolio_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            cash        REAL    NOT NULL,
            positions_value REAL NOT NULL,
            total_value REAL    NOT NULL,
            is_dry_run  INTEGER NOT NULL DEFAULT 0,
            snapshot_at TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)
    await db.commit()

    # Verify current_cash doesn't exist yet
    cols = await db.execute_fetchall("PRAGMA table_info(dry_run_sessions)")
    col_names = {r["name"] for r in cols}
    assert "current_cash" not in col_names

    # Run migration
    await _apply_migrations(db)

    # Now it should exist
    cols = await db.execute_fetchall("PRAGMA table_info(dry_run_sessions)")
    col_names = {r["name"] for r in cols}
    assert "current_cash" in col_names

    # And we can INSERT with current_cash
    await db.execute(
        "INSERT INTO dry_run_sessions (initial_cash, current_cash) VALUES (800, 800)"
    )
    await db.commit()
    rows = await db.execute_fetchall("SELECT current_cash FROM dry_run_sessions")
    assert rows[0]["current_cash"] == 800

    await db.close()


@pytest.mark.asyncio
async def test_migration_is_idempotent():
    """Running migrations twice should not fail."""
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    await create_tables(db)  # First: creates tables + runs migrations
    await _apply_migrations(db)  # Second: should be no-op
    cols = await db.execute_fetchall("PRAGMA table_info(dry_run_sessions)")
    col_names = [r["name"] for r in cols]
    assert col_names.count("current_cash") == 1
    await db.close()


@pytest.mark.asyncio
async def test_migration_adds_benchmark_columns():
    """Migration should add spy_price and benchmark_value to portfolio_snapshots."""
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row

    # Create OLD schema without benchmark columns
    await db.execute("""
        CREATE TABLE portfolio_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            cash        REAL    NOT NULL,
            positions_value REAL NOT NULL,
            total_value REAL    NOT NULL,
            is_dry_run  INTEGER NOT NULL DEFAULT 0,
            snapshot_at TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)
    # Need dry_run_sessions for the other migration
    await db.execute("""
        CREATE TABLE dry_run_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL DEFAULT (datetime('now')),
            ended_at TEXT,
            initial_cash REAL NOT NULL,
            current_cash REAL,
            final_value REAL,
            pnl REAL,
            total_trades INTEGER DEFAULT 0,
            api_cost_usd REAL DEFAULT 0,
            summary TEXT
        )
    """)
    await db.commit()

    # Verify columns don't exist yet
    cols = await db.execute_fetchall("PRAGMA table_info(portfolio_snapshots)")
    col_names = {r["name"] for r in cols}
    assert "spy_price" not in col_names
    assert "benchmark_value" not in col_names

    # Run migration
    await _apply_migrations(db)

    # Now they should exist
    cols = await db.execute_fetchall("PRAGMA table_info(portfolio_snapshots)")
    col_names = {r["name"] for r in cols}
    assert "spy_price" in col_names
    assert "benchmark_value" in col_names

    # Can insert with new columns
    await db.execute(
        "INSERT INTO portfolio_snapshots (cash, positions_value, total_value, spy_price, benchmark_value) VALUES (700, 100, 800, 450.5, 800.0)"
    )
    await db.commit()
    rows = await db.execute_fetchall("SELECT spy_price, benchmark_value FROM portfolio_snapshots")
    assert rows[0]["spy_price"] == 450.5
    assert rows[0]["benchmark_value"] == 800.0

    await db.close()


@pytest.mark.asyncio
async def test_migration_adds_fx_rate_column():
    """Migration should add usd_chf_rate column to portfolio_snapshots."""
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row

    # Create OLD schema without usd_chf_rate
    await db.execute("""
        CREATE TABLE portfolio_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            cash        REAL    NOT NULL,
            positions_value REAL NOT NULL,
            total_value REAL    NOT NULL,
            is_dry_run  INTEGER NOT NULL DEFAULT 0,
            snapshot_at TEXT    NOT NULL DEFAULT (datetime('now')),
            spy_price   REAL,
            benchmark_value REAL
        )
    """)
    # Need dry_run_sessions for other migrations
    await db.execute("""
        CREATE TABLE dry_run_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL DEFAULT (datetime('now')),
            ended_at TEXT,
            initial_cash REAL NOT NULL,
            current_cash REAL,
            final_value REAL,
            pnl REAL,
            total_trades INTEGER DEFAULT 0,
            api_cost_usd REAL DEFAULT 0,
            summary TEXT
        )
    """)
    await db.commit()

    # Verify column doesn't exist yet
    cols = await db.execute_fetchall("PRAGMA table_info(portfolio_snapshots)")
    col_names = {r["name"] for r in cols}
    assert "usd_chf_rate" not in col_names

    # Run migration
    await _apply_migrations(db)

    # Now it should exist
    cols = await db.execute_fetchall("PRAGMA table_info(portfolio_snapshots)")
    col_names = {r["name"] for r in cols}
    assert "usd_chf_rate" in col_names

    # Can insert with new column
    await db.execute(
        "INSERT INTO portfolio_snapshots (cash, positions_value, total_value, usd_chf_rate) VALUES (700, 100, 800, 0.8812)"
    )
    await db.commit()
    rows = await db.execute_fetchall("SELECT usd_chf_rate FROM portfolio_snapshots")
    assert rows[0]["usd_chf_rate"] == 0.8812

    await db.close()


@pytest.mark.asyncio
async def test_migration_adds_currency_column():
    """Migration should add currency column to positions table."""
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row

    # Create OLD schema without currency column
    await db.execute("""
        CREATE TABLE positions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT    NOT NULL,
            shares      REAL    NOT NULL CHECK (shares > 0),
            avg_cost    REAL    NOT NULL CHECK (avg_cost > 0),
            opened_at   TEXT    NOT NULL DEFAULT (datetime('now')),
            closed_at   TEXT,
            close_price REAL,
            is_dry_run  INTEGER NOT NULL DEFAULT 0,
            stop_loss_price REAL,
            risk_tier   TEXT DEFAULT 'growth',
            UNIQUE(symbol, is_dry_run, closed_at)
        )
    """)
    # Need other tables for migrations
    await db.execute("""
        CREATE TABLE dry_run_sessions (
            id INTEGER PRIMARY KEY, started_at TEXT NOT NULL DEFAULT (datetime('now')),
            ended_at TEXT, initial_cash REAL NOT NULL, current_cash REAL,
            final_value REAL, pnl REAL, total_trades INTEGER DEFAULT 0,
            api_cost_usd REAL DEFAULT 0, summary TEXT
        )
    """)
    await db.execute("""
        CREATE TABLE portfolio_snapshots (
            id INTEGER PRIMARY KEY, cash REAL NOT NULL, positions_value REAL NOT NULL,
            total_value REAL NOT NULL, is_dry_run INTEGER NOT NULL DEFAULT 0,
            snapshot_at TEXT NOT NULL DEFAULT (datetime('now')),
            spy_price REAL, benchmark_value REAL, usd_chf_rate REAL
        )
    """)
    await db.execute("""
        CREATE TABLE watchlist (
            id INTEGER PRIMARY KEY, symbol TEXT NOT NULL UNIQUE,
            reason TEXT, added_at TEXT NOT NULL DEFAULT (datetime('now')),
            removed_at TEXT, risk_tier TEXT DEFAULT 'growth'
        )
    """)
    await db.execute("""
        CREATE TABLE decisions (
            id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, action TEXT NOT NULL,
            confidence REAL NOT NULL, reasoning TEXT NOT NULL, model TEXT NOT NULL,
            is_dry_run INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            alpha_source TEXT DEFAULT ''
        )
    """)
    await db.commit()

    # Verify currency doesn't exist yet
    cols = await db.execute_fetchall("PRAGMA table_info(positions)")
    col_names = {r["name"] for r in cols}
    assert "currency" not in col_names

    # Run migration
    await _apply_migrations(db)

    # Now it should exist with default 'USD'
    cols = await db.execute_fetchall("PRAGMA table_info(positions)")
    col_names = {r["name"] for r in cols}
    assert "currency" in col_names

    # Insert and verify default
    await db.execute(
        "INSERT INTO positions (symbol, shares, avg_cost) VALUES ('AAPL', 1, 100)"
    )
    await db.commit()
    rows = await db.execute_fetchall("SELECT currency FROM positions")
    assert rows[0]["currency"] == "USD"

    await db.close()
