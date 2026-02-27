from __future__ import annotations

import pytest
import aiosqlite

from paper_trader.db.schema import create_tables
from paper_trader.db import queries


@pytest.fixture
async def db():
    """In-memory SQLite database for tests."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys=ON")
    await create_tables(conn)
    yield conn
    await conn.close()


@pytest.fixture
async def db_with_portfolio(db):
    """Database with initialized portfolio (800 CHF)."""
    await queries.init_portfolio(db, 800.0, "CHF")
    return db
