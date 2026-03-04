from __future__ import annotations

import pytest
import aiosqlite

from paper_trader.db.schema import create_tables
from paper_trader.db import queries


@pytest.fixture(autouse=True)
def _reset_settings_singleton():
    """Ensure the settings singleton is not polluted between tests.

    The dashboard test fixture creates the real app which triggers startup,
    and startup calls load_config_from_db on the production DB.  This can
    leak values (e.g. enable_tool_use=True) into the singleton and break
    subsequent tests that don't expect them.
    """
    from paper_trader.config import settings

    snapshot = {
        attr: getattr(settings, attr)
        for attr in (
            "enable_tool_use",
            "require_tool_evidence",
            "tool_use_max_turns",
            "budget_warn_usd",
            "confidence_threshold",
        )
    }
    yield
    for attr, val in snapshot.items():
        object.__setattr__(settings, attr, val)


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
