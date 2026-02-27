from __future__ import annotations

import aiosqlite
from pathlib import Path

_db: aiosqlite.Connection | None = None


async def get_db(db_path: str | None = None) -> aiosqlite.Connection:
    """Get or create the singleton database connection."""
    global _db
    if _db is None:
        if db_path is None:
            from paper_trader.config import settings
            db_path = settings.db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        _db = await aiosqlite.connect(db_path)
        _db.row_factory = aiosqlite.Row
        await _db.execute("PRAGMA journal_mode=WAL")
        await _db.execute("PRAGMA foreign_keys=ON")
    return _db


async def close_db() -> None:
    """Close the database connection."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None


async def init_db(db_path: str | None = None) -> aiosqlite.Connection:
    """Initialize DB connection and create schema."""
    from paper_trader.db.schema import create_tables
    db = await get_db(db_path)
    await create_tables(db)
    return db
