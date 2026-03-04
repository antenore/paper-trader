from __future__ import annotations

import pytest

from paper_trader.db import queries as q


class TestSaveNewsItem:
    @pytest.mark.asyncio
    async def test_save_new_item_returns_true(self, db):
        inserted = await q.save_news_item(
            db, source="test", symbol=None,
            title="Oil hits $100", summary="Crude surge",
            link="https://example.com/oil-100", published_at="Mon, 02 Mar 2026 08:00:00 GMT",
        )
        assert inserted is True

    @pytest.mark.asyncio
    async def test_duplicate_link_returns_false(self, db):
        link = "https://example.com/same-article"
        await q.save_news_item(db, "src", None, "Title", None, link, None)
        second = await q.save_news_item(db, "src", None, "Title", None, link, None)
        assert second is False

    @pytest.mark.asyncio
    async def test_saves_symbol_field(self, db):
        await q.save_news_item(
            db, source="yahoo_AAPL", symbol="AAPL",
            title="AAPL earnings beat", summary=None,
            link="https://example.com/aapl", published_at=None,
        )
        rows = await q.get_recent_news(db, hours=1, symbols=["AAPL"])
        assert len(rows) == 1
        assert rows[0]["symbol"] == "AAPL"


class TestGetRecentNews:
    @pytest.mark.asyncio
    async def test_empty_table_returns_empty(self, db):
        rows = await q.get_recent_news(db, hours=6)
        assert rows == []

    @pytest.mark.asyncio
    async def test_returns_recent_items(self, db):
        await q.save_news_item(db, "bbc", None, "War in Middle East", "Details", "https://a.com/1", None)
        await q.save_news_item(db, "eia", None, "Oil output cut", "OPEC", "https://a.com/2", None)
        rows = await q.get_recent_news(db, hours=1)
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_symbol_filter_includes_general_and_matching(self, db):
        # General news (symbol=None)
        await q.save_news_item(db, "bbc", None, "General macro news", None, "https://a.com/g", None)
        # AAPL-specific
        await q.save_news_item(db, "yahoo_AAPL", "AAPL", "AAPL up 3%", None, "https://a.com/aapl", None)
        # TSLA-specific (should NOT appear when filtering for AAPL)
        await q.save_news_item(db, "yahoo_TSLA", "TSLA", "TSLA recall", None, "https://a.com/tsla", None)

        rows = await q.get_recent_news(db, hours=1, symbols=["AAPL"])
        titles = {r["title"] for r in rows}
        assert "General macro news" in titles
        assert "AAPL up 3%" in titles
        assert "TSLA recall" not in titles

    @pytest.mark.asyncio
    async def test_no_symbol_filter_returns_all(self, db):
        await q.save_news_item(db, "bbc", None, "General", None, "https://a.com/1", None)
        await q.save_news_item(db, "yahoo_AAPL", "AAPL", "AAPL", None, "https://a.com/2", None)
        rows = await q.get_recent_news(db, hours=1)
        assert len(rows) == 2


class TestCleanupOldNews:
    @pytest.mark.asyncio
    async def test_cleanup_returns_zero_when_nothing_old(self, db):
        await q.save_news_item(db, "bbc", None, "Fresh news", None, "https://a.com/fresh", None)
        deleted = await q.cleanup_old_news(db, days=7)
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_cleanup_removes_old_items(self, db):
        # Insert an item, then manually backdating fetched_at to simulate old data
        await q.save_news_item(db, "old_source", None, "Old news", None, "https://a.com/old", None)
        await db.execute(
            "UPDATE news_items SET fetched_at = datetime('now', '-10 days') WHERE link = ?",
            ("https://a.com/old",),
        )
        await db.commit()

        deleted = await q.cleanup_old_news(db, days=7)
        assert deleted == 1

        rows = await q.get_recent_news(db, hours=24 * 365)
        assert len(rows) == 0
