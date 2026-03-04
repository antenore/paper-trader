from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from paper_trader.market.news import (
    fetch_news,
    fetch_symbol_news,
    fetch_and_save_news,
    format_news_for_ai,
    NewsItem,
)


def make_mock_feed(entries):
    """Create a mock feedparser result."""
    feed = MagicMock()
    feed.entries = [MagicMock(**e) for e in entries]
    for entry, data in zip(feed.entries, entries):
        entry.get = lambda key, default="", _data=data: _data.get(key, default)
    return feed


class TestFetchNews:
    @pytest.mark.asyncio
    async def test_fetch_with_mock_feeds(self):
        mock_feed = make_mock_feed([
            {"title": "Market Rally", "summary": "Stocks up", "link": "http://example.com/1", "published": "2026-03-02"},
            {"title": "Tech Earnings", "summary": "Mixed results", "link": "http://example.com/2", "published": "2026-03-02"},
        ])

        with patch("paper_trader.market.news.feedparser.parse", return_value=mock_feed):
            items = await fetch_news(feeds={"test": "http://fake.com/rss"})
            assert len(items) == 2
            assert items[0].title == "Market Rally"
            assert items[0].source == "test"

    @pytest.mark.asyncio
    async def test_fetch_handles_errors(self):
        with patch("paper_trader.market.news.feedparser.parse", side_effect=Exception("Network error")):
            items = await fetch_news(feeds={"broken": "http://fake.com/rss"})
            assert items == []

    @pytest.mark.asyncio
    async def test_max_per_feed(self):
        entries = [{"title": f"Item {i}", "summary": "", "link": f"http://x.com/{i}", "published": ""} for i in range(10)]
        mock_feed = make_mock_feed(entries)

        with patch("paper_trader.market.news.feedparser.parse", return_value=mock_feed):
            items = await fetch_news(feeds={"test": "http://fake.com"}, max_per_feed=3)
            assert len(items) == 3

    @pytest.mark.asyncio
    async def test_multiple_feeds_fetched_concurrently(self):
        """All feeds are gathered concurrently — results from all sources returned."""
        mock_feed = make_mock_feed([
            {"title": "News", "summary": "S", "link": "http://x.com/a", "published": ""},
        ])
        with patch("paper_trader.market.news.feedparser.parse", return_value=mock_feed):
            items = await fetch_news(feeds={"src1": "http://a.com", "src2": "http://b.com"})
            assert len(items) == 2
            sources = {i.source for i in items}
            assert sources == {"src1", "src2"}


class TestFetchSymbolNews:
    @pytest.mark.asyncio
    async def test_fetch_symbol_news(self):
        mock_feed = make_mock_feed([
            {"title": "AAPL hits new high", "summary": "Record revenue", "link": "http://example.com/aapl", "published": "2026-03-02"},
        ])

        with patch("paper_trader.market.news.feedparser.parse", return_value=mock_feed):
            items = await fetch_symbol_news("AAPL")
            assert len(items) == 1
            assert items[0].source == "yahoo_AAPL"
            assert items[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_symbol_set_on_all_items(self):
        mock_feed = make_mock_feed([
            {"title": f"News {i}", "summary": "", "link": f"http://x.com/{i}", "published": ""}
            for i in range(3)
        ])
        with patch("paper_trader.market.news.feedparser.parse", return_value=mock_feed):
            items = await fetch_symbol_news("NVDA", max_items=3)
            assert all(item.symbol == "NVDA" for item in items)


class TestFetchAndSaveNews:
    @pytest.mark.asyncio
    async def test_saves_new_items_to_db(self, db):
        mock_feed = make_mock_feed([
            {"title": "Oil at $110", "summary": "Iran crisis", "link": "https://bbc.com/oil", "published": "2026-03-02"},
        ])
        with patch("paper_trader.market.news.feedparser.parse", return_value=mock_feed):
            count = await fetch_and_save_news(db)

        assert count >= 1
        from paper_trader.db import queries as q
        rows = await q.get_recent_news(db, hours=1)
        titles = [r["title"] for r in rows]
        assert "Oil at $110" in titles

    @pytest.mark.asyncio
    async def test_deduplication_on_repeated_fetch(self, db):
        mock_feed = make_mock_feed([
            {"title": "Same article", "summary": "X", "link": "https://same.com/article", "published": ""},
        ])
        with patch("paper_trader.market.news.feedparser.parse", return_value=mock_feed):
            first = await fetch_and_save_news(db)
            second = await fetch_and_save_news(db)

        assert first >= 1
        assert second == 0  # nothing new — all already saved

    @pytest.mark.asyncio
    async def test_skips_items_without_link(self, db):
        mock_feed = make_mock_feed([
            {"title": "No link article", "summary": "X", "link": "", "published": ""},
        ])
        with patch("paper_trader.market.news.feedparser.parse", return_value=mock_feed):
            count = await fetch_and_save_news(db)

        assert count == 0


class TestFormatNews:
    def test_format_empty(self):
        assert format_news_for_ai([]) == "No recent news available."

    def test_format_items(self):
        items = [
            NewsItem(title="Big News", summary="Details here", link="", source="test"),
        ]
        result = format_news_for_ai(items)
        assert "[test]" in result
        assert "Big News" in result
        assert "Details here" in result

    def test_format_truncates_summary(self):
        long_summary = "x" * 300
        items = [NewsItem(title="T", summary=long_summary, link="", source="s")]
        result = format_news_for_ai(items)
        # summary is truncated to 200 chars in format_news_for_ai
        assert len(result) < 300
