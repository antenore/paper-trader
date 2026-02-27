import pytest
from unittest.mock import patch, MagicMock

from paper_trader.market.news import fetch_news, fetch_symbol_news, format_news_for_ai, NewsItem


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
            {"title": "Market Rally", "summary": "Stocks up", "link": "http://example.com", "published": "2025-01-01"},
            {"title": "Tech Earnings", "summary": "Mixed results", "link": "http://example.com/2", "published": "2025-01-02"},
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
        entries = [{"title": f"Item {i}", "summary": "", "link": "", "published": ""} for i in range(10)]
        mock_feed = make_mock_feed(entries)

        with patch("paper_trader.market.news.feedparser.parse", return_value=mock_feed):
            items = await fetch_news(feeds={"test": "http://fake.com"}, max_per_feed=3)
            assert len(items) == 3


class TestFetchSymbolNews:
    @pytest.mark.asyncio
    async def test_fetch_symbol_news(self):
        mock_feed = make_mock_feed([
            {"title": "AAPL hits new high", "summary": "Record revenue", "link": "http://example.com", "published": "2025-01-01"},
        ])

        with patch("paper_trader.market.news.feedparser.parse", return_value=mock_feed):
            items = await fetch_symbol_news("AAPL")
            assert len(items) == 1
            assert items[0].source == "yahoo_AAPL"


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
