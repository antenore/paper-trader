from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import feedparser

logger = logging.getLogger(__name__)

# Free RSS feeds for financial news
DEFAULT_FEEDS: dict[str, str] = {
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
    "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "cnbc": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "reuters_business": "https://www.reutersagency.com/feed/?best-topics=business-finance",
}


@dataclass
class NewsItem:
    title: str
    summary: str
    link: str
    source: str
    published: str = ""


async def fetch_news(
    feeds: dict[str, str] | None = None,
    max_per_feed: int = 5,
) -> list[NewsItem]:
    """Fetch news from RSS feeds."""
    if feeds is None:
        feeds = DEFAULT_FEEDS

    items: list[NewsItem] = []
    for source, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                items.append(NewsItem(
                    title=entry.get("title", ""),
                    summary=entry.get("summary", "")[:500],
                    link=entry.get("link", ""),
                    source=source,
                    published=entry.get("published", ""),
                ))
        except Exception as e:
            logger.warning("Failed to fetch feed %s: %s", source, e)

    return items


async def fetch_symbol_news(symbol: str, max_items: int = 5) -> list[NewsItem]:
    """Fetch news for a specific stock symbol using Yahoo Finance RSS."""
    url = f"https://finance.yahoo.com/rss/headline?s={symbol}"
    items: list[NewsItem] = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:max_items]:
            items.append(NewsItem(
                title=entry.get("title", ""),
                summary=entry.get("summary", "")[:500],
                link=entry.get("link", ""),
                source=f"yahoo_{symbol}",
                published=entry.get("published", ""),
            ))
    except Exception as e:
        logger.warning("Failed to fetch news for %s: %s", symbol, e)
    return items


def format_news_for_ai(items: list[NewsItem]) -> str:
    """Format news items into a concise string for AI context."""
    if not items:
        return "No recent news available."

    lines = []
    for item in items:
        lines.append(f"- [{item.source}] {item.title}")
        if item.summary:
            lines.append(f"  {item.summary[:200]}")
    return "\n".join(lines)
