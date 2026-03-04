from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import aiosqlite
import feedparser

logger = logging.getLogger(__name__)

# Free RSS feeds for financial news
FINANCIAL_FEEDS: dict[str, str] = {
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
    "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "cnbc": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "reuters_business": "https://www.reutersagency.com/feed/?best-topics=business-finance",
}

# Geopolitical and macro feeds — critical for events like oil shocks, wars, sanctions
GEOPOLITICAL_FEEDS: dict[str, str] = {
    "bbc_business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "eia_energy": "https://www.eia.gov/rss/press_releases.xml",
    "aljazeera": "https://www.aljazeera.com/xml/rss/all.xml",
}

DEFAULT_FEEDS: dict[str, str] = {**FINANCIAL_FEEDS, **GEOPOLITICAL_FEEDS}


@dataclass
class NewsItem:
    title: str
    summary: str
    link: str
    source: str
    published: str = ""
    symbol: str | None = None  # set for symbol-specific news


async def _fetch_one_feed(source: str, url: str, max_items: int) -> list[NewsItem]:
    """Fetch a single RSS feed. Uses run_in_executor since feedparser is blocking."""
    loop = asyncio.get_event_loop()
    try:
        feed = await loop.run_in_executor(None, feedparser.parse, url)
        items = []
        for entry in feed.entries[:max_items]:
            items.append(NewsItem(
                title=entry.get("title", ""),
                summary=entry.get("summary", "")[:500],
                link=entry.get("link", ""),
                source=source,
                published=entry.get("published", ""),
            ))
        return items
    except Exception as e:
        logger.warning("Failed to fetch feed %s: %s", source, e)
        return []


async def fetch_news(
    feeds: dict[str, str] | None = None,
    max_per_feed: int = 5,
) -> list[NewsItem]:
    """Fetch all feeds concurrently. Non-blocking."""
    if feeds is None:
        feeds = DEFAULT_FEEDS
    tasks = [_fetch_one_feed(src, url, max_per_feed) for src, url in feeds.items()]
    results = await asyncio.gather(*tasks)
    return [item for sublist in results for item in sublist]


async def fetch_symbol_news(symbol: str, max_items: int = 5) -> list[NewsItem]:
    """Fetch news for a specific stock symbol via Yahoo Finance RSS."""
    url = f"https://finance.yahoo.com/rss/headline?s={symbol}"
    items = await _fetch_one_feed(f"yahoo_{symbol}", url, max_items)
    for item in items:
        item.symbol = symbol
    return items


async def fetch_and_save_news(db: aiosqlite.Connection) -> int:
    """Fetch all feeds (general + per-symbol) and persist to DB.

    Returns the count of newly inserted items (duplicates are silently ignored).
    """
    from paper_trader.db import queries

    # General feeds — all in parallel
    all_items = await fetch_news()

    # Per-symbol feeds for watchlist + open positions — also in parallel
    watchlist = await queries.get_watchlist(db)
    positions = await queries.get_open_positions(db)
    symbols = list({w["symbol"] for w in watchlist} | {p["symbol"] for p in positions})

    if symbols:
        sym_tasks = [fetch_symbol_news(s, max_items=3) for s in symbols[:20]]
        sym_results = await asyncio.gather(*sym_tasks)
        for result_list in sym_results:
            all_items.extend(result_list)

    # Save — INSERT OR IGNORE deduplicates by link URL
    new_count = 0
    for item in all_items:
        if not item.link:
            continue
        inserted = await queries.save_news_item(
            db,
            source=item.source,
            symbol=item.symbol,
            title=item.title,
            summary=item.summary or None,
            link=item.link,
            published_at=item.published or None,
        )
        if inserted:
            new_count += 1

    logger.debug("News fetch complete: %d new items from %d fetched", new_count, len(all_items))
    return new_count


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
