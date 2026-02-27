from __future__ import annotations

import logging
from typing import Any

import aiosqlite

from paper_trader.ai.client import AIClient
from paper_trader.ai.models import ScreeningResult
from paper_trader.ai.prompts import SCREENING_SYSTEM, screening_prompt
from paper_trader.config import MODEL_HAIKU
from paper_trader.db import queries
from paper_trader.market.news import fetch_news, format_news_for_ai
from paper_trader.market.prices import MarketDataProvider

logger = logging.getLogger(__name__)


async def run_screening(
    db: aiosqlite.Connection,
    ai_client: AIClient,
    market: MarketDataProvider,
    is_dry_run: bool = False,
) -> ScreeningResult:
    """Run stock screening to update watchlist."""
    # Get current watchlist
    watchlist = await queries.get_watchlist(db)
    current_symbols = [w["symbol"] for w in watchlist]

    # Get market data
    movers = await market.get_market_movers()
    market_data = f"Market movers: {movers}"

    # Get news
    news_items = await fetch_news(max_per_feed=3)
    news_text = format_news_for_ai(news_items)

    # Call AI
    raw = await ai_client.call(
        model=MODEL_HAIKU,
        system=SCREENING_SYSTEM,
        prompt=screening_prompt(current_symbols, market_data, news_text),
        purpose="screening",
        is_dry_run=is_dry_run,
    )

    result = ScreeningResult(**raw)

    # Apply watchlist updates (skip during dry run to avoid live contamination)
    if not is_dry_run:
        for update in result.watchlist_updates:
            if update.action == "ADD":
                await queries.add_to_watchlist(db, update.symbol, update.reason)
                logger.info("Watchlist ADD: %s (%s)", update.symbol, update.reason)
            elif update.action == "REMOVE":
                await queries.remove_from_watchlist(db, update.symbol)
                logger.info("Watchlist REMOVE: %s (%s)", update.symbol, update.reason)

    return result
