from __future__ import annotations

import logging
from typing import Any

import aiosqlite

from paper_trader.ai.client import AIClient
from paper_trader.ai.models import AnalysisResult, StockDecision
from paper_trader.ai.prompts import ANALYSIS_SYSTEM, analysis_prompt
from paper_trader.config import MODEL_SONNET
from paper_trader.db import queries
from paper_trader.market.news import fetch_symbol_news, format_news_for_ai
from paper_trader.market.prices import MarketDataProvider
from paper_trader.portfolio.manager import get_portfolio_value
from paper_trader.portfolio.tools import build_tools_context

logger = logging.getLogger(__name__)


async def run_analysis(
    db: aiosqlite.Connection,
    ai_client: AIClient,
    market: MarketDataProvider,
    is_dry_run: bool = False,
) -> AnalysisResult:
    """Run intraday analysis on watchlist and open positions."""
    # Gather all symbols to analyze (with tier info)
    watchlist = await queries.get_watchlist(db)
    watchlist_symbols = [w["symbol"] for w in watchlist]
    watchlist_tiers = {w["symbol"]: w.get("risk_tier", "growth") or "growth" for w in watchlist}

    positions = await queries.get_open_positions(db, is_dry_run=is_dry_run)
    position_symbols = [p["symbol"] for p in positions]

    all_symbols = list(set(watchlist_symbols + position_symbols))
    if not all_symbols:
        logger.info("No symbols to analyze")
        return AnalysisResult(decisions=[], market_context="No symbols in watchlist or positions")

    # Get current prices
    prices = await market.get_current_prices(all_symbols)

    # Get portfolio value
    portfolio = await get_portfolio_value(db, prices, is_dry_run=is_dry_run)
    portfolio_summary = (
        f"Cash: {portfolio['cash']:.2f} CHF\n"
        f"Positions value: {portfolio['positions_value']:.2f} CHF\n"
        f"Total: {portfolio['total_value']:.2f} CHF\n"
    )
    for pos in portfolio["positions"]:
        portfolio_summary += (
            f"  {pos['symbol']}: {pos['shares']:.2f} shares @ {pos['avg_cost']:.2f} "
            f"(now {pos['current_price']:.2f}, P&L: {pos['pnl']:.2f})\n"
        )

    # Get price data summary
    price_lines = []
    for symbol in all_symbols:
        if symbol in prices:
            history = await market.get_price_history(symbol, period="5d", interval="1d")
            if len(history) >= 2:
                change = (history[-1]["close"] / history[-2]["close"] - 1) * 100
                price_lines.append(f"{symbol}: ${prices[symbol]:.2f} ({change:+.1f}%)")
            else:
                price_lines.append(f"{symbol}: ${prices[symbol]:.2f}")
    price_data = "\n".join(price_lines)

    # Get news for position symbols (more relevant)
    news_items = []
    for symbol in position_symbols[:5]:  # Limit to avoid too many requests
        news_items.extend(await fetch_symbol_news(symbol, max_items=2))
    news_text = format_news_for_ai(news_items)

    # Build tools context (stop-loss, sector, correlation, RS, ETF overlap)
    tools_context = ""
    try:
        tools_context = await build_tools_context(positions, prices, watchlist_symbols, market)
    except Exception:
        logger.debug("Tools context build failed, continuing without it", exc_info=True)

    # Call AI
    raw = await ai_client.call(
        model=MODEL_SONNET,
        system=ANALYSIS_SYSTEM,
        prompt=analysis_prompt(portfolio_summary, watchlist_symbols, price_data, news_text, tools_context=tools_context, watchlist_tiers=watchlist_tiers),
        purpose="analysis",
        is_dry_run=is_dry_run,
    )

    result = AnalysisResult(**raw)

    # Record decisions
    for decision in result.decisions:
        await queries.record_decision(
            db,
            decision.symbol,
            decision.action,
            decision.confidence,
            decision.reasoning,
            MODEL_SONNET,
            is_dry_run=is_dry_run,
        )

    return result
