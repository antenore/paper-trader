from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from paper_trader.ai.calculator import (
    CALCULATOR_TOOLS, CODE_EXECUTION_TOOL, ToolUseAudit, execute_tool,
)
from paper_trader.ai.client import AIClient
from paper_trader.ai.models import AnalysisResult, StockDecision
from paper_trader.ai.prompts import (
    ANALYSIS_SYSTEM, analysis_prompt, format_recent_trades, get_analysis_system,
)
from paper_trader.config import MODEL_SONNET, settings
from paper_trader.db import queries
from paper_trader.market.news import NewsItem, format_news_for_ai
from paper_trader.market.prices import MarketDataProvider
from paper_trader.portfolio.currency import symbol_currency
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

    # Fetch FX rate
    usd_chf_rate = 1.0
    try:
        fx_prices = await market.get_current_prices([settings.fx_pair])
        usd_chf_rate = fx_prices.get(settings.fx_pair, 1.0)
    except Exception:
        logger.debug("FX rate unavailable for analysis, using 1.0")

    # Get portfolio value (FX-aware)
    portfolio = await get_portfolio_value(db, prices, is_dry_run=is_dry_run, usd_chf_rate=usd_chf_rate)
    portfolio_summary = (
        f"Cash: {portfolio['cash']:.2f} CHF\n"
        f"Positions value: {portfolio['positions_value']:.2f} CHF\n"
        f"Total: {portfolio['total_value']:.2f} CHF\n"
    )
    if usd_chf_rate != 1.0:
        portfolio_summary += f"USD/CHF: {usd_chf_rate:.4f}\n"
    for pos in portfolio["positions"]:
        currency_label = pos.get("currency", "USD")
        portfolio_summary += (
            f"  {pos['symbol']} [{currency_label}]: {pos['shares']:.2f} shares @ {pos['avg_cost']:.2f} "
            f"(now {pos['current_price']:.2f}, P&L: {pos['pnl']:.2f})\n"
        )

    # Get price data summary (with correct currency prefix)
    price_lines = []
    for symbol in all_symbols:
        if symbol in prices:
            ccy = symbol_currency(symbol)
            prefix = "CHF" if ccy == "CHF" else "$"
            history = await market.get_price_history(symbol, period="5d", interval="1d")
            if len(history) >= 2:
                change = (history[-1]["close"] / history[-2]["close"] - 1) * 100
                price_lines.append(f"{symbol}: {prefix} {prices[symbol]:.2f} ({change:+.1f}%)")
            else:
                price_lines.append(f"{symbol}: {prefix} {prices[symbol]:.2f}")
    price_data = "\n".join(price_lines)

    # Read pre-cached news from DB (populated asynchronously by the news_fetch job).
    # This never blocks the analysis pipeline on network I/O.
    news_rows = await queries.get_recent_news(
        db, hours=settings.news_max_age_hours, symbols=position_symbols or None
    )
    news_items = [
        NewsItem(
            title=r["title"],
            summary=r.get("summary") or "",
            link=r.get("link") or "",
            source=r["source"],
            published=r.get("published_at") or "",
            symbol=r.get("symbol"),
        )
        for r in news_rows
    ]
    news_text = format_news_for_ai(news_items)

    # Query total commissions for context
    total_commissions = await queries.get_total_commissions(db, is_dry_run=is_dry_run)

    # Fetch recent trade history so the AI can see its own past actions
    recent_trades_raw = await queries.get_recent_trades(db, hours=48, is_dry_run=is_dry_run)
    recent_trades_text = format_recent_trades(recent_trades_raw)

    # Build tools context (stop-loss, sector, correlation, RS, ETF overlap, tier allocation)
    tools_context = ""
    try:
        tools_context = await build_tools_context(
            positions, prices, watchlist_symbols, market,
            portfolio_cash=portfolio["cash"],
            usd_chf_rate=usd_chf_rate,
            total_commissions=total_commissions,
        )
    except Exception:
        logger.debug("Tools context build failed, continuing without it", exc_info=True)

    # Call AI (with or without calculator tools)
    current_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    prompt_text = analysis_prompt(
        portfolio_summary, watchlist_symbols, price_data, news_text,
        tools_context=tools_context, watchlist_tiers=watchlist_tiers,
        recent_trades=recent_trades_text, current_utc=current_utc,
    )

    if settings.enable_tool_use:
        audit = ToolUseAudit()
        tools = list(CALCULATOR_TOOLS)
        if settings.enable_code_execution:
            tools.append(CODE_EXECUTION_TOOL)
        response = await ai_client.call_with_tools(
            model=MODEL_SONNET,
            system=get_analysis_system(
                enable_tools=True,
                enable_code_execution=settings.enable_code_execution,
            ),
            prompt=prompt_text,
            purpose="analysis",
            tools=tools,
            execute_tool=execute_tool,
            audit=audit,
            is_dry_run=is_dry_run,
        )
        raw = response.data
        logger.info(
            "Analysis used %d tool calls in %d turns",
            len(audit.calls), audit.turns,
        )
    else:
        raw = await ai_client.call(
            model=MODEL_SONNET,
            system=ANALYSIS_SYSTEM,
            prompt=prompt_text,
            purpose="analysis",
            is_dry_run=is_dry_run,
        )
        audit = None

    result = AnalysisResult(**raw)
    result.tool_audit = audit

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
            alpha_source=decision.alpha_source,
        )

    return result
