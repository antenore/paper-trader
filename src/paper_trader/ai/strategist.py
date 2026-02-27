from __future__ import annotations

import logging
from typing import Any

import aiosqlite

from paper_trader.ai.client import AIClient
from paper_trader.ai.models import WeeklyReview, MonthlyReview
from paper_trader.ai.prompts import (
    WEEKLY_REVIEW_SYSTEM, weekly_review_prompt,
    MONTHLY_REVIEW_SYSTEM, monthly_review_prompt,
)
from paper_trader.config import MODEL_SONNET, MODEL_OPUS, settings
from paper_trader.db import queries
from paper_trader.market.prices import MarketDataProvider
from paper_trader.portfolio.manager import get_portfolio_value

logger = logging.getLogger(__name__)


async def run_weekly_review(
    db: aiosqlite.Connection,
    ai_client: AIClient,
    market: MarketDataProvider,
    is_dry_run: bool = False,
) -> WeeklyReview:
    """Run weekly performance review (Sonnet)."""
    # Get portfolio
    positions = await queries.get_open_positions(db, is_dry_run=is_dry_run)
    symbols = [p["symbol"] for p in positions]
    prices = await market.get_current_prices(symbols) if symbols else {}
    portfolio = await get_portfolio_value(db, prices, is_dry_run=is_dry_run)

    portfolio_summary = (
        f"Cash: {portfolio['cash']:.2f} CHF | "
        f"Positions: {portfolio['positions_value']:.2f} CHF | "
        f"Total: {portfolio['total_value']:.2f} CHF"
    )

    # Get trades and decisions
    trades = await queries.get_trades(db, is_dry_run=is_dry_run, limit=20)
    trades_text = "\n".join(
        f"{t['executed_at']}: {t['action']} {t['shares']:.2f} {t['symbol']} @ {t['price']:.2f}"
        for t in trades
    ) or "No trades this week"

    decisions = await queries.get_decisions(db, is_dry_run=is_dry_run, limit=30)
    decisions_text = "\n".join(
        f"{d['created_at']}: {d['action']} {d['symbol']} (confidence: {d['confidence']:.2f}) - {d['reasoning'][:100]}"
        for d in decisions
    ) or "No decisions this week"

    # Get current journal
    journal = await queries.get_active_journal(db, is_dry_run=is_dry_run)
    journal_text = "\n".join(
        f"[{j['entry_type']}] {j['content']}" for j in journal
    ) or "Empty journal"

    # Benchmark comparison
    benchmark_text = ""
    benchmark = await queries.get_benchmark_summary(db, is_dry_run=is_dry_run)
    if benchmark:
        benchmark_text = (
            f"{settings.benchmark_symbol}: ${benchmark['initial_spy']:.2f} → ${benchmark['current_spy']:.2f} "
            f"({benchmark['spy_return_pct']:+.1f}%) | "
            f"Portfolio: {benchmark['portfolio_return_pct']:+.1f}% | "
            f"Alpha: {benchmark['alpha_pct']:+.1f}%"
        )

    # Call AI
    raw = await ai_client.call(
        model=MODEL_SONNET,
        system=WEEKLY_REVIEW_SYSTEM,
        prompt=weekly_review_prompt(
            portfolio_summary, trades_text, decisions_text, journal_text,
            benchmark_comparison=benchmark_text,
        ),
        purpose="weekly_review",
        is_dry_run=is_dry_run,
    )

    result = WeeklyReview(**raw)

    # Apply journal entries
    for entry in result.journal_entries:
        await queries.add_journal_entry(
            db, entry.entry_type, entry.content, MODEL_SONNET,
            supersedes_id=entry.supersedes_id, is_dry_run=is_dry_run,
        )

    # Apply watchlist changes (skip during dry run to avoid mutating live watchlist)
    if not is_dry_run:
        for change in result.watchlist_changes:
            if change.action == "ADD":
                await queries.add_to_watchlist(db, change.symbol, change.reason)
            elif change.action == "REMOVE":
                await queries.remove_from_watchlist(db, change.symbol)

    return result


async def run_monthly_review(
    db: aiosqlite.Connection,
    ai_client: AIClient,
    market: MarketDataProvider,
    is_dry_run: bool = False,
) -> MonthlyReview:
    """Run monthly strategic review (Opus)."""
    # Get portfolio
    positions = await queries.get_open_positions(db, is_dry_run=is_dry_run)
    symbols = [p["symbol"] for p in positions]
    prices = await market.get_current_prices(symbols) if symbols else {}
    portfolio = await get_portfolio_value(db, prices, is_dry_run=is_dry_run)

    portfolio_summary = (
        f"Cash: {portfolio['cash']:.2f} CHF | "
        f"Positions: {portfolio['positions_value']:.2f} CHF | "
        f"Total: {portfolio['total_value']:.2f} CHF"
    )

    # Get snapshots for performance
    snapshots = await queries.get_snapshots(db, is_dry_run=is_dry_run, limit=30)
    if snapshots:
        first_val = snapshots[-1]["total_value"]
        last_val = snapshots[0]["total_value"]
        change = last_val - first_val
        monthly_perf = f"Start: {first_val:.2f} → Current: {last_val:.2f} (Change: {change:+.2f})"
    else:
        monthly_perf = "No snapshot history available"

    # Get journal
    journal = await queries.get_active_journal(db, is_dry_run=is_dry_run)
    journal_text = "\n".join(
        f"[{j['entry_type']}] {j['content']}" for j in journal
    ) or "Empty journal"

    # Get API spend
    spend = await queries.get_monthly_spend(db)
    by_model = await queries.get_api_usage_by_model(db)
    spend_text = f"Total: ${spend:.2f}\n"
    for m in by_model:
        spend_text += f"  {m['model']}: {m['call_count']} calls, ${m['total_cost']:.2f}\n"

    # Benchmark comparison
    benchmark_text = ""
    benchmark = await queries.get_benchmark_summary(db, is_dry_run=is_dry_run)
    if benchmark:
        benchmark_text = (
            f"{settings.benchmark_symbol}: ${benchmark['initial_spy']:.2f} → ${benchmark['current_spy']:.2f} "
            f"({benchmark['spy_return_pct']:+.1f}%) | "
            f"Portfolio: {benchmark['portfolio_return_pct']:+.1f}% | "
            f"Alpha: {benchmark['alpha_pct']:+.1f}%"
        )

    # Call AI
    raw = await ai_client.call(
        model=MODEL_OPUS,
        system=MONTHLY_REVIEW_SYSTEM,
        prompt=monthly_review_prompt(
            portfolio_summary, monthly_perf, journal_text, spend_text,
            benchmark_comparison=benchmark_text,
        ),
        purpose="monthly_review",
        is_dry_run=is_dry_run,
    )

    result = MonthlyReview(**raw)

    # Apply journal entries
    for entry in result.journal_entries:
        await queries.add_journal_entry(
            db, entry.entry_type, entry.content, MODEL_OPUS,
            supersedes_id=entry.supersedes_id, is_dry_run=is_dry_run,
        )

    return result
