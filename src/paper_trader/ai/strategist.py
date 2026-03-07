from __future__ import annotations

import logging
from typing import Any

import aiosqlite

from paper_trader.ai.calculator import CODE_EXECUTION_TOOL, execute_tool
from paper_trader.ai.client import AIClient
from paper_trader.ai.models import WeeklyReview, MonthlyReview
from paper_trader.ai.prompts import (
    WEEKLY_REVIEW_SYSTEM, WEEKLY_CODE_EXECUTION_INSTRUCTIONS,
    weekly_review_prompt,
    MONTHLY_REVIEW_SYSTEM, MONTHLY_CODE_EXECUTION_INSTRUCTIONS,
    monthly_review_prompt,
)
from paper_trader.config import MODEL_SONNET, MODEL_OPUS, settings
from paper_trader.db import queries
from paper_trader.market.prices import MarketDataProvider
from paper_trader.portfolio.manager import get_portfolio_value

logger = logging.getLogger(__name__)


def _snapshots_to_csv(snapshots: list[dict[str, Any]]) -> str:
    """Convert snapshot rows to CSV text for code execution analysis."""
    if not snapshots:
        return ""
    header = "date,cash,positions_value,total_value,spy_price,benchmark_value,usd_chf_rate"
    lines = [header]
    for s in reversed(snapshots):  # chronological order
        lines.append(
            f"{s.get('snapshot_at', '')},{s.get('cash', 0):.2f},"
            f"{s.get('positions_value', 0):.2f},{s.get('total_value', 0):.2f},"
            f"{s.get('spy_price', '') or ''},{s.get('benchmark_value', '') or ''},"
            f"{s.get('usd_chf_rate', '') or ''}"
        )
    return "\n".join(lines)


def _trades_to_csv(trades: list[dict[str, Any]]) -> str:
    """Convert trade rows to CSV text for code execution analysis."""
    if not trades:
        return ""
    header = "date,symbol,action,shares,price,total,commission_chf,currency"
    lines = [header]
    for t in reversed(trades):  # chronological order
        lines.append(
            f"{t.get('executed_at', '')},{t.get('symbol', '')},"
            f"{t.get('action', '')},{t.get('shares', 0):.4f},"
            f"{t.get('price', 0):.4f},{t.get('total', 0):.2f},"
            f"{t.get('commission_chf', 0):.2f},{t.get('currency', 'USD')}"
        )
    return "\n".join(lines)


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
        cash_pct = portfolio['cash'] / portfolio['total_value'] if portfolio['total_value'] else 0
        cash_alpha_effect = cash_pct * abs(benchmark['spy_return_pct'])
        benchmark_text = (
            f"{settings.benchmark_symbol}: ${benchmark['initial_spy']:.2f} → ${benchmark['current_spy']:.2f} "
            f"({benchmark['spy_return_pct']:+.1f}%) | "
            f"Portfolio: {benchmark['portfolio_return_pct']:+.1f}% | "
            f"Alpha: {benchmark['alpha_pct']:+.1f}% | "
            f"Cash effect: {cash_alpha_effect:+.1f}% (cash weight: {cash_pct*100:.0f}%)"
        )
        if benchmark.get('current_fx_rate'):
            benchmark_text += f" | USD/CHF: {benchmark['current_fx_rate']:.4f} ({benchmark['fx_change_pct']:+.1f}%)"

    # Code execution: prepare snapshot CSV and switch to call_with_tools
    use_code_exec = settings.enable_code_execution
    snapshot_csv = ""
    if use_code_exec:
        snapshots = await queries.get_snapshots(db, is_dry_run=is_dry_run, limit=30)
        snapshot_csv = _snapshots_to_csv(snapshots)

    prompt_text = weekly_review_prompt(
        portfolio_summary, trades_text, decisions_text, journal_text,
        benchmark_comparison=benchmark_text,
        snapshot_csv=snapshot_csv,
    )

    if use_code_exec:
        system = WEEKLY_REVIEW_SYSTEM + WEEKLY_CODE_EXECUTION_INSTRUCTIONS
        response = await ai_client.call_with_tools(
            model=MODEL_SONNET,
            system=system,
            prompt=prompt_text,
            purpose="weekly_review",
            tools=[CODE_EXECUTION_TOOL],
            execute_tool=execute_tool,
            is_dry_run=is_dry_run,
        )
        raw = response.data
        logger.info(
            "Weekly review used code execution (%d turns, container: %s)",
            response.turns, response.container_id,
        )
    else:
        raw = await ai_client.call(
            model=MODEL_SONNET,
            system=WEEKLY_REVIEW_SYSTEM,
            prompt=prompt_text,
            purpose="weekly_review",
            is_dry_run=is_dry_run,
            max_tokens=8192,
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
    currency_text = ""
    benchmark = await queries.get_benchmark_summary(db, is_dry_run=is_dry_run)
    if benchmark:
        cash_pct = portfolio['cash'] / portfolio['total_value'] if portfolio['total_value'] else 0
        cash_alpha_effect = cash_pct * abs(benchmark['spy_return_pct'])
        benchmark_text = (
            f"{settings.benchmark_symbol}: ${benchmark['initial_spy']:.2f} → ${benchmark['current_spy']:.2f} "
            f"({benchmark['spy_return_pct']:+.1f}%) | "
            f"Portfolio: {benchmark['portfolio_return_pct']:+.1f}% | "
            f"Alpha: {benchmark['alpha_pct']:+.1f}% | "
            f"Cash effect: {cash_alpha_effect:+.1f}% (cash weight: {cash_pct*100:.0f}%)"
        )
        if benchmark.get('current_fx_rate'):
            currency_text = (
                f"USD/CHF: {benchmark['initial_fx_rate']:.4f} → {benchmark['current_fx_rate']:.4f} "
                f"({benchmark['fx_change_pct']:+.1f}%)\n"
                f"Portfolio (CHF-adjusted): {benchmark['portfolio_value_chf']:.2f} CHF\n"
                f"Benchmark (CHF-adjusted): {benchmark['benchmark_value_chf']:.2f} CHF\n"
                f"FX impact on returns: {benchmark['fx_impact_pct']:+.1f}%"
            )

    # Code execution: prepare CSV data
    use_code_exec = settings.enable_code_execution
    snapshot_csv = ""
    trades_csv = ""
    if use_code_exec:
        snapshot_csv = _snapshots_to_csv(snapshots)
        trades = await queries.get_trades(db, is_dry_run=is_dry_run, limit=50)
        trades_csv = _trades_to_csv(trades)

    prompt_text = monthly_review_prompt(
        portfolio_summary, monthly_perf, journal_text, spend_text,
        benchmark_comparison=benchmark_text,
        currency_data=currency_text,
        snapshot_csv=snapshot_csv,
        trades_csv=trades_csv,
    )

    if use_code_exec:
        system = MONTHLY_REVIEW_SYSTEM + MONTHLY_CODE_EXECUTION_INSTRUCTIONS
        response = await ai_client.call_with_tools(
            model=MODEL_OPUS,
            system=system,
            prompt=prompt_text,
            purpose="monthly_review",
            tools=[CODE_EXECUTION_TOOL],
            execute_tool=execute_tool,
            is_dry_run=is_dry_run,
        )
        raw = response.data
        logger.info(
            "Monthly review used code execution (%d turns, container: %s)",
            response.turns, response.container_id,
        )
    else:
        raw = await ai_client.call(
            model=MODEL_OPUS,
            system=MONTHLY_REVIEW_SYSTEM,
            prompt=prompt_text,
            purpose="monthly_review",
            is_dry_run=is_dry_run,
            max_tokens=8192,
        )

    result = MonthlyReview(**raw)

    # Apply journal entries
    for entry in result.journal_entries:
        await queries.add_journal_entry(
            db, entry.entry_type, entry.content, MODEL_OPUS,
            supersedes_id=entry.supersedes_id, is_dry_run=is_dry_run,
        )

    return result
