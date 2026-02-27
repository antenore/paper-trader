from __future__ import annotations

import logging
from typing import Any

import aiosqlite

from paper_trader.ai.analyst import run_analysis
from paper_trader.ai.client import AIClient, APIPausedError, BudgetExceededError
from paper_trader.ai.models import AnalysisResult, StockDecision
from paper_trader.ai.screener import run_screening
from paper_trader.config import settings
from paper_trader.db import queries
from paper_trader.market.prices import MarketDataProvider
from paper_trader.portfolio.manager import execute_buy, execute_sell, get_portfolio_value

logger = logging.getLogger(__name__)


async def run_trading_pipeline(
    db: aiosqlite.Connection,
    ai_client: AIClient,
    market: MarketDataProvider,
    run_type: str = "full",
    is_dry_run: bool = False,
) -> dict[str, Any]:
    """
    Main trading pipeline orchestrator.

    run_type: "full" (screening + analysis), "quick" (analysis only if movement), "snapshot" (just record)
    """
    result: dict[str, Any] = {"run_type": run_type, "trades": [], "errors": []}

    try:
        if run_type == "snapshot":
            await _take_snapshot(db, market, is_dry_run)
            result["snapshot"] = True
            return result

        if run_type == "quick":
            # Check if any significant movement before calling AI
            has_movement = await _check_significant_movement(db, market, is_dry_run)
            has_positions = bool(await queries.get_open_positions(db, is_dry_run=is_dry_run))
            if not has_movement and not has_positions:
                logger.info("No significant movement and no open positions, skipping AI analysis")
                result["skipped"] = True
                return result
            if not has_movement:
                logger.info("No significant movement but have open positions, checking for sell signals")

        # Step 1: Screening (full runs only)
        if run_type == "full":
            try:
                screening = await run_screening(db, ai_client, market, is_dry_run)
                result["screening"] = screening.market_summary
            except (APIPausedError, BudgetExceededError) as e:
                result["errors"].append(f"Screening skipped: {e}")
                return result

        # Step 2: Analysis
        try:
            analysis = await run_analysis(db, ai_client, market, is_dry_run)
            result["analysis"] = analysis.market_context
        except (APIPausedError, BudgetExceededError) as e:
            result["errors"].append(f"Analysis skipped: {e}")
            return result

        # Step 3: Execute decisions
        for decision in analysis.decisions:
            if decision.action == "HOLD":
                continue
            if decision.confidence < 0.4:
                logger.info("Skipping %s %s (confidence %.2f < 0.4)", decision.action, decision.symbol, decision.confidence)
                continue

            trade_result = await _execute_decision(db, market, decision, is_dry_run)
            result["trades"].append(trade_result)

        # Step 4: Take snapshot
        await _take_snapshot(db, market, is_dry_run)

    except Exception as e:
        logger.error("Pipeline error: %s", e, exc_info=True)
        result["errors"].append(str(e))

    return result


async def _check_significant_movement(
    db: aiosqlite.Connection,
    market: MarketDataProvider,
    is_dry_run: bool,
) -> bool:
    """Check if any watched symbol moved more than the threshold."""
    watchlist = await queries.get_watchlist(db)
    positions = await queries.get_open_positions(db, is_dry_run=is_dry_run)
    symbols = list({w["symbol"] for w in watchlist} | {p["symbol"] for p in positions})

    if not symbols:
        return False

    prices = await market.get_current_prices(symbols)
    threshold = settings.movement_threshold_pct

    for symbol in symbols:
        if symbol not in prices:
            continue
        history = await market.get_price_history(symbol, period="2d", interval="1d")
        if len(history) >= 2:
            prev_close = history[-2]["close"]
            current = prices[symbol]
            change_pct = abs((current / prev_close - 1) * 100)
            if change_pct >= threshold:
                logger.info("Significant movement: %s %.1f%%", symbol, change_pct)
                return True

    return False


async def _execute_decision(
    db: aiosqlite.Connection,
    market: MarketDataProvider,
    decision: StockDecision,
    is_dry_run: bool,
) -> dict[str, Any]:
    """Execute a single trading decision."""
    prices = await market.get_current_prices([decision.symbol])
    price = prices.get(decision.symbol)
    if price is None:
        return {"symbol": decision.symbol, "error": "Price unavailable"}

    # Get the latest decision ID
    decisions = await queries.get_decisions(db, is_dry_run=is_dry_run, limit=1)
    decision_id = decisions[0]["id"] if decisions else None

    if decision.action == "BUY":
        # Calculate shares based on target allocation
        portfolio = await queries.get_portfolio(db)
        if portfolio is None:
            return {"symbol": decision.symbol, "error": "Portfolio not initialized"}

        positions = await queries.get_open_positions(db, is_dry_run=is_dry_run)
        all_prices = await market.get_current_prices([p["symbol"] for p in positions])
        pv = await get_portfolio_value(db, all_prices, is_dry_run=is_dry_run)
        total_value = pv["total_value"]

        target_value = total_value * (decision.target_allocation_pct / 100) if decision.target_allocation_pct > 0 else total_value * 0.10
        existing = next((p for p in positions if p["symbol"] == decision.symbol), None)
        existing_value = existing["shares"] * price if existing else 0
        buy_value = max(0, target_value - existing_value)
        shares = buy_value / price if price > 0 else 0

        if shares < 0.01:
            return {"symbol": decision.symbol, "action": "BUY", "skipped": "Already at target"}

        result = await execute_buy(db, decision.symbol, shares, price, decision_id, is_dry_run)
        return {"symbol": decision.symbol, **result}

    elif decision.action == "SELL":
        position = await queries.get_position_by_symbol(db, decision.symbol, is_dry_run=is_dry_run)
        if position is None:
            return {"symbol": decision.symbol, "action": "SELL", "skipped": "No position"}
        result = await execute_sell(db, decision.symbol, position["shares"], price, decision_id, is_dry_run)
        return {"symbol": decision.symbol, **result}

    return {"symbol": decision.symbol, "action": decision.action, "skipped": "Unknown action"}


async def _take_snapshot(
    db: aiosqlite.Connection,
    market: MarketDataProvider,
    is_dry_run: bool,
) -> None:
    """Take a portfolio snapshot, including SPY benchmark when available."""
    positions = await queries.get_open_positions(db, is_dry_run=is_dry_run)
    symbols = [p["symbol"] for p in positions]
    prices = await market.get_current_prices(symbols) if symbols else {}
    pv = await get_portfolio_value(db, prices, is_dry_run=is_dry_run)

    # Benchmark tracking
    spy_price: float | None = None
    benchmark_value: float | None = None
    try:
        benchmark_symbol = settings.benchmark_symbol
        spy_prices = await market.get_current_prices([benchmark_symbol])
        spy_price = spy_prices.get(benchmark_symbol)
        if spy_price is not None:
            first_spy = await queries.get_first_spy_price(db, is_dry_run)
            if first_spy is None:
                # First snapshot with SPY data — benchmark starts at initial cash
                benchmark_value = settings.initial_cash_chf
            else:
                benchmark_value = settings.initial_cash_chf * (spy_price / first_spy)
    except Exception:
        logger.debug("Benchmark price unavailable, snapshot continues without it")

    await queries.record_snapshot(
        db, pv["cash"], pv["positions_value"], is_dry_run,
        spy_price=spy_price, benchmark_value=benchmark_value,
    )
