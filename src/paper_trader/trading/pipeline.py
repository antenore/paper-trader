from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiosqlite
import anthropic

from paper_trader.ai.analyst import run_analysis
from paper_trader.ai.client import AIClient, APIPausedError, BudgetExceededError
from paper_trader.ai.models import AnalysisResult, StockDecision
from paper_trader.ai.screener import run_screening
from paper_trader.config import settings
from paper_trader.db import queries
from paper_trader.market.prices import MarketDataProvider
from paper_trader.portfolio.manager import execute_buy, execute_sell, get_portfolio_value
from paper_trader.portfolio.tools import check_etf_overlap, check_stop_losses

logger = logging.getLogger(__name__)

OVERLOADED_INITIAL_DELAY = 30   # first retry delay in seconds
OVERLOADED_BACKOFF_FACTOR = 2   # multiply delay each retry
OVERLOADED_MAX_DELAY = 300      # cap per-retry wait at 5 minutes
OVERLOADED_MAX_TOTAL_SECONDS = 7200  # give up after ~2 hours total


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

    # Fetch FX rate once at start of pipeline
    usd_chf_rate = 1.0
    try:
        fx_prices = await market.get_current_prices([settings.fx_pair])
        usd_chf_rate = fx_prices.get(settings.fx_pair, 1.0)
    except Exception:
        logger.debug("FX rate unavailable, using 1.0")
    result["usd_chf_rate"] = usd_chf_rate

    try:
        # Step 0: Auto-execute stop-losses before AI analysis
        if run_type != "snapshot":
            stop_trades = await _execute_stop_losses(db, market, is_dry_run, usd_chf_rate=usd_chf_rate)
            if stop_trades:
                result["trades"].extend(stop_trades)

        if run_type == "snapshot":
            await _take_snapshot(db, market, is_dry_run, usd_chf_rate=usd_chf_rate)
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
                screening = await _retry_on_overload(
                    run_screening(db, ai_client, market, is_dry_run),
                    lambda: run_screening(db, ai_client, market, is_dry_run),
                    "Screening",
                )
                result["screening"] = screening.market_summary
            except (APIPausedError, BudgetExceededError) as e:
                result["errors"].append(f"Screening skipped: {e}")
                return result

        # Step 2: Analysis
        try:
            analysis = await _retry_on_overload(
                run_analysis(db, ai_client, market, is_dry_run),
                lambda: run_analysis(db, ai_client, market, is_dry_run),
                "Analysis",
            )
            result["analysis"] = analysis.market_context
        except (APIPausedError, BudgetExceededError) as e:
            result["errors"].append(f"Analysis skipped: {e}")
            return result

        # Step 2.5: Tool-use verification (when tool use is enabled)
        if settings.enable_tool_use and hasattr(analysis, "tool_audit") and analysis.tool_audit is not None:
            audit = analysis.tool_audit
            result["tool_use_count"] = len(audit.calls)
            result["tool_use_turns"] = audit.turns
            logger.info("Analysis used %d tool calls in %d turns", len(audit.calls), audit.turns)

            sized_symbols = audit.symbols_with_tool("position_size")
            for decision in analysis.decisions:
                if decision.action == "BUY" and decision.symbol not in sized_symbols:
                    if settings.require_tool_evidence:
                        logger.warning(
                            "REJECT BUY %s → HOLD: no position_size tool call (require_tool_evidence=True)",
                            decision.symbol,
                        )
                        decision.action = "HOLD"
                        decision.reasoning += " [REJECTED: missing position_size tool call]"
                    else:
                        logger.warning(
                            "BUY %s without position_size tool call (observation only)",
                            decision.symbol,
                        )

        # Step 3: Execute decisions (with RULE 001 session budget + RULE 003 ETF overlap)
        portfolio_for_budget = await queries.get_portfolio(db, is_dry_run=is_dry_run)
        session_budget = (portfolio_for_budget["cash"] * settings.max_session_deploy_pct) if portfolio_for_budget else 0
        session_spent = 0.0
        position_symbols = [p["symbol"] for p in await queries.get_open_positions(db, is_dry_run=is_dry_run)]

        # Build tier map from watchlist + positions
        watchlist = await queries.get_watchlist(db)
        watchlist_tiers = {w["symbol"]: w.get("risk_tier", "growth") or "growth" for w in watchlist}
        # Positions inherit their stored tier
        for p in await queries.get_open_positions(db, is_dry_run=is_dry_run):
            if p["symbol"] not in watchlist_tiers:
                watchlist_tiers[p["symbol"]] = p.get("risk_tier", "growth") or "growth"

        for decision in analysis.decisions:
            if decision.action == "HOLD":
                continue
            if decision.confidence < settings.confidence_threshold:
                logger.info("Skipping %s %s (confidence %.2f < %.2f)", decision.action, decision.symbol, decision.confidence, settings.confidence_threshold)
                continue

            # RULE 003: ETF overlap check for BUY decisions
            if decision.action == "BUY":
                overlap = check_etf_overlap(decision.symbol, position_symbols)
                if overlap:
                    logger.info("Skipping BUY %s: ETF overlap — %s", decision.symbol, overlap)
                    result["trades"].append({"symbol": decision.symbol, "action": "BUY", "skipped": f"ETF overlap: {overlap}"})
                    continue

                # RULE 001: Session budget check
                if session_spent >= session_budget:
                    logger.info("Skipping BUY %s: session budget exhausted (%.2f/%.2f)", decision.symbol, session_spent, session_budget)
                    result["trades"].append({"symbol": decision.symbol, "action": "BUY", "skipped": "Session budget exhausted"})
                    continue

            trade_result = await _execute_decision(db, market, decision, is_dry_run, watchlist_tiers, usd_chf_rate=usd_chf_rate)
            result["trades"].append(trade_result)

            # Track session spend for BUYs
            if decision.action == "BUY" and trade_result.get("ok"):
                session_spent += trade_result.get("total", 0)
                position_symbols.append(decision.symbol)

        # Step 4: Take snapshot
        await _take_snapshot(db, market, is_dry_run, usd_chf_rate=usd_chf_rate)

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
    watchlist_tiers: dict[str, str] | None = None,
    usd_chf_rate: float = 1.0,
) -> dict[str, Any]:
    """Execute a single trading decision."""
    prices = await market.get_current_prices([decision.symbol])
    price = prices.get(decision.symbol)
    if price is None:
        return {"symbol": decision.symbol, "error": "Price unavailable"}

    # Resolve risk tier: prefer decision's tier, fall back to watchlist, default to growth
    risk_tier = decision.risk_tier or (watchlist_tiers or {}).get(decision.symbol, "growth")

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
        pv = await get_portfolio_value(db, all_prices, is_dry_run=is_dry_run, usd_chf_rate=usd_chf_rate)
        total_value = pv["total_value"]

        target_value = total_value * (decision.target_allocation_pct / 100) if decision.target_allocation_pct > 0 else total_value * 0.10
        existing = next((p for p in positions if p["symbol"] == decision.symbol), None)
        existing_value = existing["shares"] * price if existing else 0
        buy_value = max(0, target_value - existing_value)
        shares = buy_value / price if price > 0 else 0

        if shares < 0.01:
            return {"symbol": decision.symbol, "action": "BUY", "skipped": "Already at target"}

        result = await execute_buy(db, decision.symbol, shares, price, decision_id, is_dry_run, risk_tier=risk_tier, usd_chf_rate=usd_chf_rate)
        return {"symbol": decision.symbol, **result}

    elif decision.action == "SELL":
        position = await queries.get_position_by_symbol(db, decision.symbol, is_dry_run=is_dry_run)
        if position is None:
            return {"symbol": decision.symbol, "action": "SELL", "skipped": "No position"}
        result = await execute_sell(db, decision.symbol, position["shares"], price, decision_id, is_dry_run, usd_chf_rate=usd_chf_rate)
        return {"symbol": decision.symbol, **result}

    return {"symbol": decision.symbol, "action": decision.action, "skipped": "Unknown action"}


async def _take_snapshot(
    db: aiosqlite.Connection,
    market: MarketDataProvider,
    is_dry_run: bool,
    usd_chf_rate: float = 1.0,
) -> None:
    """Take a portfolio snapshot, including SPY benchmark when available."""
    positions = await queries.get_open_positions(db, is_dry_run=is_dry_run)
    symbols = [p["symbol"] for p in positions]
    prices = await market.get_current_prices(symbols) if symbols else {}
    pv = await get_portfolio_value(db, prices, is_dry_run=is_dry_run, usd_chf_rate=usd_chf_rate)

    # Benchmark tracking
    spy_price: float | None = None
    benchmark_value: float | None = None
    snapshot_fx: float | None = usd_chf_rate if usd_chf_rate != 1.0 else None
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

    # FX rate tracking — prefer live rate, fall back to pipeline rate
    if snapshot_fx is None:
        try:
            fx_prices = await market.get_current_prices([settings.fx_pair])
            snapshot_fx = fx_prices.get(settings.fx_pair)
        except Exception:
            logger.debug("FX rate unavailable, snapshot continues without it")

    await queries.record_snapshot(
        db, pv["cash"], pv["positions_value"], is_dry_run,
        spy_price=spy_price, benchmark_value=benchmark_value,
        usd_chf_rate=snapshot_fx,
    )


async def _execute_stop_losses(
    db: aiosqlite.Connection,
    market: MarketDataProvider,
    is_dry_run: bool,
    usd_chf_rate: float = 1.0,
) -> list[dict[str, Any]]:
    """Step 0: Check and auto-execute stop-losses before AI analysis."""
    positions = await queries.get_open_positions(db, is_dry_run=is_dry_run)
    if not positions:
        return []

    symbols = [p["symbol"] for p in positions]
    prices = await market.get_current_prices(symbols)
    triggered = check_stop_losses(positions, prices)

    results = []
    for stop in triggered:
        # Record a system decision
        decision_id = await queries.record_decision(
            db,
            stop["symbol"],
            "SELL",
            1.0,
            f"Stop-loss triggered: price ${stop['current_price']:.2f} <= stop ${stop['stop_loss_price']:.2f} ({stop['loss_pct']:.1f}%)",
            "system/stop-loss",
            is_dry_run=is_dry_run,
        )
        result = await execute_sell(
            db, stop["symbol"], stop["shares"], stop["current_price"],
            decision_id, is_dry_run, usd_chf_rate=usd_chf_rate,
        )
        logger.warning(
            "STOP-LOSS %s: sold %.2f shares @ $%.2f (stop: $%.2f)",
            stop["symbol"], stop["shares"], stop["current_price"], stop["stop_loss_price"],
        )
        results.append({"symbol": stop["symbol"], "stop_loss": True, **result})

    return results


def _is_overloaded(exc: Exception) -> bool:
    """Check if an exception is a 529 overloaded error."""
    return isinstance(exc, anthropic.APIStatusError) and exc.status_code == 529


async def _retry_on_overload(first_attempt, factory, label: str):
    """
    Await first_attempt; on 529 overload, retry with exponential backoff.

    Backoff: 30s, 60s, 120s, 240s, 300s, 300s, ... (capped at 5 min).
    Gives up after ~2 hours total wait time.

    Args:
        first_attempt: The already-created coroutine for the first try.
        factory: A callable returning a new coroutine for retries.
        label: Human-readable name for logging (e.g. "Screening").
    """
    try:
        return await first_attempt
    except Exception as e:
        if not _is_overloaded(e):
            raise
        logger.warning("%s hit API overload (529), will retry with backoff", label)

    delay = OVERLOADED_INITIAL_DELAY
    elapsed = 0
    attempt = 0
    while elapsed < OVERLOADED_MAX_TOTAL_SECONDS:
        attempt += 1
        wait = min(delay, OVERLOADED_MAX_DELAY)
        remaining = OVERLOADED_MAX_TOTAL_SECONDS - elapsed
        if wait > remaining:
            wait = remaining
        logger.info(
            "%s retry %d — waiting %ds (elapsed %dm/%dm)...",
            label, attempt, wait,
            elapsed // 60, OVERLOADED_MAX_TOTAL_SECONDS // 60,
        )
        await asyncio.sleep(wait)
        elapsed += wait
        try:
            return await factory()
        except Exception as e:
            if not _is_overloaded(e):
                raise
            last_exc = e
            logger.warning(
                "%s retry %d still overloaded (elapsed %dm)",
                label, attempt, elapsed // 60,
            )
            delay *= OVERLOADED_BACKOFF_FACTOR

    logger.error(
        "%s giving up after %d retries (~%dm total wait)",
        label, attempt, elapsed // 60,
    )
    raise last_exc
