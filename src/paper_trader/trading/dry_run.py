from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiosqlite
import yfinance as yf

from paper_trader.ai.client import AIClient
from paper_trader.config import settings
from paper_trader.db import queries
from paper_trader.market.prices import HistoricalDataProvider
from paper_trader.portfolio.manager import get_portfolio_value
from paper_trader.trading.pipeline import run_trading_pipeline

logger = logging.getLogger(__name__)


async def load_historical_data(
    symbols: list[str], period: str = "2mo"
) -> dict[str, list[dict[str, Any]]]:
    """Load historical data from yfinance for simulation."""
    data: dict[str, list[dict[str, Any]]] = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval="1d")
            if df.empty:
                continue
            records = []
            for date, row in df.iterrows():
                records.append({
                    "date": str(date),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                })
            data[symbol] = records
        except Exception as e:
            logger.warning("Failed to load history for %s: %s", symbol, e)
    return data


async def run_dry_run(
    db: aiosqlite.Connection,
    ai_client: AIClient,
    symbols: list[str] | None = None,
    days: int = 30,
    on_progress: Any = None,
) -> dict[str, Any]:
    """
    Run a dry run simulation using historical data.
    Simulates ~days of trading with real AI calls.
    The live portfolio row is never touched — dry run cash is tracked
    in the dry_run_sessions table.
    """
    if symbols is None:
        symbols = list(settings.default_watchlist)

    # Ensure benchmark symbol is included for tracking
    if settings.benchmark_symbol not in symbols:
        symbols.append(settings.benchmark_symbol)

    # Start session atomically (returns None if one is already active)
    session_id = await queries.start_dry_run_session(db, settings.initial_cash_chf)
    if session_id is None:
        return {"error": "A dry run session is already running"}
    logger.info("Dry run session %d started with %d symbols", session_id, len(symbols))

    # Load historical data
    historical_data = await load_historical_data(symbols, period="3mo")
    if not historical_data:
        await queries.end_dry_run_session(db, session_id, settings.initial_cash_chf, 0, 0, "No data available")
        return {"session_id": session_id, "error": "No historical data available"}

    # Find the common date range
    min_len = min(len(v) for v in historical_data.values())
    sim_days = min(days, min_len - 5)  # Keep some history for analysis

    if sim_days <= 0:
        await queries.end_dry_run_session(db, session_id, settings.initial_cash_chf, 0, 0, "Insufficient historical data")
        return {"session_id": session_id, "error": "Insufficient historical data (need > 5 days)"}

    start_idx = min_len - sim_days

    market = HistoricalDataProvider(historical_data, current_index=start_idx)
    total_trades = 0
    completed_days = 0

    # Defaults in case finalization fails
    pv: dict[str, Any] = {"total_value": settings.initial_cash_chf, "cash": settings.initial_cash_chf, "positions_value": 0.0}
    api_cost = 0.0
    summary = f"Session {session_id} ended without finalization"

    try:
        # Simulate each trading day
        for day in range(sim_days):
            idx = start_idx + day
            market.current_index = idx

            # Determine run type (simulate 3x daily pattern)
            if day % 3 == 0:
                run_type = "full"
            elif day % 3 == 1:
                run_type = "quick"
            else:
                run_type = "snapshot"

            try:
                result = await run_trading_pipeline(
                    db, ai_client, market, run_type=run_type, is_dry_run=True
                )
                total_trades += len(result.get("trades", []))
                completed_days += 1

                if on_progress:
                    current_date = historical_data[symbols[0]][idx]["date"] if symbols[0] in historical_data else f"Day {day+1}"
                    await on_progress(day + 1, sim_days, current_date, result)

            except Exception as e:
                logger.error("Dry run day %d error: %s", day + 1, e, exc_info=True)
                completed_days += 1

    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.warning("Dry run session %d interrupted after %d/%d days", session_id, completed_days, sim_days)
    finally:
        # Always close the session, even on cancellation
        try:
            last_idx = start_idx + max(completed_days - 1, 0)
            market.current_index = last_idx
            positions = await queries.get_open_positions(db, is_dry_run=True)
            pos_symbols = [p["symbol"] for p in positions]
            prices = await market.get_current_prices(pos_symbols) if pos_symbols else {}
            pv = await get_portfolio_value(db, prices, is_dry_run=True)

            api_cost = await queries.get_session_api_cost(db, session_id)

            interrupted = completed_days < sim_days
            summary = (
                f"Simulated {completed_days}/{sim_days} days{' (interrupted)' if interrupted else ''}. "
                f"Final value: {pv['total_value']:.2f} CHF (P&L: {pv['total_value'] - settings.initial_cash_chf:+.2f}). "
                f"Trades: {total_trades}. API cost: ${api_cost:.2f}."
            )

            await queries.end_dry_run_session(db, session_id, pv["total_value"], total_trades, api_cost, summary)
            logger.info("Dry run session %d closed: %s", session_id, summary)
        except Exception as e:
            logger.error("Failed to close dry run session %d: %s", session_id, e)

    return {
        "session_id": session_id,
        "days_simulated": completed_days,
        "final_value": pv["total_value"],
        "pnl": pv["total_value"] - settings.initial_cash_chf,
        "total_trades": total_trades,
        "api_cost": api_cost,
        "summary": summary,
    }


async def reset_for_live(db: aiosqlite.Connection) -> None:
    """Reset portfolio for live trading. Keep journal entries."""
    await queries.update_cash(db, settings.initial_cash_chf)
    # Close all dry run positions (they stay in DB with is_dry_run=1)
    # Journal entries are kept (is_dry_run flag preserves them)
    await queries.set_setting(db, "mode", "live")
    logger.info("Portfolio reset for live trading (%.2f %s)", settings.initial_cash_chf, settings.currency)
