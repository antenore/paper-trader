from __future__ import annotations

import logging
from typing import Any

import aiosqlite

from paper_trader.config import settings
from paper_trader.db import queries
from paper_trader.portfolio.risk import check_risk

logger = logging.getLogger(__name__)


async def get_portfolio_value(db: aiosqlite.Connection, prices: dict[str, float], is_dry_run: bool = False) -> dict[str, Any]:
    """Calculate total portfolio value including positions."""
    portfolio = await queries.get_portfolio(db, is_dry_run=is_dry_run)
    if portfolio is None:
        return {"cash": 0.0, "positions_value": 0.0, "total_value": 0.0, "positions": []}

    positions = await queries.get_open_positions(db, is_dry_run=is_dry_run)
    positions_value = 0.0
    enriched = []
    for pos in positions:
        current_price = prices.get(pos["symbol"], pos["avg_cost"])
        value = pos["shares"] * current_price
        pnl = (current_price - pos["avg_cost"]) * pos["shares"]
        positions_value += value
        enriched.append({
            **pos,
            "current_price": current_price,
            "value": value,
            "pnl": pnl,
            "pnl_pct": (current_price / pos["avg_cost"] - 1) * 100 if pos["avg_cost"] else 0,
        })

    return {
        "cash": portfolio["cash"],
        "positions_value": positions_value,
        "total_value": portfolio["cash"] + positions_value,
        "positions": enriched,
    }


async def execute_buy(
    db: aiosqlite.Connection,
    symbol: str,
    shares: float,
    price: float,
    decision_id: int | None = None,
    is_dry_run: bool = False,
) -> dict[str, Any]:
    """Execute a buy order. Returns trade details or error."""
    portfolio = await queries.get_portfolio(db, is_dry_run=is_dry_run)
    if portfolio is None:
        return {"ok": False, "error": "Portfolio not initialized"}

    total_cost = shares * price

    # Check risk limits
    risk = check_risk(
        action="BUY",
        symbol=symbol,
        shares=shares,
        price=price,
        cash=portfolio["cash"],
        positions=await queries.get_open_positions(db, is_dry_run=is_dry_run),
        initial_cash=settings.initial_cash_chf,
    )
    if not risk["ok"]:
        logger.warning("Risk check failed for BUY %s: %s", symbol, risk["reason"])
        return risk

    # Use the potentially adjusted shares from risk check
    shares = risk.get("adjusted_shares", shares)
    total_cost = shares * price

    # Update cash
    new_cash = portfolio["cash"] - total_cost
    await queries.update_cash(db, new_cash, is_dry_run=is_dry_run)

    # Update or create position (RULE 005: set stop-loss at entry)
    existing = await queries.get_position_by_symbol(db, symbol, is_dry_run=is_dry_run)
    if existing:
        new_shares = existing["shares"] + shares
        new_avg_cost = (existing["shares"] * existing["avg_cost"] + total_cost) / new_shares
        stop_loss = new_avg_cost * (1 - settings.stop_loss_pct)
        await queries.update_position_shares(db, existing["id"], new_shares, new_avg_cost, stop_loss_price=stop_loss)
    else:
        stop_loss = price * (1 - settings.stop_loss_pct)
        await queries.open_position(db, symbol, shares, price, is_dry_run=is_dry_run, stop_loss_price=stop_loss)

    # Record trade
    trade_id = await queries.record_trade(
        db, symbol, "BUY", shares, price, decision_id, is_dry_run=is_dry_run
    )

    logger.info("BUY %s x%.2f @ %.2f = %.2f", symbol, shares, price, total_cost)
    return {"ok": True, "trade_id": trade_id, "shares": shares, "price": price, "total": total_cost}


async def execute_sell(
    db: aiosqlite.Connection,
    symbol: str,
    shares: float,
    price: float,
    decision_id: int | None = None,
    is_dry_run: bool = False,
) -> dict[str, Any]:
    """Execute a sell order. Returns trade details or error."""
    position = await queries.get_position_by_symbol(db, symbol, is_dry_run=is_dry_run)
    if position is None:
        return {"ok": False, "error": f"No open position for {symbol}"}

    if shares > position["shares"]:
        shares = position["shares"]  # Sell max available

    total_value = shares * price

    # Update cash
    portfolio = await queries.get_portfolio(db, is_dry_run=is_dry_run)
    if portfolio is None:
        return {"ok": False, "error": "Portfolio not initialized"}
    await queries.update_cash(db, portfolio["cash"] + total_value, is_dry_run=is_dry_run)

    # Update or close position
    remaining = position["shares"] - shares
    if remaining <= 0.001:  # Close position (float tolerance)
        await queries.close_position(db, position["id"], price)
    else:
        await queries.update_position_shares(db, position["id"], remaining, position["avg_cost"])

    # Record trade
    trade_id = await queries.record_trade(
        db, symbol, "SELL", shares, price, decision_id, is_dry_run=is_dry_run
    )

    pnl = (price - position["avg_cost"]) * shares
    logger.info("SELL %s x%.2f @ %.2f = %.2f (P&L: %.2f)", symbol, shares, price, total_value, pnl)
    return {"ok": True, "trade_id": trade_id, "shares": shares, "price": price, "total": total_value, "pnl": pnl}
