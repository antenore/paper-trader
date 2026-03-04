from __future__ import annotations

import logging
from typing import Any

import aiosqlite

from paper_trader.config import get_tier_settings, settings
from paper_trader.db import queries
from paper_trader.portfolio.currency import symbol_currency, to_chf
from paper_trader.portfolio.risk import check_risk

logger = logging.getLogger(__name__)


async def get_portfolio_value(
    db: aiosqlite.Connection,
    prices: dict[str, float],
    is_dry_run: bool = False,
    usd_chf_rate: float = 1.0,
) -> dict[str, Any]:
    """Calculate total portfolio value including positions (all in CHF)."""
    portfolio = await queries.get_portfolio(db, is_dry_run=is_dry_run)
    if portfolio is None:
        return {"cash": 0.0, "positions_value": 0.0, "total_value": 0.0, "positions": []}

    positions = await queries.get_open_positions(db, is_dry_run=is_dry_run)
    positions_value = 0.0
    enriched = []
    for pos in positions:
        current_price = prices.get(pos["symbol"], pos["avg_cost"])
        currency = pos.get("currency") or symbol_currency(pos["symbol"])
        value_chf = to_chf(pos["shares"] * current_price, currency, usd_chf_rate)
        pnl = (current_price - pos["avg_cost"]) * pos["shares"]
        positions_value += value_chf
        enriched.append({
            **pos,
            "current_price": current_price,
            "currency": currency,
            "value": value_chf,
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
    risk_tier: str = "growth",
    usd_chf_rate: float = 1.0,
) -> dict[str, Any]:
    """Execute a buy order. Returns trade details or error.

    price is in the symbol's native currency (USD or CHF).
    Cash is deducted in CHF after FX conversion.
    avg_cost is stored in the native currency.
    """
    portfolio = await queries.get_portfolio(db, is_dry_run=is_dry_run)
    if portfolio is None:
        return {"ok": False, "error": "Portfolio not initialized"}

    currency = symbol_currency(symbol)
    cost_chf = to_chf(shares * price, currency, usd_chf_rate)

    # Enrich positions with current prices for risk checks
    positions = await queries.get_open_positions(db, is_dry_run=is_dry_run)

    # Check risk limits (tier-aware, in CHF)
    risk = check_risk(
        action="BUY",
        symbol=symbol,
        shares=shares,
        price=price,
        cash=portfolio["cash"],
        positions=positions,
        initial_cash=settings.initial_cash_chf,
        risk_tier=risk_tier,
        usd_chf_rate=usd_chf_rate,
    )
    if not risk["ok"]:
        logger.warning("Risk check failed for BUY %s (%s): %s", symbol, risk_tier, risk.get("reason"))
        return risk

    # Use the potentially adjusted shares from risk check
    shares = risk.get("adjusted_shares", shares)
    cost_chf = to_chf(shares * price, currency, usd_chf_rate)

    # Update cash (deduct in CHF)
    new_cash = portfolio["cash"] - cost_chf
    await queries.update_cash(db, new_cash, is_dry_run=is_dry_run)

    # Update or create position (RULE 005: tier-aware stop-loss at entry)
    # avg_cost stored in native currency (USD or CHF)
    tier_cfg = get_tier_settings(risk_tier)
    existing = await queries.get_position_by_symbol(db, symbol, is_dry_run=is_dry_run)
    if existing:
        new_shares = existing["shares"] + shares
        new_avg_cost = (existing["shares"] * existing["avg_cost"] + shares * price) / new_shares
        stop_loss = new_avg_cost * (1 - tier_cfg["stop_loss_pct"])
        await queries.update_position_shares(db, existing["id"], new_shares, new_avg_cost, stop_loss_price=stop_loss)
    else:
        stop_loss = price * (1 - tier_cfg["stop_loss_pct"])
        await queries.open_position(
            db, symbol, shares, price, is_dry_run=is_dry_run,
            stop_loss_price=stop_loss, risk_tier=risk_tier, currency=currency,
        )

    # Record trade
    trade_id = await queries.record_trade(
        db, symbol, "BUY", shares, price, decision_id, is_dry_run=is_dry_run
    )

    logger.info("BUY %s [%s] x%.2f @ %.2f (cost CHF %.2f)", symbol, risk_tier, shares, price, cost_chf)
    return {"ok": True, "trade_id": trade_id, "shares": shares, "price": price, "total": cost_chf}


async def execute_sell(
    db: aiosqlite.Connection,
    symbol: str,
    shares: float,
    price: float,
    decision_id: int | None = None,
    is_dry_run: bool = False,
    usd_chf_rate: float = 1.0,
) -> dict[str, Any]:
    """Execute a sell order. Returns trade details or error.

    price is in native currency. Proceeds are converted to CHF and added to cash.
    """
    position = await queries.get_position_by_symbol(db, symbol, is_dry_run=is_dry_run)
    if position is None:
        return {"ok": False, "error": f"No open position for {symbol}"}

    if shares > position["shares"]:
        shares = position["shares"]  # Sell max available

    currency = position.get("currency") or symbol_currency(symbol)
    proceeds_chf = to_chf(shares * price, currency, usd_chf_rate)

    # Update cash (add proceeds in CHF)
    portfolio = await queries.get_portfolio(db, is_dry_run=is_dry_run)
    if portfolio is None:
        return {"ok": False, "error": "Portfolio not initialized"}
    await queries.update_cash(db, portfolio["cash"] + proceeds_chf, is_dry_run=is_dry_run)

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
    logger.info("SELL %s x%.2f @ %.2f (proceeds CHF %.2f, P&L: %.2f)", symbol, shares, price, proceeds_chf, pnl)
    return {"ok": True, "trade_id": trade_id, "shares": shares, "price": price, "total": proceeds_chf, "pnl": pnl}
