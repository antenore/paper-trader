from __future__ import annotations

from typing import Any

from paper_trader.config import settings
from paper_trader.portfolio.tools import check_sector_cap, get_sector


def check_risk(
    action: str,
    symbol: str,
    shares: float,
    price: float,
    cash: float,
    positions: list[dict[str, Any]],
    initial_cash: float | None = None,
) -> dict[str, Any]:
    """
    Run all risk checks before executing a trade.
    Returns {"ok": True} or {"ok": False, "reason": "..."}.
    May return {"ok": True, "adjusted_shares": N} if size was reduced.
    """
    if initial_cash is None:
        initial_cash = settings.initial_cash_chf

    # Sanity checks
    if shares <= 0:
        return {"ok": False, "reason": "Shares must be positive"}
    if price <= 0:
        return {"ok": False, "reason": "Price must be positive"}
    if action not in ("BUY", "SELL"):
        return {"ok": False, "reason": f"Invalid action: {action}"}

    # Only BUY needs risk checks (SELL is always allowed)
    if action == "SELL":
        return {"ok": True}

    total_cost = shares * price

    # 1. Safety stop: total portfolio value check
    positions_value = sum(p["shares"] * p.get("current_price", p["avg_cost"]) for p in positions)
    estimated_total = cash + positions_value
    safety_threshold = initial_cash * settings.safety_stop_pct

    if estimated_total < safety_threshold:
        return {
            "ok": False,
            "reason": f"Safety stop: portfolio ({estimated_total:.2f}) below {safety_threshold:.2f} ({settings.safety_stop_pct*100:.0f}% of initial)",
        }

    # 2. Sufficient cash
    if total_cost > cash:
        # Try reducing size to fit
        max_shares = cash / price
        if max_shares < 0.01:
            return {"ok": False, "reason": f"Insufficient cash ({cash:.2f}) for {symbol}"}
        shares = max_shares
        total_cost = shares * price

    # 3. Cash reserve: keep minimum cash
    if cash - total_cost < settings.min_cash_reserve:
        available = cash - settings.min_cash_reserve
        if available <= 0:
            return {
                "ok": False,
                "reason": f"Cash reserve: only {cash:.2f} available, need to keep {settings.min_cash_reserve:.2f}",
            }
        shares = available / price
        total_cost = shares * price

    # 4. Position limit: no single position > max_position_pct of portfolio
    portfolio_total = cash + positions_value
    max_position_value = portfolio_total * settings.max_position_pct

    # Check existing position + new purchase
    existing = next((p for p in positions if p["symbol"] == symbol), None)
    existing_value = (existing["shares"] * existing.get("current_price", existing["avg_cost"])) if existing else 0.0
    new_total_value = existing_value + total_cost

    if new_total_value > max_position_value:
        allowed_value = max_position_value - existing_value
        if allowed_value <= 0:
            return {
                "ok": False,
                "reason": f"Position limit: {symbol} already at max ({existing_value:.2f}/{max_position_value:.2f})",
            }
        shares = allowed_value / price
        total_cost = shares * price

    if shares < 0.01:
        return {"ok": False, "reason": "Trade too small after risk adjustments"}

    # 5. Sector cap: no sector > sector_cap_pct of portfolio (ETFs/Unknown exempt)
    sector_check = check_sector_cap(
        symbol, shares * price, positions,
        {p["symbol"]: p.get("current_price", p["avg_cost"]) for p in positions},
        portfolio_value=portfolio_total,
    )
    if not sector_check["ok"]:
        return sector_check

    return {"ok": True, "adjusted_shares": shares}
