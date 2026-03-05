from __future__ import annotations

from typing import Any

from paper_trader.config import get_tier_settings, settings
from paper_trader.portfolio.commission import calculate_commission, round_shares
from paper_trader.portfolio.currency import symbol_currency, to_chf
from paper_trader.portfolio.tools import check_sector_cap, get_sector


def _pos_value_chf(pos: dict[str, Any], usd_chf_rate: float) -> float:
    """Position market value in CHF."""
    price = pos.get("current_price", pos["avg_cost"])
    currency = pos.get("currency") or symbol_currency(pos["symbol"])
    return to_chf(pos["shares"] * price, currency, usd_chf_rate)


def check_risk(
    action: str,
    symbol: str,
    shares: float,
    price: float,
    cash: float,
    positions: list[dict[str, Any]],
    initial_cash: float | None = None,
    risk_tier: str = "growth",
    usd_chf_rate: float = 1.0,
) -> dict[str, Any]:
    """
    Run all risk checks before executing a trade.
    All value comparisons are in CHF (cash is already CHF).
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

    currency = symbol_currency(symbol)
    cost_chf = to_chf(shares * price, currency, usd_chf_rate)

    # 1. Safety stop: total portfolio value check (all in CHF)
    positions_value_chf = sum(_pos_value_chf(p, usd_chf_rate) for p in positions)
    estimated_total = cash + positions_value_chf
    safety_threshold = initial_cash * settings.safety_stop_pct

    if estimated_total < safety_threshold:
        return {
            "ok": False,
            "reason": f"Safety stop: portfolio ({estimated_total:.2f}) below {safety_threshold:.2f} ({settings.safety_stop_pct*100:.0f}% of initial)",
        }

    # 2. Sufficient cash (cost in CHF vs CHF cash)
    if cost_chf > cash:
        # Try reducing size to fit
        price_chf = to_chf(price, currency, usd_chf_rate)
        max_shares = cash / price_chf if price_chf > 0 else 0
        if max_shares < 0.01:
            return {"ok": False, "reason": f"Insufficient cash ({cash:.2f} CHF) for {symbol}"}
        shares = max_shares
        cost_chf = to_chf(shares * price, currency, usd_chf_rate)

    # 3. Cash reserve: keep minimum cash
    if cash - cost_chf < settings.min_cash_reserve:
        available = cash - settings.min_cash_reserve
        if available <= 0:
            return {
                "ok": False,
                "reason": f"Cash reserve: only {cash:.2f} available, need to keep {settings.min_cash_reserve:.2f}",
            }
        price_chf = to_chf(price, currency, usd_chf_rate)
        shares = available / price_chf if price_chf > 0 else 0
        cost_chf = to_chf(shares * price, currency, usd_chf_rate)

    # 4. Position limit: tier-aware (growth: 25%, moonshot: 10%)
    portfolio_total = cash + positions_value_chf
    tier_cfg = get_tier_settings(risk_tier)
    max_position_value = portfolio_total * tier_cfg["max_position_pct"]

    # Check existing position + new purchase (in CHF)
    existing = next((p for p in positions if p["symbol"] == symbol), None)
    existing_value_chf = _pos_value_chf(existing, usd_chf_rate) if existing else 0.0
    new_total_value = existing_value_chf + cost_chf

    if new_total_value > max_position_value:
        allowed_value = max_position_value - existing_value_chf
        if allowed_value <= 0:
            return {
                "ok": False,
                "reason": f"Position limit ({risk_tier}): {symbol} already at max ({existing_value_chf:.2f}/{max_position_value:.2f})",
            }
        price_chf = to_chf(price, currency, usd_chf_rate)
        shares = allowed_value / price_chf if price_chf > 0 else 0
        cost_chf = to_chf(shares * price, currency, usd_chf_rate)

    # Apply broker share constraints (SIX: whole shares, US: 4 decimals)
    shares = round_shares(symbol, shares)
    if shares <= 0:
        return {"ok": False, "reason": f"Trade too small for whole share of {symbol}"}

    cost_chf = to_chf(shares * price, currency, usd_chf_rate)

    if shares < 0.01:
        return {"ok": False, "reason": "Trade too small after risk adjustments"}

    # 4b. Bucket limit: total allocation per tier (growth: 65%, moonshot: 20%)
    tier_total = sum(
        _pos_value_chf(p, usd_chf_rate)
        for p in positions
        if p.get("risk_tier", "growth") == risk_tier
    )
    max_bucket_value = portfolio_total * tier_cfg["max_bucket_pct"]
    allowed_bucket = max_bucket_value - tier_total
    if tier_total + cost_chf > max_bucket_value:
        if allowed_bucket < 1.0:
            # Bucket already at/over limit (price appreciation can push existing
            # positions past the cap) — block new buys into this tier.
            return {
                "ok": False,
                "reason": f"Bucket limit ({risk_tier}): tier at {tier_total:.2f}/{max_bucket_value:.2f}",
            }
        price_chf = to_chf(price, currency, usd_chf_rate)
        shares = min(shares, allowed_bucket / price_chf if price_chf > 0 else 0)
        cost_chf = to_chf(shares * price, currency, usd_chf_rate)

    # Apply broker share constraints after bucket limit
    shares = round_shares(symbol, shares)
    if shares <= 0:
        return {"ok": False, "reason": f"Trade too small for whole share of {symbol}"}

    cost_chf = to_chf(shares * price, currency, usd_chf_rate)

    if shares < 0.01:
        return {"ok": False, "reason": "Trade too small after risk adjustments"}

    # 5. Sector cap: no sector > sector_cap_pct of portfolio (ETFs/Unknown exempt)
    sector_check = check_sector_cap(
        symbol, cost_chf, positions,
        {p["symbol"]: p.get("current_price", p["avg_cost"]) for p in positions},
        portfolio_value=portfolio_total,
        usd_chf_rate=usd_chf_rate,
    )
    if not sector_check["ok"]:
        return sector_check

    # 6. Final commission-aware cash check
    estimated_commission = calculate_commission(symbol, shares, price, usd_chf_rate)
    if cash - cost_chf - estimated_commission < 0:
        # Reduce shares to fit with commission
        price_chf = to_chf(price, currency, usd_chf_rate)
        if price_chf > 0:
            # Iteratively find shares that fit (commission depends on shares)
            max_shares = shares
            for _ in range(5):
                comm = calculate_commission(symbol, max_shares, price, usd_chf_rate)
                affordable = cash - comm
                max_shares = affordable / price_chf if price_chf > 0 else 0
                max_shares = round_shares(symbol, max_shares)
                if max_shares <= 0:
                    return {"ok": False, "reason": "Insufficient cash after commission"}
            shares = max_shares
            cost_chf = to_chf(shares * price, currency, usd_chf_rate)

    return {"ok": True, "adjusted_shares": shares}
