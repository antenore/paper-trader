"""IBKR commission model and share rounding constraints."""

from __future__ import annotations

import math


def calculate_commission(
    symbol: str,
    shares: float,
    price: float,
    usd_chf_rate: float = 1.0,
) -> float:
    """Calculate trading commission in CHF using IBKR fee schedule.

    US stocks: max($0.005 * shares, $1.00), capped at 1% of trade value, converted to CHF.
    SIX (.SW): max(0.05% * value, CHF 1.50).
    """
    if shares <= 0 or price <= 0:
        return 0.0

    if symbol.upper().endswith(".SW"):
        # SIX: 0.05% of trade value, minimum CHF 1.50
        value = shares * price
        return max(value * 0.0005, 1.50)
    else:
        # US: $0.005/share, min $1.00, max 1% of trade value
        trade_value_usd = shares * price
        fee_usd = max(0.005 * shares, 1.00)
        fee_usd = min(fee_usd, trade_value_usd * 0.01)
        return fee_usd * usd_chf_rate


# Estimated slippage in basis points (1 bp = 0.01%)
_SLIPPAGE_BPS_US = 5     # US large-cap: ~0.05%
_SLIPPAGE_BPS_SIX = 10   # SIX: ~0.10% (less liquid)


def apply_slippage(symbol: str, price: float, action: str) -> float:
    """Adjust price for estimated execution slippage.

    BUY: price goes UP (you pay more than the mid-quote).
    SELL: price goes DOWN (you receive less than the mid-quote).
    """
    if price <= 0:
        return price

    bps = _SLIPPAGE_BPS_SIX if symbol.upper().endswith(".SW") else _SLIPPAGE_BPS_US
    factor = bps / 10_000

    if action == "BUY":
        return price * (1 + factor)
    else:
        return price * (1 - factor)


def round_shares(symbol: str, shares: float) -> float:
    """Round shares to broker constraints.

    US stocks: round to 4 decimals (fractional OK).
    SIX (.SW): whole shares only (floor).
    """
    if shares <= 0:
        return 0.0

    if symbol.upper().endswith(".SW"):
        return float(math.floor(shares))
    else:
        return round(shares, 4)
