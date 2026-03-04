"""Currency helpers for USD/CHF conversion."""

from __future__ import annotations


def symbol_currency(symbol: str) -> str:
    """Return the native price currency for a symbol.

    Swiss SIX-listed stocks (.SW suffix) are priced in CHF.
    Everything else is assumed USD.
    """
    return "CHF" if symbol.upper().endswith(".SW") else "USD"


def to_chf(amount: float, currency: str, usd_chf_rate: float) -> float:
    """Convert an amount to CHF.

    CHF amounts pass through unchanged.
    USD amounts are multiplied by the USD/CHF exchange rate.
    """
    if currency == "CHF":
        return amount
    return amount * usd_chf_rate
