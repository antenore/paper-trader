"""Trading tools: stop-loss, correlation, sector exposure, relative strength."""

from __future__ import annotations

import logging
from typing import Any

from paper_trader.config import settings
from paper_trader.market.prices import MarketDataProvider

logger = logging.getLogger(__name__)

# ── Static data ──────────────────────────────────────────────────────

SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "GOOG": "Technology", "META": "Technology",
    "AVGO": "Technology", "ADBE": "Technology", "CRM": "Technology",
    "AMD": "Technology", "INTC": "Technology", "CSCO": "Technology",
    "ORCL": "Technology", "NOW": "Technology", "INTU": "Technology",
    "AMAT": "Technology", "MU": "Technology", "LRCX": "Technology",
    "KLAC": "Technology", "SNPS": "Technology", "CDNS": "Technology",
    "MRVL": "Technology", "PANW": "Technology", "CRWD": "Technology",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "TJX": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",
    # Communication Services
    "NFLX": "Communication Services", "DIS": "Communication Services",
    "CMCSA": "Communication Services", "T": "Communication Services",
    "VZ": "Communication Services", "TMUS": "Communication Services",
    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare",
    "PFE": "Healthcare", "ABBV": "Healthcare", "MRK": "Healthcare",
    "TMO": "Healthcare", "ABT": "Healthcare", "AMGN": "Healthcare",
    "ISRG": "Healthcare",
    # Financials
    "BRK-B": "Financials", "JPM": "Financials", "V": "Financials",
    "MA": "Financials", "BAC": "Financials", "GS": "Financials",
    "MS": "Financials", "AXP": "Financials", "BLK": "Financials",
    # Industrials
    "CAT": "Industrials", "BA": "Industrials", "HON": "Industrials",
    "UPS": "Industrials", "GE": "Industrials", "RTX": "Industrials",
    "DE": "Industrials", "LMT": "Industrials",
    # Consumer Staples
    "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "COST": "Consumer Staples",
    "WMT": "Consumer Staples",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "EOG": "Energy",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    # Real Estate
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
    # ETFs
    "SPY": "ETF", "QQQ": "ETF", "XLK": "ETF", "XLE": "ETF",
    "XLF": "ETF", "XLV": "ETF", "XLI": "ETF", "XLP": "ETF",
    "XLY": "ETF", "XLU": "ETF", "XLRE": "ETF", "XLB": "ETF",
    "XLC": "ETF", "IWM": "ETF", "DIA": "ETF", "VOO": "ETF",
    "VTI": "ETF", "ARKK": "ETF", "SOXX": "ETF", "SMH": "ETF",
}

ETF_COMPONENTS: dict[str, list[str]] = {
    "QQQ": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "COST", "NFLX"],
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "CRM", "AMD", "ADBE", "ORCL", "CSCO", "INTC"],
    "SPY": [],  # Too broad to flag overlaps
}


def get_sector(symbol: str) -> str:
    return SECTOR_MAP.get(symbol, "Unknown")


def get_sector_exposure(
    positions: list[dict[str, Any]], prices: dict[str, float]
) -> dict[str, dict[str, Any]]:
    """Tool 3: Sector exposure breakdown.

    Returns {sector: {value, pct, symbols}} sorted by value descending.
    """
    sectors: dict[str, dict[str, Any]] = {}
    total_value = 0.0

    for pos in positions:
        symbol = pos["symbol"]
        price = prices.get(symbol, pos["avg_cost"])
        value = pos["shares"] * price
        total_value += value
        sector = get_sector(symbol)

        if sector not in sectors:
            sectors[sector] = {"value": 0.0, "pct": 0.0, "symbols": []}
        sectors[sector]["value"] += value
        sectors[sector]["symbols"].append(symbol)

    if total_value > 0:
        for data in sectors.values():
            data["pct"] = (data["value"] / total_value) * 100

    return dict(sorted(sectors.items(), key=lambda x: x[1]["value"], reverse=True))


def check_sector_cap(
    symbol: str,
    buy_value: float,
    positions: list[dict[str, Any]],
    prices: dict[str, float],
    cap_pct: float | None = None,
    portfolio_value: float | None = None,
) -> dict[str, Any]:
    """RULE 004: Check if buying would breach sector cap.

    ETFs and Unknown sectors are exempt.
    portfolio_value: total portfolio value (cash + positions). If None, computed from positions + buy.
    Returns {"ok": True} or {"ok": False, "reason": "..."}.
    """
    if cap_pct is None:
        cap_pct = settings.sector_cap_pct

    sector = get_sector(symbol)
    if sector in ("ETF", "Unknown"):
        return {"ok": True}

    # Compute total portfolio value and sector value
    positions_value = 0.0
    sector_value = buy_value  # Start with the proposed purchase

    for pos in positions:
        price = prices.get(pos["symbol"], pos["avg_cost"])
        pos_value = pos["shares"] * price
        positions_value += pos_value
        if get_sector(pos["symbol"]) == sector:
            sector_value += pos_value

    # Use provided portfolio_value (includes cash), or fallback to positions + buy
    total_value = portfolio_value if portfolio_value is not None else (positions_value + buy_value)

    if total_value <= 0:
        return {"ok": True}

    sector_pct = sector_value / total_value
    if sector_pct > cap_pct:
        return {
            "ok": False,
            "reason": (
                f"Sector cap: {sector} would be {sector_pct:.0%} of portfolio "
                f"(cap: {cap_pct:.0%})"
            ),
        }

    return {"ok": True}


def check_etf_overlap(symbol: str, position_symbols: list[str]) -> str | None:
    """RULE 003: Check for ETF double-dipping.

    Returns a warning message if overlap detected, None otherwise.
    """
    symbol_sector = get_sector(symbol)

    # Case 1: Buying an individual stock that overlaps with a held ETF
    if symbol_sector != "ETF":
        for etf, components in ETF_COMPONENTS.items():
            if etf in position_symbols and symbol in components:
                return f"{symbol} is a top-10 holding of {etf} (already held)"

    # Case 2: Buying an ETF when we hold its components
    if symbol in ETF_COMPONENTS:
        components = ETF_COMPONENTS[symbol]
        overlap = [s for s in position_symbols if s in components]
        if overlap:
            return f"{symbol} overlaps with held positions: {', '.join(overlap)}"

    return None


def check_stop_losses(
    positions: list[dict[str, Any]], prices: dict[str, float]
) -> list[dict[str, Any]]:
    """Tool 1: Check all positions against their stop-loss levels.

    Returns list of triggered stops: [{symbol, shares, stop_loss_price, current_price, loss_pct}].
    """
    triggered = []
    for pos in positions:
        stop = pos.get("stop_loss_price")
        if stop is None:
            continue
        symbol = pos["symbol"]
        current = prices.get(symbol)
        if current is None:
            continue
        if current <= stop:
            loss_pct = (current / pos["avg_cost"] - 1) * 100 if pos["avg_cost"] > 0 else 0
            triggered.append({
                "symbol": symbol,
                "shares": pos["shares"],
                "stop_loss_price": stop,
                "current_price": current,
                "loss_pct": loss_pct,
                "position_id": pos["id"],
            })
    return triggered


async def compute_correlation_matrix(
    symbols: list[str], market: MarketDataProvider
) -> dict[str, Any]:
    """Tool 2: Compute pairwise correlation matrix from 30-day returns.

    Returns {matrix: {sym: {sym: corr}}, avg_correlation, warning}.
    """
    import pandas as pd

    if len(symbols) < 2:
        return {"matrix": {}, "avg_correlation": 0.0, "warning": None}

    # Collect 30-day daily close data
    closes: dict[str, list[float]] = {}
    for symbol in symbols:
        history = await market.get_price_history(symbol, period="1mo", interval="1d")
        if len(history) >= 5:
            closes[symbol] = [h["close"] for h in history]

    if len(closes) < 2:
        return {"matrix": {}, "avg_correlation": 0.0, "warning": None}

    # Align lengths to the shortest series
    min_len = min(len(v) for v in closes.values())
    df = pd.DataFrame({s: v[-min_len:] for s, v in closes.items()})
    returns = df.pct_change().dropna()

    if len(returns) < 3:
        return {"matrix": {}, "avg_correlation": 0.0, "warning": None}

    corr = returns.corr()
    matrix = {s: {t: round(corr.loc[s, t], 3) for t in corr.columns} for s in corr.index}

    # Average pairwise correlation (upper triangle)
    pairs = []
    syms = list(corr.columns)
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            pairs.append(corr.iloc[i, j])
    avg = sum(pairs) / len(pairs) if pairs else 0.0

    threshold = settings.correlation_warn_threshold
    warning = None
    if avg > threshold:
        warning = f"High average correlation ({avg:.2f}) — portfolio may lack diversification"

    return {"matrix": matrix, "avg_correlation": round(avg, 3), "warning": warning}


async def compute_relative_strength(
    symbols: list[str], market: MarketDataProvider, benchmark: str = "SPY"
) -> list[dict[str, Any]]:
    """Tool 4: Relative strength vs benchmark over 20 days.

    Returns sorted list: [{symbol, rs_ratio, performance_pct, benchmark_pct}].
    """
    # Get benchmark performance
    bench_history = await market.get_price_history(benchmark, period="1mo", interval="1d")
    if len(bench_history) < 2:
        return []

    bench_start = bench_history[0]["close"]
    bench_end = bench_history[-1]["close"]
    bench_pct = (bench_end / bench_start - 1) * 100 if bench_start > 0 else 0

    results = []
    for symbol in symbols:
        if symbol == benchmark:
            continue
        history = await market.get_price_history(symbol, period="1mo", interval="1d")
        if len(history) < 2:
            continue
        start = history[0]["close"]
        end = history[-1]["close"]
        perf_pct = (end / start - 1) * 100 if start > 0 else 0
        rs_ratio = round((1 + perf_pct / 100) / (1 + bench_pct / 100), 3) if bench_pct != -100 else 0

        results.append({
            "symbol": symbol,
            "rs_ratio": rs_ratio,
            "performance_pct": round(perf_pct, 2),
            "benchmark_pct": round(bench_pct, 2),
        })

    results.sort(key=lambda x: x["rs_ratio"], reverse=True)
    return results


async def build_tools_context(
    positions: list[dict[str, Any]],
    prices: dict[str, float],
    watchlist_symbols: list[str],
    market: MarketDataProvider,
) -> str:
    """Aggregate all tools into a text block for AI prompts."""
    sections = []

    # === STOP-LOSS ALERTS ===
    triggered = check_stop_losses(positions, prices)
    if triggered:
        lines = ["=== STOP-LOSS ALERTS ==="]
        for t in triggered:
            lines.append(
                f"  {t['symbol']}: TRIGGERED at ${t['current_price']:.2f} "
                f"(stop: ${t['stop_loss_price']:.2f}, loss: {t['loss_pct']:.1f}%)"
            )
        sections.append("\n".join(lines))

    # === SECTOR EXPOSURE ===
    exposure = get_sector_exposure(positions, prices)
    if exposure:
        lines = ["=== SECTOR EXPOSURE ==="]
        for sector, data in exposure.items():
            syms = ", ".join(data["symbols"])
            lines.append(f"  {sector}: ${data['value']:.2f} ({data['pct']:.1f}%) — {syms}")
        sections.append("\n".join(lines))

    # === CORRELATION ===
    position_symbols = [p["symbol"] for p in positions]
    if len(position_symbols) >= 2:
        try:
            corr = await compute_correlation_matrix(position_symbols, market)
            if corr["matrix"]:
                lines = ["=== CORRELATION (held positions) ==="]
                lines.append(f"  Average pairwise correlation: {corr['avg_correlation']:.3f}")
                if corr["warning"]:
                    lines.append(f"  WARNING: {corr['warning']}")
                # Show highest pairs
                pairs = []
                syms = list(corr["matrix"].keys())
                for i in range(len(syms)):
                    for j in range(i + 1, len(syms)):
                        pairs.append((syms[i], syms[j], corr["matrix"][syms[i]][syms[j]]))
                pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                for s1, s2, c in pairs[:5]:
                    lines.append(f"  {s1}/{s2}: {c:.3f}")
                sections.append("\n".join(lines))
        except Exception:
            logger.debug("Correlation computation failed", exc_info=True)

    # === RELATIVE STRENGTH ===
    all_symbols = list(set(position_symbols + watchlist_symbols))
    if all_symbols:
        try:
            rs = await compute_relative_strength(all_symbols, market)
            if rs:
                lines = ["=== RELATIVE STRENGTH (vs SPY, 1mo) ==="]
                for r in rs[:10]:
                    held = " [HELD]" if r["symbol"] in position_symbols else ""
                    lines.append(
                        f"  {r['symbol']}: RS={r['rs_ratio']:.3f} "
                        f"(stock: {r['performance_pct']:+.1f}%, SPY: {r['benchmark_pct']:+.1f}%){held}"
                    )
                sections.append("\n".join(lines))
        except Exception:
            logger.debug("Relative strength computation failed", exc_info=True)

    # === ETF OVERLAP WARNINGS ===
    overlap_warnings = []
    for symbol in watchlist_symbols:
        warning = check_etf_overlap(symbol, position_symbols)
        if warning:
            overlap_warnings.append(f"  {symbol}: {warning}")
    if overlap_warnings:
        sections.append("=== ETF OVERLAP WARNINGS ===\n" + "\n".join(overlap_warnings))

    return "\n\n".join(sections)
