"""Trading tools: stop-loss, correlation, sector exposure, relative strength, churn detection, profit-taking alerts, geopolitical catalyst monitor."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from paper_trader.config import get_tier_settings, settings
from paper_trader.market.prices import MarketDataProvider
from paper_trader.portfolio.currency import symbol_currency, to_chf

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
    # Commodities / Safe-Haven
    "GLD": "Commodities", "SLV": "Commodities", "GDX": "Commodities",
    "USO": "Commodities", "TLT": "Treasuries", "IEF": "Treasuries",
    # ETFs
    "SPY": "ETF", "QQQ": "ETF", "XLK": "ETF", "XLE": "ETF",
    "XLF": "ETF", "XLV": "ETF", "XLI": "ETF", "XLP": "ETF",
    "XLY": "ETF", "XLU": "ETF", "XLRE": "ETF", "XLB": "ETF",
    "XLC": "ETF", "IWM": "ETF", "DIA": "ETF", "VOO": "ETF",
    "VTI": "ETF", "ARKK": "ETF", "SOXX": "ETF", "SMH": "ETF",
    # Swiss SIX (.SW suffix — priced in CHF)
    "NESN.SW": "Consumer Staples", "NOVN.SW": "Healthcare",
    "ROG.SW": "Healthcare", "UBSG.SW": "Financials",
    "ABBN.SW": "Industrials", "SREN.SW": "Financials",
    "ZURN.SW": "Financials", "GIVN.SW": "Consumer Staples",
    "SIKA.SW": "Materials", "LONN.SW": "Healthcare",
    "HOLN.SW": "Materials", "PGHN.SW": "Financials",
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
    usd_chf_rate: float = 1.0,
) -> dict[str, Any]:
    """RULE 004: Check if buying would breach sector cap.

    ETFs and Unknown sectors are exempt.
    buy_value: cost in CHF.
    portfolio_value: total portfolio value in CHF (cash + positions). If None, computed from positions + buy.
    Returns {"ok": True} or {"ok": False, "reason": "..."}.
    """
    from paper_trader.portfolio.currency import symbol_currency, to_chf

    if cap_pct is None:
        cap_pct = settings.sector_cap_pct

    sector = get_sector(symbol)
    if sector in ("ETF", "Unknown"):
        return {"ok": True}

    # Compute total portfolio value and sector value (all in CHF)
    positions_value = 0.0
    sector_value = buy_value  # Already in CHF from caller

    for pos in positions:
        price = prices.get(pos["symbol"], pos["avg_cost"])
        currency = pos.get("currency") or symbol_currency(pos["symbol"])
        pos_value = to_chf(pos["shares"] * price, currency, usd_chf_rate)
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


def _pos_value_chf(pos: dict[str, Any], prices: dict[str, float], usd_chf_rate: float = 1.0) -> float:
    """Position market value in CHF using live prices."""
    price = prices.get(pos["symbol"], pos.get("current_price", pos["avg_cost"]))
    currency = pos.get("currency") or symbol_currency(pos["symbol"])
    return to_chf(pos["shares"] * price, currency, usd_chf_rate)


def detect_churn(
    churn_candidates: list[dict[str, Any]],
    cooloff_hours: int = 48,
) -> list[dict[str, Any]]:
    """Analyze churn candidates and determine cooling-off status.

    Each candidate has: symbol, trade_count, buys, sells, total_commission, last_sell_at.
    Returns list of churn alerts with cooling-off info.
    """
    now = datetime.now(timezone.utc)
    alerts: list[dict[str, Any]] = []

    for c in churn_candidates:
        round_trips = min(c["buys"], c["sells"])
        est_commission = c["total_commission"] or 0.0

        # Parse last_sell_at and check cooling-off
        cooloff_remaining_h = 0.0
        in_cooloff = False
        if c["last_sell_at"]:
            try:
                # SQLite datetime format: "YYYY-MM-DD HH:MM:SS"
                last_sell = datetime.fromisoformat(c["last_sell_at"]).replace(tzinfo=timezone.utc)
                elapsed_h = (now - last_sell).total_seconds() / 3600
                cooloff_remaining_h = max(0, cooloff_hours - elapsed_h)
                in_cooloff = cooloff_remaining_h > 0
            except (ValueError, TypeError):
                pass

        alerts.append({
            "symbol": c["symbol"],
            "round_trips": round_trips,
            "total_trades": c["trade_count"],
            "commission_cost": round(est_commission, 2),
            "in_cooloff": in_cooloff,
            "cooloff_remaining_h": round(cooloff_remaining_h, 1),
            "last_sell_at": c["last_sell_at"],
        })

    return alerts


def compute_currency_attribution(
    benchmark_summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Compute USD vs CHF performance attribution from benchmark summary.

    Returns breakdown: portfolio_chf_return, spy_usd_return, fx_effect, spy_chf_return, alpha_chf.
    """
    if benchmark_summary is None:
        return None

    portfolio_return = benchmark_summary.get("portfolio_return_pct", 0)
    spy_return = benchmark_summary.get("spy_return_pct", 0)

    initial_fx = benchmark_summary.get("initial_fx_rate")
    current_fx = benchmark_summary.get("current_fx_rate")

    if not initial_fx or not current_fx:
        return None

    fx_change_pct = ((current_fx / initial_fx) - 1) * 100
    # SPY return in CHF terms = SPY USD return + FX effect (compounded)
    spy_chf_return = ((1 + spy_return / 100) * (1 + fx_change_pct / 100) - 1) * 100

    return {
        "portfolio_chf_return_pct": round(portfolio_return, 2),
        "spy_usd_return_pct": round(spy_return, 2),
        "fx_change_pct": round(fx_change_pct, 2),
        "spy_chf_return_pct": round(spy_chf_return, 2),
        "alpha_chf_pct": round(portfolio_return - spy_chf_return, 2),
        "alpha_usd_pct": round(portfolio_return - spy_return, 2),
        "fx_drag_pct": round(spy_chf_return - spy_return, 2),
    }


# ── Geopolitical catalyst definitions ────────────────────────────────
# Each catalyst has: keywords (for news scanning), affected symbols, and
# separate escalation/de-escalation keyword sets.

GEOPOLITICAL_CATALYSTS: dict[str, dict[str, Any]] = {
    "Iran-Israel conflict": {
        "keywords": ["iran", "israel", "tehran", "idf", "irgc", "hezbollah", "houthi",
                      "strait of hormuz", "persian gulf", "middle east"],
        "escalation": ["strike", "attack", "bomb", "missile", "retaliat", "escalat",
                        "mobiliz", "offensive", "casualties", "invasion", "nuclear",
                        "sanction", "embargo", "blockade", "shut down", "intercept"],
        "de_escalation": ["ceasefire", "truce", "peace", "negotiat", "diplomati",
                          "de-escalat", "withdraw", "agreement", "talks", "summit",
                          "relief", "easing", "humanitarian", "deal"],
        "affected_symbols": ["USO", "XLE", "CVX", "XOM", "COP", "SLB", "EOG",
                             "LMT", "RTX", "GLD", "GDX"],
    },
    "Taiwan Strait tensions": {
        "keywords": ["taiwan", "taipei", "tsmc", "south china sea", "china military",
                      "pla navy", "strait"],
        "escalation": ["exercise", "blockade", "incursion", "fighter jet", "warship",
                        "drill", "mobiliz", "invasion", "sanction"],
        "de_escalation": ["diplomati", "talks", "de-escalat", "cooperat", "stabiliz",
                          "agreement", "trade deal"],
        "affected_symbols": ["TSM", "NVDA", "AMD", "AVGO", "INTC", "AMAT",
                             "LRCX", "KLAC", "SMH", "SOXX"],
    },
    "US-China trade war": {
        "keywords": ["tariff", "trade war", "china trade", "us china", "beijing",
                      "import duty", "export ban", "chip ban"],
        "escalation": ["tariff", "ban", "restrict", "retali", "escalat", "blacklist",
                        "sanction", "embargo", "export control"],
        "de_escalation": ["trade deal", "exemption", "waiver", "negotiat", "agreement",
                          "cooperat", "ease", "lift", "remove tariff", "phase"],
        "affected_symbols": ["AAPL", "NVDA", "TSLA", "AMZN", "QQQ", "SPY",
                             "FXI", "EEM"],
    },
    "Energy supply disruption": {
        "keywords": ["opec", "oil supply", "crude oil", "oil price", "natural gas",
                      "pipeline", "refinery", "energy crisis", "oil production"],
        "escalation": ["cut", "disrupt", "shortage", "surge", "spike", "halt",
                        "outage", "shut", "reduce production", "quota"],
        "de_escalation": ["increase production", "boost output", "stabiliz", "ease",
                          "oversupply", "glut", "release reserve", "strategic reserve"],
        "affected_symbols": ["USO", "XLE", "CVX", "XOM", "COP", "SLB", "EOG"],
    },
}


def check_profit_taking_alerts(
    positions: list[dict[str, Any]],
    prices: dict[str, float],
    threshold_pct: float = 0.25,
    sell_pct: float = 0.33,
    usd_chf_rate: float = 1.0,
) -> list[dict[str, Any]]:
    """RULE 014: Check positions for unrealized gains exceeding threshold.

    Returns list of alerts with recommended partial-exit quantities.
    """
    alerts: list[dict[str, Any]] = []
    for pos in positions:
        symbol = pos["symbol"]
        current = prices.get(symbol)
        if current is None:
            continue

        entry = pos["avg_cost"]
        if entry <= 0:
            continue

        gain_pct = (current / entry) - 1.0
        if gain_pct < threshold_pct:
            continue

        currency = pos.get("currency") or symbol_currency(symbol)
        position_value = to_chf(pos["shares"] * current, currency, usd_chf_rate)
        recommended_sell_shares = round(pos["shares"] * sell_pct, 4)
        recommended_sell_value = to_chf(recommended_sell_shares * current, currency, usd_chf_rate)

        alerts.append({
            "symbol": symbol,
            "gain_pct": round(gain_pct * 100, 2),
            "entry_price": round(entry, 2),
            "current_price": round(current, 2),
            "shares": pos["shares"],
            "position_value_chf": round(position_value, 2),
            "recommended_sell_shares": recommended_sell_shares,
            "recommended_sell_value_chf": round(recommended_sell_value, 2),
            "threshold_pct": round(threshold_pct * 100, 1),
        })

    # Sort by gain descending — biggest winners first
    alerts.sort(key=lambda x: x["gain_pct"], reverse=True)
    return alerts


def check_geopolitical_catalysts(
    news_items: list[dict[str, Any]],
    position_symbols: list[str],
) -> list[dict[str, Any]]:
    """Scan recent news for geopolitical catalyst signals.

    Returns list of catalyst alerts with escalation/de-escalation signals
    and confidence modifiers for affected held positions.
    """
    alerts: list[dict[str, Any]] = []

    for catalyst_name, catalyst in GEOPOLITICAL_CATALYSTS.items():
        # Check if any news matches this catalyst's keywords
        matching_headlines: list[str] = []
        escalation_hits = 0
        de_escalation_hits = 0

        for item in news_items:
            title = (item.get("title") or "").lower()
            summary = (item.get("summary") or "").lower()
            text = f"{title} {summary}"

            # Check if this news item relates to this catalyst
            if not any(kw in text for kw in catalyst["keywords"]):
                continue

            matching_headlines.append(item.get("title", "")[:100])

            # Count escalation vs de-escalation signals
            for esc_kw in catalyst["escalation"]:
                if esc_kw in text:
                    escalation_hits += 1
            for de_kw in catalyst["de_escalation"]:
                if de_kw in text:
                    de_escalation_hits += 1

        if not matching_headlines:
            continue

        # Determine signal direction and confidence modifier
        total_signals = escalation_hits + de_escalation_hits
        if total_signals == 0:
            signal = "NEUTRAL"
            confidence_modifier = 0.0
        elif escalation_hits > de_escalation_hits:
            ratio = escalation_hits / total_signals
            signal = "ESCALATION"
            # +0.05 to +0.10 boost for positions that benefit from escalation
            confidence_modifier = round(min(0.10, ratio * 0.10), 2)
        else:
            ratio = de_escalation_hits / total_signals
            signal = "DE-ESCALATION"
            # -0.05 to -0.10 for positions whose thesis depends on escalation
            confidence_modifier = round(-min(0.10, ratio * 0.10), 2)

        # Find which held positions are affected
        affected_held = [s for s in position_symbols if s in catalyst["affected_symbols"]]

        alerts.append({
            "catalyst": catalyst_name,
            "signal": signal,
            "escalation_hits": escalation_hits,
            "de_escalation_hits": de_escalation_hits,
            "confidence_modifier": confidence_modifier,
            "affected_symbols": catalyst["affected_symbols"],
            "affected_held": affected_held,
            "headline_count": len(matching_headlines),
            "sample_headlines": matching_headlines[:3],
        })

    return alerts


def recommend_position_consolidation(
    positions: list[dict[str, Any]],
    prices: dict[str, float],
    confidence_map: dict[str, float],
    usd_chf_rate: float = 1.0,
    target_count: int = 6,
) -> list[dict[str, Any]]:
    """Score positions and recommend consolidation for small/low-conviction holdings.

    Score = (confidence × value_chf) / (commission_drag_pct × 100).
    Higher score = better position to keep.
    """
    from paper_trader.portfolio.commission import calculate_commission

    scored: list[dict[str, Any]] = []
    for pos in positions:
        symbol = pos["symbol"]
        price = prices.get(symbol)
        if price is None:
            continue

        currency = pos.get("currency") or symbol_currency(symbol)
        value_chf = to_chf(pos["shares"] * price, currency, usd_chf_rate)
        confidence = confidence_map.get(symbol, 0.50)  # default to neutral

        # Estimate commission as % of position value
        comm = calculate_commission(symbol, pos["shares"], price, usd_chf_rate)
        comm_pct = (comm / value_chf * 100) if value_chf > 0 else 5.0

        # Score: high confidence + large position + low commission drag → keep
        score = (confidence * value_chf) / max(comm_pct * 100, 1.0)

        # P&L
        entry = pos.get("avg_cost", price)
        pnl_pct = ((price / entry) - 1.0) * 100 if entry > 0 else 0.0

        scored.append({
            "symbol": symbol,
            "score": round(score, 2),
            "confidence": round(confidence, 2),
            "value_chf": round(value_chf, 2),
            "commission_drag_pct": round(comm_pct, 2),
            "pnl_pct": round(pnl_pct, 2),
            "recommendation": "",  # filled below
        })

    # Sort by score descending (best positions first)
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Top N → KEEP, rest → TRIM
    for i, s in enumerate(scored):
        if i < target_count:
            s["recommendation"] = "KEEP"
        else:
            s["recommendation"] = "TRIM — low score, consider selling to reduce complexity"

    return scored


async def build_tools_context(
    positions: list[dict[str, Any]],
    prices: dict[str, float],
    watchlist_symbols: list[str],
    market: MarketDataProvider,
    portfolio_cash: float = 0.0,
    usd_chf_rate: float = 1.0,
    total_commissions: float | None = None,
    churn_alerts: list[dict[str, Any]] | None = None,
    currency_attribution: dict[str, Any] | None = None,
    symbol_trade_summary: list[dict[str, Any]] | None = None,
    news_items: list[dict[str, Any]] | None = None,
    confidence_map: dict[str, float] | None = None,
    signal_report: str = "",
    performance_report: str = "",
) -> str:
    """Aggregate all tools into a text block for AI prompts."""
    sections = []

    # === SIGNAL QUALITY REPORT === (show FIRST so AI sees it immediately)
    if signal_report:
        sections.append(signal_report)

    # === PORTFOLIO PERFORMANCE METRICS === (right after signals)
    if performance_report:
        sections.append(performance_report)

    # === POSITION CONSOLIDATION RECOMMENDER ===
    if positions and confidence_map:
        consolidation = recommend_position_consolidation(
            positions, prices, confidence_map,
            usd_chf_rate=usd_chf_rate,
        )
        trim_candidates = [c for c in consolidation if c["recommendation"].startswith("TRIM")]
        if trim_candidates:
            lines = ["=== POSITION CONSOLIDATION RECOMMENDER ==="]
            lines.append("  Positions ranked by composite score (confidence × value / commission drag):")
            for c in consolidation:
                marker = " >>> TRIM" if c["recommendation"].startswith("TRIM") else ""
                lines.append(
                    f"  {c['symbol']:6s} | score={c['score']:7.1f} | conf={c['confidence']:.2f} | "
                    f"CHF {c['value_chf']:7.2f} | comm_drag={c['commission_drag_pct']:.1f}% | "
                    f"P&L={c['pnl_pct']:+.1f}%{marker}"
                )
            lines.append(
                f"  >>> {len(trim_candidates)} position(s) scored below threshold — "
                f"consider trimming to reduce complexity and commission drag."
            )
            sections.append("\n".join(lines))

    # === PROFIT-TAKING ALERTS (RULE 014) ===
    profit_alerts = check_profit_taking_alerts(
        positions, prices,
        threshold_pct=settings.profit_taking_threshold_pct,
        sell_pct=settings.profit_taking_sell_pct,
        usd_chf_rate=usd_chf_rate,
    )
    if profit_alerts:
        lines = ["=== PROFIT-TAKING ALERTS (RULE 014) ==="]
        sell_pct_display = round(settings.profit_taking_sell_pct * 100)
        for a in profit_alerts:
            lines.append(
                f"  {a['symbol']}: +{a['gain_pct']:.1f}% unrealized gain "
                f"(entry: {a['entry_price']:.2f}, now: {a['current_price']:.2f})"
            )
            lines.append(
                f"    Position: {a['shares']:.4f} shares = CHF {a['position_value_chf']:.2f}"
            )
            lines.append(
                f"    >>> RECOMMENDED: Sell {a['recommended_sell_shares']:.4f} shares "
                f"({sell_pct_display}%) = ~CHF {a['recommended_sell_value_chf']:.2f} to lock in gains"
            )
        lines.append(
            f"  Threshold: {a['threshold_pct']:.0f}% — any position above this triggers mandatory review"
        )
        sections.append("\n".join(lines))

    # === GEOPOLITICAL CATALYST MONITOR ===
    if news_items:
        position_symbols = [p["symbol"] for p in positions]
        geo_alerts = check_geopolitical_catalysts(news_items, position_symbols)
        if geo_alerts:
            lines = ["=== GEOPOLITICAL CATALYST MONITOR ==="]
            for ga in geo_alerts:
                signal_icon = {
                    "ESCALATION": "ESCALATION",
                    "DE-ESCALATION": "DE-ESCALATION",
                    "NEUTRAL": "NEUTRAL",
                }[ga["signal"]]
                lines.append(
                    f"  {ga['catalyst']}: {signal_icon} "
                    f"(esc:{ga['escalation_hits']} / de-esc:{ga['de_escalation_hits']}, "
                    f"{ga['headline_count']} matching headlines)"
                )
                if ga["confidence_modifier"] != 0:
                    lines.append(
                        f"    Confidence modifier: {ga['confidence_modifier']:+.2f} "
                        f"for affected positions"
                    )
                if ga["affected_held"]:
                    lines.append(
                        f"    YOUR POSITIONS AFFECTED: {', '.join(ga['affected_held'])}"
                    )
                    if ga["signal"] == "DE-ESCALATION" and ga["confidence_modifier"] < 0:
                        lines.append(
                            f"    >>> SELL REVIEW FLAG: De-escalation threatens thesis "
                            f"for {', '.join(ga['affected_held'])}. Review exit."
                        )
                else:
                    lines.append(
                        f"    Affected symbols (not held): {', '.join(ga['affected_symbols'][:5])}"
                    )
                # Sample headlines
                for headline in ga["sample_headlines"]:
                    lines.append(f"      - {headline}")
            sections.append("\n".join(lines))

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

    # === TIER ALLOCATION (bucket limits) ===
    if positions:
        lines = ["=== TIER ALLOCATION (bucket limits) ==="]
        portfolio_total = sum(
            _pos_value_chf(p, prices, usd_chf_rate) for p in positions
        ) + portfolio_cash
        for tier in ("growth", "moonshot"):
            tier_cfg = get_tier_settings(tier)
            tier_value = sum(
                _pos_value_chf(p, prices, usd_chf_rate)
                for p in positions
                if p.get("risk_tier", "growth") == tier
            )
            max_value = portfolio_total * tier_cfg["max_bucket_pct"]
            remaining = max(0, max_value - tier_value)
            pct = (tier_value / portfolio_total * 100) if portfolio_total > 0 else 0
            max_pct = tier_cfg["max_bucket_pct"] * 100
            status = "FULL" if remaining < 1.0 else f"{remaining:.2f} CHF remaining"
            lines.append(
                f"  {tier}: {tier_value:.2f} CHF ({pct:.1f}%) of {max_value:.2f} CHF ({max_pct:.0f}% cap) — {status}"
            )
        sections.append("\n".join(lines))

    # === ETF OVERLAP WARNINGS ===
    overlap_warnings = []
    for symbol in watchlist_symbols:
        warning = check_etf_overlap(symbol, position_symbols)
        if warning:
            overlap_warnings.append(f"  {symbol}: {warning}")
    if overlap_warnings:
        sections.append("=== ETF OVERLAP WARNINGS ===\n" + "\n".join(overlap_warnings))

    # === COMMISSION SUMMARY ===
    if total_commissions is not None and total_commissions > 0:
        sections.append(
            f"=== COMMISSION SUMMARY ===\n"
            f"  Total commissions paid: CHF {total_commissions:.2f}"
        )

    # === SYMBOL P&L SCORECARD (7 business days) ===
    if symbol_trade_summary:
        lines = ["=== SYMBOL P&L SCORECARD (last 7 business days) ==="]
        lines.append("  Symbol | Round-trips | Commissions | Net P&L CHF | Verdict")
        lines.append("  " + "-" * 70)
        for s in symbol_trade_summary:
            rt = min(s["buys"], s["sells"])
            verdict = "LOSING MONEY" if s["net_pnl_chf"] < 0 else "OK"
            if rt >= 2 and s["net_pnl_chf"] < 0:
                verdict = "STOP TRADING THIS"
            lines.append(
                f"  {s['symbol']:6s} | {rt:11d} | CHF {s['total_commission_chf']:7.2f} | "
                f"CHF {s['net_pnl_chf']:+8.2f} | {verdict}"
            )
        total_pnl = sum(s["net_pnl_chf"] for s in symbol_trade_summary)
        total_comm = sum(s["total_commission_chf"] for s in symbol_trade_summary)
        lines.append(f"  TOTAL round-trip P&L: CHF {total_pnl:+.2f} (commissions: CHF {total_comm:.2f})")
        if total_pnl < 0:
            lines.append(f"  >>> You LOST CHF {abs(total_pnl):.2f} from round-trip trading. REDUCE ACTIVITY. <<<")
        sections.append("\n".join(lines))

    # === CHURN ALERTS (RULE 010) ===
    if churn_alerts:
        cooloff_h = settings.churn_cooloff_hours
        churn_thresh = settings.churn_confidence_threshold
        lines = ["=== CHURN ALERTS (RULE 010) ==="]
        for a in churn_alerts:
            status = (
                f"COOLING OFF ({a['cooloff_remaining_h']:.0f}h remaining — DO NOT BUY)"
                if a["in_cooloff"]
                else f"cooloff expired — re-entry requires confidence >= {churn_thresh:.2f} and NEW catalyst"
            )
            lines.append(
                f"  {a['symbol']}: {a['round_trips']} round-trip(s), "
                f"CHF {a['commission_cost']:.2f} commission drag — {status}"
            )
        sections.append("\n".join(lines))

    # === CURRENCY ATTRIBUTION ===
    if currency_attribution:
        ca = currency_attribution
        lines = [
            "=== CURRENCY PERFORMANCE ATTRIBUTION ===",
            f"  Portfolio return (CHF): {ca['portfolio_chf_return_pct']:+.2f}%",
            f"  SPY return (USD):       {ca['spy_usd_return_pct']:+.2f}%",
            f"  USD/CHF effect:         {ca['fx_change_pct']:+.2f}%",
            f"  SPY return (CHF-adj):   {ca['spy_chf_return_pct']:+.2f}%",
            f"  Alpha (CHF-adjusted):   {ca['alpha_chf_pct']:+.2f}%",
            f"  Alpha (USD-only):       {ca['alpha_usd_pct']:+.2f}%",
            f"  FX drag/boost:          {ca['fx_drag_pct']:+.2f}%",
        ]
        sections.append("\n".join(lines))

    return "\n\n".join(sections)
