"""Signal processing layer: conviction curves, news sentiment, commission trajectory, signal quality."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class ConvictionCurve:
    slope: float          # per-day change in confidence
    avg: float            # mean confidence over the window
    std: float            # standard deviation
    direction: str        # RISING, FALLING, OSCILLATING, FLAT, INSUFFICIENT
    earliest: float       # first data point value
    latest: float         # last data point value
    data_points_count: int


@dataclass
class SentimentCurve:
    avg_score: float       # rolling average sentiment [-1, +1]
    trend_direction: str   # "improving", "worsening", "stable"
    article_count: int


@dataclass
class CommissionTrajectory:
    total: float           # total commissions over the window (CHF)
    daily_avg: float       # average per active trading day
    annualized_pct: float  # annualized as % of capital (1000 CHF)
    active_days: int       # days with at least one trade
    total_days: int        # total calendar days in window


# ── Sentiment keyword lists ─────────────────────────────────────────

BULLISH_KEYWORDS = [
    "beat", "beats", "exceeded", "exceeds", "surpass", "surge", "soar",
    "rally", "upgrade", "upgrades", "outperform", "bullish", "record high",
    "strong earnings", "revenue growth", "profit growth", "breakout",
    "momentum", "accelerat", "expand", "innovation", "launch", "partner",
    "buyback", "dividend increase", "guidance raise", "raised guidance",
    "blowout", "positive", "boost", "gain", "winner", "optimis",
    "approve", "approval", "breakthrough", "contract win", "deal",
]

BEARISH_KEYWORDS = [
    "miss", "misses", "disappoint", "decline", "drop", "fall", "plunge",
    "crash", "downgrade", "downgrades", "underperform", "bearish",
    "record low", "weak earnings", "revenue decline", "profit decline",
    "breakdown", "slowdown", "decelerat", "contract", "layoff", "cut",
    "dividend cut", "guidance lower", "lowered guidance", "warning",
    "negative", "loss", "loser", "pessimis", "reject", "recall",
    "investigation", "lawsuit", "fine", "penalty", "sanction",
    "delay", "setback", "concern", "risk", "fear", "sell-off", "selloff",
]


# ── Sentiment scoring ───────────────────────────────────────────────

def score_headline_sentiment(title: str, summary: str = "") -> float:
    """Score a news headline/summary for sentiment. Returns [-1, +1]."""
    text = f"{title} {summary}".lower()

    bullish = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
    bearish = sum(1 for kw in BEARISH_KEYWORDS if kw in text)
    total = bullish + bearish

    if total == 0:
        return 0.0

    return (bullish - bearish) / total


# ── Conviction curves ───────────────────────────────────────────────

async def compute_conviction_curves(
    db: aiosqlite.Connection,
    symbols: list[str],
    days: int = 14,
) -> dict[str, ConvictionCurve]:
    """Compute conviction curves from decision history."""
    if not symbols:
        return {}

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    placeholders = ",".join("?" * len(symbols))

    rows = await db.execute_fetchall(
        f"""SELECT symbol, date(created_at) as day,
                   AVG(confidence) as avg_conf, COUNT(*) as cnt
            FROM decisions
            WHERE is_dry_run = 0
              AND created_at >= ?
              AND symbol IN ({placeholders})
            GROUP BY symbol, date(created_at)
            ORDER BY symbol, day""",
        (cutoff, *symbols),
    )

    # Group by symbol
    symbol_data: dict[str, list[tuple[str, float]]] = {}
    for r in rows:
        sym = r["symbol"]
        if sym not in symbol_data:
            symbol_data[sym] = []
        symbol_data[sym].append((r["day"], r["avg_conf"]))

    results: dict[str, ConvictionCurve] = {}
    for sym in symbols:
        data = symbol_data.get(sym, [])
        n = len(data)

        if n < 3:
            results[sym] = ConvictionCurve(
                slope=0.0, avg=0.0, std=0.0,
                direction="INSUFFICIENT",
                earliest=data[0][1] if data else 0.0,
                latest=data[-1][1] if data else 0.0,
                data_points_count=n,
            )
            continue

        values = [d[1] for d in data]
        avg = sum(values) / n
        variance = sum((v - avg) ** 2 for v in values) / n
        std = variance ** 0.5

        # Simple linear regression: slope = Σ((xi - x̄)(yi - ȳ)) / Σ((xi - x̄)²)
        x_vals = list(range(n))
        x_mean = sum(x_vals) / n
        numerator = sum((x - x_mean) * (y - avg) for x, y in zip(x_vals, values))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)
        slope = numerator / denominator if denominator > 0 else 0.0

        # Classify direction
        if n < 3:
            direction = "INSUFFICIENT"
        elif slope > 0.02:
            direction = "RISING" if std < 0.20 else "OSCILLATING"
        elif slope < -0.02:
            direction = "FALLING"
        elif std > 0.20:
            direction = "OSCILLATING"
        else:
            direction = "FLAT"

        results[sym] = ConvictionCurve(
            slope=round(slope, 4),
            avg=round(avg, 3),
            std=round(std, 3),
            direction=direction,
            earliest=round(values[0], 3),
            latest=round(values[-1], 3),
            data_points_count=n,
        )

    return results


# ── News sentiment curves ───────────────────────────────────────────

async def compute_news_sentiment(
    db: aiosqlite.Connection,
    symbols: list[str],
    days: int = 14,
) -> dict[str, SentimentCurve]:
    """Compute rolling news sentiment per symbol from DB news_items."""
    cutoff_hours = days * 24

    rows = await db.execute_fetchall(
        """SELECT symbol, title, summary, date(fetched_at) as day
           FROM news_items
           WHERE fetched_at >= datetime('now', ? || ' hours')
           ORDER BY fetched_at DESC""",
        (f"-{cutoff_hours}",),
    )

    # Group scores by symbol and day
    symbol_daily: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        sym = r["symbol"]
        if sym is None:
            continue  # skip general news for per-symbol curves
        title = r["title"] or ""
        summary = r["summary"] or ""
        day = r["day"]
        score = score_headline_sentiment(title, summary)

        if sym not in symbol_daily:
            symbol_daily[sym] = {}
        if day not in symbol_daily[sym]:
            symbol_daily[sym][day] = []
        symbol_daily[sym][day].append(score)

    results: dict[str, SentimentCurve] = {}
    for sym in symbols:
        daily_data = symbol_daily.get(sym, {})
        if not daily_data:
            continue

        # Sort days chronologically
        sorted_days = sorted(daily_data.keys())
        daily_avgs = [sum(daily_data[d]) / len(daily_data[d]) for d in sorted_days]
        article_count = sum(len(daily_data[d]) for d in sorted_days)

        # Rolling 7-day average (or all data if < 7 days)
        avg_score = sum(daily_avgs) / len(daily_avgs) if daily_avgs else 0.0

        # Trend: last 3 days avg vs previous 3 days avg
        if len(daily_avgs) >= 6:
            recent = sum(daily_avgs[-3:]) / 3
            previous = sum(daily_avgs[-6:-3]) / 3
            diff = recent - previous
        elif len(daily_avgs) >= 2:
            mid = len(daily_avgs) // 2
            recent = sum(daily_avgs[mid:]) / len(daily_avgs[mid:])
            previous = sum(daily_avgs[:mid]) / len(daily_avgs[:mid])
            diff = recent - previous
        else:
            diff = 0.0

        if diff > 0.05:
            trend = "improving"
        elif diff < -0.05:
            trend = "worsening"
        else:
            trend = "stable"

        results[sym] = SentimentCurve(
            avg_score=round(avg_score, 3),
            trend_direction=trend,
            article_count=article_count,
        )

    return results


# ── Commission trajectory ───────────────────────────────────────────

async def compute_commission_trajectory(
    db: aiosqlite.Connection,
    days: int = 14,
    capital: float = 1000.0,
) -> CommissionTrajectory:
    """Compute commission trajectory from trade history."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    rows = await db.execute_fetchall(
        """SELECT date(executed_at) as day, COUNT(*) as trades,
                  SUM(commission_chf) as commission
           FROM trades
           WHERE is_dry_run = 0 AND executed_at >= ?
           GROUP BY date(executed_at)
           ORDER BY day""",
        (cutoff,),
    )

    total = sum(r["commission"] or 0.0 for r in rows)
    active_days = len(rows)
    daily_avg = total / active_days if active_days > 0 else 0.0

    # Annualize: ~252 trading days per year
    annualized = daily_avg * 252
    annualized_pct = (annualized / capital * 100) if capital > 0 else 0.0

    return CommissionTrajectory(
        total=round(total, 2),
        daily_avg=round(daily_avg, 2),
        annualized_pct=round(annualized_pct, 1),
        active_days=active_days,
        total_days=days,
    )


# ── Signal quality (composite) ──────────────────────────────────────

def compute_signal_quality(
    conviction: dict[str, ConvictionCurve],
    sentiment: dict[str, SentimentCurve],
    rs_data: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Combine conviction + sentiment + relative strength into a quality rating.

    Returns {symbol: "HIGH"|"MEDIUM"|"LOW"|"NOISE"}.
    """
    rs_map: dict[str, float] = {}
    if rs_data:
        rs_map = {r["symbol"]: r["rs_ratio"] for r in rs_data}

    results: dict[str, str] = {}
    for sym, curve in conviction.items():
        if curve.direction in ("OSCILLATING", "INSUFFICIENT"):
            results[sym] = "NOISE"
            continue

        positives = 0

        # Conviction positive?
        if curve.direction == "RISING":
            positives += 1

        # Sentiment positive?
        sent = sentiment.get(sym)
        if sent and sent.avg_score > 0.1:
            positives += 1

        # Relative strength positive?
        rs = rs_map.get(sym, 0.0)
        if rs > 1.0:
            positives += 1

        if positives >= 3:
            results[sym] = "HIGH"
        elif positives >= 2:
            results[sym] = "MEDIUM"
        elif positives >= 1:
            results[sym] = "LOW"
        else:
            results[sym] = "NOISE"

    return results


# ── Report formatting ───────────────────────────────────────────────

_DIRECTION_ICONS = {
    "RISING": "\u2191",        # ↑
    "FALLING": "\u2193",       # ↓
    "OSCILLATING": "\u2195",   # ↕
    "FLAT": "\u2192",          # →
    "INSUFFICIENT": "?",
}

_TREND_ICONS = {
    "improving": "\u2191",
    "worsening": "\u2193",
    "stable": "\u2192",
}


def format_signal_report(
    conviction: dict[str, ConvictionCurve],
    sentiment: dict[str, SentimentCurve],
    commission: CommissionTrajectory,
    signal_quality: dict[str, str],
) -> str:
    """Format the complete signal report for the AI prompt."""
    sections = []

    # === SIGNAL QUALITY SUMMARY ===
    if signal_quality:
        lines = ["=== SIGNAL QUALITY (conviction + sentiment + relative strength) ==="]
        lines.append("  Only HIGH signals are actionable for BUY. NOISE = do not trade.")
        for sym, quality in sorted(signal_quality.items()):
            curve = conviction.get(sym)
            sent = sentiment.get(sym)
            parts = [f"  {sym:6s}: {quality:6s}"]
            if curve:
                icon = _DIRECTION_ICONS.get(curve.direction, "")
                parts.append(f"conviction {icon} {curve.direction} (slope {curve.slope:+.4f}/d, \u03c3={curve.std:.2f})")
            if sent:
                t_icon = _TREND_ICONS.get(sent.trend_direction, "")
                parts.append(f"sentiment {sent.avg_score:+.2f} {t_icon} ({sent.article_count} articles)")
            lines.append(" | ".join(parts))
        sections.append("\n".join(lines))

    # === CONVICTION CURVES ===
    if conviction:
        lines = ["=== CONVICTION CURVES (14-day decision history) ==="]
        for sym, curve in sorted(conviction.items()):
            if curve.direction == "INSUFFICIENT":
                lines.append(f"  {sym:6s}: INSUFFICIENT ({curve.data_points_count} data points, need >= 3)")
                continue
            icon = _DIRECTION_ICONS.get(curve.direction, "")
            lines.append(
                f"  {sym:6s}: {curve.direction} {icon} "
                f"({curve.earliest:.2f}\u2192{curve.latest:.2f}, "
                f"slope {curve.slope:+.4f}/d, \u03c3={curve.std:.2f})"
            )
        sections.append("\n".join(lines))

    # === NEWS SENTIMENT CURVES ===
    if sentiment:
        lines = ["=== NEWS SENTIMENT (14-day rolling) ==="]
        for sym, sent in sorted(sentiment.items()):
            icon = _TREND_ICONS.get(sent.trend_direction, "")
            lines.append(
                f"  {sym:6s}: {sent.avg_score:+.3f} ({icon} {sent.trend_direction}, "
                f"{sent.article_count} articles)"
            )
        sections.append("\n".join(lines))

    # === COMMISSION TRAJECTORY ===
    lines = ["=== COMMISSION TRAJECTORY ==="]
    lines.append(
        f"  Total: CHF {commission.total:.2f} over {commission.total_days} days "
        f"({commission.active_days} active trading days)"
    )
    lines.append(f"  Daily average: CHF {commission.daily_avg:.2f}/active day")
    lines.append(
        f"  Annualized: CHF {commission.daily_avg * 252:.0f}/year "
        f"= {commission.annualized_pct:.1f}% OF CAPITAL"
    )
    if commission.annualized_pct > 20:
        lines.append(
            f"  >>> At this pace, commissions alone eat {commission.annualized_pct:.0f}% "
            f"of your portfolio per year. REDUCE TRADING FREQUENCY."
        )
    sections.append("\n".join(lines))

    return "\n\n".join(sections)
