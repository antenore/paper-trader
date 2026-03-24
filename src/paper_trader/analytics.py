"""Portfolio analytics powered by QuantStats.

Reads portfolio_snapshots from DB and computes performance metrics.
"""
from __future__ import annotations

import io
import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless rendering, must be before any other matplotlib import

import aiosqlite
import pandas as pd
import quantstats as qs

logger = logging.getLogger(__name__)


async def _load_daily_series(
    db: aiosqlite.Connection, is_dry_run: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """Load daily portfolio and benchmark returns from snapshots.

    Takes the last snapshot per day (end-of-day value).
    Returns (portfolio_returns, benchmark_returns) as pandas Series.
    """
    rows = await db.execute_fetchall(
        """
        SELECT date(snapshot_at) as day,
               total_value,
               spy_price,
               usd_chf_rate
        FROM portfolio_snapshots
        WHERE is_dry_run = ?
        ORDER BY snapshot_at
        """,
        (int(is_dry_run),),
    )

    if not rows:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Build DataFrame, keep last snapshot per day
    df = pd.DataFrame([dict(r) for r in rows])
    df["day"] = pd.to_datetime(df["day"])
    daily = df.groupby("day").last().sort_index()

    # Portfolio returns from total_value
    portfolio = daily["total_value"]
    portfolio.index.name = None
    portfolio_returns = portfolio.pct_change().dropna()

    # Benchmark: SPY in CHF (spy_price * usd_chf_rate)
    if daily["spy_price"].notna().any() and daily["usd_chf_rate"].notna().any():
        spy_chf = daily["spy_price"] * daily["usd_chf_rate"]
        benchmark_returns = spy_chf.pct_change().dropna()
    else:
        benchmark_returns = pd.Series(dtype=float)

    return portfolio_returns, benchmark_returns


async def compute_metrics(
    db: aiosqlite.Connection, is_dry_run: bool = False,
) -> dict[str, Any]:
    """Compute key performance metrics."""
    portfolio_returns, benchmark_returns = await _load_daily_series(db, is_dry_run)

    if portfolio_returns.empty or len(portfolio_returns) < 2:
        return {"error": "Not enough data (need at least 3 daily snapshots)"}

    # Align benchmark to portfolio dates
    bench = benchmark_returns.reindex(portfolio_returns.index).dropna() if not benchmark_returns.empty else None

    metrics: dict[str, Any] = {}

    # Basic stats
    metrics["total_return"] = float(qs.stats.comp(portfolio_returns))
    metrics["cagr"] = float(qs.stats.cagr(portfolio_returns))
    metrics["sharpe"] = float(qs.stats.sharpe(portfolio_returns))
    metrics["sortino"] = float(qs.stats.sortino(portfolio_returns))
    metrics["max_drawdown"] = float(qs.stats.max_drawdown(portfolio_returns))
    metrics["volatility"] = float(qs.stats.volatility(portfolio_returns))
    metrics["calmar"] = float(qs.stats.calmar(portfolio_returns))

    # Win/loss
    metrics["win_rate"] = float(qs.stats.win_rate(portfolio_returns))
    metrics["best_day"] = float(qs.stats.best(portfolio_returns))
    metrics["worst_day"] = float(qs.stats.worst(portfolio_returns))
    metrics["avg_return"] = float(qs.stats.avg_return(portfolio_returns))

    # Value at Risk
    metrics["var_95"] = float(qs.stats.value_at_risk(portfolio_returns))

    # vs benchmark
    if bench is not None and len(bench) >= 2:
        metrics["benchmark_return"] = float(qs.stats.comp(bench))
        metrics["benchmark_sharpe"] = float(qs.stats.sharpe(bench))
        # Information ratio
        try:
            metrics["information_ratio"] = float(
                qs.stats.information_ratio(portfolio_returns, bench)
            )
        except Exception:
            metrics["information_ratio"] = None

    metrics["trading_days"] = len(portfolio_returns)
    metrics["start_date"] = str(portfolio_returns.index[0].date())
    metrics["end_date"] = str(portfolio_returns.index[-1].date())

    return metrics


def format_metrics_for_ai(metrics: dict[str, Any]) -> str:
    """Format key metrics as a text block for the AI analyst context."""
    if "error" in metrics:
        return ""

    lines = ["=== PORTFOLIO PERFORMANCE METRICS ==="]

    total_ret = metrics["total_return"] * 100
    sharpe = metrics["sharpe"]
    sortino = metrics["sortino"]
    max_dd = metrics["max_drawdown"] * 100
    win_rate = metrics["win_rate"] * 100
    vol = metrics["volatility"] * 100
    days = metrics["trading_days"]

    lines.append(f"  Total return: {total_ret:+.2f}% over {days} trading days")
    lines.append(f"  Sharpe ratio: {sharpe:.2f}  |  Sortino: {sortino:.2f}")
    lines.append(f"  Max drawdown: {max_dd:.2f}%  |  Volatility (ann.): {vol:.1f}%")
    lines.append(f"  Win rate: {win_rate:.0f}% of days positive")

    if "benchmark_return" in metrics:
        bench_ret = metrics["benchmark_return"] * 100
        alpha = total_ret - bench_ret
        lines.append(f"  Benchmark (SPY CHF): {bench_ret:+.2f}%  |  Alpha: {alpha:+.2f}%")

    # Interpretive guidance
    if sharpe < -2:
        lines.append("  WARNING: Sharpe deeply negative. Current strategy is destroying value.")
    elif sharpe < 0:
        lines.append("  NOTE: Sharpe negative. Underperforming risk-free rate.")

    if win_rate < 40:
        lines.append("  NOTE: Win rate below 40%. Most trading days lose money.")

    return "\n".join(lines)


async def generate_tearsheet_html(
    db: aiosqlite.Connection, is_dry_run: bool = False,
) -> str:
    """Generate a QuantStats HTML tearsheet as a string."""
    import tempfile
    import os

    portfolio_returns, benchmark_returns = await _load_daily_series(db, is_dry_run)

    if portfolio_returns.empty or len(portfolio_returns) < 2:
        return "<p>Not enough data for tearsheet (need at least 3 daily snapshots).</p>"

    bench = benchmark_returns.reindex(portfolio_returns.index).dropna() if not benchmark_returns.empty else None

    # QuantStats writes HTML to a file path, not a file object
    fd, tmppath = tempfile.mkstemp(suffix=".html")
    os.close(fd)
    try:
        qs.reports.html(
            portfolio_returns,
            benchmark=bench,
            title="Paper Trader Performance",
            output=tmppath,
        )
        with open(tmppath, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        os.unlink(tmppath)
