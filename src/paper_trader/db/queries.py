from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import aiosqlite


# ── Settings ──────────────────────────────────────────────────────────

async def get_setting(db: aiosqlite.Connection, key: str) -> str | None:
    row = await db.execute_fetchall("SELECT value FROM settings WHERE key = ?", (key,))
    return row[0]["value"] if row else None


async def set_setting(db: aiosqlite.Connection, key: str, value: str) -> None:
    await db.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = ?",
        (key, value, value),
    )
    await db.commit()


# ── Portfolio ─────────────────────────────────────────────────────────

async def get_portfolio(
    db: aiosqlite.Connection, is_dry_run: bool = False, session_id: int | None = None
) -> dict[str, Any] | None:
    if is_dry_run:
        cash = await get_dry_run_cash(db, session_id)
        if cash is None:
            return None
        return {"cash": cash, "currency": "CHF"}
    rows = await db.execute_fetchall("SELECT * FROM portfolio WHERE id = 1")
    return dict(rows[0]) if rows else None


async def init_portfolio(db: aiosqlite.Connection, cash: float, currency: str = "CHF") -> None:
    await db.execute(
        "INSERT OR IGNORE INTO portfolio (id, cash, currency) VALUES (1, ?, ?)",
        (cash, currency),
    )
    await db.commit()


async def update_cash(
    db: aiosqlite.Connection, new_cash: float, is_dry_run: bool = False, session_id: int | None = None
) -> None:
    if is_dry_run:
        await update_dry_run_cash(db, new_cash, session_id)
        return
    await db.execute(
        "UPDATE portfolio SET cash = ?, updated_at = datetime('now') WHERE id = 1",
        (new_cash,),
    )
    await db.commit()


# ── Positions ─────────────────────────────────────────────────────────

async def get_open_positions(db: aiosqlite.Connection, is_dry_run: bool = False) -> list[dict[str, Any]]:
    rows = await db.execute_fetchall(
        "SELECT * FROM positions WHERE closed_at IS NULL AND is_dry_run = ?",
        (int(is_dry_run),),
    )
    return [dict(r) for r in rows]


async def get_position_by_symbol(
    db: aiosqlite.Connection, symbol: str, is_dry_run: bool = False
) -> dict[str, Any] | None:
    rows = await db.execute_fetchall(
        "SELECT * FROM positions WHERE symbol = ? AND closed_at IS NULL AND is_dry_run = ?",
        (symbol, int(is_dry_run)),
    )
    return dict(rows[0]) if rows else None


async def open_position(
    db: aiosqlite.Connection,
    symbol: str,
    shares: float,
    avg_cost: float,
    is_dry_run: bool = False,
    stop_loss_price: float | None = None,
    risk_tier: str = "growth",
    currency: str = "USD",
) -> int:
    cursor = await db.execute(
        "INSERT INTO positions (symbol, shares, avg_cost, is_dry_run, stop_loss_price, risk_tier, currency) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (symbol, shares, avg_cost, int(is_dry_run), stop_loss_price, risk_tier, currency),
    )
    await db.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def close_position(
    db: aiosqlite.Connection, position_id: int, close_price: float
) -> None:
    await db.execute(
        "UPDATE positions SET closed_at = datetime('now'), close_price = ? WHERE id = ?",
        (close_price, position_id),
    )
    await db.commit()


async def update_position_shares(
    db: aiosqlite.Connection,
    position_id: int,
    new_shares: float,
    new_avg_cost: float,
    stop_loss_price: float | None = None,
) -> None:
    if stop_loss_price is not None:
        await db.execute(
            "UPDATE positions SET shares = ?, avg_cost = ?, stop_loss_price = ? WHERE id = ?",
            (new_shares, new_avg_cost, stop_loss_price, position_id),
        )
    else:
        await db.execute(
            "UPDATE positions SET shares = ?, avg_cost = ? WHERE id = ?",
            (new_shares, new_avg_cost, position_id),
        )
    await db.commit()


# ── Trades ────────────────────────────────────────────────────────────

async def record_trade(
    db: aiosqlite.Connection,
    symbol: str,
    action: str,
    shares: float,
    price: float,
    decision_id: int | None = None,
    is_dry_run: bool = False,
) -> int:
    total = shares * price
    cursor = await db.execute(
        """INSERT INTO trades (symbol, action, shares, price, total, decision_id, is_dry_run)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (symbol, action, shares, price, total, decision_id, int(is_dry_run)),
    )
    await db.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def get_trades(
    db: aiosqlite.Connection, is_dry_run: bool = False, limit: int = 50
) -> list[dict[str, Any]]:
    rows = await db.execute_fetchall(
        "SELECT * FROM trades WHERE is_dry_run = ? ORDER BY executed_at DESC LIMIT ?",
        (int(is_dry_run), limit),
    )
    return [dict(r) for r in rows]


# ── Decisions ─────────────────────────────────────────────────────────

async def record_decision(
    db: aiosqlite.Connection,
    symbol: str,
    action: str,
    confidence: float,
    reasoning: str,
    model: str,
    is_dry_run: bool = False,
    alpha_source: str = "",
) -> int:
    cursor = await db.execute(
        """INSERT INTO decisions (symbol, action, confidence, reasoning, model, is_dry_run, alpha_source)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (symbol, action, confidence, reasoning, model, int(is_dry_run), alpha_source),
    )
    await db.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def get_decisions(
    db: aiosqlite.Connection, is_dry_run: bool = False, limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    rows = await db.execute_fetchall(
        "SELECT * FROM decisions WHERE is_dry_run = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (int(is_dry_run), limit, offset),
    )
    return [dict(r) for r in rows]


async def count_decisions(
    db: aiosqlite.Connection, is_dry_run: bool = False,
) -> int:
    row = await db.execute_fetchall(
        "SELECT COUNT(*) FROM decisions WHERE is_dry_run = ?",
        (int(is_dry_run),),
    )
    return row[0][0] if row else 0


# ── API Calls ─────────────────────────────────────────────────────────

async def record_api_call(
    db: aiosqlite.Connection,
    model: str,
    purpose: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    is_dry_run: bool = False,
) -> int:
    cursor = await db.execute(
        """INSERT INTO api_calls (model, purpose, input_tokens, output_tokens, cost_usd, is_dry_run)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (model, purpose, input_tokens, output_tokens, cost_usd, int(is_dry_run)),
    )
    await db.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def get_total_spend(db: aiosqlite.Connection, include_dry_run: bool = True) -> float:
    """Get total API spend across all time (matches the Anthropic account credit model)."""
    if include_dry_run:
        rows = await db.execute_fetchall(
            "SELECT COALESCE(SUM(cost_usd), 0) as total FROM api_calls",
        )
    else:
        rows = await db.execute_fetchall(
            "SELECT COALESCE(SUM(cost_usd), 0) as total FROM api_calls WHERE is_dry_run = 0",
        )
    return rows[0]["total"]


async def get_monthly_spend(db: aiosqlite.Connection, include_dry_run: bool = True) -> float:
    """Get total API spend for the current month (informational only — budget uses total spend)."""
    if include_dry_run:
        rows = await db.execute_fetchall(
            """SELECT COALESCE(SUM(cost_usd), 0) as total
               FROM api_calls
               WHERE called_at >= date('now', 'start of month')""",
        )
    else:
        rows = await db.execute_fetchall(
            """SELECT COALESCE(SUM(cost_usd), 0) as total
               FROM api_calls
               WHERE called_at >= date('now', 'start of month')
                 AND is_dry_run = 0""",
        )
    return rows[0]["total"]


async def get_api_usage_by_model(
    db: aiosqlite.Connection, include_dry_run: bool = True
) -> list[dict[str, Any]]:
    """Get all-time API usage breakdown by model."""
    if include_dry_run:
        rows = await db.execute_fetchall(
            """SELECT model,
                      COUNT(*) as call_count,
                      SUM(input_tokens) as total_input,
                      SUM(output_tokens) as total_output,
                      SUM(cost_usd) as total_cost
               FROM api_calls
               GROUP BY model
               ORDER BY total_cost DESC""",
        )
    else:
        rows = await db.execute_fetchall(
            """SELECT model,
                      COUNT(*) as call_count,
                      SUM(input_tokens) as total_input,
                      SUM(output_tokens) as total_output,
                      SUM(cost_usd) as total_cost
               FROM api_calls
               WHERE is_dry_run = 0
               GROUP BY model
               ORDER BY total_cost DESC""",
        )
    return [dict(r) for r in rows]


async def get_daily_spend(
    db: aiosqlite.Connection, days: int = 30, include_dry_run: bool = True
) -> list[dict[str, Any]]:
    """Get daily API spend for the last N days."""
    if include_dry_run:
        rows = await db.execute_fetchall(
            """SELECT date(called_at) as day,
                      SUM(cost_usd) as total_cost,
                      COUNT(*) as call_count
               FROM api_calls
               WHERE called_at >= date('now', ? || ' days')
               GROUP BY date(called_at)
               ORDER BY day""",
            (f"-{days}",),
        )
    else:
        rows = await db.execute_fetchall(
            """SELECT date(called_at) as day,
                      SUM(cost_usd) as total_cost,
                      COUNT(*) as call_count
               FROM api_calls
               WHERE called_at >= date('now', ? || ' days')
                 AND is_dry_run = 0
               GROUP BY date(called_at)
               ORDER BY day""",
            (f"-{days}",),
        )
    return [dict(r) for r in rows]


# ── Portfolio Snapshots ───────────────────────────────────────────────

async def record_snapshot(
    db: aiosqlite.Connection,
    cash: float,
    positions_value: float,
    is_dry_run: bool = False,
    spy_price: float | None = None,
    benchmark_value: float | None = None,
    usd_chf_rate: float | None = None,
) -> int:
    total = cash + positions_value
    cursor = await db.execute(
        """INSERT INTO portfolio_snapshots (cash, positions_value, total_value, is_dry_run, spy_price, benchmark_value, usd_chf_rate)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (cash, positions_value, total, int(is_dry_run), spy_price, benchmark_value, usd_chf_rate),
    )
    await db.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def get_snapshots(
    db: aiosqlite.Connection,
    is_dry_run: bool = False,
    limit: int = 90,
    since: datetime | None = None,
) -> list[dict[str, Any]]:
    if since is not None:
        rows = await db.execute_fetchall(
            """SELECT * FROM portfolio_snapshots
               WHERE is_dry_run = ? AND snapshot_at >= ?
               ORDER BY snapshot_at DESC""",
            (int(is_dry_run), since.isoformat()),
        )
    else:
        rows = await db.execute_fetchall(
            """SELECT * FROM portfolio_snapshots
               WHERE is_dry_run = ?
               ORDER BY snapshot_at DESC LIMIT ?""",
            (int(is_dry_run), limit),
        )
    return [dict(r) for r in rows]


async def get_first_spy_price(db: aiosqlite.Connection, is_dry_run: bool = False) -> float | None:
    """Return the earliest non-NULL spy_price for the given run mode."""
    rows = await db.execute_fetchall(
        """SELECT spy_price FROM portfolio_snapshots
           WHERE is_dry_run = ? AND spy_price IS NOT NULL
           ORDER BY snapshot_at ASC LIMIT 1""",
        (int(is_dry_run),),
    )
    return rows[0]["spy_price"] if rows else None


async def get_first_fx_rate(db: aiosqlite.Connection, is_dry_run: bool = False) -> float | None:
    """Return the earliest non-NULL usd_chf_rate for the given run mode."""
    rows = await db.execute_fetchall(
        """SELECT usd_chf_rate FROM portfolio_snapshots
           WHERE is_dry_run = ? AND usd_chf_rate IS NOT NULL
           ORDER BY snapshot_at ASC LIMIT 1""",
        (int(is_dry_run),),
    )
    return rows[0]["usd_chf_rate"] if rows else None


async def get_latest_fx_rate(db: aiosqlite.Connection, is_dry_run: bool = False) -> float | None:
    """Return the most recent non-NULL usd_chf_rate from snapshots."""
    rows = await db.execute_fetchall(
        """SELECT usd_chf_rate FROM portfolio_snapshots
           WHERE is_dry_run = ? AND usd_chf_rate IS NOT NULL
           ORDER BY snapshot_at DESC LIMIT 1""",
        (int(is_dry_run),),
    )
    return rows[0]["usd_chf_rate"] if rows else None


async def get_benchmark_summary(db: aiosqlite.Connection, is_dry_run: bool = False) -> dict[str, Any] | None:
    """Return benchmark comparison data for AI reviews.

    Returns dict with initial/current SPY price, returns, portfolio value,
    and FX (USD/CHF) data when available. Returns None if no benchmark data exists yet.
    """
    first = await db.execute_fetchall(
        """SELECT spy_price, benchmark_value, total_value, usd_chf_rate FROM portfolio_snapshots
           WHERE is_dry_run = ? AND spy_price IS NOT NULL
           ORDER BY snapshot_at ASC LIMIT 1""",
        (int(is_dry_run),),
    )
    if not first:
        return None

    latest = await db.execute_fetchall(
        """SELECT spy_price, benchmark_value, total_value, usd_chf_rate FROM portfolio_snapshots
           WHERE is_dry_run = ? AND spy_price IS NOT NULL
           ORDER BY snapshot_at DESC LIMIT 1""",
        (int(is_dry_run),),
    )
    if not latest:
        return None

    initial_spy = first[0]["spy_price"]
    current_spy = latest[0]["spy_price"]
    spy_return_pct = ((current_spy / initial_spy) - 1) * 100 if initial_spy else 0
    initial_value = first[0]["total_value"]
    current_value = latest[0]["total_value"]
    portfolio_return_pct = ((current_value / initial_value) - 1) * 100 if initial_value else 0

    result: dict[str, Any] = {
        "initial_spy": initial_spy,
        "current_spy": current_spy,
        "spy_return_pct": spy_return_pct,
        "benchmark_value": latest[0]["benchmark_value"],
        "portfolio_value": current_value,
        "portfolio_return_pct": portfolio_return_pct,
        "alpha_pct": portfolio_return_pct - spy_return_pct,
    }

    # FX data (USD/CHF) — include when available
    initial_fx = first[0]["usd_chf_rate"]
    current_fx = latest[0]["usd_chf_rate"]
    if initial_fx and current_fx:
        fx_change_pct = ((current_fx / initial_fx) - 1) * 100
        # Benchmark adjusted for FX: what SPY buy-and-hold returns in CHF
        benchmark_value_chf = initial_value * (current_spy / initial_spy) * (current_fx / initial_fx) if initial_spy else 0
        # Portfolio in CHF: cash is already CHF, positions are USD * fx_rate
        # (positions_value from latest snapshot is in the same mixed unit as total_value)
        portfolio_value_chf = current_value  # already mixed CHF+USD, best approximation
        # FX impact: difference between USD-denominated and CHF-adjusted alpha
        usd_alpha = portfolio_return_pct - spy_return_pct
        chf_spy_return = ((current_spy / initial_spy) * (current_fx / initial_fx) - 1) * 100 if initial_spy else 0
        chf_alpha = portfolio_return_pct - chf_spy_return
        fx_impact_pct = chf_alpha - usd_alpha

        result.update({
            "initial_fx_rate": initial_fx,
            "current_fx_rate": current_fx,
            "fx_change_pct": fx_change_pct,
            "benchmark_value_chf": benchmark_value_chf,
            "portfolio_value_chf": portfolio_value_chf,
            "fx_impact_pct": fx_impact_pct,
        })

    return result


# ── Strategy Journal ──────────────────────────────────────────────────

async def add_journal_entry(
    db: aiosqlite.Connection,
    entry_type: str,
    content: str,
    model: str,
    supersedes_id: int | None = None,
    is_dry_run: bool = False,
) -> int:
    if supersedes_id is not None:
        await db.execute(
            "UPDATE strategy_journal SET is_active = 0 WHERE id = ?",
            (supersedes_id,),
        )
    cursor = await db.execute(
        """INSERT INTO strategy_journal (entry_type, content, model, supersedes_id, is_dry_run)
           VALUES (?, ?, ?, ?, ?)""",
        (entry_type, content, model, supersedes_id, int(is_dry_run)),
    )
    await db.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def get_active_journal(
    db: aiosqlite.Connection, is_dry_run: bool = False
) -> list[dict[str, Any]]:
    rows = await db.execute_fetchall(
        """SELECT * FROM strategy_journal
           WHERE is_active = 1 AND is_dry_run = ?
           ORDER BY created_at DESC""",
        (int(is_dry_run),),
    )
    return [dict(r) for r in rows]


# ── Watchlist ─────────────────────────────────────────────────────────

async def get_watchlist(db: aiosqlite.Connection) -> list[dict[str, Any]]:
    rows = await db.execute_fetchall(
        "SELECT * FROM watchlist WHERE removed_at IS NULL ORDER BY added_at DESC",
    )
    return [dict(r) for r in rows]


async def add_to_watchlist(
    db: aiosqlite.Connection, symbol: str, reason: str | None = None, risk_tier: str = "growth",
) -> None:
    await db.execute(
        """INSERT INTO watchlist (symbol, reason, risk_tier) VALUES (?, ?, ?)
           ON CONFLICT(symbol) DO UPDATE SET reason = ?, risk_tier = ?, removed_at = NULL, added_at = datetime('now')""",
        (symbol, reason, risk_tier, reason, risk_tier),
    )
    await db.commit()


async def remove_from_watchlist(db: aiosqlite.Connection, symbol: str) -> None:
    await db.execute(
        "UPDATE watchlist SET removed_at = datetime('now') WHERE symbol = ? AND removed_at IS NULL",
        (symbol,),
    )
    await db.commit()


# ── Dry Run Sessions ─────────────────────────────────────────────────

async def start_dry_run_session(db: aiosqlite.Connection, initial_cash: float) -> int | None:
    """Start a new dry run session. Returns None if one is already active (atomic check)."""
    cursor = await db.execute(
        """INSERT INTO dry_run_sessions (initial_cash, current_cash)
           SELECT ?, ?
           WHERE NOT EXISTS (SELECT 1 FROM dry_run_sessions WHERE ended_at IS NULL)""",
        (initial_cash, initial_cash),
    )
    await db.commit()
    if cursor.lastrowid == 0 or cursor.rowcount == 0:
        return None
    return cursor.lastrowid


async def get_dry_run_cash(db: aiosqlite.Connection, session_id: int | None = None) -> float | None:
    """Get current cash for an active dry run session."""
    if session_id is not None:
        rows = await db.execute_fetchall(
            "SELECT current_cash FROM dry_run_sessions WHERE id = ?", (session_id,)
        )
    else:
        rows = await db.execute_fetchall(
            "SELECT current_cash FROM dry_run_sessions WHERE ended_at IS NULL ORDER BY id DESC LIMIT 1"
        )
    return rows[0]["current_cash"] if rows else None


async def update_dry_run_cash(db: aiosqlite.Connection, new_cash: float, session_id: int | None = None) -> None:
    """Update current cash for an active dry run session (targets only the most recent if no session_id)."""
    if session_id is not None:
        await db.execute(
            "UPDATE dry_run_sessions SET current_cash = ? WHERE id = ?", (new_cash, session_id)
        )
    else:
        await db.execute(
            """UPDATE dry_run_sessions SET current_cash = ?
               WHERE id = (SELECT id FROM dry_run_sessions WHERE ended_at IS NULL ORDER BY id DESC LIMIT 1)""",
            (new_cash,),
        )
    await db.commit()


async def end_dry_run_session(
    db: aiosqlite.Connection,
    session_id: int,
    final_value: float,
    total_trades: int,
    api_cost: float,
    summary: str,
) -> None:
    pnl = final_value - (await get_dry_run_session(db, session_id))["initial_cash"]
    await db.execute(
        """UPDATE dry_run_sessions
           SET ended_at = datetime('now'), final_value = ?, pnl = ?,
               total_trades = ?, api_cost_usd = ?, summary = ?
           WHERE id = ?""",
        (final_value, pnl, total_trades, api_cost, summary, session_id),
    )
    await db.commit()


async def get_dry_run_session(db: aiosqlite.Connection, session_id: int) -> dict[str, Any]:
    rows = await db.execute_fetchall(
        "SELECT * FROM dry_run_sessions WHERE id = ?", (session_id,)
    )
    return dict(rows[0])


async def get_session_api_cost(db: aiosqlite.Connection, session_id: int) -> float:
    """Get API cost for a specific dry run session (bounded to its time window)."""
    session = await get_dry_run_session(db, session_id)
    rows = await db.execute_fetchall(
        """SELECT COALESCE(SUM(cost_usd), 0) as total
           FROM api_calls
           WHERE is_dry_run = 1
             AND called_at >= ?
             AND called_at <= COALESCE(?, datetime('now'))""",
        (session["started_at"], session["ended_at"]),
    )
    return rows[0]["total"]


async def get_active_dry_run_session(db: aiosqlite.Connection) -> dict[str, Any] | None:
    """Get the currently running dry run session, if any."""
    rows = await db.execute_fetchall(
        "SELECT * FROM dry_run_sessions WHERE ended_at IS NULL ORDER BY id DESC LIMIT 1"
    )
    return dict(rows[0]) if rows else None


async def get_dry_run_sessions(db: aiosqlite.Connection) -> list[dict[str, Any]]:
    rows = await db.execute_fetchall(
        "SELECT * FROM dry_run_sessions ORDER BY started_at DESC"
    )
    return [dict(r) for r in rows]


async def reset_database(
    db: aiosqlite.Connection, initial_cash: float, currency: str = "CHF"
) -> None:
    """Delete all data from all tables and reinitialize portfolio."""
    tables = [
        "portfolio_snapshots", "trades", "decisions", "api_calls",
        "strategy_journal", "dry_run_sessions", "positions", "watchlist",
        "news_items", "settings", "portfolio",
    ]
    for table in tables:
        await db.execute(f"DELETE FROM {table}")  # noqa: S608 — table names are hardcoded above
    await db.commit()
    await init_portfolio(db, initial_cash, currency)


async def count_snapshots(db: aiosqlite.Connection, is_dry_run: bool = False) -> int:
    """Count portfolio snapshots for the given run mode."""
    rows = await db.execute_fetchall(
        "SELECT COUNT(*) as cnt FROM portfolio_snapshots WHERE is_dry_run = ?",
        (int(is_dry_run),),
    )
    return rows[0]["cnt"]


# ── News Items ────────────────────────────────────────────────────────

async def save_news_item(
    db: aiosqlite.Connection,
    source: str,
    symbol: str | None,
    title: str,
    summary: str | None,
    link: str,
    published_at: str | None,
) -> bool:
    """Save a news item. Returns True if inserted, False if duplicate (same link)."""
    cursor = await db.execute(
        """INSERT OR IGNORE INTO news_items (source, symbol, title, summary, link, published_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (source, symbol, title, summary, link, published_at),
    )
    await db.commit()
    return cursor.rowcount > 0


async def get_recent_news(
    db: aiosqlite.Connection,
    hours: int = 6,
    symbols: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Get news from DB fetched within the last N hours.

    If symbols given, returns general news (symbol IS NULL) plus
    symbol-specific news matching any of the given tickers.
    """
    if symbols:
        placeholders = ",".join("?" * len(symbols))
        rows = await db.execute_fetchall(
            f"""SELECT * FROM news_items
               WHERE fetched_at >= datetime('now', ? || ' hours')
                 AND (symbol IS NULL OR symbol IN ({placeholders}))
               ORDER BY fetched_at DESC LIMIT 50""",
            (f"-{hours}", *symbols),
        )
    else:
        rows = await db.execute_fetchall(
            """SELECT * FROM news_items
               WHERE fetched_at >= datetime('now', ? || ' hours')
               ORDER BY fetched_at DESC LIMIT 30""",
            (f"-{hours}",),
        )
    return [dict(r) for r in rows]


async def cleanup_old_news(db: aiosqlite.Connection, days: int = 7) -> int:
    """Delete news items older than N days. Returns count deleted."""
    cursor = await db.execute(
        "DELETE FROM news_items WHERE fetched_at < datetime('now', ? || ' days')",
        (f"-{days}",),
    )
    await db.commit()
    return cursor.rowcount


async def has_monthly_review_this_month(db: aiosqlite.Connection, is_dry_run: bool = False) -> bool:
    """Check if a monthly_review API call has already been made this month."""
    rows = await db.execute_fetchall(
        """SELECT 1 FROM api_calls
           WHERE purpose = 'monthly_review'
             AND called_at >= date('now', 'start of month')
             AND is_dry_run = ?
           LIMIT 1""",
        (int(is_dry_run),),
    )
    return bool(rows)


async def close_orphaned_dry_run_sessions(db: aiosqlite.Connection) -> int:
    """Close any dry run sessions that were left open (e.g. process killed)."""
    cursor = await db.execute(
        """UPDATE dry_run_sessions
           SET ended_at = datetime('now'),
               summary = COALESCE(summary, 'Session interrupted (process terminated)')
           WHERE ended_at IS NULL""",
    )
    await db.commit()
    return cursor.rowcount
