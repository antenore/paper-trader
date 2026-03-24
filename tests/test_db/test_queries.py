from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from paper_trader.db import queries
from paper_trader.db.queries import _business_days_ago


@pytest.mark.asyncio
async def test_settings_roundtrip(db):
    await queries.set_setting(db, "mode", "live")
    assert await queries.get_setting(db, "mode") == "live"

    await queries.set_setting(db, "mode", "dry_run")
    assert await queries.get_setting(db, "mode") == "dry_run"


@pytest.mark.asyncio
async def test_settings_missing_key(db):
    assert await queries.get_setting(db, "nonexistent") is None


@pytest.mark.asyncio
async def test_portfolio_init(db):
    await queries.init_portfolio(db, 800.0, "CHF")
    p = await queries.get_portfolio(db)
    assert p is not None
    assert p["cash"] == 800.0
    assert p["currency"] == "CHF"


@pytest.mark.asyncio
async def test_portfolio_init_idempotent(db):
    await queries.init_portfolio(db, 800.0)
    await queries.init_portfolio(db, 1000.0)  # Should not overwrite
    p = await queries.get_portfolio(db)
    assert p["cash"] == 800.0


@pytest.mark.asyncio
async def test_update_cash(db_with_portfolio):
    await queries.update_cash(db_with_portfolio, 750.0)
    p = await queries.get_portfolio(db_with_portfolio)
    assert p["cash"] == 750.0


@pytest.mark.asyncio
async def test_position_lifecycle(db):
    pos_id = await queries.open_position(db, "AAPL", 10.0, 150.0)
    assert pos_id is not None

    positions = await queries.get_open_positions(db)
    assert len(positions) == 1
    assert positions[0]["symbol"] == "AAPL"
    assert positions[0]["shares"] == 10.0

    pos = await queries.get_position_by_symbol(db, "AAPL")
    assert pos is not None
    assert pos["avg_cost"] == 150.0

    await queries.close_position(db, pos_id, 160.0)
    positions = await queries.get_open_positions(db)
    assert len(positions) == 0


@pytest.mark.asyncio
async def test_trade_recording(db):
    trade_id = await queries.record_trade(db, "SPY", "BUY", 5.0, 450.0)
    assert trade_id is not None

    trades = await queries.get_trades(db)
    assert len(trades) == 1
    assert trades[0]["symbol"] == "SPY"
    assert trades[0]["total"] == 2250.0


@pytest.mark.asyncio
async def test_decision_recording(db):
    dec_id = await queries.record_decision(
        db, "AAPL", "BUY", 0.85, "Strong earnings", "claude-haiku-4-5-20251001"
    )
    assert dec_id is not None

    decs = await queries.get_decisions(db)
    assert len(decs) == 1
    assert decs[0]["confidence"] == 0.85


@pytest.mark.asyncio
async def test_api_call_tracking(db):
    await queries.record_api_call(db, "claude-haiku-4-5-20251001", "screening", 1000, 500, 0.003)
    await queries.record_api_call(db, "claude-haiku-4-5-20251001", "analysis", 2000, 800, 0.005)

    spend = await queries.get_monthly_spend(db)
    assert abs(spend - 0.008) < 0.0001

    by_model = await queries.get_api_usage_by_model(db)
    assert len(by_model) == 1
    assert by_model[0]["call_count"] == 2


@pytest.mark.asyncio
async def test_api_spend_excludes_dry_run(db):
    """get_monthly_spend(include_dry_run=False) should exclude dry run costs."""
    await queries.record_api_call(db, "claude-haiku-4-5-20251001", "screening", 1000, 500, 0.003, is_dry_run=False)
    await queries.record_api_call(db, "claude-haiku-4-5-20251001", "dry-screening", 1000, 500, 0.010, is_dry_run=True)

    total = await queries.get_monthly_spend(db, include_dry_run=True)
    assert abs(total - 0.013) < 0.0001

    live_only = await queries.get_monthly_spend(db, include_dry_run=False)
    assert abs(live_only - 0.003) < 0.0001


@pytest.mark.asyncio
async def test_api_usage_by_model_excludes_dry_run(db):
    """get_api_usage_by_model(include_dry_run=False) should exclude dry run costs."""
    await queries.record_api_call(db, "claude-haiku-4-5-20251001", "screening", 1000, 500, 0.003, is_dry_run=False)
    await queries.record_api_call(db, "claude-haiku-4-5-20251001", "dry-screening", 2000, 800, 0.010, is_dry_run=True)

    all_usage = await queries.get_api_usage_by_model(db, include_dry_run=True)
    assert len(all_usage) == 1
    assert all_usage[0]["call_count"] == 2

    live_usage = await queries.get_api_usage_by_model(db, include_dry_run=False)
    assert len(live_usage) == 1
    assert live_usage[0]["call_count"] == 1
    assert abs(live_usage[0]["total_cost"] - 0.003) < 0.0001


@pytest.mark.asyncio
async def test_snapshot_recording(db):
    await queries.record_snapshot(db, 700.0, 100.0)
    snaps = await queries.get_snapshots(db)
    assert len(snaps) == 1
    assert snaps[0]["total_value"] == 800.0


@pytest.mark.asyncio
async def test_snapshot_with_benchmark(db):
    """record_snapshot stores spy_price and benchmark_value."""
    await queries.record_snapshot(db, 700.0, 100.0, spy_price=450.0, benchmark_value=800.0)
    snaps = await queries.get_snapshots(db)
    assert len(snaps) == 1
    assert snaps[0]["spy_price"] == 450.0
    assert snaps[0]["benchmark_value"] == 800.0


@pytest.mark.asyncio
async def test_snapshot_without_benchmark(db):
    """record_snapshot works without benchmark params (backward compat)."""
    await queries.record_snapshot(db, 700.0, 100.0)
    snaps = await queries.get_snapshots(db)
    assert snaps[0]["spy_price"] is None
    assert snaps[0]["benchmark_value"] is None


@pytest.mark.asyncio
async def test_get_first_spy_price_with_data(db):
    """get_first_spy_price returns earliest non-NULL spy_price."""
    await queries.record_snapshot(db, 700.0, 100.0)  # No SPY
    await queries.record_snapshot(db, 700.0, 100.0, spy_price=450.0, benchmark_value=800.0)
    await queries.record_snapshot(db, 700.0, 100.0, spy_price=455.0, benchmark_value=808.9)

    first = await queries.get_first_spy_price(db)
    assert first == 450.0


@pytest.mark.asyncio
async def test_get_first_spy_price_no_data(db):
    """get_first_spy_price returns None when no SPY data exists."""
    await queries.record_snapshot(db, 700.0, 100.0)
    first = await queries.get_first_spy_price(db)
    assert first is None


@pytest.mark.asyncio
async def test_get_benchmark_summary_with_data(db):
    """get_benchmark_summary returns comparison dict."""
    # Use explicit timestamps to ensure ordering
    await db.execute(
        """INSERT INTO portfolio_snapshots (cash, positions_value, total_value, is_dry_run, spy_price, benchmark_value, snapshot_at)
           VALUES (?, ?, ?, 0, ?, ?, ?)""",
        (700.0, 100.0, 800.0, 450.0, 800.0, "2025-06-01 10:00:00"),
    )
    await db.execute(
        """INSERT INTO portfolio_snapshots (cash, positions_value, total_value, is_dry_run, spy_price, benchmark_value, snapshot_at)
           VALUES (?, ?, ?, 0, ?, ?, ?)""",
        (720.0, 120.0, 840.0, 460.0, 817.8, "2025-06-02 10:00:00"),
    )
    await db.commit()

    summary = await queries.get_benchmark_summary(db)
    assert summary is not None
    assert summary["initial_spy"] == 450.0
    assert summary["current_spy"] == 460.0
    assert abs(summary["spy_return_pct"] - 2.222) < 0.01
    assert summary["portfolio_value"] == 840.0
    assert summary["benchmark_value"] == 817.8
    assert abs(summary["portfolio_return_pct"] - 5.0) < 0.01
    assert abs(summary["alpha_pct"] - 2.778) < 0.01


@pytest.mark.asyncio
async def test_get_benchmark_summary_no_data(db):
    """get_benchmark_summary returns None when no SPY data."""
    await queries.record_snapshot(db, 700.0, 100.0)  # No spy_price
    summary = await queries.get_benchmark_summary(db)
    assert summary is None


@pytest.mark.asyncio
async def test_journal_superseding(db):
    id1 = await queries.add_journal_entry(db, "observation", "Market is bullish", "haiku")
    id2 = await queries.add_journal_entry(db, "revision", "Market turning bearish", "sonnet", supersedes_id=id1)

    active = await queries.get_active_journal(db)
    assert len(active) == 1
    assert active[0]["id"] == id2


@pytest.mark.asyncio
async def test_watchlist_operations(db):
    await queries.add_to_watchlist(db, "AAPL", "Strong momentum")
    await queries.add_to_watchlist(db, "GOOGL", "Undervalued")

    wl = await queries.get_watchlist(db)
    assert len(wl) == 2

    await queries.remove_from_watchlist(db, "AAPL")
    wl = await queries.get_watchlist(db)
    assert len(wl) == 1
    assert wl[0]["symbol"] == "GOOGL"


@pytest.mark.asyncio
async def test_dry_run_session(db):
    sid = await queries.start_dry_run_session(db, 800.0)
    await queries.end_dry_run_session(db, sid, 850.0, 10, 0.05, "Good run")

    session = await queries.get_dry_run_session(db, sid)
    assert session["pnl"] == 50.0
    assert session["total_trades"] == 10


@pytest.mark.asyncio
async def test_start_dry_run_session_atomic_guard(db):
    """start_dry_run_session returns None if a session is already active."""
    sid1 = await queries.start_dry_run_session(db, 800.0)
    assert sid1 is not None

    # Second attempt while first is active
    sid2 = await queries.start_dry_run_session(db, 800.0)
    assert sid2 is None, "Should reject when active session exists"

    # End the first session
    await queries.end_dry_run_session(db, sid1, 800.0, 0, 0, "Done")

    # Now it should work again
    sid3 = await queries.start_dry_run_session(db, 800.0)
    assert sid3 is not None


@pytest.mark.asyncio
async def test_session_api_cost_time_bounded(db):
    """get_session_api_cost only counts calls within the session's time window."""
    # Set up two sessions with distinct time windows using explicit timestamps
    await db.execute(
        "INSERT INTO dry_run_sessions (started_at, ended_at, initial_cash, current_cash) VALUES (?, ?, 800, 800)",
        ("2025-06-01 10:00:00", "2025-06-01 11:00:00"),
    )
    await db.execute(
        "INSERT INTO dry_run_sessions (started_at, ended_at, initial_cash, current_cash) VALUES (?, ?, 800, 800)",
        ("2025-06-01 12:00:00", "2025-06-01 13:00:00"),
    )
    await db.commit()
    sessions = await queries.get_dry_run_sessions(db)
    sid2, sid1 = sessions[0]["id"], sessions[1]["id"]  # DESC order

    # API call during session 1
    await db.execute(
        "INSERT INTO api_calls (model, purpose, input_tokens, output_tokens, cost_usd, is_dry_run, called_at) VALUES (?, ?, ?, ?, ?, 1, ?)",
        ("haiku", "s1-call", 1000, 500, 0.005, "2025-06-01 10:30:00"),
    )
    # API call during session 2
    await db.execute(
        "INSERT INTO api_calls (model, purpose, input_tokens, output_tokens, cost_usd, is_dry_run, called_at) VALUES (?, ?, ?, ?, ?, 1, ?)",
        ("haiku", "s2-call", 2000, 800, 0.010, "2025-06-01 12:30:00"),
    )
    await db.commit()

    cost1 = await queries.get_session_api_cost(db, sid1)
    cost2 = await queries.get_session_api_cost(db, sid2)

    assert abs(cost1 - 0.005) < 0.0001, "Session 1 should only count its own calls"
    assert abs(cost2 - 0.010) < 0.0001, "Session 2 should only count its own calls"


@pytest.mark.asyncio
async def test_dry_run_cash_tracking(db):
    """Dry run cash is tracked in dry_run_sessions, independent of portfolio."""
    await queries.init_portfolio(db, 800.0, "CHF")

    sid = await queries.start_dry_run_session(db, 1000.0)

    # Dry run cash starts at initial_cash
    cash = await queries.get_dry_run_cash(db, sid)
    assert cash == 1000.0

    # Update dry run cash
    await queries.update_dry_run_cash(db, 900.0, sid)
    cash = await queries.get_dry_run_cash(db, sid)
    assert cash == 900.0

    # Live portfolio is untouched
    p = await queries.get_portfolio(db)
    assert p["cash"] == 800.0


@pytest.mark.asyncio
async def test_get_portfolio_dry_run_mode(db):
    """get_portfolio with is_dry_run=True reads from dry_run_sessions."""
    await queries.init_portfolio(db, 800.0, "CHF")
    sid = await queries.start_dry_run_session(db, 1000.0)
    await queries.update_dry_run_cash(db, 950.0, sid)

    live = await queries.get_portfolio(db)
    assert live["cash"] == 800.0

    dry = await queries.get_portfolio(db, is_dry_run=True, session_id=sid)
    assert dry["cash"] == 950.0


@pytest.mark.asyncio
async def test_update_cash_dry_run_mode(db):
    """update_cash with is_dry_run=True writes to dry_run_sessions, not portfolio."""
    await queries.init_portfolio(db, 800.0, "CHF")
    sid = await queries.start_dry_run_session(db, 1000.0)

    await queries.update_cash(db, 500.0, is_dry_run=True, session_id=sid)

    # Dry run cash updated
    cash = await queries.get_dry_run_cash(db, sid)
    assert cash == 500.0

    # Live cash unchanged
    p = await queries.get_portfolio(db)
    assert p["cash"] == 800.0


@pytest.mark.asyncio
async def test_session_api_cost(db):
    """get_session_api_cost returns only dry run API costs since session start."""
    sid = await queries.start_dry_run_session(db, 800.0)

    # Record a live call and a dry run call
    await queries.record_api_call(db, "haiku", "live-screening", 1000, 500, 0.003, is_dry_run=False)
    await queries.record_api_call(db, "haiku", "dry-screening", 2000, 800, 0.010, is_dry_run=True)
    await queries.record_api_call(db, "haiku", "dry-analysis", 1500, 600, 0.007, is_dry_run=True)

    cost = await queries.get_session_api_cost(db, sid)
    assert abs(cost - 0.017) < 0.0001, "Should only include dry run calls"


@pytest.mark.asyncio
async def test_update_dry_run_cash_no_session_id_targets_latest(db):
    """update_dry_run_cash(session_id=None) must only update the most recent active session."""
    # Create two sessions with explicit timestamps (one closed, one open)
    await db.execute(
        "INSERT INTO dry_run_sessions (id, started_at, ended_at, initial_cash, current_cash) VALUES (?, ?, ?, 800, 800)",
        (100, "2025-06-01 10:00:00", "2025-06-01 11:00:00"),
    )
    await db.execute(
        "INSERT INTO dry_run_sessions (id, started_at, initial_cash, current_cash) VALUES (?, ?, 1000, 1000)",
        (101, "2025-06-02 10:00:00"),
    )
    await db.commit()

    # Update without session_id — should hit only session 101 (the active one)
    await queries.update_dry_run_cash(db, 900.0)

    s100 = await queries.get_dry_run_session(db, 100)
    s101 = await queries.get_dry_run_session(db, 101)
    assert s100["current_cash"] == 800, "Closed session must not be modified"
    assert s101["current_cash"] == 900.0, "Active session should be updated"


@pytest.mark.asyncio
async def test_snapshot_with_fx_rate(db):
    """record_snapshot stores usd_chf_rate."""
    await queries.record_snapshot(db, 700.0, 100.0, spy_price=450.0, benchmark_value=800.0, usd_chf_rate=0.8812)
    snaps = await queries.get_snapshots(db)
    assert len(snaps) == 1
    assert snaps[0]["usd_chf_rate"] == 0.8812


@pytest.mark.asyncio
async def test_snapshot_without_fx_rate(db):
    """record_snapshot works without usd_chf_rate (backward compat)."""
    await queries.record_snapshot(db, 700.0, 100.0)
    snaps = await queries.get_snapshots(db)
    assert snaps[0]["usd_chf_rate"] is None


@pytest.mark.asyncio
async def test_get_first_fx_rate_with_data(db):
    """get_first_fx_rate returns earliest non-NULL usd_chf_rate."""
    await queries.record_snapshot(db, 700.0, 100.0)  # No FX
    await queries.record_snapshot(db, 700.0, 100.0, usd_chf_rate=0.8800)
    await queries.record_snapshot(db, 700.0, 100.0, usd_chf_rate=0.8850)

    first = await queries.get_first_fx_rate(db)
    assert first == 0.8800


@pytest.mark.asyncio
async def test_get_first_fx_rate_no_data(db):
    """get_first_fx_rate returns None when no FX data exists."""
    await queries.record_snapshot(db, 700.0, 100.0)
    first = await queries.get_first_fx_rate(db)
    assert first is None


@pytest.mark.asyncio
async def test_benchmark_summary_with_fx(db):
    """get_benchmark_summary includes FX fields when usd_chf_rate is present."""
    await db.execute(
        """INSERT INTO portfolio_snapshots (cash, positions_value, total_value, is_dry_run, spy_price, benchmark_value, usd_chf_rate, snapshot_at)
           VALUES (?, ?, ?, 0, ?, ?, ?, ?)""",
        (700.0, 100.0, 800.0, 450.0, 800.0, 0.8800, "2025-06-01 10:00:00"),
    )
    await db.execute(
        """INSERT INTO portfolio_snapshots (cash, positions_value, total_value, is_dry_run, spy_price, benchmark_value, usd_chf_rate, snapshot_at)
           VALUES (?, ?, ?, 0, ?, ?, ?, ?)""",
        (720.0, 120.0, 840.0, 460.0, 817.8, 0.8700, "2025-06-02 10:00:00"),
    )
    await db.commit()

    summary = await queries.get_benchmark_summary(db)
    assert summary is not None
    assert summary["initial_fx_rate"] == 0.8800
    assert summary["current_fx_rate"] == 0.8700
    assert "fx_change_pct" in summary
    assert "benchmark_value_chf" in summary
    assert "fx_impact_pct" in summary
    # USD weakened (rate dropped): fx_change_pct should be negative
    assert summary["fx_change_pct"] < 0


@pytest.mark.asyncio
async def test_benchmark_summary_without_fx(db):
    """get_benchmark_summary omits FX fields when usd_chf_rate is NULL."""
    await db.execute(
        """INSERT INTO portfolio_snapshots (cash, positions_value, total_value, is_dry_run, spy_price, benchmark_value, snapshot_at)
           VALUES (?, ?, ?, 0, ?, ?, ?)""",
        (700.0, 100.0, 800.0, 450.0, 800.0, "2025-06-01 10:00:00"),
    )
    await db.execute(
        """INSERT INTO portfolio_snapshots (cash, positions_value, total_value, is_dry_run, spy_price, benchmark_value, snapshot_at)
           VALUES (?, ?, ?, 0, ?, ?, ?)""",
        (720.0, 120.0, 840.0, 460.0, 817.8, "2025-06-02 10:00:00"),
    )
    await db.commit()

    summary = await queries.get_benchmark_summary(db)
    assert summary is not None
    assert "current_fx_rate" not in summary


@pytest.mark.asyncio
async def test_daily_spend_excludes_dry_run(db):
    """get_daily_spend(include_dry_run=False) should exclude dry run costs."""
    await queries.record_api_call(db, "haiku", "screening", 1000, 500, 0.003, is_dry_run=False)
    await queries.record_api_call(db, "haiku", "dry-screening", 2000, 800, 0.010, is_dry_run=True)

    all_days = await queries.get_daily_spend(db, days=30, include_dry_run=True)
    live_days = await queries.get_daily_spend(db, days=30, include_dry_run=False)

    all_total = sum(d["total_cost"] for d in all_days)
    live_total = sum(d["total_cost"] for d in live_days)

    assert abs(all_total - 0.013) < 0.0001
    assert abs(live_total - 0.003) < 0.0001


# ── Business days helper ─────────────────────────────────────────────


class TestBusinessDaysAgo:
    def test_weekday_counts_only_business_days(self):
        """On a Wednesday, 5 business days ago is the previous Wednesday."""
        # Fix "now" to Wednesday 2026-03-11 12:00 UTC
        fake_now = datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc)
        with patch("paper_trader.db.queries.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = _business_days_ago(5)
        # 5 business days back from Wed Mar 11:
        # Mar 10 (Mon), Mar 9 (Sun skip), Mar 8 (Sat skip), Mar 6 (Fri),
        # Mar 5 (Thu), Mar 4 (Wed) = 5 business days
        assert result.weekday() < 5  # Must be a weekday
        assert result.date() == datetime(2026, 3, 4).date()

    def test_monday_skips_weekend(self):
        """On Monday, 1 business day ago is Friday."""
        fake_now = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)  # Monday
        with patch("paper_trader.db.queries.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = _business_days_ago(1)
        assert result.date() == datetime(2026, 3, 6).date()  # Friday

    def test_seven_business_days(self):
        """7 business days back spans ~9 calendar days (crossing a weekend)."""
        fake_now = datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc)  # Wednesday
        with patch("paper_trader.db.queries.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = _business_days_ago(7)
        # 7 biz days back from Wed Mar 11:
        # Tue 10, Mon 9, (skip Sun 8, Sat 7), Fri 6, Thu 5, Wed 4, Tue 3, Mon 2
        # = Mon Mar 2 (7th business day back)
        assert result.date() == datetime(2026, 3, 2).date()


# ── Recent trades with business days ─────────────────────────────────


@pytest.mark.asyncio
async def test_get_recent_trades_business_days(db):
    """get_recent_trades(business_days=N) returns trades from N business days back."""
    # Insert trades: one 2 business days ago, one 10 business days ago
    recent_dt = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    old_dt = (datetime.now(timezone.utc) - timedelta(days=15)).strftime("%Y-%m-%d %H:%M:%S")

    await db.execute(
        "INSERT INTO trades (symbol, action, shares, price, total, is_dry_run, executed_at) VALUES (?, ?, ?, ?, ?, 0, ?)",
        ("AAPL", "BUY", 1.0, 150.0, 150.0, recent_dt),
    )
    await db.execute(
        "INSERT INTO trades (symbol, action, shares, price, total, is_dry_run, executed_at) VALUES (?, ?, ?, ?, ?, 0, ?)",
        ("GOOGL", "BUY", 1.0, 140.0, 140.0, old_dt),
    )
    await db.commit()

    trades_7d = await queries.get_recent_trades(db, business_days=7)
    assert len(trades_7d) == 1
    assert trades_7d[0]["symbol"] == "AAPL"

    trades_20d = await queries.get_recent_trades(db, business_days=20)
    assert len(trades_20d) == 2


@pytest.mark.asyncio
async def test_get_recent_trades_hours_backward_compat(db):
    """get_recent_trades(hours=N) still works for backward compatibility."""
    recent_dt = (datetime.now(timezone.utc) - timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S")
    await db.execute(
        "INSERT INTO trades (symbol, action, shares, price, total, is_dry_run, executed_at) VALUES (?, ?, ?, ?, ?, 0, ?)",
        ("AAPL", "BUY", 1.0, 150.0, 150.0, recent_dt),
    )
    await db.commit()

    trades = await queries.get_recent_trades(db, hours=48)
    assert len(trades) == 1


# ── Symbol trade summary ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_symbol_trade_summary(db):
    """get_symbol_trade_summary returns net P&L per symbol with round-trips."""
    now = datetime.now(timezone.utc)
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

    # RTX: BUY 100, SELL 95, comm 1.56 each → net = 95 - 100 - 3.12 = -8.12
    await db.execute(
        "INSERT INTO trades (symbol, action, shares, price, total, commission_chf, is_dry_run, executed_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
        ("RTX", "BUY", 1.0, 200.0, 100.0, 1.56, yesterday),
    )
    await db.execute(
        "INSERT INTO trades (symbol, action, shares, price, total, commission_chf, is_dry_run, executed_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
        ("RTX", "SELL", 1.0, 195.0, 95.0, 1.56, yesterday),
    )
    # AAPL: only a BUY → should NOT appear (no round-trip)
    await db.execute(
        "INSERT INTO trades (symbol, action, shares, price, total, commission_chf, is_dry_run, executed_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
        ("AAPL", "BUY", 1.0, 150.0, 150.0, 0.78, yesterday),
    )
    await db.commit()

    summary = await queries.get_symbol_trade_summary(db, business_days=7)
    assert len(summary) == 1
    assert summary[0]["symbol"] == "RTX"
    assert summary[0]["buys"] == 1
    assert summary[0]["sells"] == 1
    assert summary[0]["net_pnl_chf"] == pytest.approx(-8.12, abs=0.01)


# ── Churn candidates with business days ──────────────────────────────


@pytest.mark.asyncio
async def test_get_churn_candidates_business_days(db):
    """get_churn_candidates(business_days=N) uses business-day lookback."""
    recent_dt = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

    await db.execute(
        "INSERT INTO trades (symbol, action, shares, price, total, commission_chf, is_dry_run, executed_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
        ("RTX", "BUY", 1.0, 200.0, 200.0, 0.78, recent_dt),
    )
    await db.execute(
        "INSERT INTO trades (symbol, action, shares, price, total, commission_chf, is_dry_run, executed_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
        ("RTX", "SELL", 1.0, 195.0, 195.0, 0.78, recent_dt),
    )
    await db.commit()

    candidates = await queries.get_churn_candidates(db, business_days=7)
    assert len(candidates) == 1
    assert candidates[0]["symbol"] == "RTX"


# ── Weekly trade count ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_count_trades_this_week_empty(db):
    """Returns 0 when no trades exist."""
    count = await queries.count_trades_this_week(db)
    assert count == 0


@pytest.mark.asyncio
async def test_count_trades_this_week_counts_recent(db):
    """Counts trades from the current week only."""
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d %H:%M:%S")
    two_weeks_ago = (now - timedelta(days=14)).strftime("%Y-%m-%d %H:%M:%S")

    # Trade today (should be counted)
    await db.execute(
        "INSERT INTO trades (symbol, action, shares, price, total, is_dry_run, executed_at) VALUES (?, ?, ?, ?, ?, 0, ?)",
        ("AAPL", "BUY", 1.0, 150.0, 150.0, today),
    )
    # Trade 2 weeks ago (should NOT be counted)
    await db.execute(
        "INSERT INTO trades (symbol, action, shares, price, total, is_dry_run, executed_at) VALUES (?, ?, ?, ?, ?, 0, ?)",
        ("GOOGL", "BUY", 1.0, 140.0, 140.0, two_weeks_ago),
    )
    await db.commit()

    count = await queries.count_trades_this_week(db)
    assert count == 1


# ── Latest confidence per symbol ─────────────────────────────────────


@pytest.mark.asyncio
async def test_latest_confidence_per_symbol(db):
    """Returns the most recent confidence for each requested symbol."""
    # Two decisions for AAPL (should return the latest)
    await queries.record_decision(db, "AAPL", "HOLD", 0.3, "weak", "test-model")
    await queries.record_decision(db, "AAPL", "HOLD", 0.7, "strong", "test-model")
    # One for GOOGL
    await queries.record_decision(db, "GOOGL", "BUY", 0.8, "catalyst", "test-model")

    result = await queries.get_latest_confidence_per_symbol(db, ["AAPL", "GOOGL", "MSFT"])
    assert result["AAPL"] == pytest.approx(0.7)
    assert result["GOOGL"] == pytest.approx(0.8)
    assert "MSFT" not in result  # No decision for MSFT


@pytest.mark.asyncio
async def test_latest_confidence_empty_symbols(db):
    """Returns empty dict for empty symbol list."""
    result = await queries.get_latest_confidence_per_symbol(db, [])
    assert result == {}
