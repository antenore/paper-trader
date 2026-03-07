"""Tests for churn detection (RULE 010) and currency attribution."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from paper_trader.portfolio.tools import (
    compute_currency_attribution,
    detect_churn,
)


# ── detect_churn ─────────────────────────────────────────────────────


class TestDetectChurn:
    def _now_minus(self, hours: int) -> str:
        """UTC datetime string N hours ago."""
        dt = datetime.now(timezone.utc) - timedelta(hours=hours)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def test_in_cooloff(self):
        """Symbol sold 12h ago should be in 48h cooloff."""
        candidates = [{
            "symbol": "RTX",
            "trade_count": 4,
            "buys": 2,
            "sells": 2,
            "total_commission": 4.50,
            "last_sell_at": self._now_minus(12),
        }]
        alerts = detect_churn(candidates, cooloff_hours=48)
        assert len(alerts) == 1
        assert alerts[0]["symbol"] == "RTX"
        assert alerts[0]["in_cooloff"] is True
        assert alerts[0]["cooloff_remaining_h"] > 30
        assert alerts[0]["round_trips"] == 2
        assert alerts[0]["commission_cost"] == 4.50

    def test_cooloff_expired(self):
        """Symbol sold 60h ago should NOT be in 48h cooloff."""
        candidates = [{
            "symbol": "XLE",
            "trade_count": 2,
            "buys": 1,
            "sells": 1,
            "total_commission": 2.00,
            "last_sell_at": self._now_minus(60),
        }]
        alerts = detect_churn(candidates, cooloff_hours=48)
        assert len(alerts) == 1
        assert alerts[0]["in_cooloff"] is False
        assert alerts[0]["cooloff_remaining_h"] == 0

    def test_empty_candidates(self):
        assert detect_churn([]) == []

    def test_round_trips_min_of_buys_sells(self):
        """Round trips = min(buys, sells)."""
        candidates = [{
            "symbol": "NVDA",
            "trade_count": 5,
            "buys": 3,
            "sells": 2,
            "total_commission": 5.00,
            "last_sell_at": self._now_minus(1),
        }]
        alerts = detect_churn(candidates, cooloff_hours=48)
        assert alerts[0]["round_trips"] == 2

    def test_null_last_sell(self):
        """Handle missing last_sell_at gracefully."""
        candidates = [{
            "symbol": "TEST",
            "trade_count": 2,
            "buys": 1,
            "sells": 1,
            "total_commission": 1.00,
            "last_sell_at": None,
        }]
        alerts = detect_churn(candidates, cooloff_hours=48)
        assert alerts[0]["in_cooloff"] is False


# ── compute_currency_attribution ─────────────────────────────────────


class TestCurrencyAttribution:
    def test_basic_attribution(self):
        """USD weakening should show negative FX drag."""
        summary = {
            "portfolio_return_pct": 2.0,
            "spy_return_pct": 1.5,
            "initial_fx_rate": 0.90,   # USD/CHF was 0.90
            "current_fx_rate": 0.88,   # USD weakened to 0.88
        }
        result = compute_currency_attribution(summary)
        assert result is not None
        assert result["portfolio_chf_return_pct"] == 2.0
        assert result["spy_usd_return_pct"] == 1.5
        assert result["fx_change_pct"] < 0  # USD weakened
        assert result["fx_drag_pct"] < 0    # Negative drag
        assert result["alpha_usd_pct"] == pytest.approx(0.5)

    def test_usd_strengthening(self):
        """USD strengthening should show positive FX boost."""
        summary = {
            "portfolio_return_pct": 3.0,
            "spy_return_pct": 2.0,
            "initial_fx_rate": 0.88,
            "current_fx_rate": 0.92,   # USD strengthened
        }
        result = compute_currency_attribution(summary)
        assert result is not None
        assert result["fx_change_pct"] > 0
        assert result["fx_drag_pct"] > 0  # Actually a boost

    def test_no_fx_data(self):
        """Returns None when FX data is missing."""
        summary = {
            "portfolio_return_pct": 2.0,
            "spy_return_pct": 1.5,
        }
        result = compute_currency_attribution(summary)
        assert result is None

    def test_none_summary(self):
        assert compute_currency_attribution(None) is None

    def test_alpha_consistency(self):
        """alpha_chf = alpha_usd - fx_drag (identity)."""
        summary = {
            "portfolio_return_pct": 5.0,
            "spy_return_pct": 3.0,
            "initial_fx_rate": 0.90,
            "current_fx_rate": 0.87,
        }
        result = compute_currency_attribution(summary)
        assert result is not None
        # alpha_chf = portfolio_return - spy_chf_return
        # alpha_usd = portfolio_return - spy_usd_return
        # fx_drag = spy_chf_return - spy_usd_return
        # So: alpha_chf = alpha_usd - fx_drag
        assert result["alpha_chf_pct"] == pytest.approx(
            result["alpha_usd_pct"] - result["fx_drag_pct"], abs=0.01,
        )


# ── churn query integration ──────────────────────────────────────────


class TestChurnQuery:
    @pytest.mark.asyncio
    async def test_get_churn_candidates(self, db_with_portfolio):
        """Symbols with BUY+SELL in last 5 days should appear as churn candidates."""
        from paper_trader.db import queries

        db = db_with_portfolio
        # RTX: buy then sell (churn candidate)
        await queries.record_trade(db, "RTX", "BUY", 1.0, 200.0, commission_chf=1.0)
        await queries.record_trade(db, "RTX", "SELL", 1.0, 195.0, commission_chf=1.0)
        # AAPL: only buy (NOT a churn candidate)
        await queries.record_trade(db, "AAPL", "BUY", 2.0, 150.0, commission_chf=1.0)

        candidates = await queries.get_churn_candidates(db, days=5)
        assert len(candidates) == 1
        assert candidates[0]["symbol"] == "RTX"
        assert candidates[0]["buys"] == 1
        assert candidates[0]["sells"] == 1
        assert candidates[0]["total_commission"] == 2.0

    @pytest.mark.asyncio
    async def test_no_churn_candidates(self, db_with_portfolio):
        """Only buys → no churn candidates."""
        from paper_trader.db import queries

        db = db_with_portfolio
        await queries.record_trade(db, "AAPL", "BUY", 2.0, 150.0, commission_chf=1.0)
        candidates = await queries.get_churn_candidates(db, days=5)
        assert len(candidates) == 0
