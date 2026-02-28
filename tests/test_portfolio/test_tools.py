"""Tests for portfolio/tools.py — trading tools and rule enforcement."""

from __future__ import annotations

import pytest

from paper_trader.portfolio.tools import (
    check_etf_overlap,
    check_sector_cap,
    check_stop_losses,
    compute_correlation_matrix,
    compute_relative_strength,
    get_sector,
    get_sector_exposure,
)


def _pos(symbol: str, shares: float, avg_cost: float, stop: float | None = None, pid: int = 1) -> dict:
    return {"id": pid, "symbol": symbol, "shares": shares, "avg_cost": avg_cost, "stop_loss_price": stop}


# ── get_sector ───────────────────────────────────────────────────────

class TestGetSector:
    def test_known_stock(self):
        assert get_sector("AAPL") == "Technology"

    def test_unknown_stock(self):
        assert get_sector("ZZZZ") == "Unknown"

    def test_etf(self):
        assert get_sector("QQQ") == "ETF"


# ── get_sector_exposure ──────────────────────────────────────────────

class TestSectorExposure:
    def test_single_position(self):
        positions = [_pos("AAPL", 2, 150)]
        prices = {"AAPL": 160.0}
        result = get_sector_exposure(positions, prices)
        assert "Technology" in result
        assert result["Technology"]["pct"] == pytest.approx(100.0)
        assert result["Technology"]["symbols"] == ["AAPL"]

    def test_two_sectors(self):
        positions = [_pos("AAPL", 1, 100, pid=1), _pos("JPM", 1, 100, pid=2)]
        prices = {"AAPL": 100.0, "JPM": 100.0}
        result = get_sector_exposure(positions, prices)
        assert "Technology" in result
        assert "Financials" in result
        assert result["Technology"]["pct"] == pytest.approx(50.0)
        assert result["Financials"]["pct"] == pytest.approx(50.0)


# ── check_sector_cap ────────────────────────────────────────────────

class TestSectorCap:
    def test_within_cap(self):
        # Portfolio: 500 cash + 100 MSFT = 600, buying 50 AAPL → Tech = 150/600 = 25% < 60%
        positions = [_pos("MSFT", 1, 100)]
        prices = {"MSFT": 100.0}
        result = check_sector_cap("AAPL", 50.0, positions, prices, cap_pct=0.60, portfolio_value=600.0)
        assert result["ok"]

    def test_exceeds_cap(self):
        # Portfolio total = 350, Tech = 300/350 = 85.7% > 60%
        positions = [_pos("MSFT", 1, 100), _pos("JPM", 1, 50, pid=2)]
        prices = {"MSFT": 100.0, "JPM": 50.0}
        result = check_sector_cap("AAPL", 200.0, positions, prices, cap_pct=0.60, portfolio_value=350.0)
        assert not result["ok"]
        assert "sector cap" in result["reason"].lower()

    def test_etf_exempt(self):
        positions = [_pos("QQQ", 10, 400)]
        prices = {"QQQ": 400.0}
        result = check_sector_cap("SPY", 500.0, positions, prices, cap_pct=0.60, portfolio_value=4500.0)
        assert result["ok"]


# ── check_etf_overlap ───────────────────────────────────────────────

class TestETFOverlap:
    def test_no_overlap(self):
        result = check_etf_overlap("JPM", ["AAPL", "MSFT"])
        assert result is None

    def test_stock_overlaps_with_held_etf(self):
        result = check_etf_overlap("AAPL", ["QQQ", "JPM"])
        assert result is not None
        assert "QQQ" in result

    def test_buying_etf_with_held_components(self):
        result = check_etf_overlap("XLK", ["AAPL", "NVDA"])
        assert result is not None
        assert "AAPL" in result
        assert "NVDA" in result

    def test_spy_exempt(self):
        # SPY has empty components list → no overlap flagged
        result = check_etf_overlap("SPY", ["AAPL", "MSFT", "GOOGL"])
        assert result is None


# ── check_stop_losses ────────────────────────────────────────────────

class TestStopLoss:
    def test_not_triggered(self):
        positions = [_pos("AAPL", 2, 150, stop=140)]
        prices = {"AAPL": 155.0}
        result = check_stop_losses(positions, prices)
        assert len(result) == 0

    def test_triggered(self):
        positions = [_pos("AAPL", 2, 150, stop=140)]
        prices = {"AAPL": 135.0}
        result = check_stop_losses(positions, prices)
        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["current_price"] == 135.0

    def test_no_stop_set(self):
        positions = [_pos("AAPL", 2, 150, stop=None)]
        prices = {"AAPL": 100.0}
        result = check_stop_losses(positions, prices)
        assert len(result) == 0

    def test_exact_stop_price(self):
        positions = [_pos("AAPL", 2, 150, stop=140)]
        prices = {"AAPL": 140.0}
        result = check_stop_losses(positions, prices)
        assert len(result) == 1  # Triggered at exact stop


# ── compute_correlation_matrix ───────────────────────────────────────

class TestCorrelation:
    @pytest.mark.asyncio
    async def test_two_correlated_symbols(self):
        class MockMarket:
            async def get_current_prices(self, symbols):
                return {}
            async def get_price_history(self, symbol, period="1mo", interval="1d"):
                if symbol == "A":
                    return [{"close": 100 + i} for i in range(20)]
                else:
                    return [{"close": 50 + i * 0.5} for i in range(20)]
            async def get_market_movers(self):
                return {}

        result = await compute_correlation_matrix(["A", "B"], MockMarket())
        assert "matrix" in result
        assert result["avg_correlation"] > 0.9  # Perfectly correlated

    @pytest.mark.asyncio
    async def test_single_symbol_empty(self):
        class MockMarket:
            async def get_current_prices(self, symbols):
                return {}
            async def get_price_history(self, symbol, period="1mo", interval="1d"):
                return [{"close": 100 + i} for i in range(20)]
            async def get_market_movers(self):
                return {}

        result = await compute_correlation_matrix(["A"], MockMarket())
        assert result["matrix"] == {}
        assert result["avg_correlation"] == 0.0


# ── compute_relative_strength ────────────────────────────────────────

class TestRelativeStrength:
    @pytest.mark.asyncio
    async def test_basic_ranking(self):
        class MockMarket:
            async def get_current_prices(self, symbols):
                return {}
            async def get_price_history(self, symbol, period="1mo", interval="1d"):
                if symbol == "SPY":
                    return [{"close": 100}, {"close": 105}]  # +5%
                elif symbol == "STRONG":
                    return [{"close": 100}, {"close": 115}]  # +15%
                else:
                    return [{"close": 100}, {"close": 102}]  # +2%
            async def get_market_movers(self):
                return {}

        result = await compute_relative_strength(["STRONG", "WEAK"], MockMarket(), benchmark="SPY")
        assert len(result) == 2
        assert result[0]["symbol"] == "STRONG"
        assert result[0]["rs_ratio"] > 1.0
        assert result[1]["symbol"] == "WEAK"
        assert result[1]["rs_ratio"] < 1.0
