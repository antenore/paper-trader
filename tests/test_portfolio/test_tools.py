"""Tests for portfolio/tools.py — trading tools and rule enforcement."""

from __future__ import annotations

import pytest

from paper_trader.portfolio.tools import (
    check_etf_overlap,
    check_geopolitical_catalysts,
    check_profit_taking_alerts,
    check_sector_cap,
    check_stop_losses,
    compute_correlation_matrix,
    compute_relative_strength,
    get_sector,
    get_sector_exposure,
    recommend_position_consolidation,
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


# ── check_profit_taking_alerts (RULE 014) ─────────────────────────

class TestProfitTakingAlerts:
    def test_no_alert_below_threshold(self):
        positions = [_pos("USO", 10, 80)]
        prices = {"USO": 90.0}  # +12.5%, below 25%
        result = check_profit_taking_alerts(positions, prices, threshold_pct=0.25)
        assert len(result) == 0

    def test_alert_above_threshold(self):
        positions = [_pos("USO", 10, 80)]
        prices = {"USO": 110.0}  # +37.5%
        result = check_profit_taking_alerts(positions, prices, threshold_pct=0.25)
        assert len(result) == 1
        assert result[0]["symbol"] == "USO"
        assert result[0]["gain_pct"] == pytest.approx(37.5, rel=0.01)
        assert result[0]["recommended_sell_shares"] == pytest.approx(3.3, rel=0.1)

    def test_exact_threshold(self):
        positions = [_pos("USO", 4, 100)]
        prices = {"USO": 125.0}  # exactly +25%
        result = check_profit_taking_alerts(positions, prices, threshold_pct=0.25)
        assert len(result) == 1

    def test_multiple_alerts_sorted_by_gain(self):
        positions = [
            _pos("USO", 5, 80, pid=1),
            _pos("CVX", 3, 100, pid=2),
        ]
        prices = {"USO": 120.0, "CVX": 130.0}  # USO +50%, CVX +30%
        result = check_profit_taking_alerts(positions, prices, threshold_pct=0.25)
        assert len(result) == 2
        assert result[0]["symbol"] == "USO"  # Higher gain first
        assert result[1]["symbol"] == "CVX"

    def test_custom_sell_pct(self):
        positions = [_pos("USO", 10, 80)]
        prices = {"USO": 110.0}
        result = check_profit_taking_alerts(positions, prices, threshold_pct=0.25, sell_pct=0.50)
        assert len(result) == 1
        assert result[0]["recommended_sell_shares"] == pytest.approx(5.0)

    def test_fx_conversion(self):
        positions = [_pos("USO", 10, 80)]
        prices = {"USO": 110.0}
        result = check_profit_taking_alerts(positions, prices, threshold_pct=0.25, usd_chf_rate=0.80)
        assert len(result) == 1
        # Position value should be in CHF: 10 * 110 * 0.80 = 880
        assert result[0]["position_value_chf"] == pytest.approx(880.0, rel=0.01)

    def test_no_price_available(self):
        positions = [_pos("USO", 10, 80)]
        prices = {}  # No price
        result = check_profit_taking_alerts(positions, prices, threshold_pct=0.25)
        assert len(result) == 0

    def test_swiss_stock_chf_no_conversion(self):
        positions = [{"id": 1, "symbol": "NESN.SW", "shares": 5, "avg_cost": 100,
                       "stop_loss_price": None, "currency": "CHF"}]
        prices = {"NESN.SW": 130.0}  # +30%
        result = check_profit_taking_alerts(positions, prices, threshold_pct=0.25, usd_chf_rate=0.80)
        assert len(result) == 1
        # CHF position — no FX conversion
        assert result[0]["position_value_chf"] == pytest.approx(650.0)


# ── check_geopolitical_catalysts ──────────────────────────────────

class TestGeopoliticalCatalysts:
    def test_no_matching_news(self):
        news = [{"title": "Apple releases new iPhone", "summary": "Consumer electronics update"}]
        result = check_geopolitical_catalysts(news, ["AAPL"])
        assert len(result) == 0

    def test_escalation_signal(self):
        news = [
            {"title": "Iran strikes back at Israel with missile barrage", "summary": "Escalation in Middle East"},
            {"title": "Oil prices surge on Iran attack fears", "summary": "Strait of Hormuz blockade risk"},
        ]
        result = check_geopolitical_catalysts(news, ["USO", "CVX"])
        assert len(result) >= 1
        iran_alert = next(a for a in result if "Iran" in a["catalyst"])
        assert iran_alert["signal"] == "ESCALATION"
        assert iran_alert["confidence_modifier"] > 0
        assert "USO" in iran_alert["affected_held"]
        assert "CVX" in iran_alert["affected_held"]

    def test_de_escalation_signal(self):
        news = [
            {"title": "Iran and Israel agree to ceasefire talks", "summary": "Diplomatic breakthrough in Middle East peace negotiations"},
            {"title": "Iran diplomatic agreement easing tensions", "summary": "Sanctions relief expected"},
        ]
        result = check_geopolitical_catalysts(news, ["USO", "LMT"])
        assert len(result) >= 1
        iran_alert = next(a for a in result if "Iran" in a["catalyst"])
        assert iran_alert["signal"] == "DE-ESCALATION"
        assert iran_alert["confidence_modifier"] < 0
        assert "USO" in iran_alert["affected_held"]
        assert "LMT" in iran_alert["affected_held"]

    def test_no_held_positions_affected(self):
        news = [
            {"title": "Iran missile strike on Israel escalates conflict", "summary": "Middle East crisis"},
        ]
        result = check_geopolitical_catalysts(news, ["AAPL", "MSFT"])
        assert len(result) >= 1
        iran_alert = next(a for a in result if "Iran" in a["catalyst"])
        assert iran_alert["affected_held"] == []  # No held positions are in the affected list

    def test_taiwan_catalyst(self):
        news = [
            {"title": "China military exercises near Taiwan intensify", "summary": "PLA Navy warship drills near strait"},
        ]
        result = check_geopolitical_catalysts(news, ["NVDA", "AMD"])
        assert len(result) >= 1
        taiwan_alert = next(a for a in result if "Taiwan" in a["catalyst"])
        assert taiwan_alert["signal"] == "ESCALATION"
        assert "NVDA" in taiwan_alert["affected_held"]
        assert "AMD" in taiwan_alert["affected_held"]

    def test_energy_disruption(self):
        news = [
            {"title": "OPEC announces production cut", "summary": "Oil supply shortage expected"},
        ]
        result = check_geopolitical_catalysts(news, ["USO", "XOM"])
        assert len(result) >= 1
        energy_alert = next(a for a in result if "Energy" in a["catalyst"])
        assert energy_alert["signal"] == "ESCALATION"
        assert "USO" in energy_alert["affected_held"]

    def test_multiple_catalysts_detected(self):
        news = [
            {"title": "Iran attacks Israel with missiles", "summary": "Escalation in Middle East"},
            {"title": "China military drills near Taiwan", "summary": "PLA exercises"},
        ]
        result = check_geopolitical_catalysts(news, ["USO", "NVDA"])
        assert len(result) >= 2

    def test_neutral_signal(self):
        news = [
            {"title": "Iran discussed at UN General Assembly", "summary": "Middle East situation review"},
        ]
        result = check_geopolitical_catalysts(news, ["USO"])
        # May or may not match depending on keywords — if it matches, signal should be NEUTRAL
        if result:
            iran_alert = next((a for a in result if "Iran" in a["catalyst"]), None)
            if iran_alert:
                assert iran_alert["signal"] == "NEUTRAL"
                assert iran_alert["confidence_modifier"] == 0.0

    def test_sample_headlines_limited_to_3(self):
        news = [
            {"title": f"Iran conflict escalation report #{i}", "summary": "Middle East update"}
            for i in range(10)
        ]
        result = check_geopolitical_catalysts(news, ["USO"])
        assert len(result) >= 1
        iran_alert = next(a for a in result if "Iran" in a["catalyst"])
        assert len(iran_alert["sample_headlines"]) <= 3

    def test_empty_news(self):
        result = check_geopolitical_catalysts([], ["USO", "NVDA"])
        assert len(result) == 0

    def test_us_china_trade_war(self):
        news = [
            {"title": "New tariff on China imports announced", "summary": "US China trade war escalates with retaliatory measures"},
        ]
        result = check_geopolitical_catalysts(news, ["AAPL", "NVDA"])
        assert len(result) >= 1
        trade_alert = next(a for a in result if "China" in a["catalyst"])
        assert trade_alert["signal"] == "ESCALATION"
        assert "AAPL" in trade_alert["affected_held"]


# ── Position Consolidation Recommender ───────────────────────────────

class TestPositionConsolidation:
    def test_ranks_by_score(self):
        positions = [
            _pos("AAPL", 2.0, 150.0, pid=1),
            _pos("NVDA", 0.5, 800.0, pid=2),
            _pos("ZZZZ", 0.1, 10.0, pid=3),  # tiny position
        ]
        prices = {"AAPL": 155.0, "NVDA": 820.0, "ZZZZ": 9.0}
        confidence = {"AAPL": 0.75, "NVDA": 0.80, "ZZZZ": 0.30}

        result = recommend_position_consolidation(
            positions, prices, confidence, target_count=2,
        )
        assert len(result) == 3
        # First two should be KEEP, last TRIM
        assert result[0]["recommendation"] == "KEEP"
        assert result[1]["recommendation"] == "KEEP"
        assert result[2]["recommendation"].startswith("TRIM")
        # ZZZZ (tiny, low confidence) should be last
        assert result[2]["symbol"] == "ZZZZ"

    def test_no_trim_under_target(self):
        """No TRIM recommendations when positions <= target_count."""
        positions = [_pos("AAPL", 2.0, 150.0, pid=1)]
        prices = {"AAPL": 155.0}
        confidence = {"AAPL": 0.70}

        result = recommend_position_consolidation(
            positions, prices, confidence, target_count=6,
        )
        assert len(result) == 1
        assert result[0]["recommendation"] == "KEEP"

    def test_empty_positions(self):
        result = recommend_position_consolidation([], {}, {})
        assert result == []

    def test_missing_confidence_defaults(self):
        """Positions without confidence data use 0.50 default."""
        positions = [_pos("AAPL", 1.0, 150.0, pid=1)]
        prices = {"AAPL": 155.0}
        confidence = {}  # empty

        result = recommend_position_consolidation(
            positions, prices, confidence, target_count=1,
        )
        assert len(result) == 1
        assert result[0]["confidence"] == 0.50
        assert result[0]["recommendation"] == "KEEP"
