import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from paper_trader.ai.client import AIClient, APIPausedError
from paper_trader.config import MODEL_HAIKU
from paper_trader.db import queries
from paper_trader.market.prices import HistoricalDataProvider
from paper_trader.trading.pipeline import run_trading_pipeline


def make_history(prices: list[float]) -> list[dict]:
    return [
        {"date": f"2025-01-{i+1:02d}", "open": p, "high": p+1, "low": p-1, "close": p, "volume": 1000}
        for i, p in enumerate(prices)
    ]


@pytest.fixture
async def pipeline_setup(db_with_portfolio):
    """Set up pipeline with mock AI and historical market data."""
    db = db_with_portfolio
    ai_client = AIClient(db, api_key="test-key")

    # Historical data for simulation
    market = HistoricalDataProvider(
        {
            "AAPL": make_history([150, 152, 155, 153, 158]),
            "GOOGL": make_history([140, 138, 142, 145, 143]),
            "SPY": make_history([450, 452, 455, 453, 458]),
        },
        current_index=4,  # Latest day
    )

    # Add some watchlist symbols
    await queries.add_to_watchlist(db, "AAPL", "Test")
    await queries.add_to_watchlist(db, "GOOGL", "Test")

    return db, ai_client, market


def mock_screening_response():
    return {
        "watchlist_updates": [
            {"symbol": "AAPL", "action": "KEEP", "reason": "Strong momentum"},
        ],
        "market_summary": "Bullish market",
    }


def mock_analysis_response():
    return {
        "decisions": [
            {"symbol": "AAPL", "action": "BUY", "confidence": 0.8, "reasoning": "Strong earnings", "target_allocation_pct": 15},
        ],
        "market_context": "Positive sentiment",
    }


def mock_analysis_hold_response():
    return {
        "decisions": [
            {"symbol": "AAPL", "action": "HOLD", "confidence": 0.5, "reasoning": "Wait and see"},
        ],
        "market_context": "Uncertain",
    }


def mock_analysis_low_confidence():
    return {
        "decisions": [
            {"symbol": "AAPL", "action": "BUY", "confidence": 0.3, "reasoning": "Weak signal"},
        ],
        "market_context": "Mixed",
    }


class TestTradingPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_buy(self, pipeline_setup):
        db, ai_client, market = pipeline_setup

        call_count = 0
        async def mock_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("purpose") == "screening":
                return mock_screening_response()
            return mock_analysis_response()

        with patch.object(ai_client, "call", side_effect=mock_call):
            with patch("paper_trader.ai.screener.fetch_news", new_callable=AsyncMock, return_value=[]):
                with patch("paper_trader.ai.analyst.fetch_symbol_news", new_callable=AsyncMock, return_value=[]):
                    result = await run_trading_pipeline(db, ai_client, market, run_type="full")

        assert len(result["errors"]) == 0
        assert len(result["trades"]) == 1
        assert result["trades"][0]["ok"]

        # Verify trade was recorded
        trades = await queries.get_trades(db)
        assert len(trades) == 1
        assert trades[0]["action"] == "BUY"

    @pytest.mark.asyncio
    async def test_hold_decision_no_trade(self, pipeline_setup):
        db, ai_client, market = pipeline_setup

        async def mock_call(*args, **kwargs):
            if kwargs.get("purpose") == "screening":
                return mock_screening_response()
            return mock_analysis_hold_response()

        with patch.object(ai_client, "call", side_effect=mock_call):
            with patch("paper_trader.ai.screener.fetch_news", new_callable=AsyncMock, return_value=[]):
                with patch("paper_trader.ai.analyst.fetch_symbol_news", new_callable=AsyncMock, return_value=[]):
                    result = await run_trading_pipeline(db, ai_client, market, run_type="full")

        assert len(result["trades"]) == 0

    @pytest.mark.asyncio
    async def test_low_confidence_skipped(self, pipeline_setup):
        db, ai_client, market = pipeline_setup

        async def mock_call(*args, **kwargs):
            if kwargs.get("purpose") == "screening":
                return mock_screening_response()
            return mock_analysis_low_confidence()

        with patch.object(ai_client, "call", side_effect=mock_call):
            with patch("paper_trader.ai.screener.fetch_news", new_callable=AsyncMock, return_value=[]):
                with patch("paper_trader.ai.analyst.fetch_symbol_news", new_callable=AsyncMock, return_value=[]):
                    result = await run_trading_pipeline(db, ai_client, market, run_type="full")

        assert len(result["trades"]) == 0

    @pytest.mark.asyncio
    async def test_snapshot_only(self, pipeline_setup):
        db, ai_client, market = pipeline_setup

        result = await run_trading_pipeline(db, ai_client, market, run_type="snapshot")
        assert result["snapshot"] is True

        snaps = await queries.get_snapshots(db)
        assert len(snaps) == 1

    @pytest.mark.asyncio
    async def test_quick_skip_no_movement_no_positions(self, pipeline_setup):
        db, ai_client, market = pipeline_setup

        # Use data with <1% movements (new threshold)
        small_market = HistoricalDataProvider(
            {
                "AAPL": make_history([150, 150.5, 150.2, 150.8, 151.0]),
                "GOOGL": make_history([140, 140.2, 140.1, 140.3, 140.5]),
            },
            current_index=4,
        )

        result = await run_trading_pipeline(db, ai_client, small_market, run_type="quick")
        assert result.get("skipped") is True

    @pytest.mark.asyncio
    async def test_quick_with_positions_analyzes_anyway(self, pipeline_setup):
        """Even without significant movement, midday check should analyze if we have open positions."""
        db, ai_client, market = pipeline_setup

        # Create an open position so we have positions to check
        await db.execute(
            "INSERT INTO positions (symbol, shares, avg_cost, is_dry_run) VALUES (?, ?, ?, ?)",
            ("AAPL", 1.0, 150.0, 0),
        )
        await db.commit()

        # Use data with <1% movements
        small_market = HistoricalDataProvider(
            {
                "AAPL": make_history([150, 150.5, 150.2, 150.8, 151.0]),
                "GOOGL": make_history([140, 140.2, 140.1, 140.3, 140.5]),
            },
            current_index=4,
        )

        async def mock_call(*args, **kwargs):
            return mock_analysis_hold_response()

        with patch.object(ai_client, "call", side_effect=mock_call):
            with patch("paper_trader.ai.analyst.fetch_symbol_news", new_callable=AsyncMock, return_value=[]):
                result = await run_trading_pipeline(db, ai_client, small_market, run_type="quick")

        # Should NOT be skipped — we have open positions
        assert result.get("skipped") is not True
        assert "analysis" in result

    @pytest.mark.asyncio
    async def test_api_paused_handles_gracefully(self, pipeline_setup):
        db, ai_client, market = pipeline_setup
        await queries.set_setting(db, "api_paused", "true")
        await queries.set_setting(db, "pause_reason", "Test pause")

        result = await run_trading_pipeline(db, ai_client, market, run_type="full")
        assert len(result["errors"]) > 0
        assert "paused" in result["errors"][0].lower()

    @pytest.mark.asyncio
    async def test_snapshot_records_spy_benchmark(self, pipeline_setup):
        """Snapshot should record SPY price and benchmark_value."""
        db, ai_client, market = pipeline_setup

        result = await run_trading_pipeline(db, ai_client, market, run_type="snapshot")
        assert result["snapshot"] is True

        snaps = await queries.get_snapshots(db)
        assert len(snaps) == 1
        # SPY is in the market fixture at index 4 → price 458
        assert snaps[0]["spy_price"] == 458.0
        assert snaps[0]["benchmark_value"] == 800.0  # First snapshot = initial cash
