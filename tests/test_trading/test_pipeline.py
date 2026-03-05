import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx

from paper_trader.ai.calculator import ToolUseAudit
from paper_trader.ai.client import AIClient, APIPausedError
from paper_trader.ai.models import AnalysisResult, StockDecision
from paper_trader.config import MODEL_HAIKU, settings
from paper_trader.db import queries
from paper_trader.market.prices import HistoricalDataProvider
from paper_trader.trading.pipeline import (
    run_trading_pipeline,
    _retry_on_overload,
    _is_overloaded,
    OVERLOADED_INITIAL_DELAY,
    OVERLOADED_BACKOFF_FACTOR,
    OVERLOADED_MAX_DELAY,
    OVERLOADED_MAX_TOTAL_SECONDS,
)


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
        assert snaps[0]["benchmark_value"] == 1000.0  # First snapshot = initial cash


class TestToolUseVerification:
    """Test post-analysis tool-use evidence checks."""

    @pytest.mark.asyncio
    async def test_buy_without_position_size_warning(self, pipeline_setup, caplog):
        """When require_tool_evidence=False, BUY without position_size logs a warning but proceeds."""
        db, ai_client, market = pipeline_setup

        # Enable tool use but NOT strict mode
        original_enable = settings.enable_tool_use
        original_require = settings.require_tool_evidence
        object.__setattr__(settings, "enable_tool_use", True)
        object.__setattr__(settings, "require_tool_evidence", False)

        # Build an AnalysisResult with tool_audit that has NO position_size call
        audit = ToolUseAudit()
        audit.turns = 1
        # Record only a pnl call (no position_size for AAPL)
        audit.record("calculate_pnl", {"symbol": "GOOGL"}, {"pnl_chf": 10})

        analysis = AnalysisResult(
            decisions=[StockDecision(symbol="AAPL", action="BUY", confidence=0.8, reasoning="test", target_allocation_pct=15)],
            market_context="test",
            tool_audit=audit,
        )

        async def mock_screening(*a, **kw):
            from paper_trader.ai.models import ScreeningResult
            return ScreeningResult(market_summary="ok")

        try:
            with patch("paper_trader.trading.pipeline.run_analysis", new_callable=AsyncMock, return_value=analysis):
                with patch("paper_trader.trading.pipeline.run_screening", new_callable=AsyncMock, side_effect=mock_screening):
                    import logging
                    with caplog.at_level(logging.WARNING):
                        result = await run_trading_pipeline(db, ai_client, market, run_type="full")

            # BUY should NOT be rejected — just a warning
            assert any("BUY AAPL without position_size" in r.message for r in caplog.records)
            # The decision should still be BUY (trade attempt), not downgraded to HOLD
            assert result["tool_use_count"] == 1
        finally:
            object.__setattr__(settings, "enable_tool_use", original_enable)
            object.__setattr__(settings, "require_tool_evidence", original_require)

    @pytest.mark.asyncio
    async def test_buy_without_position_size_rejected(self, pipeline_setup, caplog):
        """When require_tool_evidence=True, BUY without position_size is downgraded to HOLD."""
        db, ai_client, market = pipeline_setup

        original_enable = settings.enable_tool_use
        original_require = settings.require_tool_evidence
        object.__setattr__(settings, "enable_tool_use", True)
        object.__setattr__(settings, "require_tool_evidence", True)

        audit = ToolUseAudit()
        audit.turns = 1
        # No position_size call at all

        analysis = AnalysisResult(
            decisions=[StockDecision(symbol="AAPL", action="BUY", confidence=0.8, reasoning="test", target_allocation_pct=15)],
            market_context="test",
            tool_audit=audit,
        )

        async def mock_screening(*a, **kw):
            from paper_trader.ai.models import ScreeningResult
            return ScreeningResult(market_summary="ok")

        try:
            with patch("paper_trader.trading.pipeline.run_analysis", new_callable=AsyncMock, return_value=analysis):
                with patch("paper_trader.trading.pipeline.run_screening", new_callable=AsyncMock, side_effect=mock_screening):
                    import logging
                    with caplog.at_level(logging.WARNING):
                        result = await run_trading_pipeline(db, ai_client, market, run_type="full")

            # BUY should be rejected → HOLD → no trades
            assert any("REJECT BUY AAPL" in r.message for r in caplog.records)
            # No trades executed (HOLD is skipped)
            assert len(result["trades"]) == 0
        finally:
            object.__setattr__(settings, "enable_tool_use", original_enable)
            object.__setattr__(settings, "require_tool_evidence", original_require)

    @pytest.mark.asyncio
    async def test_buy_with_position_size_passes(self, pipeline_setup):
        """When position_size was called for the symbol, BUY proceeds normally."""
        db, ai_client, market = pipeline_setup

        original_enable = settings.enable_tool_use
        original_require = settings.require_tool_evidence
        object.__setattr__(settings, "enable_tool_use", True)
        object.__setattr__(settings, "require_tool_evidence", True)

        audit = ToolUseAudit()
        audit.turns = 2
        audit.record("position_size", {"symbol": "AAPL"}, {"shares": 0.8})

        analysis = AnalysisResult(
            decisions=[StockDecision(symbol="AAPL", action="BUY", confidence=0.8, reasoning="test", target_allocation_pct=15)],
            market_context="test",
            tool_audit=audit,
        )

        async def mock_screening(*a, **kw):
            from paper_trader.ai.models import ScreeningResult
            return ScreeningResult(market_summary="ok")

        try:
            with patch("paper_trader.trading.pipeline.run_analysis", new_callable=AsyncMock, return_value=analysis):
                with patch("paper_trader.trading.pipeline.run_screening", new_callable=AsyncMock, side_effect=mock_screening):
                    result = await run_trading_pipeline(db, ai_client, market, run_type="full")

            # BUY should proceed — position_size was called for AAPL
            assert len(result["trades"]) == 1
            assert result["trades"][0].get("ok") or result["trades"][0].get("action") == "BUY"
        finally:
            object.__setattr__(settings, "enable_tool_use", original_enable)
            object.__setattr__(settings, "require_tool_evidence", original_require)


def _make_overloaded_error():
    """Create a mock 529 overloaded error."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 529
    mock_response.headers = httpx.Headers({})
    mock_response.json.return_value = {
        "type": "error",
        "error": {"type": "overloaded_error", "message": "Overloaded"},
    }
    return anthropic.APIStatusError(
        message="Overloaded",
        response=mock_response,
        body={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    )


class TestRetryOnOverload:
    """Test exponential backoff retry logic for 529 overloaded errors."""

    def test_is_overloaded_detects_529(self):
        err = _make_overloaded_error()
        assert _is_overloaded(err)

    def test_is_overloaded_ignores_other_errors(self):
        assert not _is_overloaded(ValueError("nope"))

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """No retry needed when first attempt succeeds."""
        first = AsyncMock(return_value="ok")
        factory = AsyncMock()

        result = await _retry_on_overload(first(), factory, "Test")
        assert result == "ok"
        factory.assert_not_called()

    @pytest.mark.asyncio
    async def test_success_after_overload_retry(self):
        """Recovers after one 529 then succeeds."""
        err = _make_overloaded_error()

        async def failing_first():
            raise err

        factory = AsyncMock(return_value="recovered")

        with patch("paper_trader.trading.pipeline.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await _retry_on_overload(failing_first(), factory, "Test")

        assert result == "recovered"
        factory.assert_called_once()
        # First retry should wait OVERLOADED_INITIAL_DELAY seconds
        mock_sleep.assert_called_once()
        assert mock_sleep.call_args[0][0] == OVERLOADED_INITIAL_DELAY

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self):
        """Verify delay doubles each retry: 30, 60, 120, 240, 300 (capped)."""
        err = _make_overloaded_error()
        call_count = 0

        async def failing_first():
            raise err

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 5:
                raise err
            return "finally"

        with patch("paper_trader.trading.pipeline.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await _retry_on_overload(failing_first(), fail_then_succeed, "Test")

        assert result == "finally"
        assert call_count == 5

        delays = [c[0][0] for c in mock_sleep.call_args_list]
        assert delays == [30, 60, 120, 240, 300]

    @pytest.mark.asyncio
    async def test_non_overload_error_not_retried(self):
        """Non-529 errors propagate immediately, no retry."""
        async def failing_first():
            raise ValueError("bad input")

        factory = AsyncMock()

        with pytest.raises(ValueError, match="bad input"):
            await _retry_on_overload(failing_first(), factory, "Test")

        factory.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_overload_on_retry_propagates(self):
        """If a retry raises a non-529 error, it propagates immediately."""
        err_529 = _make_overloaded_error()

        async def failing_first():
            raise err_529

        factory = AsyncMock(side_effect=ValueError("something else broke"))

        with patch("paper_trader.trading.pipeline.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ValueError, match="something else broke"):
                await _retry_on_overload(failing_first(), factory, "Test")

    @pytest.mark.asyncio
    async def test_gives_up_after_max_total_time(self):
        """Stops retrying after OVERLOADED_MAX_TOTAL_SECONDS elapsed."""
        err = _make_overloaded_error()

        async def failing_first():
            raise err

        factory = AsyncMock(side_effect=err)

        with patch("paper_trader.trading.pipeline.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with patch(
                "paper_trader.trading.pipeline.OVERLOADED_MAX_TOTAL_SECONDS", 180,
            ):
                # With max 180s: delays 30+60+120 = 210 > 180
                # So it should do 30+60+90(capped to remaining) = 180
                with pytest.raises(anthropic.APIStatusError):
                    await _retry_on_overload(failing_first(), factory, "Test")

        total_waited = sum(c[0][0] for c in mock_sleep.call_args_list)
        assert total_waited <= 180
