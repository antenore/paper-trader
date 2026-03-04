import pytest
from unittest.mock import AsyncMock, patch

from paper_trader.ai.client import AIClient
from paper_trader.db import queries
from paper_trader.market.prices import HistoricalDataProvider
from paper_trader.trading.dry_run import run_dry_run, reset_for_live


def make_history(base: float, days: int = 40) -> list[dict]:
    """Generate synthetic price history with small random-ish movements."""
    import math
    prices = []
    for i in range(days):
        price = base + math.sin(i * 0.3) * base * 0.05
        prices.append({
            "date": f"2025-01-{(i % 28) + 1:02d}",
            "open": price * 0.998,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": 1_000_000,
        })
    return prices


@pytest.fixture
async def dry_run_setup(db_with_portfolio):
    db = db_with_portfolio
    ai_client = AIClient(db, api_key="test-key")
    return db, ai_client


class TestDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_with_mock_data(self, dry_run_setup):
        db, ai_client = dry_run_setup

        mock_data = {
            "AAPL": make_history(150, 40),
            "GOOGL": make_history(140, 40),
        }

        screening_response = {
            "watchlist_updates": [{"symbol": "AAPL", "action": "KEEP", "reason": "test"}],
            "market_summary": "Test",
        }
        analysis_response = {
            "decisions": [
                {"symbol": "AAPL", "action": "BUY", "confidence": 0.8, "reasoning": "Test", "target_allocation_pct": 15},
            ],
            "market_context": "Test",
        }

        async def mock_call(*args, **kwargs):
            if kwargs.get("purpose") == "screening":
                return screening_response
            return analysis_response

        with patch.object(ai_client, "call", side_effect=mock_call):
            with patch("paper_trader.trading.dry_run.load_historical_data", new_callable=AsyncMock, return_value=mock_data):
                with patch("paper_trader.ai.screener.fetch_news", new_callable=AsyncMock, return_value=[]):
                    result = await run_dry_run(db, ai_client, symbols=["AAPL", "GOOGL"], days=10)

        assert "session_id" in result
        assert result["days_simulated"] > 0
        assert "error" not in result

        # Check session was recorded
        session = await queries.get_dry_run_session(db, result["session_id"])
        assert session["ended_at"] is not None

    @pytest.mark.asyncio
    async def test_dry_run_no_data(self, dry_run_setup):
        db, ai_client = dry_run_setup

        with patch("paper_trader.trading.dry_run.load_historical_data", new_callable=AsyncMock, return_value={}):
            result = await run_dry_run(db, ai_client, symbols=["FAKE"], days=10)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_dry_run_progress_callback(self, dry_run_setup):
        db, ai_client = dry_run_setup

        mock_data = {"AAPL": make_history(150, 40)}
        progress_calls = []

        async def on_progress(day, total, date, result):
            progress_calls.append((day, total))

        async def mock_call(*args, **kwargs):
            return {
                "watchlist_updates": [],
                "market_summary": "Test",
                "decisions": [],
                "market_context": "Test",
            }

        with patch.object(ai_client, "call", side_effect=mock_call):
            with patch("paper_trader.trading.dry_run.load_historical_data", new_callable=AsyncMock, return_value=mock_data):
                with patch("paper_trader.ai.screener.fetch_news", new_callable=AsyncMock, return_value=[]):
                    await run_dry_run(db, ai_client, symbols=["AAPL"], days=5, on_progress=on_progress)

        assert len(progress_calls) > 0


class TestDryRunCashIsolation:
    """Tests that dry run never touches the live portfolio row."""

    @pytest.mark.asyncio
    async def test_live_cash_untouched_during_dry_run(self, dry_run_setup):
        """Live portfolio cash must not change at all during a dry run."""
        db, ai_client = dry_run_setup
        await queries.update_cash(db, 750.0)  # Simulate live trading state

        mock_data = {"AAPL": make_history(150, 40)}

        async def mock_call(*args, **kwargs):
            return {
                "watchlist_updates": [],
                "market_summary": "Test",
                "decisions": [],
                "market_context": "Test",
            }

        with patch.object(ai_client, "call", side_effect=mock_call):
            with patch("paper_trader.trading.dry_run.load_historical_data", new_callable=AsyncMock, return_value=mock_data):
                with patch("paper_trader.ai.screener.fetch_news", new_callable=AsyncMock, return_value=[]):
                    await run_dry_run(db, ai_client, symbols=["AAPL"], days=5)

        portfolio = await queries.get_portfolio(db)
        assert portfolio["cash"] == 750.0, "Live cash must not be modified by dry run"

    @pytest.mark.asyncio
    async def test_live_cash_untouched_on_no_data(self, dry_run_setup):
        """Live cash untouched even when dry run exits early (no data)."""
        db, ai_client = dry_run_setup
        await queries.update_cash(db, 600.0)

        with patch("paper_trader.trading.dry_run.load_historical_data", new_callable=AsyncMock, return_value={}):
            await run_dry_run(db, ai_client, symbols=["FAKE"], days=10)

        portfolio = await queries.get_portfolio(db)
        assert portfolio["cash"] == 600.0, "Live cash must not be modified on early exit"

    @pytest.mark.asyncio
    async def test_live_cash_untouched_on_finalization_error(self, dry_run_setup):
        """Live cash untouched even if finalization throws."""
        db, ai_client = dry_run_setup
        await queries.update_cash(db, 500.0)

        mock_data = {"AAPL": make_history(150, 40)}

        async def mock_call(*args, **kwargs):
            return {
                "watchlist_updates": [],
                "market_summary": "Test",
                "decisions": [],
                "market_context": "Test",
            }

        with patch.object(ai_client, "call", side_effect=mock_call):
            with patch("paper_trader.trading.dry_run.load_historical_data", new_callable=AsyncMock, return_value=mock_data):
                with patch("paper_trader.ai.screener.fetch_news", new_callable=AsyncMock, return_value=[]):
                    with patch("paper_trader.portfolio.manager.get_portfolio_value", new_callable=AsyncMock, side_effect=Exception("DB error")):
                        result = await run_dry_run(db, ai_client, symbols=["AAPL"], days=5)

        portfolio = await queries.get_portfolio(db)
        assert portfolio["cash"] == 500.0, "Live cash must not be modified even if finalization fails"
        assert "session_id" in result

    @pytest.mark.asyncio
    async def test_dry_run_cash_tracked_in_session(self, dry_run_setup):
        """Dry run should track cash in dry_run_sessions, not portfolio."""
        db, ai_client = dry_run_setup

        mock_data = {"AAPL": make_history(150, 40)}

        async def mock_call(*args, **kwargs):
            return {
                "watchlist_updates": [],
                "market_summary": "Test",
                "decisions": [],
                "market_context": "Test",
            }

        with patch.object(ai_client, "call", side_effect=mock_call):
            with patch("paper_trader.trading.dry_run.load_historical_data", new_callable=AsyncMock, return_value=mock_data):
                with patch("paper_trader.ai.screener.fetch_news", new_callable=AsyncMock, return_value=[]):
                    result = await run_dry_run(db, ai_client, symbols=["AAPL"], days=5)

        session = await queries.get_dry_run_session(db, result["session_id"])
        assert session["current_cash"] is not None, "Dry run session should track current_cash"


class TestDryRunSmallDataset:
    """Tests that dry run handles insufficient historical data gracefully."""

    @pytest.mark.asyncio
    async def test_too_few_days_returns_error(self, dry_run_setup):
        """Dataset with <= 5 days should return an error, not crash."""
        db, ai_client = dry_run_setup

        # Only 4 days of data — sim_days = min(30, 4-5) = -1
        mock_data = {"AAPL": make_history(150, 4)}

        with patch("paper_trader.trading.dry_run.load_historical_data", new_callable=AsyncMock, return_value=mock_data):
            result = await run_dry_run(db, ai_client, symbols=["AAPL"], days=30)

        assert "error" in result
        assert "insufficient" in result["error"].lower() or "data" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_exactly_six_days_works(self, dry_run_setup):
        """6 days of data — sim_days = 1, should work."""
        db, ai_client = dry_run_setup

        mock_data = {"AAPL": make_history(150, 6)}

        async def mock_call(*args, **kwargs):
            return {
                "watchlist_updates": [],
                "market_summary": "Test",
                "decisions": [],
                "market_context": "Test",
            }

        with patch.object(ai_client, "call", side_effect=mock_call):
            with patch("paper_trader.trading.dry_run.load_historical_data", new_callable=AsyncMock, return_value=mock_data):
                with patch("paper_trader.ai.screener.fetch_news", new_callable=AsyncMock, return_value=[]):
                    result = await run_dry_run(db, ai_client, symbols=["AAPL"], days=30)

        assert "error" not in result
        assert result["days_simulated"] == 1


class TestDryRunNoWatchlistMutation:
    """Tests that dry run does not modify the live watchlist."""

    @pytest.mark.asyncio
    async def test_watchlist_unchanged_after_dry_run(self, dry_run_setup):
        db, ai_client = dry_run_setup
        # Set up a known watchlist state
        await queries.add_to_watchlist(db, "SPY", "Live watchlist")
        before = await queries.get_watchlist(db)

        mock_data = {"AAPL": make_history(150, 40)}

        async def mock_call(*args, **kwargs):
            return {
                "watchlist_updates": [],
                "market_summary": "Test",
                "decisions": [],
                "market_context": "Test",
            }

        with patch.object(ai_client, "call", side_effect=mock_call):
            with patch("paper_trader.trading.dry_run.load_historical_data", new_callable=AsyncMock, return_value=mock_data):
                with patch("paper_trader.ai.screener.fetch_news", new_callable=AsyncMock, return_value=[]):
                    await run_dry_run(db, ai_client, symbols=["AAPL", "GOOGL"], days=5)

        after = await queries.get_watchlist(db)
        before_symbols = {w["symbol"] for w in before}
        after_symbols = {w["symbol"] for w in after}
        assert before_symbols == after_symbols, "Dry run should not modify the live watchlist"


class TestDryRunConcurrency:
    """Tests that concurrent dry runs are rejected."""

    @pytest.mark.asyncio
    async def test_second_dry_run_rejected(self, dry_run_setup):
        """Starting a dry run while one is active should return an error."""
        db, ai_client = dry_run_setup

        # Manually start a session without ending it
        await queries.start_dry_run_session(db, 800.0)

        # Now try to start a dry run — should be rejected immediately
        result = await run_dry_run(db, ai_client, symbols=["AAPL"], days=5)
        assert "error" in result
        assert "already running" in result["error"].lower()


class TestDryRunBenchmarkSymbol:
    """Tests that benchmark symbol is always included in dry run symbols."""

    @pytest.mark.asyncio
    async def test_spy_appended_to_custom_symbols(self, dry_run_setup):
        """SPY should be appended when not in the custom symbol list."""
        db, ai_client = dry_run_setup

        mock_data = {
            "AAPL": make_history(150, 40),
            "SPY": make_history(450, 40),
        }

        async def mock_call(*args, **kwargs):
            return {
                "watchlist_updates": [],
                "market_summary": "Test",
                "decisions": [],
                "market_context": "Test",
            }

        with patch.object(ai_client, "call", side_effect=mock_call):
            with patch("paper_trader.trading.dry_run.load_historical_data", new_callable=AsyncMock, return_value=mock_data) as mock_load:
                with patch("paper_trader.ai.screener.fetch_news", new_callable=AsyncMock, return_value=[]):
                    await run_dry_run(db, ai_client, symbols=["AAPL"], days=5)

        # The symbols list passed to load_historical_data should include SPY
        called_symbols = mock_load.call_args[0][0]
        assert "SPY" in called_symbols

    @pytest.mark.asyncio
    async def test_spy_not_duplicated(self, dry_run_setup):
        """SPY should not be duplicated when already in the symbol list."""
        db, ai_client = dry_run_setup

        mock_data = {
            "AAPL": make_history(150, 40),
            "SPY": make_history(450, 40),
        }

        async def mock_call(*args, **kwargs):
            return {
                "watchlist_updates": [],
                "market_summary": "Test",
                "decisions": [],
                "market_context": "Test",
            }

        with patch.object(ai_client, "call", side_effect=mock_call):
            with patch("paper_trader.trading.dry_run.load_historical_data", new_callable=AsyncMock, return_value=mock_data) as mock_load:
                with patch("paper_trader.ai.screener.fetch_news", new_callable=AsyncMock, return_value=[]):
                    await run_dry_run(db, ai_client, symbols=["AAPL", "SPY"], days=5)

        called_symbols = mock_load.call_args[0][0]
        assert called_symbols.count("SPY") == 1


class TestResetForLive:
    @pytest.mark.asyncio
    async def test_reset(self, dry_run_setup):
        db, _ = dry_run_setup
        await queries.update_cash(db, 500.0)  # Simulate trading

        await reset_for_live(db)

        portfolio = await queries.get_portfolio(db)
        assert portfolio["cash"] == 800.0

        mode = await queries.get_setting(db, "mode")
        assert mode == "live"
