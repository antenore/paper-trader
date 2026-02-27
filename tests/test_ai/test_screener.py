import pytest
from unittest.mock import AsyncMock, patch

from paper_trader.ai.client import AIClient
from paper_trader.ai.screener import run_screening
from paper_trader.db import queries
from paper_trader.market.prices import HistoricalDataProvider


@pytest.mark.asyncio
async def test_screening_dry_run_skips_watchlist_mutations(db_with_portfolio):
    """run_screening with is_dry_run=True must NOT call add_to_watchlist or remove_from_watchlist."""
    db = db_with_portfolio
    ai_client = AIClient(db, api_key="test-key")

    # Seed a known watchlist state
    await queries.add_to_watchlist(db, "SPY", "Live entry")
    before = await queries.get_watchlist(db)

    screening_response = {
        "watchlist_updates": [
            {"symbol": "AAPL", "action": "ADD", "reason": "Hot stock"},
            {"symbol": "SPY", "action": "REMOVE", "reason": "Cooling off"},
        ],
        "market_summary": "Test market",
    }

    mock_market = AsyncMock(spec=HistoricalDataProvider)
    mock_market.get_market_movers = AsyncMock(return_value=[])

    with patch.object(ai_client, "call", new_callable=AsyncMock, return_value=screening_response):
        with patch("paper_trader.ai.screener.fetch_news", new_callable=AsyncMock, return_value=[]):
            with patch("paper_trader.ai.screener.format_news_for_ai", return_value=""):
                result = await run_screening(db, ai_client, mock_market, is_dry_run=True)

    # The result should still contain the AI's suggestions
    assert len(result.watchlist_updates) == 2

    # But the live watchlist must be unchanged
    after = await queries.get_watchlist(db)
    before_symbols = {w["symbol"] for w in before}
    after_symbols = {w["symbol"] for w in after}
    assert before_symbols == after_symbols, "Dry run screening must not mutate the live watchlist"


@pytest.mark.asyncio
async def test_screening_live_applies_watchlist_mutations(db_with_portfolio):
    """run_screening with is_dry_run=False SHOULD apply watchlist mutations."""
    db = db_with_portfolio
    ai_client = AIClient(db, api_key="test-key")

    screening_response = {
        "watchlist_updates": [
            {"symbol": "AAPL", "action": "ADD", "reason": "Hot stock"},
        ],
        "market_summary": "Test market",
    }

    mock_market = AsyncMock(spec=HistoricalDataProvider)
    mock_market.get_market_movers = AsyncMock(return_value=[])

    with patch.object(ai_client, "call", new_callable=AsyncMock, return_value=screening_response):
        with patch("paper_trader.ai.screener.fetch_news", new_callable=AsyncMock, return_value=[]):
            with patch("paper_trader.ai.screener.format_news_for_ai", return_value=""):
                await run_screening(db, ai_client, mock_market, is_dry_run=False)

    wl = await queries.get_watchlist(db)
    symbols = {w["symbol"] for w in wl}
    assert "AAPL" in symbols, "Live screening should add to watchlist"
