import pytest

from paper_trader.market.prices import HistoricalDataProvider, LiveDataProvider


def make_history(prices: list[float]) -> list[dict]:
    return [{"date": f"2025-01-{i+1:02d}", "open": p, "high": p+1, "low": p-1, "close": p, "volume": 1000} for i, p in enumerate(prices)]


class TestHistoricalDataProvider:
    @pytest.mark.asyncio
    async def test_get_current_prices(self):
        data = {"AAPL": make_history([150, 155, 160])}
        provider = HistoricalDataProvider(data, current_index=1)
        prices = await provider.get_current_prices(["AAPL"])
        assert prices["AAPL"] == 155.0

    @pytest.mark.asyncio
    async def test_get_current_prices_missing_symbol(self):
        data = {"AAPL": make_history([150, 155])}
        provider = HistoricalDataProvider(data, current_index=0)
        prices = await provider.get_current_prices(["GOOGL"])
        assert "GOOGL" not in prices

    @pytest.mark.asyncio
    async def test_get_price_history_up_to_index(self):
        data = {"AAPL": make_history([150, 155, 160, 165])}
        provider = HistoricalDataProvider(data, current_index=2)
        history = await provider.get_price_history("AAPL")
        assert len(history) == 3  # indices 0, 1, 2

    @pytest.mark.asyncio
    async def test_advance_index(self):
        data = {"AAPL": make_history([150, 155, 160])}
        provider = HistoricalDataProvider(data, current_index=0)

        prices = await provider.get_current_prices(["AAPL"])
        assert prices["AAPL"] == 150.0

        provider.current_index = 2
        prices = await provider.get_current_prices(["AAPL"])
        assert prices["AAPL"] == 160.0

    @pytest.mark.asyncio
    async def test_market_movers_empty(self):
        provider = HistoricalDataProvider({})
        movers = await provider.get_market_movers()
        assert movers == {"gainers": [], "losers": []}


class TestLiveDataProvider:
    @pytest.mark.asyncio
    async def test_instantiation(self):
        provider = LiveDataProvider()
        assert provider is not None
