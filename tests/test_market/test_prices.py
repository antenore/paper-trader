from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from paper_trader.market.prices import (
    HistoricalDataProvider,
    LiveDataProvider,
    _is_swiss,
)


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


class TestIsSwiss:
    def test_swiss_ticker(self):
        assert _is_swiss("NESN.SW") is True
        assert _is_swiss("nesn.sw") is True

    def test_us_ticker(self):
        assert _is_swiss("AAPL") is False
        assert _is_swiss("SPY") is False


class TestAlpacaPrices:
    """Tests for Alpaca integration with mocked HTTP."""

    def _make_provider(self) -> LiveDataProvider:
        """Create a provider with Alpaca enabled via mock settings."""
        with patch("paper_trader.market.prices.settings") as mock_settings:
            mock_settings.alpaca_api_key = "test-key"
            mock_settings.alpaca_api_secret = "test-secret"
            mock_settings.alpaca_data_url = "https://data.alpaca.markets"
            provider = LiveDataProvider()
        return provider

    def _make_disabled_provider(self) -> LiveDataProvider:
        """Create a provider with Alpaca disabled."""
        with patch("paper_trader.market.prices.settings") as mock_settings:
            mock_settings.alpaca_api_key = ""
            mock_settings.alpaca_api_secret = ""
            mock_settings.alpaca_data_url = "https://data.alpaca.markets"
            provider = LiveDataProvider()
        return provider

    @pytest.mark.asyncio
    async def test_alpaca_prices_us_stocks(self):
        provider = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "AAPL": {"latestTrade": {"p": 175.50}},
            "MSFT": {"latestTrade": {"p": 410.25}},
        }
        provider._client.get = AsyncMock(return_value=mock_resp)

        prices = await provider.get_current_prices(["AAPL", "MSFT"])

        assert prices["AAPL"] == 175.50
        assert prices["MSFT"] == 410.25
        provider._client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_swiss_tickers_bypass_alpaca(self):
        provider = self._make_provider()
        provider._client.get = AsyncMock()

        with patch.object(provider, "_yfinance_prices", new_callable=AsyncMock) as yf_mock:
            yf_mock.return_value = {"NESN.SW": 95.0}
            prices = await provider.get_current_prices(["NESN.SW"])

        assert prices["NESN.SW"] == 95.0
        provider._client.get.assert_not_called()
        yf_mock.assert_called_once_with(["NESN.SW"])

    @pytest.mark.asyncio
    async def test_alpaca_fallback_on_error(self):
        provider = self._make_provider()
        provider._client.get = AsyncMock(side_effect=httpx.ConnectError("timeout"))

        with patch.object(provider, "_yfinance_prices", new_callable=AsyncMock) as yf_mock:
            yf_mock.return_value = {"AAPL": 175.0}
            prices = await provider.get_current_prices(["AAPL"])

        assert prices["AAPL"] == 175.0
        yf_mock.assert_called_once_with(["AAPL"])

    @pytest.mark.asyncio
    async def test_alpaca_fallback_partial(self):
        """Alpaca returns data for AAPL but not MSFT - MSFT falls back to yfinance."""
        provider = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "AAPL": {"latestTrade": {"p": 175.50}},
        }
        provider._client.get = AsyncMock(return_value=mock_resp)

        with patch.object(provider, "_yfinance_prices", new_callable=AsyncMock) as yf_mock:
            yf_mock.return_value = {"MSFT": 410.0}
            prices = await provider.get_current_prices(["AAPL", "MSFT"])

        assert prices["AAPL"] == 175.50
        assert prices["MSFT"] == 410.0
        yf_mock.assert_called_once_with(["MSFT"])

    @pytest.mark.asyncio
    async def test_alpaca_disabled_uses_yfinance(self):
        provider = self._make_disabled_provider()

        with patch.object(provider, "_yfinance_prices", new_callable=AsyncMock) as yf_mock:
            yf_mock.return_value = {"AAPL": 175.0}
            prices = await provider.get_current_prices(["AAPL"])

        assert prices["AAPL"] == 175.0
        yf_mock.assert_called_once_with(["AAPL"])

    @pytest.mark.asyncio
    async def test_mixed_symbols(self):
        """US stocks via Alpaca, Swiss via yfinance, in one call."""
        provider = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "AAPL": {"latestTrade": {"p": 175.50}},
        }
        provider._client.get = AsyncMock(return_value=mock_resp)

        with patch.object(provider, "_yfinance_prices", new_callable=AsyncMock) as yf_mock:
            yf_mock.return_value = {"NESN.SW": 95.0}
            prices = await provider.get_current_prices(["AAPL", "NESN.SW"])

        assert prices["AAPL"] == 175.50
        assert prices["NESN.SW"] == 95.0
        yf_mock.assert_called_once_with(["NESN.SW"])


class TestAlpacaHistory:
    """Tests for Alpaca historical bars."""

    def _make_provider(self) -> LiveDataProvider:
        with patch("paper_trader.market.prices.settings") as mock_settings:
            mock_settings.alpaca_api_key = "test-key"
            mock_settings.alpaca_api_secret = "test-secret"
            mock_settings.alpaca_data_url = "https://data.alpaca.markets"
            provider = LiveDataProvider()
        return provider

    @pytest.mark.asyncio
    async def test_alpaca_history_bars(self):
        provider = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "bars": [
                {"t": "2026-03-01T05:00:00Z", "o": 150.0, "h": 155.0, "l": 149.0, "c": 153.0, "v": 1000000},
                {"t": "2026-03-02T05:00:00Z", "o": 153.0, "h": 158.0, "l": 152.0, "c": 157.0, "v": 1200000},
            ],
        }
        provider._client.get = AsyncMock(return_value=mock_resp)

        history = await provider.get_price_history("AAPL", period="1mo", interval="1d")

        assert len(history) == 2
        assert history[0]["close"] == 153.0
        assert history[1]["open"] == 153.0
        assert history[1]["volume"] == 1200000

    @pytest.mark.asyncio
    async def test_alpaca_history_fallback(self):
        provider = self._make_provider()
        provider._client.get = AsyncMock(side_effect=httpx.ConnectError("timeout"))

        with patch.object(provider, "_yfinance_history", new_callable=AsyncMock) as yf_mock:
            yf_mock.return_value = [{"date": "2026-03-01", "close": 153.0}]
            history = await provider.get_price_history("AAPL", period="1mo", interval="1d")

        assert len(history) == 1
        yf_mock.assert_called_once_with("AAPL", "1mo", "1d")

    @pytest.mark.asyncio
    async def test_swiss_history_uses_yfinance(self):
        provider = self._make_provider()

        with patch.object(provider, "_yfinance_history", new_callable=AsyncMock) as yf_mock:
            yf_mock.return_value = [{"date": "2026-03-01", "close": 95.0}]
            history = await provider.get_price_history("NESN.SW", period="1mo", interval="1d")

        assert len(history) == 1
        yf_mock.assert_called_once_with("NESN.SW", "1mo", "1d")
