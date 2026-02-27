from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Protocol, runtime_checkable

import yfinance as yf

logger = logging.getLogger(__name__)


@runtime_checkable
class MarketDataProvider(Protocol):
    """Protocol for market data providers (live vs historical)."""

    async def get_current_prices(self, symbols: list[str]) -> dict[str, float]: ...
    async def get_price_history(
        self, symbol: str, period: str, interval: str
    ) -> list[dict[str, Any]]: ...
    async def get_market_movers(self) -> dict[str, list[dict[str, Any]]]: ...


class LiveDataProvider:
    """Fetches live market data from yfinance."""

    async def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """Get current prices for a list of symbols."""
        prices: dict[str, float] = {}
        try:
            tickers = yf.Tickers(" ".join(symbols))
            for symbol in symbols:
                try:
                    ticker = tickers.tickers.get(symbol)
                    if ticker is None:
                        continue
                    info = ticker.fast_info
                    price = getattr(info, "last_price", None)
                    if price is not None and price > 0:
                        prices[symbol] = float(price)
                except Exception as e:
                    logger.warning("Failed to get price for %s: %s", symbol, e)
        except Exception as e:
            logger.error("Failed to fetch prices: %s", e)
        return prices

    async def get_price_history(
        self, symbol: str, period: str = "1mo", interval: str = "1d"
    ) -> list[dict[str, Any]]:
        """Get historical price data."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                return []
            records = []
            for date, row in df.iterrows():
                records.append({
                    "date": str(date),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                })
            return records
        except Exception as e:
            logger.error("Failed to get history for %s: %s", symbol, e)
            return []

    async def get_market_movers(self) -> dict[str, list[dict[str, Any]]]:
        """Get market movers (top gainers/losers from major indices)."""
        movers: dict[str, list[dict[str, Any]]] = {"gainers": [], "losers": []}
        try:
            # Use SPY components as a proxy for market activity
            spy = yf.Ticker("SPY")
            df = spy.history(period="2d", interval="1d")
            if len(df) >= 2:
                change = (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100
                movers["market_change_pct"] = float(change)
        except Exception as e:
            logger.warning("Failed to get market movers: %s", e)
        return movers


class HistoricalDataProvider:
    """Provides historical data for dry run simulations."""

    def __init__(self, symbol_data: dict[str, list[dict[str, Any]]], current_index: int = 0):
        self._data = symbol_data
        self._index = current_index

    @property
    def current_index(self) -> int:
        return self._index

    @current_index.setter
    def current_index(self, value: int) -> None:
        self._index = value

    async def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """Get prices at the current simulation index."""
        prices: dict[str, float] = {}
        for symbol in symbols:
            history = self._data.get(symbol, [])
            if history and self._index < len(history):
                prices[symbol] = history[self._index]["close"]
        return prices

    async def get_price_history(
        self, symbol: str, period: str = "1mo", interval: str = "1d"
    ) -> list[dict[str, Any]]:
        """Get history up to the current simulation index."""
        history = self._data.get(symbol, [])
        end = min(self._index + 1, len(history))
        return history[:end]

    async def get_market_movers(self) -> dict[str, list[dict[str, Any]]]:
        """Simulated market movers based on historical data."""
        return {"gainers": [], "losers": []}
