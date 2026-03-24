from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol, runtime_checkable

import httpx
import yfinance as yf

from paper_trader.config import settings

logger = logging.getLogger(__name__)


@runtime_checkable
class MarketDataProvider(Protocol):
    """Protocol for market data providers (live vs historical)."""

    async def get_current_prices(self, symbols: list[str]) -> dict[str, float]: ...
    async def get_price_history(
        self, symbol: str, period: str, interval: str
    ) -> list[dict[str, Any]]: ...
    async def get_market_movers(self) -> dict[str, list[dict[str, Any]]]: ...


def _is_swiss(symbol: str) -> bool:
    return symbol.upper().endswith(".SW")


# Map yfinance period strings to days for Alpaca start date calculation
_PERIOD_DAYS = {
    "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730,
}

# Map yfinance interval strings to Alpaca timeframe values
_INTERVAL_MAP = {
    "1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min",
    "1h": "1Hour", "1d": "1Day", "1wk": "1Week", "1mo": "1Month",
}


class LiveDataProvider:
    """Fetches live market data. Alpaca for US stocks, yfinance for Swiss and fallback."""

    def __init__(self) -> None:
        self._alpaca_enabled = bool(
            settings.alpaca_api_key and settings.alpaca_api_secret
        )
        if self._alpaca_enabled:
            self._alpaca_headers = {
                "APCA-API-KEY-ID": settings.alpaca_api_key,
                "APCA-API-SECRET-KEY": settings.alpaca_api_secret,
            }
            self._client = httpx.AsyncClient(
                base_url=settings.alpaca_data_url,
                headers=self._alpaca_headers,
                timeout=10.0,
            )
            logger.info("Alpaca market data enabled")
        else:
            self._client = None
            logger.info("Alpaca not configured, using yfinance only")

    async def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """Get current prices. Alpaca for US, yfinance for Swiss and fallback."""
        prices: dict[str, float] = {}
        swiss = [s for s in symbols if _is_swiss(s)]
        us = [s for s in symbols if not _is_swiss(s)]

        alpaca_failed: list[str] = []
        if us and self._alpaca_enabled:
            try:
                resp = await self._client.get(
                    "/v2/stocks/snapshots",
                    params={"symbols": ",".join(us), "feed": "iex"},
                )
                resp.raise_for_status()
                data = resp.json()
                for sym in us:
                    snap = data.get(sym)
                    if snap:
                        price = snap.get("latestTrade", {}).get("p")
                        if price and price > 0:
                            prices[sym] = float(price)
                            continue
                    alpaca_failed.append(sym)
                if alpaca_failed:
                    logger.debug(
                        "Alpaca missing prices for %s, trying yfinance",
                        alpaca_failed,
                    )
            except Exception as e:
                logger.warning("Alpaca snapshot failed, falling back to yfinance: %s", e)
                alpaca_failed = us
        else:
            alpaca_failed = us

        yf_symbols = swiss + alpaca_failed
        if yf_symbols:
            prices.update(await self._yfinance_prices(yf_symbols))

        return prices

    async def get_price_history(
        self, symbol: str, period: str = "1mo", interval: str = "1d"
    ) -> list[dict[str, Any]]:
        """Get historical price data. Alpaca for US, yfinance for Swiss and fallback."""
        if not _is_swiss(symbol) and self._alpaca_enabled:
            try:
                return await self._alpaca_bars(symbol, period, interval)
            except Exception as e:
                logger.warning(
                    "Alpaca bars failed for %s, falling back to yfinance: %s",
                    symbol, e,
                )

        return await self._yfinance_history(symbol, period, interval)

    async def get_market_movers(self) -> dict[str, list[dict[str, Any]]]:
        """Get market movers (SPY-based proxy). Always uses yfinance."""
        movers: dict[str, list[dict[str, Any]]] = {"gainers": [], "losers": []}
        try:
            spy = yf.Ticker("SPY")
            df = spy.history(period="2d", interval="1d")
            if len(df) >= 2:
                change = (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100
                movers["market_change_pct"] = float(change)
        except Exception as e:
            logger.warning("Failed to get market movers: %s", e)
        return movers

    # -- Alpaca helpers --

    async def _alpaca_bars(
        self, symbol: str, period: str, interval: str
    ) -> list[dict[str, Any]]:
        """Fetch historical bars from Alpaca."""
        days = _PERIOD_DAYS.get(period, 30)
        start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
            "%Y-%m-%dT00:00:00Z"
        )
        timeframe = _INTERVAL_MAP.get(interval, "1Day")

        resp = await self._client.get(
            f"/v2/stocks/{symbol}/bars",
            params={
                "timeframe": timeframe,
                "start": start,
                "feed": "sip",
                "limit": 10000,
                "adjustment": "split",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        records = []
        for bar in data.get("bars", []):
            records.append({
                "date": bar["t"],
                "open": float(bar["o"]),
                "high": float(bar["h"]),
                "low": float(bar["l"]),
                "close": float(bar["c"]),
                "volume": int(bar["v"]),
            })
        return records

    # -- yfinance helpers --

    async def _yfinance_prices(self, symbols: list[str]) -> dict[str, float]:
        """Fetch current prices from yfinance."""
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
            logger.error("Failed to fetch yfinance prices: %s", e)
        return prices

    async def _yfinance_history(
        self, symbol: str, period: str = "1mo", interval: str = "1d"
    ) -> list[dict[str, Any]]:
        """Fetch historical data from yfinance."""
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
            logger.error("Failed to get yfinance history for %s: %s", symbol, e)
            return []


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
