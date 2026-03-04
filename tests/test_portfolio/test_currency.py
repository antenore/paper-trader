"""Tests for currency module."""

from paper_trader.portfolio.currency import symbol_currency, to_chf


class TestSymbolCurrency:
    def test_us_stock(self):
        assert symbol_currency("AAPL") == "USD"

    def test_swiss_stock(self):
        assert symbol_currency("NESN.SW") == "CHF"

    def test_swiss_stock_lowercase(self):
        assert symbol_currency("nesn.sw") == "CHF"

    def test_etf(self):
        assert symbol_currency("SPY") == "USD"

    def test_unknown(self):
        assert symbol_currency("XYZ") == "USD"


class TestToChf:
    def test_chf_passthrough(self):
        assert to_chf(100.0, "CHF", 0.88) == 100.0

    def test_usd_conversion(self):
        assert to_chf(100.0, "USD", 0.88) == 88.0

    def test_usd_rate_one(self):
        assert to_chf(100.0, "USD", 1.0) == 100.0

    def test_zero_amount(self):
        assert to_chf(0.0, "USD", 0.88) == 0.0
