import pytest

from paper_trader.portfolio.commission import apply_slippage, calculate_commission, round_shares


class TestUSCommission:
    def test_basic(self):
        # 100 shares at $150 → 0.005 * 100 = $0.50, min $1.00 → $1.00
        fee = calculate_commission("AAPL", 100, 150.0)
        assert fee == 1.00

    def test_minimum_floor(self):
        # 10 shares → 0.005 * 10 = $0.05, min $1.00 → $1.00
        fee = calculate_commission("AAPL", 10, 150.0)
        assert fee == 1.00

    def test_large_order(self):
        # 1000 shares at $150 → 0.005 * 1000 = $5.00
        fee = calculate_commission("AAPL", 1000, 150.0)
        assert fee == 5.00

    def test_one_percent_cap(self):
        # 1 share at $50 → 0.005 * 1 = $0.005, min $1.00 → $1.00
        # 1% cap = $0.50 → capped at $0.50
        fee = calculate_commission("AAPL", 1, 50.0)
        assert fee == 0.50

    def test_fx_conversion(self):
        # 100 shares at $150, rate 0.88 → $1.00 * 0.88 = CHF 0.88
        fee = calculate_commission("AAPL", 100, 150.0, usd_chf_rate=0.88)
        assert fee == pytest.approx(0.88)


class TestSIXCommission:
    def test_basic(self):
        # 10 shares at CHF 100 = CHF 1000 → 0.05% = CHF 0.50, min CHF 1.50 → CHF 1.50
        fee = calculate_commission("NESN.SW", 10, 100.0)
        assert fee == 1.50

    def test_large_order(self):
        # 100 shares at CHF 100 = CHF 10000 → 0.05% = CHF 5.00 > 1.50
        fee = calculate_commission("NESN.SW", 100, 100.0)
        assert fee == 5.00

    def test_minimum_floor(self):
        # 1 share at CHF 100 = CHF 100 → 0.05% = CHF 0.05, min CHF 1.50
        fee = calculate_commission("NESN.SW", 1, 100.0)
        assert fee == 1.50

    def test_fx_rate_ignored(self):
        # SIX stocks are in CHF — rate should not affect commission
        fee1 = calculate_commission("NESN.SW", 10, 100.0, usd_chf_rate=1.0)
        fee2 = calculate_commission("NESN.SW", 10, 100.0, usd_chf_rate=0.50)
        assert fee1 == fee2


class TestEdgeCases:
    def test_zero_shares(self):
        assert calculate_commission("AAPL", 0, 150.0) == 0.0

    def test_zero_price(self):
        assert calculate_commission("AAPL", 10, 0.0) == 0.0

    def test_negative_shares(self):
        assert calculate_commission("AAPL", -5, 150.0) == 0.0


class TestRoundShares:
    def test_us_fractional(self):
        assert round_shares("AAPL", 1.23456) == 1.2346

    def test_us_small(self):
        assert round_shares("AAPL", 0.0001) == 0.0001

    def test_six_whole_floor(self):
        assert round_shares("NESN.SW", 2.9) == 2.0

    def test_six_exact(self):
        assert round_shares("NESN.SW", 3.0) == 3.0

    def test_six_less_than_one(self):
        assert round_shares("NESN.SW", 0.7) == 0.0

    def test_zero_shares(self):
        assert round_shares("AAPL", 0) == 0.0

    def test_negative_shares(self):
        assert round_shares("AAPL", -1) == 0.0


class TestSlippage:
    def test_buy_us_price_increases(self):
        # US: 5 bps = 0.05% → $100 * 1.0005 = $100.05
        result = apply_slippage("AAPL", 100.0, "BUY")
        assert result == pytest.approx(100.05)

    def test_sell_us_price_decreases(self):
        # US: 5 bps → $100 * 0.9995 = $99.95
        result = apply_slippage("AAPL", 100.0, "SELL")
        assert result == pytest.approx(99.95)

    def test_buy_six_larger_slippage(self):
        # SIX: 10 bps = 0.10% → CHF 100 * 1.001 = CHF 100.10
        result = apply_slippage("NESN.SW", 100.0, "BUY")
        assert result == pytest.approx(100.10)

    def test_sell_six_larger_slippage(self):
        # SIX: 10 bps → CHF 100 * 0.999 = CHF 99.90
        result = apply_slippage("NESN.SW", 100.0, "SELL")
        assert result == pytest.approx(99.90)

    def test_zero_price(self):
        assert apply_slippage("AAPL", 0.0, "BUY") == 0.0

    def test_negative_price(self):
        assert apply_slippage("AAPL", -10.0, "BUY") == -10.0

    def test_buy_always_worse_than_quote(self):
        price = 150.0
        assert apply_slippage("AAPL", price, "BUY") > price

    def test_sell_always_worse_than_quote(self):
        price = 150.0
        assert apply_slippage("AAPL", price, "SELL") < price
