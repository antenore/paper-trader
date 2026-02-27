import pytest

from paper_trader.portfolio.risk import check_risk


def make_position(symbol: str, shares: float, avg_cost: float) -> dict:
    return {"symbol": symbol, "shares": shares, "avg_cost": avg_cost}


class TestSanityChecks:
    def test_negative_shares(self):
        result = check_risk("BUY", "AAPL", -1, 150, 800, [], 800)
        assert not result["ok"]
        assert "positive" in result["reason"].lower()

    def test_zero_price(self):
        result = check_risk("BUY", "AAPL", 1, 0, 800, [], 800)
        assert not result["ok"]

    def test_invalid_action(self):
        result = check_risk("SHORT", "AAPL", 1, 150, 800, [], 800)
        assert not result["ok"]
        assert "invalid" in result["reason"].lower()

    def test_sell_always_passes(self):
        result = check_risk("SELL", "AAPL", 1, 150, 0, [], 800)
        assert result["ok"]


class TestSafetyStop:
    def test_portfolio_below_safety_threshold(self):
        # Safety stop at 50% of 800 = 400. With 350 cash and no positions → stop
        result = check_risk("BUY", "AAPL", 1, 10, 350.0, [], 800)
        assert not result["ok"]
        assert "safety stop" in result["reason"].lower()

    def test_portfolio_above_safety_threshold(self):
        result = check_risk("BUY", "AAPL", 1, 10, 500.0, [], 800)
        assert result["ok"]

    def test_portfolio_below_with_positions(self):
        # 100 cash + 200 in positions = 300 < 400 safety stop
        positions = [make_position("SPY", 1, 200)]
        result = check_risk("BUY", "AAPL", 1, 10, 100.0, positions, 800)
        assert not result["ok"]
        assert "safety stop" in result["reason"].lower()


class TestCashReserve:
    def test_buy_leaves_enough_reserve(self):
        # 800 cash, buy 1 share at 200 → 600 left > 100 reserve
        result = check_risk("BUY", "AAPL", 1, 200, 800.0, [], 800)
        assert result["ok"]

    def test_buy_would_breach_reserve(self):
        # 800 cash, buy 1 share at 750 → 50 left < 100 reserve
        # Should auto-reduce to (800-100)/750 = 0.933 shares
        result = check_risk("BUY", "AAPL", 1, 750, 800.0, [], 800)
        assert result["ok"]
        assert result["adjusted_shares"] < 1.0
        assert result["adjusted_shares"] * 750 <= 700.0 + 0.01

    def test_not_enough_cash_for_reserve(self):
        # 500 cash, 0 positions → above safety stop (500 > 400)
        # But want to buy at 500, leaving 0 < 100 reserve
        # Available = 500 - 100 = 400, so 400/500 = 0.8 shares
        # Position limit: 30% of 500 = 150, so 150/500 = 0.3 shares
        result = check_risk("BUY", "AAPL", 1, 500, 500.0, [], 800)
        assert result["ok"]
        assert result["adjusted_shares"] <= 0.3 + 0.01


class TestPositionLimit:
    def test_within_limit(self):
        # Portfolio 800, max 30% = 240. Buy 1 share at 200 → ok
        result = check_risk("BUY", "AAPL", 1, 200, 800.0, [], 800)
        assert result["ok"]

    def test_exceeds_limit_auto_reduce(self):
        # Portfolio ~800, max 30% = 240. Buy 2 shares at 200 = 400 → reduce to 240/200 = 1.2
        result = check_risk("BUY", "AAPL", 2, 200, 800.0, [], 800)
        assert result["ok"]
        assert result["adjusted_shares"] < 2.0
        assert result["adjusted_shares"] * 200 <= 240.0 + 0.01

    def test_existing_position_at_limit(self):
        # Portfolio = 560 cash + 240 position = 800. 30% = 240.
        # Existing AAPL already at 240 → no room
        positions = [make_position("AAPL", 1, 240)]
        result = check_risk("BUY", "AAPL", 1, 240, 560.0, positions, 800)
        assert not result["ok"]
        assert "position limit" in result["reason"].lower()


class TestInsufficientCash:
    def test_not_enough_cash(self):
        # 50 cash → below safety stop (50 < 400)
        result = check_risk("BUY", "AAPL", 10, 150, 50.0, [], 800)
        assert not result["ok"]

    def test_auto_reduce_to_fit_cash(self):
        # 500 cash, want 2 shares at 150 (=300). Fits in cash.
        # Cash reserve: 500-300=200 > 100 → ok
        # Position limit: 30% of 500 = 150. So reduce to 150/150 = 1.0 shares
        result = check_risk("BUY", "AAPL", 2, 150, 500.0, [], 800)
        assert result["ok"]
        assert result["adjusted_shares"] <= 1.0 + 0.01


class TestEdgeCases:
    def test_very_small_trade_rejected(self):
        # After adjustments, if trade < 0.01 shares → reject
        result = check_risk("BUY", "AAPL", 0.001, 150, 800.0, [], 800)
        assert not result["ok"]
        assert "too small" in result["reason"].lower()

    def test_normal_buy_returns_adjusted_shares(self):
        # Simple buy that passes all checks
        result = check_risk("BUY", "AAPL", 1, 100, 800.0, [], 800)
        assert result["ok"]
        assert "adjusted_shares" in result
