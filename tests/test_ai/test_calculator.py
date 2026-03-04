import pytest

from paper_trader.ai.calculator import (
    ToolUseAudit,
    _calc_break_even,
    _calc_pnl,
    _calc_portfolio_weight,
    _calc_position_size,
    _calc_risk_reward,
    _calc_what_if_buy,
    execute_tool,
)


class TestPositionSize:
    def test_basic_buy(self):
        r = _calc_position_size("AAPL", price=150, cash=800, portfolio_total=800, target_pct=15)
        assert r["symbol"] == "AAPL"
        # target_value = 800 * 0.15 = 120, shares = 120/150 = 0.8
        assert r["shares"] == 0.8
        assert r["cost_chf"] == 120.0
        assert r["remaining_cash"] == 680.0

    def test_with_existing_position(self):
        r = _calc_position_size("AAPL", price=150, cash=800, portfolio_total=800, target_pct=15, existing_value=60)
        # target = 120, additional = 60, shares = 60/150 = 0.4
        assert r["shares"] == 0.4
        assert r["cost_chf"] == 60.0

    def test_already_at_target(self):
        r = _calc_position_size("AAPL", price=150, cash=800, portfolio_total=800, target_pct=15, existing_value=120)
        assert r["shares"] == 0.0
        assert r["cost_chf"] == 0.0

    def test_cash_limited(self):
        r = _calc_position_size("AAPL", price=150, cash=50, portfolio_total=800, target_pct=15)
        # target = 120 but only 50 cash → buy 50/150 = 0.3333
        assert r["shares"] == pytest.approx(0.3333, abs=0.001)
        assert r["cost_chf"] == pytest.approx(50.0, abs=0.01)

    def test_zero_price_error(self):
        r = _calc_position_size("AAPL", price=0, cash=800, portfolio_total=800, target_pct=15)
        assert "error" in r

    def test_zero_portfolio_error(self):
        r = _calc_position_size("AAPL", price=150, cash=800, portfolio_total=0, target_pct=15)
        assert "error" in r

    def test_weight_calculation(self):
        r = _calc_position_size("AAPL", price=100, cash=800, portfolio_total=800, target_pct=25)
        # cost = 200, weight = 200/800 = 25%
        assert r["new_weight_pct"] == 25.0


class TestPnL:
    def test_profit(self):
        r = _calc_pnl("AAPL", entry_price=100, current_price=115, shares=10)
        assert r["pnl_chf"] == 150.0
        assert r["pnl_pct"] == 15.0

    def test_loss(self):
        r = _calc_pnl("AAPL", entry_price=100, current_price=90, shares=5)
        assert r["pnl_chf"] == -50.0
        assert r["pnl_pct"] == -10.0

    def test_breakeven(self):
        r = _calc_pnl("AAPL", entry_price=100, current_price=100, shares=10)
        assert r["pnl_chf"] == 0.0
        assert r["pnl_pct"] == 0.0

    def test_zero_entry_error(self):
        r = _calc_pnl("AAPL", entry_price=0, current_price=100, shares=10)
        assert "error" in r

    def test_position_value(self):
        r = _calc_pnl("AAPL", entry_price=100, current_price=110, shares=5)
        assert r["position_value"] == 550.0


class TestRiskReward:
    def test_basic_ratio(self):
        r = _calc_risk_reward("AAPL", entry=100, stop_loss=93, target=121)
        # risk = 7, reward = 21, ratio = 3.0
        assert r["risk_reward_ratio"] == 3.0
        assert r["risk_per_share"] == 7.0
        assert r["reward_per_share"] == 21.0

    def test_risk_pct(self):
        r = _calc_risk_reward("AAPL", entry=100, stop_loss=93, target=121)
        assert r["risk_pct"] == 7.0
        assert r["reward_pct"] == 21.0

    def test_stop_above_entry_error(self):
        r = _calc_risk_reward("AAPL", entry=100, stop_loss=105, target=120)
        assert "error" in r

    def test_target_below_entry_error(self):
        r = _calc_risk_reward("AAPL", entry=100, stop_loss=93, target=95)
        assert "error" in r


class TestPortfolioWeight:
    def test_basic(self):
        r = _calc_portfolio_weight("AAPL", position_value=200, total_value=800)
        assert r["weight_pct"] == 25.0

    def test_zero_total_error(self):
        r = _calc_portfolio_weight("AAPL", position_value=200, total_value=0)
        assert "error" in r

    def test_small_position(self):
        r = _calc_portfolio_weight("AAPL", position_value=8, total_value=800)
        assert r["weight_pct"] == 1.0


class TestBreakEven:
    def test_no_commission(self):
        r = _calc_break_even("AAPL", entry_price=150, shares=10)
        assert r["break_even_price"] == 150.0

    def test_with_commission(self):
        r = _calc_break_even("AAPL", entry_price=150, shares=10, commission=5)
        # total_cost = 1500 + 5 = 1505, break_even = 150.5
        assert r["break_even_price"] == 150.5
        assert r["commission_per_share"] == 0.5

    def test_zero_shares_error(self):
        r = _calc_break_even("AAPL", entry_price=150, shares=0)
        assert "error" in r


class TestWhatIfBuy:
    def test_basic(self):
        r = _calc_what_if_buy("AAPL", shares=1, price=150, cash=800, positions_value=0)
        assert r["cost_chf"] == 150.0
        assert r["new_cash"] == 650.0
        assert r["new_total"] == 800.0
        assert r["sector"] == "Technology"

    def test_insufficient_cash(self):
        r = _calc_what_if_buy("AAPL", shares=10, price=150, cash=100, positions_value=0)
        assert "error" in r

    def test_sector_exposure(self):
        r = _calc_what_if_buy(
            "AAPL", shares=1, price=100, cash=800, positions_value=200,
            sector_values={"Technology": 200},
        )
        # cost = 100, new_total = 700 + 300 = 1000
        # tech = 200 + 100 = 300, pct = 30%
        assert r["sector_exposure_pct"]["Technology"] == 30.0

    def test_unknown_sector(self):
        r = _calc_what_if_buy("ZZZZ", shares=1, price=50, cash=800, positions_value=0)
        assert r["sector"] == "Unknown"


class TestDispatch:
    def test_position_size_dispatch(self):
        r = execute_tool("position_size", {
            "symbol": "AAPL", "price": 150, "cash": 800,
            "portfolio_total": 800, "target_pct": 15,
        })
        assert r["shares"] == 0.8

    def test_calculate_pnl_dispatch(self):
        r = execute_tool("calculate_pnl", {
            "symbol": "AAPL", "entry_price": 100, "current_price": 110, "shares": 10,
        })
        assert r["pnl_chf"] == 100.0

    def test_risk_reward_dispatch(self):
        r = execute_tool("risk_reward", {
            "symbol": "AAPL", "entry": 100, "stop_loss": 93, "target": 121,
        })
        assert r["risk_reward_ratio"] == 3.0

    def test_unknown_tool(self):
        r = execute_tool("nonexistent", {"symbol": "AAPL"})
        assert "error" in r
        assert "Unknown tool" in r["error"]

    def test_missing_input(self):
        r = execute_tool("position_size", {"symbol": "AAPL"})
        assert "error" in r


class TestAudit:
    def test_record_and_query(self):
        audit = ToolUseAudit()
        audit.record("position_size", {"symbol": "AAPL"}, {"shares": 1})
        audit.record("calculate_pnl", {"symbol": "NVDA"}, {"pnl_chf": 50})
        audit.record("position_size", {"symbol": "GOOGL"}, {"shares": 2})

        assert len(audit.calls) == 3
        assert audit.symbols_with_tool("position_size") == {"AAPL", "GOOGL"}
        assert audit.symbols_with_tool("calculate_pnl") == {"NVDA"}
        assert audit.symbols_with_tool("risk_reward") == set()

    def test_empty_audit(self):
        audit = ToolUseAudit()
        assert len(audit.calls) == 0
        assert audit.symbols_with_tool("position_size") == set()

    def test_turns_tracking(self):
        audit = ToolUseAudit()
        audit.turns = 3
        assert audit.turns == 3
