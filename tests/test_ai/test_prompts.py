"""Tests for prompt construction and trade history formatting."""

from paper_trader.ai.prompts import analysis_prompt, format_recent_trades


class TestFormatRecentTrades:
    def test_empty(self):
        assert format_recent_trades([]) == "No recent trades."

    def test_single_trade(self):
        trades = [
            {
                "symbol": "NVDA",
                "action": "BUY",
                "shares": 1.0917,
                "price": 183.13,
                "total": 199.92,
                "commission_chf": 0.78,
                "executed_at": "2026-03-05 14:09:22",
            }
        ]
        result = format_recent_trades(trades)
        assert "NVDA" in result
        assert "BUY" in result
        assert "183.13" in result
        assert "0.78" in result
        assert "2026-03-05 14:09:22" in result

    def test_multiple_trades_preserves_order(self):
        trades = [
            {
                "symbol": "NVDA",
                "action": "BUY",
                "shares": 1.0,
                "price": 183.0,
                "total": 183.0,
                "commission_chf": 0.78,
                "executed_at": "2026-03-05 14:00:00",
            },
            {
                "symbol": "RTX",
                "action": "SELL",
                "shares": 0.7,
                "price": 202.0,
                "total": 141.4,
                "commission_chf": 0.78,
                "executed_at": "2026-03-05 18:00:00",
            },
        ]
        result = format_recent_trades(trades)
        nvda_pos = result.index("NVDA")
        rtx_pos = result.index("RTX")
        assert nvda_pos < rtx_pos

    def test_missing_commission_defaults_zero(self):
        trades = [
            {
                "symbol": "AAPL",
                "action": "BUY",
                "shares": 1.0,
                "price": 150.0,
                "total": 150.0,
                "executed_at": "2026-03-05 10:00:00",
            }
        ]
        result = format_recent_trades(trades)
        assert "AAPL" in result
        assert " 0.00" in result


class TestAnalysisPrompt:
    def test_includes_current_time(self):
        result = analysis_prompt(
            portfolio_summary="Cash: 500 CHF",
            watchlist_symbols=["NVDA"],
            price_data="NVDA: $183",
            news="No news",
            current_utc="2026-03-06 12:00 UTC",
        )
        assert "2026-03-06 12:00 UTC" in result

    def test_includes_recent_trades(self):
        result = analysis_prompt(
            portfolio_summary="Cash: 500 CHF",
            watchlist_symbols=["NVDA"],
            price_data="NVDA: $183",
            news="No news",
            recent_trades="2026-03-05 14:00 | BUY | NVDA | 1.0000 | 183.00",
        )
        assert "Recent trades (last 7 business days" in result
        assert "BUY | NVDA" in result

    def test_omits_recent_trades_when_empty(self):
        result = analysis_prompt(
            portfolio_summary="Cash: 500 CHF",
            watchlist_symbols=["NVDA"],
            price_data="NVDA: $183",
            news="No news",
            recent_trades="",
        )
        assert "Recent trades" not in result

    def test_anti_churning_in_system_prompt(self):
        from paper_trader.ai.prompts import ANALYSIS_SYSTEM

        assert "Trading History Awareness" in ANALYSIS_SYSTEM
        assert "Do NOT re-buy a stock you sold" in ANALYSIS_SYSTEM
        assert "ANTI-CHURN" in ANALYSIS_SYSTEM
