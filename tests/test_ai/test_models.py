import pytest
from pydantic import ValidationError

from paper_trader.ai.models import (
    StockDecision,
    ScreeningResult,
    WatchlistUpdate,
    AnalysisResult,
    WeeklyReview,
    MonthlyReview,
    JournalEntry,
)


class TestStockDecision:
    def test_valid_buy(self):
        d = StockDecision(symbol="AAPL", action="BUY", confidence=0.8, reasoning="Strong earnings")
        assert d.action == "BUY"
        assert d.target_allocation_pct == 0.0

    def test_valid_with_allocation(self):
        d = StockDecision(symbol="AAPL", action="BUY", confidence=0.9, reasoning="Go", target_allocation_pct=15.0)
        assert d.target_allocation_pct == 15.0

    def test_invalid_action(self):
        with pytest.raises(ValidationError):
            StockDecision(symbol="AAPL", action="SHORT", confidence=0.5, reasoning="Test")

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            StockDecision(symbol="AAPL", action="BUY", confidence=1.5, reasoning="Test")
        with pytest.raises(ValidationError):
            StockDecision(symbol="AAPL", action="BUY", confidence=-0.1, reasoning="Test")

    def test_allocation_max(self):
        with pytest.raises(ValidationError):
            StockDecision(symbol="AAPL", action="BUY", confidence=0.8, reasoning="Test", target_allocation_pct=35.0)


class TestScreeningResult:
    def test_valid(self):
        r = ScreeningResult(
            watchlist_updates=[WatchlistUpdate(symbol="AAPL", action="ADD", reason="Momentum")],
            market_summary="Bullish day",
        )
        assert len(r.watchlist_updates) == 1

    def test_empty(self):
        r = ScreeningResult()
        assert r.watchlist_updates == []


class TestAnalysisResult:
    def test_from_dict(self):
        data = {
            "decisions": [
                {"symbol": "AAPL", "action": "BUY", "confidence": 0.8, "reasoning": "Strong"},
                {"symbol": "GOOGL", "action": "HOLD", "confidence": 0.5, "reasoning": "Flat"},
            ],
            "market_context": "Mixed signals",
        }
        r = AnalysisResult(**data)
        assert len(r.decisions) == 2
        assert r.decisions[0].action == "BUY"


class TestWeeklyReview:
    def test_valid(self):
        r = WeeklyReview(
            performance_summary="Good week",
            patterns=["Momentum works on Mondays"],
            journal_entries=[JournalEntry(entry_type="observation", content="Test")],
        )
        assert len(r.journal_entries) == 1


class TestMonthlyReview:
    def test_valid(self):
        r = MonthlyReview(
            strategic_summary="On track",
            risk_assessment="Low risk",
            recommended_changes=["Increase position sizes"],
        )
        assert len(r.recommended_changes) == 1

    def test_with_currency_fields(self):
        r = MonthlyReview(
            strategic_summary="On track",
            risk_assessment="Low risk",
            currency_recommendation="STAY_USD",
            currency_reasoning="USD expected to strengthen",
        )
        assert r.currency_recommendation == "STAY_USD"
        assert r.currency_reasoning == "USD expected to strengthen"

    def test_currency_fields_optional(self):
        r = MonthlyReview(
            strategic_summary="On track",
            risk_assessment="Low risk",
        )
        assert r.currency_recommendation == ""
        assert r.currency_reasoning == ""
