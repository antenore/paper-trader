from __future__ import annotations

from pydantic import BaseModel, Field


class StockDecision(BaseModel):
    """AI decision for a single stock."""
    symbol: str
    action: str = Field(pattern="^(BUY|SELL|HOLD)$")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    target_allocation_pct: float = Field(
        ge=0.0, le=30.0, default=0.0,
        description="Target portfolio allocation percentage (0-30%)",
    )


class ScreeningResult(BaseModel):
    """Result from stock screening."""
    watchlist_updates: list[WatchlistUpdate] = []
    market_summary: str = ""


class WatchlistUpdate(BaseModel):
    """Update to the watchlist."""
    symbol: str
    action: str = Field(pattern="^(ADD|REMOVE|KEEP)$")
    reason: str


# Fix forward reference
ScreeningResult.model_rebuild()


class AnalysisResult(BaseModel):
    """Result from intraday analysis."""
    decisions: list[StockDecision] = []
    market_context: str = ""


class WeeklyReview(BaseModel):
    """Weekly performance review."""
    performance_summary: str
    patterns: list[str] = []
    journal_entries: list[JournalEntry] = []
    watchlist_changes: list[WatchlistUpdate] = []


class JournalEntry(BaseModel):
    """Strategy journal entry."""
    entry_type: str = Field(pattern="^(observation|pattern|rule|revision)$")
    content: str
    supersedes_id: int | None = None


# Fix forward reference
WeeklyReview.model_rebuild()


class MonthlyReview(BaseModel):
    """Monthly strategic review."""
    strategic_summary: str
    risk_assessment: str
    recommended_changes: list[str] = []
    journal_entries: list[JournalEntry] = []
