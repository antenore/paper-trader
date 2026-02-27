from __future__ import annotations

from pydantic_settings import BaseSettings


# Model pricing per million tokens (USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-5-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-5-20250514": {"input": 15.00, "output": 75.00},
}

# Default model aliases
MODEL_HAIKU = "claude-haiku-4-5-20251001"
MODEL_SONNET = "claude-sonnet-4-5-20250514"
MODEL_OPUS = "claude-opus-4-5-20250514"


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_prefix": "PT_"}

    # Portfolio
    initial_cash_chf: float = 800.0
    currency: str = "CHF"

    # Risk limits
    safety_stop_pct: float = 0.50  # pause if portfolio < 50% of initial
    max_position_pct: float = 0.40  # no single position > 40%
    min_cash_reserve: float = 50.0  # always keep 50 CHF

    # API budget (USD)
    budget_warn_usd: float = 45.0
    budget_hard_stop_usd: float = 50.0

    # Anthropic
    anthropic_api_key: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8420

    # Database
    db_path: str = "data/paper_trader.db"

    # Market data
    default_watchlist: list[str] = [
        "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    ]

    # Scheduling (timezone)
    timezone: str = "US/Eastern"

    # Movement threshold for midday smart-skip (%)
    movement_threshold_pct: float = 1.0

    # Benchmark for alpha/beta comparison
    benchmark_symbol: str = "SPY"


settings = Settings()
