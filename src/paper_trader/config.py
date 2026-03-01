from __future__ import annotations

import logging

import aiosqlite
from pydantic_settings import BaseSettings

_config_logger = logging.getLogger(__name__)


# Model pricing per million tokens (USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
}

# Default model aliases
MODEL_HAIKU = "claude-haiku-4-5-20251001"
MODEL_SONNET = "claude-sonnet-4-6"
MODEL_OPUS = "claude-opus-4-6"


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

    # Confidence threshold for executing trades
    confidence_threshold: float = 0.4

    # Benchmark for alpha/beta comparison
    benchmark_symbol: str = "SPY"

    # Stop-loss: auto-sell when price drops this fraction below entry
    stop_loss_pct: float = 0.07

    # Sector cap: max fraction of portfolio in one sector
    sector_cap_pct: float = 0.60

    # Staggered entry: max fraction of cash deployed per session
    max_session_deploy_pct: float = 0.40

    # Correlation warning threshold (0-1)
    correlation_warn_threshold: float = 0.75

    # ── Risk tiers (growth vs moonshot) ──────────────────────────────
    # Growth: established companies with clear momentum (AAPL, NVDA, etc.)
    growth_max_position_pct: float = 0.25   # max 25% per growth position
    growth_stop_loss_pct: float = 0.07      # 7% stop-loss
    growth_max_bucket_pct: float = 0.65     # max 65% total in growth tier

    # Moonshot: higher-risk plays, small caps, speculative bets
    moonshot_max_position_pct: float = 0.10  # max 10% per moonshot position
    moonshot_stop_loss_pct: float = 0.15     # 15% stop-loss (more volatile, needs room)
    moonshot_max_bucket_pct: float = 0.20    # max 20% total in moonshot tier


settings = Settings()

# Valid risk tiers
RISK_TIERS = ("growth", "moonshot")


def get_tier_settings(tier: str) -> dict[str, float]:
    """Return risk parameters for a given tier."""
    if tier == "moonshot":
        return {
            "max_position_pct": settings.moonshot_max_position_pct,
            "stop_loss_pct": settings.moonshot_stop_loss_pct,
            "max_bucket_pct": settings.moonshot_max_bucket_pct,
        }
    # Default to growth for unknown tiers
    return {
        "max_position_pct": settings.growth_max_position_pct,
        "stop_loss_pct": settings.growth_stop_loss_pct,
        "max_bucket_pct": settings.growth_max_bucket_pct,
    }


# ── Dynamic config persistence ────────────────────────────────────────

def _parse_list(value: str) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


def _serialize_list(value: list[str]) -> str:
    return ", ".join(value)


CONFIG_KEYS: dict[str, tuple[str, type]] = {
    "config.budget_warn_usd": ("budget_warn_usd", float),
    "config.budget_hard_stop_usd": ("budget_hard_stop_usd", float),
    "config.safety_stop_pct": ("safety_stop_pct", float),
    "config.max_position_pct": ("max_position_pct", float),
    "config.min_cash_reserve": ("min_cash_reserve", float),
    "config.movement_threshold_pct": ("movement_threshold_pct", float),
    "config.confidence_threshold": ("confidence_threshold", float),
    "config.timezone": ("timezone", str),
    "config.benchmark_symbol": ("benchmark_symbol", str),
    "config.default_watchlist": ("default_watchlist", list),
    "config.stop_loss_pct": ("stop_loss_pct", float),
    "config.sector_cap_pct": ("sector_cap_pct", float),
    "config.max_session_deploy_pct": ("max_session_deploy_pct", float),
    "config.correlation_warn_threshold": ("correlation_warn_threshold", float),
    "config.growth_max_position_pct": ("growth_max_position_pct", float),
    "config.growth_stop_loss_pct": ("growth_stop_loss_pct", float),
    "config.growth_max_bucket_pct": ("growth_max_bucket_pct", float),
    "config.moonshot_max_position_pct": ("moonshot_max_position_pct", float),
    "config.moonshot_stop_loss_pct": ("moonshot_stop_loss_pct", float),
    "config.moonshot_max_bucket_pct": ("moonshot_max_bucket_pct", float),
}


async def load_config_from_db(db: aiosqlite.Connection) -> None:
    """Load saved config overrides from the settings table into the singleton."""
    rows = await db.execute_fetchall(
        "SELECT key, value FROM settings WHERE key LIKE 'config.%'"
    )
    for row in rows:
        key, raw_value = row["key"], row["value"]
        if key not in CONFIG_KEYS:
            continue
        attr_name, type_conv = CONFIG_KEYS[key]
        try:
            if type_conv is list:
                value = _parse_list(raw_value)
            else:
                value = type_conv(raw_value)
            object.__setattr__(settings, attr_name, value)
            _config_logger.debug("Loaded config %s = %s", attr_name, value)
        except (ValueError, TypeError):
            _config_logger.warning("Invalid saved config for %s: %s", key, raw_value)


async def save_config_to_db(db: aiosqlite.Connection, key: str, value: str) -> bool:
    """Save a config value to DB and update the in-memory singleton. Returns True on success."""
    if key not in CONFIG_KEYS:
        return False
    attr_name, type_conv = CONFIG_KEYS[key]
    try:
        if type_conv is list:
            parsed = _parse_list(value)
        else:
            parsed = type_conv(value)
    except (ValueError, TypeError):
        return False

    # Persist to DB
    await db.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = ?",
        (key, value, value),
    )
    await db.commit()

    # Update in-memory singleton
    object.__setattr__(settings, attr_name, parsed)
    _config_logger.info("Saved config %s = %s", key, parsed)
    return True
