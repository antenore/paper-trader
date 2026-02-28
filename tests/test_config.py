import pytest

from paper_trader.config import (
    Settings, MODEL_PRICING, MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS,
    CONFIG_KEYS, load_config_from_db, save_config_to_db, settings,
)


def test_default_settings():
    s = Settings(anthropic_api_key="test-key")
    assert s.initial_cash_chf == 800.0
    assert s.currency == "CHF"
    assert s.port == 8420
    assert s.budget_hard_stop_usd == 50.0
    assert s.safety_stop_pct == 0.50
    assert s.max_position_pct == 0.40
    assert s.min_cash_reserve == 50.0
    assert s.confidence_threshold == 0.4


def test_model_pricing_has_all_models():
    for model in [MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS]:
        assert model in MODEL_PRICING
        assert "input" in MODEL_PRICING[model]
        assert "output" in MODEL_PRICING[model]


def test_default_watchlist():
    s = Settings(anthropic_api_key="test-key")
    assert "SPY" in s.default_watchlist
    assert len(s.default_watchlist) > 0


class TestConfigPersistence:
    @pytest.mark.asyncio
    async def test_save_config_to_db(self, db):
        original = settings.budget_warn_usd
        try:
            ok = await save_config_to_db(db, "config.budget_warn_usd", "30.0")
            assert ok is True
            assert settings.budget_warn_usd == 30.0
        finally:
            object.__setattr__(settings, "budget_warn_usd", original)

    @pytest.mark.asyncio
    async def test_load_config_from_db(self, db):
        original = settings.confidence_threshold
        try:
            await db.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?)",
                ("config.confidence_threshold", "0.75"),
            )
            await db.commit()
            await load_config_from_db(db)
            assert settings.confidence_threshold == 0.75
        finally:
            object.__setattr__(settings, "confidence_threshold", original)

    @pytest.mark.asyncio
    async def test_unknown_key_rejected(self, db):
        ok = await save_config_to_db(db, "config.nonexistent_key", "123")
        assert ok is False
