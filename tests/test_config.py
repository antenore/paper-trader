from paper_trader.config import Settings, MODEL_PRICING, MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS


def test_default_settings():
    s = Settings(anthropic_api_key="test-key")
    assert s.initial_cash_chf == 800.0
    assert s.currency == "CHF"
    assert s.port == 8420
    assert s.budget_hard_stop_usd == 50.0
    assert s.safety_stop_pct == 0.50
    assert s.max_position_pct == 0.30
    assert s.min_cash_reserve == 100.0


def test_model_pricing_has_all_models():
    for model in [MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS]:
        assert model in MODEL_PRICING
        assert "input" in MODEL_PRICING[model]
        assert "output" in MODEL_PRICING[model]


def test_default_watchlist():
    s = Settings(anthropic_api_key="test-key")
    assert "SPY" in s.default_watchlist
    assert len(s.default_watchlist) > 0
