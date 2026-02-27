import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from paper_trader.ai.client import AIClient, BudgetExceededError, APIPausedError
from paper_trader.config import MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS
from paper_trader.db import queries


def make_mock_response(text: str, input_tokens: int = 100, output_tokens: int = 50):
    """Create a mock Anthropic API response."""
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return response


@pytest.fixture
async def ai_client(db_with_portfolio):
    client = AIClient(db_with_portfolio, api_key="test-key")
    return client, db_with_portfolio


class TestCostCalculation:
    def test_haiku_cost(self):
        # 1000 input ($0.80/M) + 500 output ($4/M)
        cost = AIClient._calculate_cost(MODEL_HAIKU, 1000, 500)
        expected = (1000 / 1e6) * 0.80 + (500 / 1e6) * 4.00
        assert abs(cost - expected) < 0.0001

    def test_sonnet_cost(self):
        cost = AIClient._calculate_cost(MODEL_SONNET, 1000, 500)
        expected = (1000 / 1e6) * 3.00 + (500 / 1e6) * 15.00
        assert abs(cost - expected) < 0.0001

    def test_opus_cost(self):
        cost = AIClient._calculate_cost(MODEL_OPUS, 1000, 500)
        expected = (1000 / 1e6) * 15.00 + (500 / 1e6) * 75.00
        assert abs(cost - expected) < 0.0001

    def test_unknown_model_uses_haiku_fallback(self):
        cost = AIClient._calculate_cost("unknown-model", 1000, 500)
        haiku_cost = AIClient._calculate_cost(MODEL_HAIKU, 1000, 500)
        assert cost == haiku_cost


class TestBudgetEnforcement:
    @pytest.mark.asyncio
    async def test_call_records_cost(self, ai_client):
        client, db = ai_client
        mock_response = make_mock_response('{"result": "ok"}', 500, 200)

        with patch.object(client._client.messages, "create", new_callable=AsyncMock, return_value=mock_response):
            result = await client.call(MODEL_HAIKU, "system", "prompt", "test")
            assert result == {"result": "ok"}

        spend = await queries.get_monthly_spend(db)
        assert spend > 0

    @pytest.mark.asyncio
    async def test_budget_hard_stop(self, ai_client):
        client, db = ai_client
        # Simulate $50 already spent
        await queries.record_api_call(db, MODEL_HAIKU, "test", 0, 0, 50.0)

        with pytest.raises(BudgetExceededError):
            await client.call(MODEL_HAIKU, "system", "prompt", "test")

        # Should have set api_paused
        assert await queries.get_setting(db, "api_paused") == "true"

    @pytest.mark.asyncio
    async def test_api_paused_rejects(self, ai_client):
        client, db = ai_client
        await queries.set_setting(db, "api_paused", "true")
        await queries.set_setting(db, "pause_reason", "Manual pause")

        with pytest.raises(APIPausedError):
            await client.call(MODEL_HAIKU, "system", "prompt", "test")

    @pytest.mark.asyncio
    async def test_budget_warning_still_works(self, ai_client):
        client, db = ai_client
        # Simulate $46 spent (above warning, below hard stop)
        await queries.record_api_call(db, MODEL_HAIKU, "test", 0, 0, 46.0)

        mock_response = make_mock_response('{"result": "ok"}')
        with patch.object(client._client.messages, "create", new_callable=AsyncMock, return_value=mock_response):
            result = await client.call(MODEL_HAIKU, "system", "prompt", "test")
            assert result == {"result": "ok"}


class TestResponseParsing:
    @pytest.mark.asyncio
    async def test_parse_json(self, ai_client):
        client, db = ai_client
        mock_response = make_mock_response('{"action": "BUY", "symbol": "AAPL"}')

        with patch.object(client._client.messages, "create", new_callable=AsyncMock, return_value=mock_response):
            result = await client.call(MODEL_HAIKU, "system", "prompt", "test")
            assert result["action"] == "BUY"

    @pytest.mark.asyncio
    async def test_parse_json_in_code_block(self, ai_client):
        client, db = ai_client
        text = '```json\n{"action": "SELL"}\n```'
        mock_response = make_mock_response(text)

        with patch.object(client._client.messages, "create", new_callable=AsyncMock, return_value=mock_response):
            result = await client.call(MODEL_HAIKU, "system", "prompt", "test")
            assert result["action"] == "SELL"

    @pytest.mark.asyncio
    async def test_parse_json_in_generic_code_block(self, ai_client):
        client, db = ai_client
        text = '```\n{"action": "HOLD"}\n```'
        mock_response = make_mock_response(text)

        with patch.object(client._client.messages, "create", new_callable=AsyncMock, return_value=mock_response):
            result = await client.call(MODEL_HAIKU, "system", "prompt", "test")
            assert result["action"] == "HOLD"
