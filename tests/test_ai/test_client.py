import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from paper_trader.ai.client import AIClient, BudgetExceededError, APIPausedError, ToolUseResponse, _repair_json
from paper_trader.ai.calculator import ToolUseAudit, execute_tool, CALCULATOR_TOOLS
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
        # 1000 input ($1.00/M) + 500 output ($5/M)
        cost = AIClient._calculate_cost(MODEL_HAIKU, 1000, 500)
        expected = (1000 / 1e6) * 1.00 + (500 / 1e6) * 5.00
        assert abs(cost - expected) < 0.0001

    def test_sonnet_cost(self):
        cost = AIClient._calculate_cost(MODEL_SONNET, 1000, 500)
        expected = (1000 / 1e6) * 3.00 + (500 / 1e6) * 15.00
        assert abs(cost - expected) < 0.0001

    def test_opus_cost(self):
        cost = AIClient._calculate_cost(MODEL_OPUS, 1000, 500)
        expected = (1000 / 1e6) * 5.00 + (500 / 1e6) * 25.00
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
        # Simulate total spend at hard stop threshold ($95 default)
        await queries.record_api_call(db, MODEL_HAIKU, "test", 0, 0, 95.0)

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
        # Simulate $81 spent (above warning $80, below hard stop $95)
        await queries.record_api_call(db, MODEL_HAIKU, "test", 0, 0, 81.0)

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


def make_tool_use_block(tool_id: str, name: str, tool_input: dict):
    """Create a mock tool_use content block."""
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = tool_input
    return block


def make_text_block(text: str):
    """Create a mock text content block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_tool_response(content_blocks, stop_reason="end_turn", input_tokens=100, output_tokens=50):
    """Create a mock response with mixed content blocks."""
    response = MagicMock()
    response.content = content_blocks
    response.stop_reason = stop_reason
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return response


class TestCallWithTools:
    @pytest.mark.asyncio
    async def test_direct_response_no_tool_use(self, ai_client):
        """Claude answers directly without calling any tools."""
        client, db = ai_client
        final = make_tool_response(
            [make_text_block('{"decisions": [], "market_context": "ok"}')],
            stop_reason="end_turn",
        )

        with patch.object(client._client.messages, "create", new_callable=AsyncMock, return_value=final):
            result = await client.call_with_tools(
                MODEL_HAIKU, "system", "prompt", "test",
                tools=CALCULATOR_TOOLS, execute_tool=execute_tool,
            )

        assert isinstance(result, ToolUseResponse)
        assert result.data == {"decisions": [], "market_context": "ok"}
        assert result.turns == 1
        assert result.total_input_tokens == 100
        assert result.total_output_tokens == 50

    @pytest.mark.asyncio
    async def test_single_tool_turn_then_final(self, ai_client):
        """Claude calls one tool, gets result, then gives final answer."""
        client, db = ai_client
        audit = ToolUseAudit()

        # Turn 1: Claude requests position_size tool
        tool_block = make_tool_use_block(
            "tool_1", "position_size",
            {"symbol": "AAPL", "price": 150, "cash": 800, "portfolio_total": 800, "target_pct": 15},
        )
        turn1 = make_tool_response([tool_block], stop_reason="tool_use", input_tokens=200, output_tokens=80)

        # Turn 2: Final text response
        turn2 = make_tool_response(
            [make_text_block('{"decisions": [{"symbol": "AAPL", "action": "BUY", "confidence": 0.8, "reasoning": "good", "target_allocation_pct": 15}], "market_context": "ok"}')],
            stop_reason="end_turn", input_tokens=300, output_tokens=150,
        )

        with patch.object(client._client.messages, "create", new_callable=AsyncMock, side_effect=[turn1, turn2]):
            result = await client.call_with_tools(
                MODEL_HAIKU, "system", "prompt", "test",
                tools=CALCULATOR_TOOLS, execute_tool=execute_tool, audit=audit,
            )

        assert result.turns == 2
        assert result.total_input_tokens == 500  # 200 + 300
        assert result.total_output_tokens == 230  # 80 + 150
        assert len(result.data["decisions"]) == 1
        # Audit should have recorded the tool call
        assert len(audit.calls) == 1
        assert audit.calls[0].name == "position_size"
        assert "AAPL" in audit.symbols_with_tool("position_size")

    @pytest.mark.asyncio
    async def test_max_turns_respected(self, ai_client):
        """Tool-use loop stops at max_turns, then makes a final text-only call."""
        client, db = ai_client

        tool_block = make_tool_use_block(
            "tool_1", "position_size",
            {"symbol": "AAPL", "price": 150, "cash": 800, "portfolio_total": 800, "target_pct": 15},
        )
        tool_response = make_tool_response([tool_block], stop_reason="tool_use")

        # After max_turns, the last response is tool_use, so we get an extra
        # final call without tools that returns JSON text.
        final_text = make_tool_response(
            [make_text_block('{"decisions": [], "market_context": "max turns"}')],
            stop_reason="end_turn",
        )

        # 2 tool turns + 1 final text-only call = 3 API calls
        with patch.object(
            client._client.messages, "create",
            new_callable=AsyncMock,
            side_effect=[tool_response, tool_response, final_text],
        ):
            result = await client.call_with_tools(
                MODEL_HAIKU, "system", "prompt", "test",
                tools=CALCULATOR_TOOLS, execute_tool=execute_tool,
                max_turns=2,
            )

        # 2 tool turns + 1 final = 3 turns total
        assert result.turns == 3
        assert result.data == {"decisions": [], "market_context": "max turns"}

    @pytest.mark.asyncio
    async def test_token_cost_accumulation(self, ai_client):
        """Tokens and costs from all turns are summed."""
        client, db = ai_client

        tool_block = make_tool_use_block(
            "tool_1", "calculate_pnl",
            {"symbol": "AAPL", "entry_price": 100, "current_price": 110, "shares": 10},
        )
        turn1 = make_tool_response([tool_block], stop_reason="tool_use", input_tokens=100, output_tokens=40)
        turn2 = make_tool_response(
            [make_text_block('{"decisions": [], "market_context": "ok"}')],
            stop_reason="end_turn", input_tokens=200, output_tokens=60,
        )

        with patch.object(client._client.messages, "create", new_callable=AsyncMock, side_effect=[turn1, turn2]):
            result = await client.call_with_tools(
                MODEL_HAIKU, "system", "prompt", "test",
                tools=CALCULATOR_TOOLS, execute_tool=execute_tool,
            )

        assert result.total_input_tokens == 300
        assert result.total_output_tokens == 100
        assert result.total_cost > 0

        # Cost should have been recorded in DB
        spend = await queries.get_monthly_spend(db)
        assert spend > 0

    @pytest.mark.asyncio
    async def test_budget_enforcement(self, ai_client):
        """Budget check fires before the tool-use loop starts."""
        client, db = ai_client
        await queries.record_api_call(db, MODEL_HAIKU, "test", 0, 0, 95.0)

        with pytest.raises(BudgetExceededError):
            await client.call_with_tools(
                MODEL_HAIKU, "system", "prompt", "test",
                tools=CALCULATOR_TOOLS, execute_tool=execute_tool,
            )

    @pytest.mark.asyncio
    async def test_api_paused_enforcement(self, ai_client):
        """Paused API check fires before the tool-use loop starts."""
        client, db = ai_client
        await queries.set_setting(db, "api_paused", "true")

        with pytest.raises(APIPausedError):
            await client.call_with_tools(
                MODEL_HAIKU, "system", "prompt", "test",
                tools=CALCULATOR_TOOLS, execute_tool=execute_tool,
            )


class TestRepairJson:
    def test_valid_json_unchanged(self):
        s = '{"key": "value"}'
        assert json.loads(_repair_json(s)) == {"key": "value"}

    def test_truncated_string(self):
        s = '{"key": "unterminated'
        result = json.loads(_repair_json(s))
        assert result["key"] == "unterminated"

    def test_truncated_nested_object(self):
        s = '{"decisions": [{"action": "BUY", "reason": "good"'
        result = json.loads(_repair_json(s))
        assert result["decisions"][0]["action"] == "BUY"

    def test_truncated_array(self):
        s = '{"items": [1, 2, 3'
        result = json.loads(_repair_json(s))
        assert result["items"] == [1, 2, 3]

    def test_truncated_string_in_array(self):
        s = '{"decisions": [{"action": "HOLD", "reasoning": "The market is showing signs of'
        result = json.loads(_repair_json(s))
        assert result["decisions"][0]["action"] == "HOLD"

    def test_deeply_nested_truncation(self):
        s = '{"a": {"b": [{"c": "val'
        result = json.loads(_repair_json(s))
        assert result["a"]["b"][0]["c"] == "val"
