"""Tests for server-side code execution (Anthropic sandbox) integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from paper_trader.ai.calculator import CODE_EXECUTION_TOOL, CALCULATOR_TOOLS
from paper_trader.ai.client import AIClient, ToolUseResponse
from paper_trader.ai.prompts import (
    CODE_EXECUTION_INSTRUCTIONS,
    WEEKLY_CODE_EXECUTION_INSTRUCTIONS,
    MONTHLY_CODE_EXECUTION_INSTRUCTIONS,
    get_analysis_system,
)
from paper_trader.ai.strategist import _snapshots_to_csv, _trades_to_csv
from paper_trader.config import MODEL_HAIKU, MODEL_SONNET, settings


# ── Tool definition tests ─────────────────────────────────────────────


class TestCodeExecutionToolDefinition:
    def test_tool_format(self):
        assert CODE_EXECUTION_TOOL == {
            "type": "code_execution_20250825",
            "name": "code_execution",
        }

    def test_tool_is_separate_from_calculator(self):
        """Code execution tool shouldn't be in CALCULATOR_TOOLS."""
        names = [t["name"] for t in CALCULATOR_TOOLS]
        assert "code_execution" not in names


# ── Prompt instruction tests ──────────────────────────────────────────


class TestCodeExecutionPrompts:
    def test_analysis_system_without_code_exec(self):
        system = get_analysis_system(enable_tools=True, enable_code_execution=False)
        assert "Python Sandbox" not in system
        assert "Calculator Tools" in system

    def test_analysis_system_with_code_exec(self):
        system = get_analysis_system(enable_tools=True, enable_code_execution=True)
        assert "Python Sandbox" in system
        assert "Calculator Tools" in system
        assert "pandas" in system

    def test_analysis_system_code_exec_only(self):
        """Code execution without calculator tools."""
        system = get_analysis_system(enable_tools=False, enable_code_execution=True)
        assert "Python Sandbox" in system
        assert "Calculator Tools" not in system

    def test_weekly_instructions_content(self):
        assert "win rate" in WEEKLY_CODE_EXECUTION_INSTRUCTIONS
        assert "Sharpe ratio" in WEEKLY_CODE_EXECUTION_INSTRUCTIONS

    def test_monthly_instructions_content(self):
        assert "Sharpe ratio" in MONTHLY_CODE_EXECUTION_INSTRUCTIONS
        assert "Monte Carlo" in MONTHLY_CODE_EXECUTION_INSTRUCTIONS
        assert "Sortino" in MONTHLY_CODE_EXECUTION_INSTRUCTIONS


# ── CSV conversion tests ─────────────────────────────────────────────


class TestSnapshotsToCsv:
    def test_empty_snapshots(self):
        assert _snapshots_to_csv([]) == ""

    def test_single_snapshot(self):
        snapshots = [{
            "snapshot_at": "2026-03-01T22:15:00",
            "cash": 500.0,
            "positions_value": 500.0,
            "total_value": 1000.0,
            "spy_price": 510.5,
            "benchmark_value": 1020.0,
            "usd_chf_rate": 0.89,
        }]
        csv = _snapshots_to_csv(snapshots)
        lines = csv.strip().split("\n")
        assert lines[0] == "date,cash,positions_value,total_value,spy_price,benchmark_value,usd_chf_rate"
        assert "2026-03-01" in lines[1]
        assert "500.00" in lines[1]
        assert "0.89" in lines[1]

    def test_chronological_order(self):
        """Snapshots come from DB as DESC; CSV should be chronological (ASC)."""
        snapshots = [
            {"snapshot_at": "2026-03-03", "cash": 300, "positions_value": 700,
             "total_value": 1000, "spy_price": None, "benchmark_value": None, "usd_chf_rate": None},
            {"snapshot_at": "2026-03-02", "cash": 400, "positions_value": 600,
             "total_value": 1000, "spy_price": None, "benchmark_value": None, "usd_chf_rate": None},
            {"snapshot_at": "2026-03-01", "cash": 500, "positions_value": 500,
             "total_value": 1000, "spy_price": None, "benchmark_value": None, "usd_chf_rate": None},
        ]
        csv = _snapshots_to_csv(snapshots)
        lines = csv.strip().split("\n")
        # Data lines (skip header): oldest first
        assert "2026-03-01" in lines[1]
        assert "2026-03-03" in lines[3]

    def test_none_values_handled(self):
        """None values for optional fields should produce empty CSV cells."""
        snapshots = [{
            "snapshot_at": "2026-03-01",
            "cash": 1000, "positions_value": 0, "total_value": 1000,
            "spy_price": None, "benchmark_value": None, "usd_chf_rate": None,
        }]
        csv = _snapshots_to_csv(snapshots)
        # Should have empty fields, not 'None'
        assert "None" not in csv


class TestTradesToCsv:
    def test_empty_trades(self):
        assert _trades_to_csv([]) == ""

    def test_single_trade(self):
        trades = [{
            "executed_at": "2026-03-01T10:00:00",
            "symbol": "AAPL",
            "action": "BUY",
            "shares": 0.5,
            "price": 150.0,
            "total": 75.0,
            "commission_chf": 1.0,
            "currency": "USD",
        }]
        csv = _trades_to_csv(trades)
        lines = csv.strip().split("\n")
        assert lines[0] == "date,symbol,action,shares,price,total,commission_chf,currency"
        assert "AAPL" in lines[1]
        assert "BUY" in lines[1]

    def test_chronological_order(self):
        """Trades come from DB as DESC; CSV should be chronological (ASC)."""
        trades = [
            {"executed_at": "2026-03-03", "symbol": "AAPL", "action": "SELL",
             "shares": 1, "price": 160, "total": 160, "commission_chf": 1, "currency": "USD"},
            {"executed_at": "2026-03-01", "symbol": "AAPL", "action": "BUY",
             "shares": 1, "price": 150, "total": 150, "commission_chf": 1, "currency": "USD"},
        ]
        csv = _trades_to_csv(trades)
        lines = csv.strip().split("\n")
        assert "BUY" in lines[1]  # oldest first
        assert "SELL" in lines[2]


# ── Client pause_turn handling ────────────────────────────────────────


def _make_text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_response(content_blocks, stop_reason="end_turn", input_tokens=100, output_tokens=50):
    response = MagicMock()
    response.content = content_blocks
    response.stop_reason = stop_reason
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    response.container = None
    return response


def _make_server_tool_block():
    """Create a mock server_tool_use block (code execution)."""
    block = MagicMock()
    block.type = "server_tool_use"
    block.id = "srvtoolu_123"
    block.name = "bash_code_execution"
    return block


def _make_code_result_block():
    """Create a mock bash_code_execution_tool_result block."""
    block = MagicMock()
    block.type = "bash_code_execution_tool_result"
    block.tool_use_id = "srvtoolu_123"
    return block


class TestPauseTurnHandling:
    @pytest.fixture
    async def ai_client(self, db_with_portfolio):
        client = AIClient(db_with_portfolio, api_key="test-key")
        return client, db_with_portfolio

    @pytest.mark.asyncio
    async def test_pause_turn_continues(self, ai_client):
        """When server returns pause_turn, the client should continue the conversation."""
        client, db = ai_client

        # Turn 1: Claude starts code execution, gets paused
        pause_response = _make_response(
            [_make_server_tool_block(), _make_code_result_block()],
            stop_reason="pause_turn",
        )

        # Turn 2: Continuation completes with final text
        final_response = _make_response(
            [_make_text_block('{"performance_summary": "Good week", "patterns": [], "journal_entries": [], "watchlist_changes": []}')],
            stop_reason="end_turn",
        )

        with patch.object(
            client._client.messages, "create",
            new_callable=AsyncMock,
            side_effect=[pause_response, final_response],
        ):
            result = await client.call_with_tools(
                MODEL_HAIKU, "system", "prompt", "test",
                tools=[CODE_EXECUTION_TOOL],
                execute_tool=lambda n, i: {"error": "not expected"},
            )

        assert result.turns == 2
        assert result.data["performance_summary"] == "Good week"

    @pytest.mark.asyncio
    async def test_server_tool_blocks_not_treated_as_client_tools(self, ai_client):
        """Server-side code execution blocks should not trigger client-side tool handling."""
        client, db = ai_client

        # Response with server tool use + text — stop_reason is end_turn
        response = _make_response(
            [
                _make_server_tool_block(),
                _make_code_result_block(),
                _make_text_block('{"result": "computed"}'),
            ],
            stop_reason="end_turn",
        )

        with patch.object(
            client._client.messages, "create",
            new_callable=AsyncMock,
            return_value=response,
        ):
            result = await client.call_with_tools(
                MODEL_HAIKU, "system", "prompt", "test",
                tools=[CODE_EXECUTION_TOOL],
                execute_tool=lambda n, i: {"error": "should not be called"},
            )

        assert result.turns == 1
        assert result.data == {"result": "computed"}


class TestContainerTracking:
    @pytest.fixture
    async def ai_client(self, db_with_portfolio):
        client = AIClient(db_with_portfolio, api_key="test-key")
        return client, db_with_portfolio

    @pytest.mark.asyncio
    async def test_container_id_captured(self, ai_client):
        """Container ID from code execution response should be captured."""
        client, db = ai_client

        response = _make_response(
            [_make_text_block('{"result": "ok"}')],
            stop_reason="end_turn",
        )
        # Simulate container in response
        response.container = MagicMock()
        response.container.id = "container_abc123"

        with patch.object(
            client._client.messages, "create",
            new_callable=AsyncMock,
            return_value=response,
        ):
            result = await client.call_with_tools(
                MODEL_HAIKU, "system", "prompt", "test",
                tools=[CODE_EXECUTION_TOOL],
                execute_tool=lambda n, i: {},
            )

        assert result.container_id == "container_abc123"

    @pytest.mark.asyncio
    async def test_no_container_when_not_used(self, ai_client):
        """No container ID when code execution isn't used."""
        client, db = ai_client

        response = _make_response(
            [_make_text_block('{"result": "ok"}')],
            stop_reason="end_turn",
        )
        response.container = None

        with patch.object(
            client._client.messages, "create",
            new_callable=AsyncMock,
            return_value=response,
        ):
            result = await client.call_with_tools(
                MODEL_HAIKU, "system", "prompt", "test",
                tools=CALCULATOR_TOOLS,
                execute_tool=lambda n, i: {},
            )

        assert result.container_id is None


# ── Config tests ──────────────────────────────────────────────────────


class TestCodeExecutionConfig:
    def test_default_disabled(self):
        assert settings.enable_code_execution is False

    def test_setting_name(self):
        """Ensure the setting exists on the Settings class."""
        from paper_trader.config import Settings
        s = Settings(anthropic_api_key="test")
        assert hasattr(s, "enable_code_execution")
        assert s.enable_code_execution is False
