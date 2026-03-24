from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic
import aiosqlite

from paper_trader.config import MODEL_HAIKU, MODEL_PRICING, settings
from paper_trader.db import queries

logger = logging.getLogger(__name__)


def _repair_json(s: str) -> str:
    """Attempt to repair truncated JSON by closing open strings, arrays, and objects."""
    # Strip trailing whitespace
    s = s.rstrip()

    # Track nesting to know what to close
    in_string = False
    escape = False
    stack: list[str] = []  # '{' or '['

    for ch in s:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()

    # If we ended inside a string, close it
    if in_string:
        s += '"'

    # Close open brackets/braces in reverse order
    for opener in reversed(stack):
        s += ']' if opener == '[' else '}'

    return s


class BudgetExceededError(Exception):
    """Raised when API budget is exceeded."""


class APIPausedError(Exception):
    """Raised when API calls are paused."""


@dataclass
class ToolUseResponse:
    """Response from a multi-turn tool-use call."""
    data: dict[str, Any]
    audit: Any = None  # ToolUseAudit from calculator module
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    turns: int = 0
    container_id: str | None = None  # Code execution container for reuse


class AIClient:
    """
    Single chokepoint for all AI calls.
    Checks budget, makes call, records cost.
    """

    def __init__(self, db: aiosqlite.Connection, api_key: str | None = None):
        self._db = db
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or settings.anthropic_api_key,
            max_retries=5,
        )

    async def call(
        self,
        model: str,
        system: str,
        prompt: str,
        purpose: str,
        is_dry_run: bool = False,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """
        Make an AI call with budget enforcement.
        Returns parsed JSON response.
        """
        # Check if paused
        paused = await queries.get_setting(self._db, "api_paused")
        if paused == "true":
            reason = await queries.get_setting(self._db, "pause_reason") or "Manual pause"
            raise APIPausedError(f"API calls paused: {reason}")

        # Check total (all-time) spend against account balance — exclude dry run costs
        total_spend = await queries.get_total_spend(self._db, include_dry_run=False)

        if total_spend >= settings.budget_hard_stop_usd:
            await queries.set_setting(self._db, "api_paused", "true")
            await queries.set_setting(
                self._db, "pause_reason",
                f"API budget hard stop (${total_spend:.2f} spent of ${settings.api_account_balance_usd:.2f})",
            )
            raise BudgetExceededError(
                f"Total spend ${total_spend:.2f} >= hard stop ${settings.budget_hard_stop_usd:.2f}"
            )

        if total_spend >= settings.budget_warn_usd:
            logger.warning(
                "Budget warning: total spend $%.2f approaching limit $%.2f (balance: $%.2f)",
                total_spend, settings.budget_hard_stop_usd, settings.api_account_balance_usd,
            )

        # Make the API call
        response = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        # Calculate cost
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = self._calculate_cost(model, input_tokens, output_tokens)

        # Record the call
        await queries.record_api_call(
            self._db, model, purpose, input_tokens, output_tokens, cost, is_dry_run
        )

        logger.info(
            "AI call [%s] %s: %d in / %d out = $%.4f",
            model.split("-")[1] if "-" in model else model,
            purpose,
            input_tokens,
            output_tokens,
            cost,
        )

        # Parse response
        text = response.content[0].text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
            else:
                json_str = text

            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning(
                    "JSON parse failed for %s, attempting repair (stop_reason=%s)",
                    purpose, response.stop_reason,
                )
                return json.loads(_repair_json(json_str))

    async def call_with_tools(
        self,
        model: str,
        system: str,
        prompt: str,
        purpose: str,
        tools: list[dict[str, Any]],
        execute_tool: Callable[[str, dict[str, Any]], dict[str, Any]],
        audit: Any = None,
        is_dry_run: bool = False,
        max_tokens: int = 4096,
        max_turns: int | None = None,
    ) -> ToolUseResponse:
        """
        Make an AI call with tool use (multi-turn loop).

        Sends the prompt with tool definitions. If Claude responds with tool_use
        blocks, executes them locally and feeds results back. Repeats until Claude
        returns end_turn or max_turns is reached.
        """
        if max_turns is None:
            max_turns = settings.tool_use_max_turns

        # Budget/pause checks (same as call())
        paused = await queries.get_setting(self._db, "api_paused")
        if paused == "true":
            reason = await queries.get_setting(self._db, "pause_reason") or "Manual pause"
            raise APIPausedError(f"API calls paused: {reason}")

        total_spend = await queries.get_total_spend(self._db, include_dry_run=False)
        if total_spend >= settings.budget_hard_stop_usd:
            await queries.set_setting(self._db, "api_paused", "true")
            await queries.set_setting(
                self._db, "pause_reason",
                f"API budget hard stop (${total_spend:.2f} spent of ${settings.api_account_balance_usd:.2f})",
            )
            raise BudgetExceededError(
                f"Total spend ${total_spend:.2f} >= hard stop ${settings.budget_hard_stop_usd:.2f}"
            )

        if total_spend >= settings.budget_warn_usd:
            logger.warning(
                "Budget warning: total spend $%.2f approaching limit $%.2f",
                total_spend, settings.budget_hard_stop_usd,
            )

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        total_input = 0
        total_output = 0
        total_cost = 0.0
        turns = 0

        for turn in range(max_turns):
            turns += 1
            response = await self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                tools=tools,
            )

            in_tok = response.usage.input_tokens
            out_tok = response.usage.output_tokens
            cost = self._calculate_cost(model, in_tok, out_tok)
            total_input += in_tok
            total_output += out_tok
            total_cost += cost

            # Handle pause_turn: server-side code execution paused a long turn.
            # Send the response back as-is to let Claude continue.
            if response.stop_reason == "pause_turn":
                logger.info("Code execution pause_turn on turn %d, continuing...", turn + 1)
                messages.append({"role": "assistant", "content": response.content})
                continue

            # Check if Claude wants to use client-side tools
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            if response.stop_reason != "tool_use" or not tool_use_blocks:
                # Final response — extract text and parse JSON
                text_blocks = [b for b in response.content if b.type == "text"]
                text = text_blocks[0].text if text_blocks else ""
                break

            # Execute tool calls and build result messages
            assistant_content = response.content
            tool_results = []
            for block in tool_use_blocks:
                result = execute_tool(block.name, block.input)
                logger.info(
                    "Tool use turn %d: %s(%s) → %s",
                    turn + 1, block.name, block.input.get("symbol", ""), result,
                )
                if audit is not None:
                    audit.record(block.name, block.input, result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})
        else:
            # Max turns exhausted
            logger.warning("Tool use max turns (%d) reached for %s", max_turns, purpose)

            # If last response was tool_use, execute the tools and make one
            # final call WITHOUT tools to force a JSON text response.
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if tool_use_blocks:
                assistant_content = response.content
                tool_results = []
                for block in tool_use_blocks:
                    result = execute_tool(block.name, block.input)
                    logger.info(
                        "Tool use final turn: %s(%s) → %s",
                        block.name, block.input.get("symbol", ""), result,
                    )
                    if audit is not None:
                        audit.record(block.name, block.input, result)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": tool_results})

                # Ask explicitly for the JSON response
                messages.append({
                    "role": "user",
                    "content": (
                        "You have used all available tool turns. "
                        "Now produce your final JSON response based on the "
                        "tool results and analysis so far."
                    ),
                })

                # Final call without tools — forces text-only JSON response
                response = await self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                )
                in_tok = response.usage.input_tokens
                out_tok = response.usage.output_tokens
                cost = self._calculate_cost(model, in_tok, out_tok)
                total_input += in_tok
                total_output += out_tok
                total_cost += cost
                turns += 1

            text_blocks = [b for b in response.content if b.type == "text"]
            text = text_blocks[0].text if text_blocks else ""
            if not text.strip():
                logger.error(
                    "Empty text after max turns for %s. Response blocks: %s",
                    purpose, [b.type for b in response.content],
                )

        # Record aggregated cost
        await queries.record_api_call(
            self._db, model, purpose, total_input, total_output, total_cost, is_dry_run,
        )

        logger.info(
            "AI call [%s] %s (tool-use): %d turns, %d in / %d out = $%.4f",
            model.split("-")[1] if "-" in model else model,
            purpose, turns, total_input, total_output, total_cost,
        )

        # Parse JSON from final text
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
            else:
                json_str = text

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning(
                    "JSON parse failed for %s, attempting repair (stop_reason=%s)",
                    purpose, response.stop_reason,
                )
                repaired = _repair_json(json_str)
                data = json.loads(repaired)

        if audit is not None:
            audit.turns = turns

        # Track container ID for code execution reuse
        container_id = None
        if hasattr(response, "container") and response.container:
            container_id = response.container.id

        return ToolUseResponse(
            data=data,
            audit=audit,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_cost=total_cost,
            turns=turns,
            container_id=container_id,
        )

    @staticmethod
    def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost of an API call in USD."""
        pricing = MODEL_PRICING.get(model)
        if pricing is None:
            logger.warning("Unknown model %s, using Haiku pricing as fallback", model)
            pricing = MODEL_PRICING[MODEL_HAIKU]

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
