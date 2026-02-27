from __future__ import annotations

import json
import logging
from typing import Any

import anthropic
import aiosqlite

from paper_trader.config import MODEL_PRICING, settings
from paper_trader.db import queries

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when API budget is exceeded."""


class APIPausedError(Exception):
    """Raised when API calls are paused."""


class AIClient:
    """
    Single chokepoint for all AI calls.
    Checks budget, makes call, records cost.
    """

    def __init__(self, db: aiosqlite.Connection, api_key: str | None = None):
        self._db = db
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or settings.anthropic_api_key
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

        # Check monthly budget (exclude dry run costs from budget enforcement)
        monthly_spend = await queries.get_monthly_spend(self._db, include_dry_run=False)

        if monthly_spend >= settings.budget_hard_stop_usd:
            await queries.set_setting(self._db, "api_paused", "true")
            await queries.set_setting(self._db, "pause_reason", "Monthly budget hard stop ($50)")
            raise BudgetExceededError(
                f"Monthly spend ${monthly_spend:.2f} >= hard stop ${settings.budget_hard_stop_usd:.2f}"
            )

        if monthly_spend >= settings.budget_warn_usd:
            logger.warning(
                "Budget warning: monthly spend $%.2f approaching limit $%.2f",
                monthly_spend, settings.budget_hard_stop_usd,
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
            # Try to extract JSON from markdown code block
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            raise

    @staticmethod
    def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost of an API call in USD."""
        pricing = MODEL_PRICING.get(model)
        if pricing is None:
            logger.warning("Unknown model %s, using Haiku pricing as fallback", model)
            pricing = {"input": 0.80, "output": 4.00}

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
