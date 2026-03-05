"""Calculator tools for Claude API tool use — exact arithmetic instead of LLM guessing."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from paper_trader.portfolio.tools import get_sector

logger = logging.getLogger(__name__)


# ── Audit dataclasses ─────────────────────────────────────────────────


@dataclass
class ToolCall:
    """Record of a single tool invocation."""
    name: str
    inputs: dict[str, Any]
    result: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolUseAudit:
    """Audit trail for all tool calls in one analysis session."""
    calls: list[ToolCall] = field(default_factory=list)
    turns: int = 0

    def record(self, name: str, inputs: dict[str, Any], result: dict[str, Any]) -> None:
        self.calls.append(ToolCall(name=name, inputs=inputs, result=result))

    def symbols_with_tool(self, tool_name: str) -> set[str]:
        """Return the set of symbols for which a given tool was called."""
        return {
            c.inputs.get("symbol", "")
            for c in self.calls
            if c.name == tool_name and "symbol" in c.inputs
        }


# ── Server-side code execution tool (Anthropic sandbox) ───────────────

CODE_EXECUTION_TOOL: dict[str, str] = {
    "type": "code_execution_20250825",
    "name": "code_execution",
}

# ── Tool definitions (Anthropic JSON Schema format) ──────────────────

CALCULATOR_TOOLS: list[dict[str, Any]] = [
    {
        "name": "position_size",
        "description": (
            "Calculate how many shares to buy of a stock given a target portfolio "
            "allocation percentage. Returns exact share count, cost in CHF (FX-converted "
            "for USD stocks), and resulting portfolio weight. MUST be called before every BUY decision."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol"},
                "price": {"type": "number", "description": "Current share price in native currency (USD or CHF for .SW stocks)"},
                "cash": {"type": "number", "description": "Available cash in CHF"},
                "portfolio_total": {"type": "number", "description": "Total portfolio value in CHF"},
                "target_pct": {"type": "number", "description": "Target allocation percentage (e.g. 15 for 15%)"},
                "existing_value": {"type": "number", "description": "Current position value in CHF (0 if new)", "default": 0},
                "usd_chf_rate": {"type": "number", "description": "USD/CHF exchange rate (default 1.0, used for USD stocks)", "default": 1.0},
            },
            "required": ["symbol", "price", "cash", "portfolio_total", "target_pct"],
        },
    },
    {
        "name": "calculate_pnl",
        "description": (
            "Calculate exact profit/loss for a position in CHF and percentage. "
            "Optionally include buy/sell commissions for net P&L. "
            "MUST be called before every SELL decision."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol"},
                "entry_price": {"type": "number", "description": "Average entry price per share"},
                "current_price": {"type": "number", "description": "Current share price"},
                "shares": {"type": "number", "description": "Number of shares held"},
                "buy_commission": {"type": "number", "description": "Commission paid on buy in CHF", "default": 0},
                "sell_commission": {"type": "number", "description": "Estimated commission on sell in CHF", "default": 0},
            },
            "required": ["symbol", "entry_price", "current_price", "shares"],
        },
    },
    {
        "name": "risk_reward",
        "description": (
            "Calculate risk/reward ratio given entry, stop-loss, and target prices."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol"},
                "entry": {"type": "number", "description": "Entry price"},
                "stop_loss": {"type": "number", "description": "Stop-loss price"},
                "target": {"type": "number", "description": "Target/take-profit price"},
            },
            "required": ["symbol", "entry", "stop_loss", "target"],
        },
    },
    {
        "name": "portfolio_weight",
        "description": "Calculate the portfolio weight percentage of a position.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol"},
                "position_value": {"type": "number", "description": "Current value of the position in CHF"},
                "total_value": {"type": "number", "description": "Total portfolio value in CHF"},
            },
            "required": ["symbol", "position_value", "total_value"],
        },
    },
    {
        "name": "break_even",
        "description": "Calculate the break-even price for a position accounting for commission costs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol"},
                "entry_price": {"type": "number", "description": "Average entry price per share"},
                "shares": {"type": "number", "description": "Number of shares held"},
                "commission": {"type": "number", "description": "Total commission/fees paid in USD", "default": 0},
            },
            "required": ["symbol", "entry_price", "shares"],
        },
    },
    {
        "name": "what_if_buy",
        "description": (
            "Simulate the impact of buying a stock on portfolio composition. "
            "Shows resulting cash, weight, and sector exposure. Cost is FX-converted to CHF for USD stocks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol to buy"},
                "shares": {"type": "number", "description": "Number of shares to buy"},
                "price": {"type": "number", "description": "Price per share in native currency (USD or CHF for .SW stocks)"},
                "cash": {"type": "number", "description": "Current cash in CHF"},
                "positions_value": {"type": "number", "description": "Current total positions value in CHF"},
                "sector_values": {
                    "type": "object",
                    "description": "Current sector values as {sector_name: value_in_CHF}",
                    "additionalProperties": {"type": "number"},
                },
                "usd_chf_rate": {"type": "number", "description": "USD/CHF exchange rate (default 1.0, used for USD stocks)", "default": 1.0},
            },
            "required": ["symbol", "shares", "price", "cash", "positions_value"],
        },
    },
]


# ── Pure calculation functions ────────────────────────────────────────


def _calc_position_size(
    symbol: str,
    price: float,
    cash: float,
    portfolio_total: float,
    target_pct: float,
    existing_value: float = 0,
    usd_chf_rate: float = 1.0,
) -> dict[str, Any]:
    """Calculate shares to buy for a target allocation (cost in CHF)."""
    from paper_trader.portfolio.commission import calculate_commission, round_shares
    from paper_trader.portfolio.currency import symbol_currency, to_chf

    if price <= 0:
        return {"error": "Price must be positive", "symbol": symbol}
    if portfolio_total <= 0:
        return {"error": "Portfolio total must be positive", "symbol": symbol}

    currency = symbol_currency(symbol)
    price_chf = to_chf(price, currency, usd_chf_rate)

    target_value = portfolio_total * (target_pct / 100)
    additional_value = max(0, target_value - existing_value)
    affordable = min(additional_value, cash)
    shares = affordable / price_chf if price_chf > 0 else 0
    shares = round_shares(symbol, shares)
    cost_chf = shares * price_chf
    commission_chf = calculate_commission(symbol, shares, price, usd_chf_rate)
    total_cost_chf = cost_chf + commission_chf
    # If total cost exceeds cash, reduce shares
    if total_cost_chf > cash and price_chf > 0:
        shares = round_shares(symbol, (cash - commission_chf) / price_chf)
        if shares <= 0:
            shares = 0.0
        cost_chf = shares * price_chf
        commission_chf = calculate_commission(symbol, shares, price, usd_chf_rate)
        total_cost_chf = cost_chf + commission_chf
    new_weight = ((existing_value + cost_chf) / portfolio_total) * 100 if portfolio_total > 0 else 0

    return {
        "symbol": symbol,
        "shares": round(shares, 4),
        "cost_chf": round(cost_chf, 2),
        "commission_chf": round(commission_chf, 2),
        "total_cost_chf": round(total_cost_chf, 2),
        "remaining_cash": round(cash - total_cost_chf, 2),
        "new_weight_pct": round(new_weight, 2),
        "target_value": round(target_value, 2),
        "currency": currency,
        "usd_chf_rate": usd_chf_rate,
    }


def _calc_pnl(
    symbol: str,
    entry_price: float,
    current_price: float,
    shares: float,
    buy_commission: float = 0,
    sell_commission: float = 0,
) -> dict[str, Any]:
    """Calculate P&L in CHF and percentage, optionally net of commissions."""
    if entry_price <= 0:
        return {"error": "Entry price must be positive", "symbol": symbol}

    pnl_per_share = current_price - entry_price
    pnl_total = pnl_per_share * shares
    total_commissions = buy_commission + sell_commission
    net_pnl = pnl_total - total_commissions
    pnl_pct = (pnl_per_share / entry_price) * 100

    result: dict[str, Any] = {
        "symbol": symbol,
        "pnl_chf": round(pnl_total, 2),
        "pnl_pct": round(pnl_pct, 2),
        "entry_price": entry_price,
        "current_price": current_price,
        "shares": shares,
        "position_value": round(current_price * shares, 2),
    }
    if total_commissions > 0:
        result["total_commissions"] = round(total_commissions, 2)
        result["net_pnl_chf"] = round(net_pnl, 2)
    return result


def _calc_risk_reward(
    symbol: str,
    entry: float,
    stop_loss: float,
    target: float,
) -> dict[str, Any]:
    """Calculate risk/reward ratio."""
    risk = entry - stop_loss
    if risk <= 0:
        return {"error": "Stop-loss must be below entry price", "symbol": symbol}

    reward = target - entry
    if reward <= 0:
        return {"error": "Target must be above entry price", "symbol": symbol}

    ratio = reward / risk

    return {
        "symbol": symbol,
        "risk_reward_ratio": round(ratio, 2),
        "risk_per_share": round(risk, 2),
        "reward_per_share": round(reward, 2),
        "risk_pct": round((risk / entry) * 100, 2),
        "reward_pct": round((reward / entry) * 100, 2),
    }


def _calc_portfolio_weight(
    symbol: str,
    position_value: float,
    total_value: float,
) -> dict[str, Any]:
    """Calculate portfolio weight of a position."""
    if total_value <= 0:
        return {"error": "Total value must be positive", "symbol": symbol}

    weight = (position_value / total_value) * 100

    return {
        "symbol": symbol,
        "weight_pct": round(weight, 2),
        "position_value": round(position_value, 2),
        "total_value": round(total_value, 2),
    }


def _calc_break_even(
    symbol: str,
    entry_price: float,
    shares: float,
    commission: float = 0,
) -> dict[str, Any]:
    """Calculate break-even price accounting for commissions."""
    if shares <= 0:
        return {"error": "Shares must be positive", "symbol": symbol}

    total_cost = (entry_price * shares) + commission
    break_even_price = total_cost / shares

    return {
        "symbol": symbol,
        "break_even_price": round(break_even_price, 4),
        "entry_price": entry_price,
        "commission": commission,
        "commission_per_share": round(commission / shares, 4),
    }


def _calc_what_if_buy(
    symbol: str,
    shares: float,
    price: float,
    cash: float,
    positions_value: float,
    sector_values: dict[str, float] | None = None,
    usd_chf_rate: float = 1.0,
) -> dict[str, Any]:
    """Simulate the impact of a buy on portfolio composition (all values in CHF)."""
    from paper_trader.portfolio.commission import calculate_commission
    from paper_trader.portfolio.currency import symbol_currency, to_chf

    currency = symbol_currency(symbol)
    cost_chf = to_chf(shares * price, currency, usd_chf_rate)
    commission_chf = calculate_commission(symbol, shares, price, usd_chf_rate)
    total_cost = cost_chf + commission_chf
    if total_cost > cash:
        return {"error": f"Insufficient cash: need {total_cost:.2f} CHF (incl. {commission_chf:.2f} commission), have {cash:.2f} CHF", "symbol": symbol}

    new_cash = cash - total_cost
    new_positions_value = positions_value + cost_chf
    new_total = new_cash + new_positions_value
    new_weight = (cost_chf / new_total) * 100 if new_total > 0 else 0

    sector = get_sector(symbol)
    result: dict[str, Any] = {
        "symbol": symbol,
        "cost_chf": round(cost_chf, 2),
        "commission_chf": round(commission_chf, 2),
        "total_cost_chf": round(total_cost, 2),
        "new_cash": round(new_cash, 2),
        "new_total": round(new_total, 2),
        "new_weight_pct": round(new_weight, 2),
        "sector": sector,
        "currency": currency,
    }

    # Sector exposure after buy
    if sector_values is not None:
        updated_sectors = dict(sector_values)
        updated_sectors[sector] = updated_sectors.get(sector, 0) + cost_chf
        sector_pcts = {
            s: round((v / new_total) * 100, 2)
            for s, v in updated_sectors.items()
            if v > 0
        }
        result["sector_exposure_pct"] = sector_pcts

    return result


# ── Dispatch ──────────────────────────────────────────────────────────

_DISPATCH: dict[str, Any] = {
    "position_size": lambda inputs: _calc_position_size(
        symbol=inputs["symbol"],
        price=inputs["price"],
        cash=inputs["cash"],
        portfolio_total=inputs["portfolio_total"],
        target_pct=inputs["target_pct"],
        existing_value=inputs.get("existing_value", 0),
        usd_chf_rate=inputs.get("usd_chf_rate", 1.0),
    ),
    "calculate_pnl": lambda inputs: _calc_pnl(
        symbol=inputs["symbol"],
        entry_price=inputs["entry_price"],
        current_price=inputs["current_price"],
        shares=inputs["shares"],
        buy_commission=inputs.get("buy_commission", 0),
        sell_commission=inputs.get("sell_commission", 0),
    ),
    "risk_reward": lambda inputs: _calc_risk_reward(
        symbol=inputs["symbol"],
        entry=inputs["entry"],
        stop_loss=inputs["stop_loss"],
        target=inputs["target"],
    ),
    "portfolio_weight": lambda inputs: _calc_portfolio_weight(
        symbol=inputs["symbol"],
        position_value=inputs["position_value"],
        total_value=inputs["total_value"],
    ),
    "break_even": lambda inputs: _calc_break_even(
        symbol=inputs["symbol"],
        entry_price=inputs["entry_price"],
        shares=inputs["shares"],
        commission=inputs.get("commission", 0),
    ),
    "what_if_buy": lambda inputs: _calc_what_if_buy(
        symbol=inputs["symbol"],
        shares=inputs["shares"],
        price=inputs["price"],
        cash=inputs["cash"],
        positions_value=inputs["positions_value"],
        sector_values=inputs.get("sector_values"),
        usd_chf_rate=inputs.get("usd_chf_rate", 1.0),
    ),
}


def execute_tool(name: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call and return the result. Catches errors gracefully."""
    handler = _DISPATCH.get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        result = handler(inputs)
        logger.info("Tool call: %s(%s) → %s", name, inputs.get("symbol", ""), result)
        return result
    except (KeyError, TypeError) as exc:
        return {"error": f"Invalid inputs for {name}: {exc}"}
    except Exception as exc:
        logger.error("Tool %s failed: %s", name, exc, exc_info=True)
        return {"error": f"Tool {name} error: {exc}"}
