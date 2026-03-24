# Paper Trader

An AI-driven paper trading bot that simulates a CHF portfolio against real market data.
Built as an infrastructure-first learning platform: the goal is to understand what works
before risking real money, and to build the kind of tooling that can evolve toward
production-grade quantitative trading.

## Current State

The bot runs autonomously as a systemd user service. It screens stocks, fetches news,
computes signal quality, and lets a Claude LLM make trade decisions with full tool access
(P&L calculator, risk/reward, portfolio weights, correlation matrix, sector exposure).

This is a working system, not a demo. It trades on schedule, tracks commissions with IBKR's
real fee model, converts USD/CHF, enforces risk limits, and records every decision with
full reasoning. It also has real limitations: see "Where This Stands" below.

## Quick Start

```bash
./install.sh
# Edit .env with your Anthropic API key
uv run paper-trader
# Open http://0.0.0.0:8420
```

## Architecture

```
scheduler (APScheduler)
  |
  +-- news_fetch (every 30min, 24/7)
  |
  +-- trading sessions (SIX + US schedule)
        |
        +-- screener (watchlist mutations)
        +-- signal processing (conviction curves, sentiment, commission trajectory)
        +-- analyst (Claude + tools)
        +-- pipeline (risk checks -> execution)
        +-- portfolio manager (buy/sell, commissions, FX)
```

### Signal Processing Layer

Added March 2026 after the bot burned 30 CHF in commissions on 44 trades in three weeks.
The problem was not that the agent lacked constraints: it lacked vision. It saw each session
as a blank page, reacting to noise instead of trends.

Four components, no new DB tables (aggregates existing data):

- **Conviction curves**: 14-day decision history, linear regression per symbol. Classification: RISING, FALLING, OSCILLATING, FLAT, INSUFFICIENT.
- **News sentiment**: keyword scoring on headlines, daily aggregation, rolling trend (improving/worsening/stable).
- **Commission trajectory**: daily commission stats, annualized as % of capital. The "gut punch" metric.
- **Signal quality composite**: conviction + sentiment + relative strength. HIGH/MEDIUM/LOW/NOISE. Only HIGH is actionable for BUY.

The agent is free to trade (confidence threshold 0.40), but it sees the full signal report
before deciding. The idea: informed agents make better decisions than constrained ones.

### Trading Tools

The LLM has access to calculation tools during analysis:

- `calculate_pnl`: per-position P&L in CHF with entry/current prices
- `portfolio_weight`: position weight as % of total portfolio
- `risk_reward`: risk/reward ratio with stop-loss and target prices
- `correlation_matrix`: cross-correlation between held positions
- `sector_exposure`: sector concentration analysis
- `relative_strength`: RS scanner against SPY benchmark

### Risk Management

- Safety stop: 50% drawdown halts trading
- Position limit: 40% max per position
- Cash reserve: 50 CHF minimum
- Sector cap: 60%
- Stop-loss: 7% (growth), 15% (moonshot)
- Session deploy: max 40% of available cash per session, max 2 trades
- Churn detection: 72h cooloff after round-trip, 0.85 confidence required for re-entry
- Profit-taking: alert at +25%, recommend partial sell at 33%

### Commission Model

IBKR Tiered pricing:
- US equities: max($0.005/share, $1.00)
- SIX (Swiss): max(0.05% of value, CHF 1.50)
- Slippage: 5bps US, 10bps SIX
- Fractional shares: 4 decimals US, whole shares SIX

### Schedule (all times CET)

| Time | Type | Session |
|------|------|---------|
| 09:30 | full | SIX open scan |
| 12:00 | quick | European midday |
| 15:45 | full | US open scan |
| 18:00 | quick | US midday |
| 20:00 | full | US afternoon |
| 22:15 | snapshot | End of day |
| Fri 22:30 | review | Weekly (Sonnet) |
| 1st-3rd 22:45 | review | Monthly (Opus) |

News fetch runs every 30 minutes, 24/7.

## Dashboard

| Route | What it shows |
|-------|---------------|
| `/` | Portfolio, positions, equity curve, sector chart |
| `/decisions` | Decision log with full AI reasoning |
| `/journal` | Strategy journal (patterns, observations) |
| `/api-usage` | API cost breakdown by model and day |
| `/dry-run` | Historical simulation mode |
| `/settings` | Risk parameters, schedule, watchlist |

## Running as a Service

```bash
systemctl --user status paper-trader
systemctl --user restart paper-trader
systemctl --user stop paper-trader
journalctl --user -u paper-trader -f    # live logs
```

After modifying Python code, restart the service. Jinja templates reload automatically.

## Tests

```bash
uv run pytest -v    # 405 tests
```

## Tech Stack

FastAPI, SQLite (aiosqlite), yfinance, Anthropic SDK, APScheduler, Jinja2 + HTMX + Chart.js + Pico CSS.

## Where This Stands

This system works as infrastructure. The scheduler, pipeline, dashboard, risk management,
commission model, FX handling, and signal processing are solid. The test suite covers 405 cases.

The weak point is the decision engine. An LLM making BUY/SELL calls has no statistical edge:
it processes language, not price distributions. Experienced quant traders are clear on this:
LLMs are useful for NLP (parsing news, summarizing filings, sentiment extraction) but not
for trading decisions that need to be backtested against historical data.

That said, the criticism often comes from asking LLMs the wrong questions. Asking an LLM
"should I buy NVDA?" is like asking it to solve a differential equation by talking about it.
The right approach is to use the LLM as an orchestrator: let it call quantitative tools,
interpret their outputs, and synthesize a decision grounded in real calculations. That is
closer to what this system does, and closer to where the real opportunity might be.

### What a production trading system looks like

For reference, and because this is the direction we are evaluating:

- **Backtesting**: NautilusTrader (Rust core, Python API, same code for backtest and live) or VectorBT (vectorized research)
- **Indicators**: TA-Lib (200+ battle-tested indicators) or pandas-ta
- **Data**: Polygon.io, Databento, or Alpaca data. yfinance is unreliable for anything beyond prototyping.
- **Execution**: ib_async (IBKR), Alpaca, CCXT (crypto)
- **Risk/optimization**: Riskfolio-Lib, PyPortfolioOpt, CVXPY
- **Analytics**: QuantStats (tear sheets, Sharpe, drawdown, alpha/beta)
- **Architecture**: event-driven with message bus, deterministic replay, state machines for order lifecycle

The gap between this project and that stack is real, but the infrastructure we have
(scheduler, pipeline, tools, risk checks, commission model) maps to components that
production systems also need. The question is whether to evolve incrementally or adopt
a framework like NautilusTrader and rebuild around it.

No decision made yet. We are watching how the signal processing layer performs first.
