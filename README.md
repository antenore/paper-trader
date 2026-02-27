# Paper Trader

A local paper trading bot that simulates trading with a virtual 800 CHF portfolio, using real market data (yfinance) and Claude AI for decisions.

**Purpose**: Learn what works before risking real money.

## Quick Start

```bash
./install.sh
# Edit .env with your Anthropic API key
uv run paper-trader
# Open http://0.0.0.0:8420
```

## How It Works

- **3x daily** during US market hours (10:00, 13:00, 16:15 ET): stock screening, analysis, and trading decisions using Haiku
- **Weekly** (Friday 17:00 ET): performance review with Sonnet
- **Monthly** (1st trading day, 17:30 ET): strategic review with Opus
- **Midday smart-skip**: if no stock moved >2%, skips the AI call to save budget
- **Estimated cost**: ~$2/month (hard cap at $50)

## Features

- Real market data via yfinance (free, no API key needed)
- AI decisions with confidence scoring and reasoning
- Risk management: safety stop (50%), position limit (30%), cash reserve (100 CHF)
- Dry run mode: simulate ~30 days with historical data
- Strategy journal: AI learns and records patterns over time
- Dashboard with equity curve, decision log, API usage tracking

## Dashboard Pages

| Route | Description |
|---|---|
| `/` | Main dashboard: portfolio, positions, equity curve |
| `/decisions` | Full decision log with reasoning |
| `/journal` | Strategy journal (observations, patterns, rules) |
| `/api-usage` | API cost breakdown by model and day |
| `/dry-run` | Start/view dry run simulations |

## Running Tests

```bash
uv run pytest -v
```

## Configuration

All settings via environment variables (prefix `PT_`) or `.env` file. See `.env.example`.

## Tech Stack

FastAPI, SQLite (aiosqlite), yfinance, Anthropic SDK, APScheduler, Jinja2 + HTMX + Chart.js + Pico CSS
