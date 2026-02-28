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

| Time (ET) | Type | What it does |
|-----------|------|-------------|
| 10:00 | full | Screen + analyze + trade (Haiku) |
| 13:00 | quick | Check positions + trade if movement >1% |
| 14:30 | full | Screen + analyze + trade — afternoon session (Haiku) |
| 16:15 | snapshot | Record closing prices |
| Fri 17:00 | weekly | Performance review (Sonnet) |
| 1st of month 17:30 | monthly | Strategic review (Opus) |

- **Midday smart-skip**: if no stock moved >1% **and** no open positions, skips the AI call to save budget
- **Estimated cost**: ~$2/month (hard cap at $50)

## Features

- Real market data via yfinance (free, no API key needed)
- AI decisions with confidence scoring and reasoning
- Risk management: safety stop (50%), position limit (40%), cash reserve (50 CHF), sector cap (60%), stop-loss (7%)
- Trading tools: stop-loss automation, correlation matrix, sector exposure analysis, relative strength scanner
- Rule enforcement: staggered entry (40% session cap), ETF overlap detection, sector concentration limits
- Aggressive paper trading stance: confidence threshold 0.4, prefers action over inaction
- Dry run mode: simulate ~30 days with historical data
- Strategy journal: AI learns and records patterns over time
- SPY benchmark comparison with alpha tracking
- Dashboard with equity curve, sector allocation chart, decision log, API usage tracking

## Dashboard Pages

| Route | Description |
|---|---|
| `/` | Main dashboard: portfolio, positions, equity curve, sector chart |
| `/decisions` | Full decision log with reasoning |
| `/journal` | Strategy journal (observations, patterns, rules) |
| `/api-usage` | API cost breakdown by model and day |
| `/dry-run` | Start/view dry run simulations |
| `/settings` | Configure risk parameters, schedule, watchlist |

## Running as a Service

```bash
# The bot runs as a systemd user service (no sudo needed)
systemctl --user status paper-trader    # check status
systemctl --user restart paper-trader   # restart
systemctl --user stop paper-trader      # stop
journalctl --user -u paper-trader -f    # live logs
```

## Running Tests

```bash
uv run pytest -v
```

## Configuration

All settings via environment variables (prefix `PT_`) or `.env` file. See `.env.example`.

## Tech Stack

FastAPI, SQLite (aiosqlite), yfinance, Anthropic SDK, APScheduler, Jinja2 + HTMX + Chart.js + Pico CSS
