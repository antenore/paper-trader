from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from pathlib import Path

from paper_trader.ai.client import AIClient
from paper_trader.config import settings, CONFIG_KEYS, _serialize_list, save_config_to_db
from paper_trader.dashboard.csrf import csrf_token
from paper_trader.db.connection import get_db
from paper_trader.db import queries
from paper_trader.market.prices import LiveDataProvider
from paper_trader.portfolio.tools import get_sector, get_sector_exposure
from paper_trader.scheduler.jobs import (
    get_current_schedule,
    get_job_statuses,
    reschedule_job,
    SCHEDULE,
    trigger_job,
)
from paper_trader.trading.dry_run import reset_for_live, run_dry_run

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _csrf_hidden(request: Request) -> str:
    """Return an HTML hidden input with the CSRF token."""
    token = csrf_token(request)
    return Markup(f'<input type="hidden" name="csrf_token" value="{token}">')


def _ctx(request: Request, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build template context with common variables (CSRF, etc.)."""
    ctx: dict[str, Any] = {"csrf_hidden": _csrf_hidden(request)}
    if extra:
        ctx.update(extra)
    return ctx


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    db = await get_db()
    portfolio = await queries.get_portfolio(db)
    positions = await queries.get_open_positions(db)
    snapshots = await queries.get_snapshots(db, limit=90)
    decisions = await queries.get_decisions(db, limit=5)
    monthly_spend = await queries.get_monthly_spend(db, include_dry_run=False)
    mode = await queries.get_setting(db, "mode") or "live"
    api_paused = await queries.get_setting(db, "api_paused") == "true"
    pause_reason = await queries.get_setting(db, "pause_reason") or ""

    # Enrich positions with sector, current price, P&L%, stop distance
    enriched_positions = []
    prices: dict[str, float] = {}
    if positions:
        try:
            market = LiveDataProvider()
            symbols = [p["symbol"] for p in positions]
            prices = await market.get_current_prices(symbols)
        except Exception:
            logger.debug("Failed to fetch live prices for dashboard")
    for p in positions:
        current_price = prices.get(p["symbol"], p["avg_cost"])
        pnl_pct = (current_price / p["avg_cost"] - 1) * 100 if p["avg_cost"] > 0 else 0
        stop = p.get("stop_loss_price")
        stop_dist = ((current_price - stop) / current_price * 100) if stop and current_price > 0 else None
        enriched_positions.append({
            **p,
            "sector": get_sector(p["symbol"]),
            "current_price": current_price,
            "pnl_pct": pnl_pct,
            "stop_distance_pct": stop_dist,
        })

    # Total portfolio value = cash + market value of all positions
    positions_value = sum(
        prices.get(p["symbol"], p["avg_cost"]) * p["shares"]
        for p in positions
    ) if positions else 0.0
    total_value = (portfolio["cash"] if portfolio else 0.0) + positions_value

    # Sector exposure for doughnut chart
    sector_exposure = get_sector_exposure(positions, prices) if positions else {}

    # Equity curve data (reversed for chronological order)
    snapshots_chrono = list(reversed(snapshots))
    equity_dates = [s["snapshot_at"][:10] for s in snapshots_chrono]
    equity_values = [s["total_value"] for s in snapshots_chrono]
    benchmark_values = [s.get("benchmark_value") for s in snapshots_chrono]
    # Only pass benchmark if at least one non-None value exists
    has_benchmark = any(v is not None for v in benchmark_values)

    return templates.TemplateResponse(request, "index.html", _ctx(request, {
        "portfolio": portfolio,
        "positions": enriched_positions,
        "snapshots": snapshots,
        "decisions": decisions,
        "monthly_spend": monthly_spend,
        "budget_warn": settings.budget_warn_usd,
        "budget_limit": settings.budget_hard_stop_usd,
        "mode": mode,
        "api_paused": api_paused,
        "pause_reason": pause_reason,
        "equity_dates": equity_dates,
        "equity_values": equity_values,
        "benchmark_values": benchmark_values if has_benchmark else [],
        "total_value": total_value,
        "sector_exposure": sector_exposure,
    }))


@router.get("/decisions", response_class=HTMLResponse)
async def decisions_page(request: Request):
    per_page = 50
    try:
        page = max(1, int(request.query_params.get("page", "1")))
    except ValueError:
        page = 1
    offset = (page - 1) * per_page

    db = await get_db()
    total = await queries.count_decisions(db)
    decisions = await queries.get_decisions(db, limit=per_page, offset=offset)
    total_pages = max(1, (total + per_page - 1) // per_page)

    return templates.TemplateResponse(request, "decisions.html", {
        "decisions": decisions,
        "page": page,
        "total_pages": total_pages,
    })


@router.get("/journal", response_class=HTMLResponse)
async def journal_page(request: Request):
    db = await get_db()
    journal = await queries.get_active_journal(db)
    return templates.TemplateResponse(request, "journal.html", {
        "journal": journal,
    })


@router.get("/api-usage", response_class=HTMLResponse)
async def api_usage_page(request: Request):
    db = await get_db()
    by_model = await queries.get_api_usage_by_model(db, include_dry_run=False)
    daily = await queries.get_daily_spend(db, days=30, include_dry_run=False)
    monthly_spend = await queries.get_monthly_spend(db, include_dry_run=False)
    return templates.TemplateResponse(request, "api_usage.html", {
        "by_model": by_model,
        "daily": daily,
        "monthly_spend": monthly_spend,
        "budget_warn": settings.budget_warn_usd,
        "budget_limit": settings.budget_hard_stop_usd,
    })


@router.get("/dry-run", response_class=HTMLResponse)
async def dry_run_page(request: Request):
    db = await get_db()
    sessions = await queries.get_dry_run_sessions(db)
    return templates.TemplateResponse(request, "dry_run.html", _ctx(request, {
        "sessions": sessions,
        "default_symbols": settings.default_watchlist,
    }))


@router.post("/resume-trading")
async def resume_trading():
    db = await get_db()
    await queries.set_setting(db, "api_paused", "false")
    await queries.set_setting(db, "pause_reason", "")
    logger.info("Trading resumed via dashboard")
    return RedirectResponse(url="/", status_code=303)


@router.post("/go-live")
async def go_live():
    db = await get_db()
    await reset_for_live(db)
    logger.info("Switched to live mode via dashboard")
    return RedirectResponse(url="/", status_code=303)


@router.post("/start-dry-run")
async def start_dry_run_handler():
    """Start a dry run in the background."""
    db = await get_db()
    ai_client = AIClient(db)
    task = asyncio.create_task(run_dry_run(db, ai_client))
    task.add_done_callback(_log_task_result)
    return RedirectResponse(url="/dry-run", status_code=303)


def _log_task_result(task: asyncio.Task) -> None:
    """Log exceptions from background tasks instead of silently dropping them."""
    if task.cancelled():
        logger.warning("Background task %s was cancelled", task.get_name())
    elif exc := task.exception():
        logger.error("Background task %s failed: %s", task.get_name(), exc, exc_info=exc)


# ── Settings ─────────────────────────────────────────────────────────

@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    current = {}
    for db_key, (attr_name, type_conv) in CONFIG_KEYS.items():
        val = getattr(settings, attr_name)
        if type_conv is list:
            current[db_key] = _serialize_list(val)
        else:
            current[db_key] = val

    schedule = get_current_schedule()
    saved = request.query_params.get("saved")

    return templates.TemplateResponse(request, "settings.html", _ctx(request, {
        "config": current,
        "schedule": schedule,
        "saved": saved,
    }))


@router.post("/settings/config")
async def save_config(request: Request):
    db = await get_db()
    form = await request.form()

    for db_key in CONFIG_KEYS:
        value = form.get(db_key)
        if value is not None:
            ok = await save_config_to_db(db, db_key, str(value))
            if not ok:
                return RedirectResponse(url="/settings?saved=error", status_code=303)

    return RedirectResponse(url="/settings?saved=config", status_code=303)


@router.post("/settings/schedule")
async def save_schedule(request: Request):
    db = await get_db()
    form = await request.form()

    for _, default_cron, job_id, _ in SCHEDULE:
        hour = form.get(f"{job_id}.hour")
        minute = form.get(f"{job_id}.minute")
        day_of_week = form.get(f"{job_id}.day_of_week")
        if hour is None or minute is None:
            continue

        cron_kw: dict = {}
        try:
            cron_kw["hour"] = int(hour)
            cron_kw["minute"] = int(minute)
        except ValueError:
            return RedirectResponse(url="/settings?saved=error", status_code=303)

        if day_of_week:
            cron_kw["day_of_week"] = day_of_week
        if "day" in default_cron:
            day = form.get(f"{job_id}.day")
            if day:
                cron_kw["day"] = day

        reschedule_job(job_id, cron_kw)

        # Persist to DB
        await queries.set_setting(db, f"schedule.{job_id}", json.dumps(cron_kw))

    return RedirectResponse(url="/settings?saved=schedule", status_code=303)


# ── Manual Job Triggers ──────────────────────────────────────────────

@router.post("/trigger/{job_id}")
async def trigger_job_route(job_id: str):
    success, message = trigger_job(job_id)
    status_code = 200 if success else 409
    return JSONResponse({"success": success, "message": message}, status_code=status_code)


@router.get("/api/job-status")
async def job_status():
    return get_job_statuses()


# ── HTMX Partials ────────────────────────────────────────────────────

@router.get("/partials/portfolio", response_class=HTMLResponse)
async def partial_portfolio(request: Request):
    db = await get_db()
    portfolio = await queries.get_portfolio(db)
    positions = await queries.get_open_positions(db)

    enriched = []
    prices: dict[str, float] = {}
    if positions:
        try:
            market = LiveDataProvider()
            prices = await market.get_current_prices([p["symbol"] for p in positions])
        except Exception:
            pass
    for p in positions:
        cp = prices.get(p["symbol"], p["avg_cost"])
        pnl = (cp / p["avg_cost"] - 1) * 100 if p["avg_cost"] > 0 else 0
        stop = p.get("stop_loss_price")
        enriched.append({
            **p,
            "sector": get_sector(p["symbol"]),
            "current_price": cp,
            "pnl_pct": pnl,
            "stop_distance_pct": ((cp - stop) / cp * 100) if stop and cp > 0 else None,
        })

    return templates.TemplateResponse(request, "partials/portfolio.html", {
        "portfolio": portfolio,
        "positions": enriched,
    })


@router.get("/partials/decisions", response_class=HTMLResponse)
async def partial_decisions(request: Request):
    db = await get_db()
    decisions = await queries.get_decisions(db, limit=5)
    return templates.TemplateResponse(request, "partials/decisions.html", {
        "decisions": decisions,
    })
