from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from paper_trader.db.connection import get_db
from paper_trader.db import queries
from paper_trader.config import settings
from paper_trader.portfolio.tools import get_sector, get_sector_exposure

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


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
        from paper_trader.market.prices import LiveDataProvider
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

    # Sector exposure for doughnut chart
    sector_exposure = get_sector_exposure(positions, prices) if positions else {}

    # Equity curve data (reversed for chronological order)
    snapshots_chrono = list(reversed(snapshots))
    equity_dates = [s["snapshot_at"][:10] for s in snapshots_chrono]
    equity_values = [s["total_value"] for s in snapshots_chrono]
    benchmark_values = [s.get("benchmark_value") for s in snapshots_chrono]
    # Only pass benchmark if at least one non-None value exists
    has_benchmark = any(v is not None for v in benchmark_values)

    return templates.TemplateResponse(request, "index.html", {
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
        "sector_exposure": sector_exposure,
    })


@router.get("/decisions", response_class=HTMLResponse)
async def decisions_page(request: Request):
    db = await get_db()
    decisions = await queries.get_decisions(db, limit=100)
    return templates.TemplateResponse(request, "decisions.html", {
        "decisions": decisions,
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
    return templates.TemplateResponse(request, "dry_run.html", {
        "sessions": sessions,
        "default_symbols": settings.default_watchlist,
    })


@router.post("/resume-trading")
async def resume_trading():
    db = await get_db()
    await queries.set_setting(db, "api_paused", "false")
    await queries.set_setting(db, "pause_reason", "")
    logger.info("Trading resumed via dashboard")
    return RedirectResponse(url="/", status_code=303)


@router.post("/go-live")
async def go_live():
    from paper_trader.trading.dry_run import reset_for_live
    db = await get_db()
    await reset_for_live(db)
    logger.info("Switched to live mode via dashboard")
    return RedirectResponse(url="/", status_code=303)


@router.post("/start-dry-run")
async def start_dry_run_handler():
    """Start a dry run in the background."""
    import asyncio
    from paper_trader.ai.client import AIClient
    from paper_trader.trading.dry_run import run_dry_run

    db = await get_db()
    ai_client = AIClient(db)
    asyncio.create_task(run_dry_run(db, ai_client))
    return RedirectResponse(url="/dry-run", status_code=303)


# ── Settings ─────────────────────────────────────────────────────────

@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    from paper_trader.config import CONFIG_KEYS, _serialize_list
    from paper_trader.scheduler.jobs import get_current_schedule

    current = {}
    for db_key, (attr_name, type_conv) in CONFIG_KEYS.items():
        val = getattr(settings, attr_name)
        if type_conv is list:
            current[db_key] = _serialize_list(val)
        else:
            current[db_key] = val

    schedule = get_current_schedule()
    saved = request.query_params.get("saved")

    return templates.TemplateResponse(request, "settings.html", {
        "config": current,
        "schedule": schedule,
        "saved": saved,
    })


@router.post("/settings/config")
async def save_config(request: Request):
    from paper_trader.config import CONFIG_KEYS, save_config_to_db

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
    import json
    from paper_trader.scheduler.jobs import SCHEDULE, reschedule_job

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
    from paper_trader.scheduler.jobs import trigger_job
    success, message = trigger_job(job_id)
    status_code = 200 if success else 409
    return {"success": success, "message": message}


@router.get("/api/job-status")
async def job_status():
    from paper_trader.scheduler.jobs import get_job_statuses
    return get_job_statuses()


# ── HTMX Partials ────────────────────────────────────────────────────

@router.get("/partials/portfolio", response_class=HTMLResponse)
async def partial_portfolio(request: Request):
    db = await get_db()
    portfolio = await queries.get_portfolio(db)
    positions = await queries.get_open_positions(db)
    return templates.TemplateResponse(request, "partials/portfolio.html", {
        "portfolio": portfolio,
        "positions": positions,
    })


@router.get("/partials/decisions", response_class=HTMLResponse)
async def partial_decisions(request: Request):
    db = await get_db()
    decisions = await queries.get_decisions(db, limit=5)
    return templates.TemplateResponse(request, "partials/decisions.html", {
        "decisions": decisions,
    })
