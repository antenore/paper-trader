from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from paper_trader.config import settings  # noqa: E402 — used in SCHEDULE and run_job

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None
_running_jobs: dict[str, asyncio.Task] = {}
_job_results: dict[str, dict] = {}


def is_market_open() -> bool:
    """Check if any tracked market is currently open (Mon-Fri).

    Covers both:
      - Swiss SIX: 09:00-17:30 Europe/Zurich
      - US (NYSE/NASDAQ): 09:30-16:00 US/Eastern
    """
    now_zurich = datetime.now(ZoneInfo("Europe/Zurich"))
    if now_zurich.weekday() >= 5:  # Weekend
        return False

    # SIX: 09:00-17:30 CET
    six_open = now_zurich.replace(hour=9, minute=0, second=0, microsecond=0)
    six_close = now_zurich.replace(hour=17, minute=30, second=0, microsecond=0)
    if six_open <= now_zurich <= six_close:
        return True

    # US: 09:30-16:00 ET
    now_et = datetime.now(ZoneInfo("US/Eastern"))
    us_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    us_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return us_open <= now_et <= us_close


async def run_job(run_type: str) -> None:
    """Generic job runner for all scheduled tasks."""
    from paper_trader.db.connection import get_db

    logger.info("Running %s", run_type)
    try:
        db = await get_db()

        # News fetch is independent — no AI client or market data needed
        if run_type == "news_fetch":
            from paper_trader.db import queries as q
            from paper_trader.market.news import fetch_and_save_news
            new_count = await fetch_and_save_news(db)
            cleaned = await q.cleanup_old_news(db, days=settings.news_cleanup_days)
            logger.info("News fetch: %d new items, %d old items cleaned up", new_count, cleaned)
            return

        from paper_trader.ai.client import AIClient
        from paper_trader.market.prices import LiveDataProvider

        ai_client = AIClient(db)
        market = LiveDataProvider()

        if run_type == "weekly_review":
            from paper_trader.ai.strategist import run_weekly_review
            result = await run_weekly_review(db, ai_client, market)
            logger.info("Weekly review complete: %s", result.performance_summary[:100])
        elif run_type == "monthly_review":
            from paper_trader.db import queries as q
            if await q.has_monthly_review_this_month(db):
                logger.info("Monthly review already ran this month, skipping duplicate")
                return
            from paper_trader.ai.strategist import run_monthly_review
            result = await run_monthly_review(db, ai_client, market)
            logger.info("Monthly review complete: %s", result.strategic_summary[:100])
        else:
            from paper_trader.trading.pipeline import run_trading_pipeline
            result = await run_trading_pipeline(db, ai_client, market, run_type=run_type)
            logger.info("%s complete: %s", run_type, result)
    except Exception as e:
        logger.error("%s failed: %s", run_type, e, exc_info=True)


# Schedule configuration: (run_type, cron_kwargs, job_id, job_name)
# All times in Europe/Zurich (CET/CEST)
# SIX: 09:00-17:30 CET | US: 15:30-22:00 CET | Overlap: 15:30-17:30
SCHEDULE = [
    # News: runs 24/7 every N minutes — captures weekend geopolitical events
    ("news_fetch",     dict(minute=f"*/{settings.news_fetch_interval_minutes}"),    "news_fetch",      "News Fetch"),
    # SIX session
    ("full",           dict(day_of_week="mon-fri", hour=9, minute=30),              "six_open_scan",   "SIX Open Scan"),
    ("quick",          dict(day_of_week="mon-fri", hour=12, minute=0),              "europe_midday",   "European Midday"),
    # US session (overlap with SIX tail)
    ("full",           dict(day_of_week="mon-fri", hour=15, minute=45),             "us_open_scan",    "US Open Scan"),
    ("quick",          dict(day_of_week="mon-fri", hour=18, minute=0),              "us_midday",       "US Midday"),
    ("full",           dict(day_of_week="mon-fri", hour=20, minute=0),              "us_afternoon",    "US Afternoon"),
    # End of day — after US close
    ("snapshot",       dict(day_of_week="mon-fri", hour=22, minute=15),             "end_of_day",      "End of Day"),
    # Reviews
    ("weekly_review",  dict(day_of_week="fri", hour=22, minute=30),                 "weekly_review",   "Weekly Review"),
    ("monthly_review", dict(day="1-3", day_of_week="mon-fri", hour=22, minute=45),  "monthly_review",  "Monthly Review"),
]


_VALID_JOB_IDS = {entry[2] for entry in SCHEDULE}


def trigger_job(job_id: str) -> tuple[bool, str]:
    """Trigger a manual run of a scheduled job. Returns (success, message)."""
    if job_id not in _VALID_JOB_IDS:
        return False, f"Unknown job: {job_id}"

    if job_id in _running_jobs and not _running_jobs[job_id].done():
        return False, f"Job {job_id} is already running"

    run_type = next(entry[0] for entry in SCHEDULE if entry[2] == job_id)

    async def _tracked_run():
        _job_results[job_id] = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "error": None,
        }
        try:
            await run_job(run_type)
            _job_results[job_id]["status"] = "completed"
        except Exception as e:
            _job_results[job_id]["status"] = "failed"
            _job_results[job_id]["error"] = str(e)
        finally:
            _job_results[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()

    task = asyncio.ensure_future(_tracked_run())
    _running_jobs[job_id] = task
    return True, f"Job {job_id} started"


def get_job_statuses() -> list[dict]:
    """Return status of all scheduled jobs for the UI."""
    statuses = []
    for run_type, cron_kw, job_id, job_name in SCHEDULE:
        is_running = job_id in _running_jobs and not _running_jobs[job_id].done()
        last = _job_results.get(job_id)
        statuses.append({
            "job_id": job_id,
            "name": job_name,
            "run_type": run_type,
            "is_running": is_running,
            "last_status": last["status"] if last else None,
            "last_started": last["started_at"] if last else None,
            "last_finished": last["finished_at"] if last else None,
            "last_error": last["error"] if last else None,
        })
    return statuses


def reschedule_job(job_id: str, cron_kw: dict) -> bool:
    """Reschedule an existing job with new cron parameters."""
    if _scheduler is None or job_id not in _VALID_JOB_IDS:
        return False
    try:
        _scheduler.reschedule_job(
            job_id, trigger=CronTrigger(**cron_kw, timezone=settings.timezone),
        )
        return True
    except Exception:
        logger.exception("Failed to reschedule %s", job_id)
        return False


def get_current_schedule() -> list[dict]:
    """Read live schedule from APScheduler."""
    if _scheduler is None:
        # Fall back to static SCHEDULE
        result = []
        for run_type, cron_kw, job_id, job_name in SCHEDULE:
            result.append({"job_id": job_id, "name": job_name, **cron_kw})
        return result

    result = []
    for run_type, _, job_id, job_name in SCHEDULE:
        job = _scheduler.get_job(job_id)
        if job is None:
            continue
        trigger = job.trigger
        fields = {}
        for field in trigger.fields:
            fields[field.name] = str(field)
        result.append({"job_id": job_id, "name": job_name, **fields})
    return result


async def start_scheduler() -> None:
    """Start the APScheduler with all trading jobs."""
    import json
    from paper_trader.db.connection import get_db
    from paper_trader.db import queries as q

    global _scheduler
    tz = settings.timezone

    _scheduler = AsyncIOScheduler(timezone=tz)

    # Load saved schedule overrides from DB
    db = await get_db()
    overrides: dict[str, dict] = {}
    for _, _, job_id, _ in SCHEDULE:
        saved = await q.get_setting(db, f"schedule.{job_id}")
        if saved:
            try:
                overrides[job_id] = json.loads(saved)
            except json.JSONDecodeError:
                pass

    for run_type, cron_kw, job_id, job_name in SCHEDULE:
        effective_cron = overrides.get(job_id, cron_kw)
        _scheduler.add_job(
            run_job, args=[run_type],
            trigger=CronTrigger(**effective_cron, timezone=tz),
            id=job_id, name=job_name,
        )

    _scheduler.start()
    logger.info("Scheduler started with %d jobs", len(_scheduler.get_jobs()))


async def stop_scheduler() -> None:
    """Stop the scheduler."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduler stopped")
