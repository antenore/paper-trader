from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from paper_trader.config import settings

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None
_running_jobs: dict[str, asyncio.Task] = {}
_job_results: dict[str, dict] = {}


def is_market_open() -> bool:
    """Check if US market is currently open (Mon-Fri 9:30-16:00 ET)."""
    now = datetime.now(ZoneInfo(settings.timezone))
    if now.weekday() >= 5:  # Weekend
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


async def run_job(run_type: str) -> None:
    """Generic job runner for all scheduled tasks."""
    from paper_trader.ai.client import AIClient
    from paper_trader.db.connection import get_db
    from paper_trader.market.prices import LiveDataProvider

    logger.info("Running %s", run_type)
    try:
        db = await get_db()
        ai_client = AIClient(db)
        market = LiveDataProvider()

        if run_type == "weekly_review":
            from paper_trader.ai.strategist import run_weekly_review
            result = await run_weekly_review(db, ai_client, market)
            logger.info("Weekly review complete: %s", result.performance_summary[:100])
        elif run_type == "monthly_review":
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
SCHEDULE = [
    ("full",           dict(day_of_week="mon-fri", hour=10, minute=0),              "morning_scan",    "Morning Scan"),
    ("quick",          dict(day_of_week="mon-fri", hour=13, minute=0),              "midday_check",    "Midday Check"),
    ("full",           dict(day_of_week="mon-fri", hour=14, minute=30),             "afternoon_scan",  "Afternoon Scan"),
    ("snapshot",       dict(day_of_week="mon-fri", hour=16, minute=15),             "end_of_day",      "End of Day"),
    ("weekly_review",  dict(day_of_week="fri", hour=17, minute=0),                  "weekly_review",   "Weekly Review"),
    ("monthly_review", dict(day="1-3", day_of_week="mon-fri", hour=17, minute=30),  "monthly_review",  "Monthly Review"),
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
