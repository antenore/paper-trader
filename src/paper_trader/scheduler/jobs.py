from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from paper_trader.config import settings

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


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
    ("full",           dict(day_of_week="mon-fri", hour=10, minute=0),              "morning_scan",   "Morning Scan"),
    ("quick",          dict(day_of_week="mon-fri", hour=13, minute=0),              "midday_check",   "Midday Check"),
    ("snapshot",       dict(day_of_week="mon-fri", hour=16, minute=15),             "end_of_day",     "End of Day"),
    ("weekly_review",  dict(day_of_week="fri", hour=17, minute=0),                  "weekly_review",  "Weekly Review"),
    ("monthly_review", dict(day="1-3", day_of_week="mon-fri", hour=17, minute=30),  "monthly_review", "Monthly Review"),
]


async def start_scheduler() -> None:
    """Start the APScheduler with all trading jobs."""
    global _scheduler
    tz = settings.timezone

    _scheduler = AsyncIOScheduler(timezone=tz)

    for run_type, cron_kw, job_id, job_name in SCHEDULE:
        _scheduler.add_job(
            run_job, args=[run_type],
            trigger=CronTrigger(**cron_kw, timezone=tz),
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
