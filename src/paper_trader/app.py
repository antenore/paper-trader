from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from paper_trader.config import settings
from paper_trader.db.connection import init_db, close_db
from paper_trader.db import queries

logger = logging.getLogger("paper_trader")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.info("Starting Paper Trader...")

    # Initialize database
    db = await init_db(settings.db_path)
    await queries.init_portfolio(db, settings.initial_cash_chf, settings.currency)

    # Close any dry run sessions left open from a previous crash/kill
    orphaned = await queries.close_orphaned_dry_run_sessions(db)
    if orphaned:
        logger.info("Closed %d orphaned dry run session(s) from previous run", orphaned)
        await queries.update_cash(db, settings.initial_cash_chf)
        logger.info("Restored portfolio cash to %.2f %s", settings.initial_cash_chf, settings.currency)

    logger.info("Database initialized, portfolio ready (%.2f %s)", settings.initial_cash_chf, settings.currency)

    # Load saved config overrides from DB
    from paper_trader.config import load_config_from_db
    await load_config_from_db(db)

    # Start scheduler
    try:
        from paper_trader.scheduler.jobs import start_scheduler, stop_scheduler
        await start_scheduler()
        logger.info("Scheduler started")
    except ImportError:
        logger.info("Scheduler not yet implemented, skipping")

    # Run first scan if portfolio is fresh and market is open
    snapshot_count = await queries.count_snapshots(db, is_dry_run=False)
    if snapshot_count == 0:
        from paper_trader.scheduler.jobs import is_market_open, run_job
        if is_market_open():
            logger.info("Fresh portfolio detected during market hours — running initial scan")
            asyncio.create_task(run_job("full"))
        else:
            logger.info("Fresh portfolio detected but market is closed — first scan will run at next scheduled window")

    yield

    # Shutdown
    try:
        from paper_trader.scheduler.jobs import stop_scheduler
        await stop_scheduler()
    except ImportError:
        pass
    await close_db()
    logger.info("Paper Trader stopped.")


def create_app() -> FastAPI:
    app = FastAPI(title="Paper Trader", version="0.1.0", lifespan=lifespan)

    # Health check
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # CSRF protection
    from paper_trader.dashboard.csrf import CSRFMiddleware
    app.add_middleware(CSRFMiddleware)

    # Mount dashboard routes
    try:
        from paper_trader.dashboard.routes import router as dashboard_router
        app.include_router(dashboard_router)
    except ImportError:
        pass

    # Mount static files
    static_dir = Path(__file__).parent / "dashboard" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


def main():
    app = create_app()
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
