import pytest
from unittest.mock import patch, AsyncMock

from fastapi.testclient import TestClient

from paper_trader.app import create_app
from paper_trader.db.connection import get_db, close_db, init_db
from paper_trader.db import queries


@pytest.fixture
async def client():
    """Create a test client with in-memory DB."""
    import aiosqlite
    from paper_trader.db import connection as conn_module
    from paper_trader.db.schema import create_tables

    # Set up in-memory DB
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA foreign_keys=ON")
    await create_tables(db)
    await queries.init_portfolio(db, 800.0, "CHF")

    # Patch get_db to return our test DB
    async def mock_get_db(db_path=None):
        return db

    with patch("paper_trader.dashboard.routes.get_db", side_effect=mock_get_db):
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as tc:
            yield tc, db

    await db.close()


class TestDashboardRoutes:
    @pytest.mark.asyncio
    async def test_index(self, client):
        tc, db = client
        response = tc.get("/")
        assert response.status_code == 200
        assert "Paper Trader" in response.text
        assert "Dashboard" in response.text

    @pytest.mark.asyncio
    async def test_decisions_page(self, client):
        tc, db = client
        await queries.record_decision(db, "AAPL", "BUY", 0.8, "Test", "haiku")
        response = tc.get("/decisions")
        assert response.status_code == 200
        assert "AAPL" in response.text

    @pytest.mark.asyncio
    async def test_journal_page(self, client):
        tc, db = client
        response = tc.get("/journal")
        assert response.status_code == 200
        assert "Journal" in response.text

    @pytest.mark.asyncio
    async def test_api_usage_page(self, client):
        tc, db = client
        response = tc.get("/api-usage")
        assert response.status_code == 200
        assert "API Usage" in response.text

    @pytest.mark.asyncio
    async def test_dry_run_page(self, client):
        tc, db = client
        response = tc.get("/dry-run")
        assert response.status_code == 200
        assert "Dry Run" in response.text

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        tc, db = client
        response = tc.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_resume_trading(self, client):
        tc, db = client
        await queries.set_setting(db, "api_paused", "true")
        response = tc.post("/resume-trading", follow_redirects=False)
        assert response.status_code == 303

        paused = await queries.get_setting(db, "api_paused")
        assert paused == "false"

    @pytest.mark.asyncio
    async def test_index_shows_pause_alert(self, client):
        tc, db = client
        await queries.set_setting(db, "api_paused", "true")
        await queries.set_setting(db, "pause_reason", "Budget exceeded")
        response = tc.get("/")
        assert "Trading Paused" in response.text
        assert "Budget exceeded" in response.text

    @pytest.mark.asyncio
    async def test_partials_portfolio(self, client):
        tc, db = client
        response = tc.get("/partials/portfolio")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_partials_decisions(self, client):
        tc, db = client
        response = tc.get("/partials/decisions")
        assert response.status_code == 200
