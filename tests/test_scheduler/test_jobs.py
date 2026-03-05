import asyncio
from datetime import datetime
from unittest.mock import patch, AsyncMock

import pytest

from paper_trader.scheduler.jobs import (
    SCHEDULE, trigger_job, get_job_statuses, is_market_open,
    _running_jobs, _job_results,
)


class TestSchedule:
    def test_schedule_has_expected_entries(self):
        job_ids = [entry[2] for entry in SCHEDULE]
        assert "six_open_scan" in job_ids
        assert "europe_midday" in job_ids
        assert "us_open_scan" in job_ids
        assert "us_midday" in job_ids
        assert "us_afternoon" in job_ids
        assert "end_of_day" in job_ids
        assert "weekly_review" in job_ids
        assert "monthly_review" in job_ids

    def test_us_afternoon_is_full_type(self):
        us_afternoon = next(e for e in SCHEDULE if e[2] == "us_afternoon")
        assert us_afternoon[0] == "full"
        assert us_afternoon[1]["hour"] == 20
        assert us_afternoon[1]["minute"] == 0

    def test_six_open_scan_is_full_type(self):
        six_open = next(e for e in SCHEDULE if e[2] == "six_open_scan")
        assert six_open[0] == "full"
        assert six_open[1]["hour"] == 9
        assert six_open[1]["minute"] == 30

    def test_all_job_ids_unique(self):
        job_ids = [entry[2] for entry in SCHEDULE]
        assert len(job_ids) == len(set(job_ids))

    def test_schedule_count(self):
        assert len(SCHEDULE) == 9

    def test_news_fetch_job_exists(self):
        job_ids = [entry[2] for entry in SCHEDULE]
        assert "news_fetch" in job_ids


class TestIsMarketOpen:
    """Test multi-market is_market_open() covering SIX and US hours."""

    def _mock_now(self, zurich_dt=None, et_dt=None):
        """Patch datetime.now to return specific times per timezone."""
        original_now = datetime.now

        def fake_now(tz=None):
            tz_name = str(tz)
            if "Zurich" in tz_name and zurich_dt:
                return zurich_dt
            if "Eastern" in tz_name and et_dt:
                return et_dt
            return original_now(tz)

        return patch("paper_trader.scheduler.jobs.datetime") if False else \
               patch("paper_trader.scheduler.jobs.datetime", wraps=datetime)

    def test_six_open_weekday(self):
        """SIX open at 10:00 CET on Monday."""
        from zoneinfo import ZoneInfo
        zurich_tz = ZoneInfo("Europe/Zurich")
        # Monday 10:00 CET — SIX is open
        mock_zurich = datetime(2026, 3, 2, 10, 0, tzinfo=zurich_tz)

        with patch("paper_trader.scheduler.jobs.datetime") as mock_dt:
            mock_dt.now.return_value = mock_zurich
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert is_market_open() is True

    def test_weekend_closed(self):
        """Both markets closed on Saturday."""
        from zoneinfo import ZoneInfo
        zurich_tz = ZoneInfo("Europe/Zurich")
        # Saturday 10:00 CET
        mock_zurich = datetime(2026, 3, 7, 10, 0, tzinfo=zurich_tz)

        with patch("paper_trader.scheduler.jobs.datetime") as mock_dt:
            mock_dt.now.return_value = mock_zurich
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert is_market_open() is False

    def test_us_open_evening_cet(self):
        """US open at 20:00 CET (14:00 ET) — SIX closed but US open."""
        from zoneinfo import ZoneInfo
        zurich_tz = ZoneInfo("Europe/Zurich")
        et_tz = ZoneInfo("US/Eastern")
        # Monday 20:00 CET = 14:00 ET
        mock_zurich = datetime(2026, 3, 2, 20, 0, tzinfo=zurich_tz)
        mock_et = datetime(2026, 3, 2, 14, 0, tzinfo=et_tz)

        call_count = [0]

        def fake_now(tz=None):
            call_count[0] += 1
            tz_name = str(tz) if tz else ""
            if "Eastern" in tz_name:
                return mock_et
            return mock_zurich

        with patch("paper_trader.scheduler.jobs.datetime") as mock_dt:
            mock_dt.now.side_effect = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert is_market_open() is True

    def test_all_closed_late_night(self):
        """Both markets closed at 23:00 CET."""
        from zoneinfo import ZoneInfo
        zurich_tz = ZoneInfo("Europe/Zurich")
        et_tz = ZoneInfo("US/Eastern")
        # Monday 23:00 CET = 17:00 ET — both closed
        mock_zurich = datetime(2026, 3, 2, 23, 0, tzinfo=zurich_tz)
        mock_et = datetime(2026, 3, 2, 17, 0, tzinfo=et_tz)

        def fake_now(tz=None):
            tz_name = str(tz) if tz else ""
            if "Eastern" in tz_name:
                return mock_et
            return mock_zurich

        with patch("paper_trader.scheduler.jobs.datetime") as mock_dt:
            mock_dt.now.side_effect = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert is_market_open() is False


class TestManualTrigger:
    def setup_method(self):
        _running_jobs.clear()
        _job_results.clear()

    def test_trigger_unknown_job(self):
        success, msg = trigger_job("nonexistent_job")
        assert success is False
        assert "Unknown" in msg

    @pytest.mark.asyncio
    async def test_trigger_valid_job(self):
        with patch("paper_trader.scheduler.jobs.run_job", new_callable=AsyncMock):
            success, msg = trigger_job("weekly_review")
            assert success is True
            assert "started" in msg
            # Let the task complete
            await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_double_trigger_rejected(self):
        async def slow_job(run_type):
            await asyncio.sleep(10)

        with patch("paper_trader.scheduler.jobs.run_job", side_effect=slow_job):
            s1, _ = trigger_job("weekly_review")
            assert s1 is True
            s2, msg2 = trigger_job("weekly_review")
            assert s2 is False
            assert "already running" in msg2

        # Cancel the lingering task
        if "weekly_review" in _running_jobs:
            _running_jobs["weekly_review"].cancel()

    def test_get_job_statuses_returns_all(self):
        statuses = get_job_statuses()
        assert len(statuses) == 9
        job_ids = [s["job_id"] for s in statuses]
        assert "six_open_scan" in job_ids
        assert "weekly_review" in job_ids
        assert "news_fetch" in job_ids
