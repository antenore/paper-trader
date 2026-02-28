import asyncio
from unittest.mock import patch, AsyncMock

import pytest

from paper_trader.scheduler.jobs import (
    SCHEDULE, trigger_job, get_job_statuses,
    _running_jobs, _job_results,
)


class TestSchedule:
    def test_schedule_has_expected_entries(self):
        job_ids = [entry[2] for entry in SCHEDULE]
        assert "morning_scan" in job_ids
        assert "midday_check" in job_ids
        assert "afternoon_scan" in job_ids
        assert "end_of_day" in job_ids
        assert "weekly_review" in job_ids
        assert "monthly_review" in job_ids

    def test_afternoon_scan_is_full_type(self):
        afternoon = next(e for e in SCHEDULE if e[2] == "afternoon_scan")
        assert afternoon[0] == "full"
        assert afternoon[1]["hour"] == 14
        assert afternoon[1]["minute"] == 30

    def test_all_job_ids_unique(self):
        job_ids = [entry[2] for entry in SCHEDULE]
        assert len(job_ids) == len(set(job_ids))

    def test_schedule_count(self):
        assert len(SCHEDULE) == 6


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
        assert len(statuses) == 6
        job_ids = [s["job_id"] for s in statuses]
        assert "morning_scan" in job_ids
        assert "weekly_review" in job_ids
