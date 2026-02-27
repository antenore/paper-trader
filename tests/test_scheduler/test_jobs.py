from paper_trader.scheduler.jobs import SCHEDULE


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
