import pytest

from paper_trader.analytics import compute_metrics, format_metrics_for_ai, _load_daily_series


async def _seed_snapshots(db, values, spy_prices=None, fx_rates=None):
    """Insert fake daily snapshots for testing."""
    for i, val in enumerate(values):
        spy = (spy_prices or [100] * len(values))[i]
        fx = (fx_rates or [0.80] * len(values))[i]
        day = f"2026-03-{5 + i:02d}T22:15:00Z"
        await db.execute(
            """INSERT INTO portfolio_snapshots
               (cash, positions_value, total_value, is_dry_run, snapshot_at, spy_price, benchmark_value, usd_chf_rate)
               VALUES (?, ?, ?, 0, ?, ?, ?, ?)""",
            (300, val - 300, val, day, spy, 0, fx),
        )
    await db.commit()


class TestLoadDailySeries:
    @pytest.mark.asyncio
    async def test_empty_db(self, db):
        portfolio, benchmark = await _load_daily_series(db)
        assert portfolio.empty
        assert benchmark.empty

    @pytest.mark.asyncio
    async def test_returns_calculated(self, db):
        await _seed_snapshots(db, [1000, 990, 1010, 1005])
        portfolio, benchmark = await _load_daily_series(db)
        assert len(portfolio) == 3  # 4 values -> 3 returns
        assert portfolio.iloc[0] == pytest.approx(-0.01)  # 1000 -> 990

    @pytest.mark.asyncio
    async def test_benchmark_in_chf(self, db):
        """Benchmark should be SPY * USD/CHF rate."""
        await _seed_snapshots(
            db,
            [1000, 990, 980],
            spy_prices=[100, 102, 101],
            fx_rates=[0.80, 0.80, 0.80],
        )
        _, benchmark = await _load_daily_series(db)
        assert len(benchmark) == 2
        # First return: (102*0.80)/(100*0.80) - 1 = 0.02
        assert benchmark.iloc[0] == pytest.approx(0.02)


class TestComputeMetrics:
    @pytest.mark.asyncio
    async def test_not_enough_data(self, db):
        await _seed_snapshots(db, [1000])
        metrics = await compute_metrics(db)
        assert "error" in metrics

    @pytest.mark.asyncio
    async def test_basic_metrics(self, db):
        await _seed_snapshots(db, [1000, 990, 980, 970, 975, 985])
        metrics = await compute_metrics(db)
        assert "error" not in metrics
        assert metrics["total_return"] < 0
        assert metrics["max_drawdown"] < 0
        assert "sharpe" in metrics
        assert "sortino" in metrics
        assert "volatility" in metrics
        assert "win_rate" in metrics
        assert metrics["trading_days"] == 5

    @pytest.mark.asyncio
    async def test_benchmark_comparison(self, db):
        await _seed_snapshots(
            db,
            [1000, 990, 980, 975],
            spy_prices=[100, 101, 102, 103],
            fx_rates=[0.80, 0.80, 0.80, 0.80],
        )
        metrics = await compute_metrics(db)
        assert "benchmark_return" in metrics
        assert metrics["benchmark_return"] > 0  # SPY went up
        assert metrics["total_return"] < 0  # Portfolio went down

    @pytest.mark.asyncio
    async def test_all_positive_returns(self, db):
        await _seed_snapshots(db, [1000, 1010, 1025, 1040])
        metrics = await compute_metrics(db)
        assert metrics["total_return"] > 0
        assert metrics["sharpe"] > 0
        assert metrics["win_rate"] == 1.0


class TestFormatMetricsForAI:
    def test_empty_on_error(self):
        assert format_metrics_for_ai({"error": "not enough data"}) == ""

    def test_contains_key_metrics(self):
        metrics = {
            "total_return": -0.05,
            "sharpe": -6.7,
            "sortino": -6.9,
            "max_drawdown": -0.06,
            "volatility": 0.16,
            "win_rate": 0.31,
            "trading_days": 13,
            "benchmark_return": -0.03,
        }
        text = format_metrics_for_ai(metrics)
        assert "PORTFOLIO PERFORMANCE METRICS" in text
        assert "Sharpe ratio: -6.70" in text
        assert "Win rate: 31%" in text
        assert "Alpha: -2.00%" in text
        assert "WARNING" in text  # Sharpe < -2

    def test_no_warning_on_good_sharpe(self):
        metrics = {
            "total_return": 0.10,
            "sharpe": 1.5,
            "sortino": 2.0,
            "max_drawdown": -0.02,
            "volatility": 0.10,
            "win_rate": 0.65,
            "trading_days": 30,
        }
        text = format_metrics_for_ai(metrics)
        assert "WARNING" not in text
        assert "Sharpe ratio: 1.50" in text
