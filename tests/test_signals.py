"""Tests for the signal processing layer."""

import pytest
from datetime import datetime, timedelta, timezone

from paper_trader.signals import (
    ConvictionCurve,
    SentimentCurve,
    CommissionTrajectory,
    score_headline_sentiment,
    compute_conviction_curves,
    compute_news_sentiment,
    compute_commission_trajectory,
    compute_signal_quality,
    format_signal_report,
)
from paper_trader.db import queries


class TestScoreHeadlineSentiment:
    def test_bullish_headline(self):
        score = score_headline_sentiment("NVDA beats earnings expectations with record revenue growth")
        assert score > 0

    def test_bearish_headline(self):
        score = score_headline_sentiment("Stock plunges after disappointing earnings miss and layoff announcement")
        assert score < 0

    def test_neutral_headline(self):
        score = score_headline_sentiment("Company announces new office location")
        assert score == pytest.approx(0.0, abs=0.01)

    def test_mixed_headline(self):
        score = score_headline_sentiment("Revenue beats expectations but guidance lowered amid slowdown concerns")
        # Mixed signal — should be close to 0
        assert -0.5 <= score <= 0.5

    def test_empty_headline(self):
        score = score_headline_sentiment("")
        assert score == 0.0


class TestConvictionCurves:
    @pytest.mark.asyncio
    async def test_rising_conviction(self, db):
        """Steadily increasing confidence → RISING direction."""
        now = datetime.now(timezone.utc)
        for i in range(5):
            day = now - timedelta(days=5 - i)
            conf = 0.40 + i * 0.08  # 0.40, 0.48, 0.56, 0.64, 0.72
            await db.execute(
                "INSERT INTO decisions (symbol, action, confidence, reasoning, model, is_dry_run, created_at) VALUES (?, ?, ?, ?, ?, 0, ?)",
                ("NVDA", "HOLD", conf, "test", "test", day.strftime("%Y-%m-%d 12:00:00")),
            )
        await db.commit()

        result = await compute_conviction_curves(db, ["NVDA"])
        assert "NVDA" in result
        assert result["NVDA"].direction == "RISING"
        assert result["NVDA"].slope > 0.02

    @pytest.mark.asyncio
    async def test_oscillating_conviction(self, db):
        """High variance oscillation → OSCILLATING direction."""
        now = datetime.now(timezone.utc)
        confidences = [0.30, 0.80, 0.25, 0.75, 0.30, 0.80, 0.25]
        for i, conf in enumerate(confidences):
            day = now - timedelta(days=7 - i)
            await db.execute(
                "INSERT INTO decisions (symbol, action, confidence, reasoning, model, is_dry_run, created_at) VALUES (?, ?, ?, ?, ?, 0, ?)",
                ("RTX", "HOLD", conf, "test", "test", day.strftime("%Y-%m-%d 12:00:00")),
            )
        await db.commit()

        result = await compute_conviction_curves(db, ["RTX"])
        assert "RTX" in result
        assert result["RTX"].direction == "OSCILLATING"
        assert result["RTX"].std > 0.20

    @pytest.mark.asyncio
    async def test_insufficient_data(self, db):
        """Less than 3 data points → INSUFFICIENT."""
        now = datetime.now(timezone.utc)
        await db.execute(
            "INSERT INTO decisions (symbol, action, confidence, reasoning, model, is_dry_run, created_at) VALUES (?, ?, ?, ?, ?, 0, ?)",
            ("MU", "HOLD", 0.50, "test", "test", now.strftime("%Y-%m-%d 12:00:00")),
        )
        await db.commit()

        result = await compute_conviction_curves(db, ["MU"])
        assert "MU" in result
        assert result["MU"].direction == "INSUFFICIENT"
        assert result["MU"].data_points_count < 3

    @pytest.mark.asyncio
    async def test_empty_symbols(self, db):
        result = await compute_conviction_curves(db, [])
        assert result == {}


class TestCommissionTrajectory:
    @pytest.mark.asyncio
    async def test_annualized_calculation(self, db):
        """Verify annualized commission as % of capital."""
        now = datetime.now(timezone.utc)
        # Insert trades on 3 different days
        for i in range(3):
            day = now - timedelta(days=i + 1)
            await db.execute(
                "INSERT INTO trades (symbol, action, shares, price, total, commission_chf, is_dry_run, executed_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
                ("AAPL", "BUY", 1.0, 150.0, 150.0, 2.0, day.strftime("%Y-%m-%d 10:00:00")),
            )
        await db.commit()

        result = await compute_commission_trajectory(db, days=14, capital=1000.0)
        assert result.total == 6.0
        assert result.active_days == 3
        assert result.daily_avg == 2.0
        # Annualized: 2.0 * 252 = 504, 504/1000 = 50.4%
        assert result.annualized_pct == pytest.approx(50.4, abs=0.1)

    @pytest.mark.asyncio
    async def test_no_trades(self, db):
        result = await compute_commission_trajectory(db, days=14)
        assert result.total == 0.0
        assert result.active_days == 0
        assert result.daily_avg == 0.0
        assert result.annualized_pct == 0.0


class TestSignalQuality:
    def test_high_quality(self):
        """All 3 positives → HIGH."""
        conviction = {"NVDA": ConvictionCurve(
            slope=0.04, avg=0.65, std=0.08, direction="RISING",
            earliest=0.42, latest=0.68, data_points_count=7,
        )}
        sentiment = {"NVDA": SentimentCurve(avg_score=0.45, trend_direction="improving", article_count=12)}
        rs_data = [{"symbol": "NVDA", "rs_ratio": 1.2}]

        result = compute_signal_quality(conviction, sentiment, rs_data)
        assert result["NVDA"] == "HIGH"

    def test_noise_from_oscillating(self):
        """OSCILLATING conviction → NOISE regardless of other signals."""
        conviction = {"RTX": ConvictionCurve(
            slope=0.01, avg=0.50, std=0.25, direction="OSCILLATING",
            earliest=0.25, latest=0.70, data_points_count=7,
        )}
        sentiment = {"RTX": SentimentCurve(avg_score=0.50, trend_direction="improving", article_count=10)}
        rs_data = [{"symbol": "RTX", "rs_ratio": 1.5}]

        result = compute_signal_quality(conviction, sentiment, rs_data)
        assert result["RTX"] == "NOISE"

    def test_noise_from_insufficient(self):
        """INSUFFICIENT conviction → NOISE."""
        conviction = {"MU": ConvictionCurve(
            slope=0.0, avg=0.0, std=0.0, direction="INSUFFICIENT",
            earliest=0.0, latest=0.0, data_points_count=1,
        )}
        result = compute_signal_quality(conviction, {}, None)
        assert result["MU"] == "NOISE"

    def test_medium_quality(self):
        """2 of 3 positives → MEDIUM."""
        conviction = {"AAPL": ConvictionCurve(
            slope=0.03, avg=0.60, std=0.10, direction="RISING",
            earliest=0.50, latest=0.70, data_points_count=5,
        )}
        sentiment = {"AAPL": SentimentCurve(avg_score=0.30, trend_direction="improving", article_count=8)}
        # No RS data → rs defaults to 0

        result = compute_signal_quality(conviction, sentiment, None)
        assert result["AAPL"] == "MEDIUM"

    def test_low_quality(self):
        """1 of 3 positives → LOW."""
        conviction = {"TSLA": ConvictionCurve(
            slope=-0.01, avg=0.45, std=0.10, direction="FLAT",
            earliest=0.45, latest=0.44, data_points_count=5,
        )}
        sentiment = {"TSLA": SentimentCurve(avg_score=0.20, trend_direction="stable", article_count=5)}

        result = compute_signal_quality(conviction, sentiment, None)
        assert result["TSLA"] == "LOW"


class TestFormatSignalReport:
    def test_format_includes_all_sections(self):
        conviction = {"NVDA": ConvictionCurve(
            slope=0.04, avg=0.65, std=0.08, direction="RISING",
            earliest=0.42, latest=0.68, data_points_count=7,
        )}
        sentiment = {"NVDA": SentimentCurve(avg_score=0.45, trend_direction="improving", article_count=12)}
        commission = CommissionTrajectory(total=19.50, daily_avg=2.17, annualized_pct=54.7, active_days=9, total_days=14)
        quality = {"NVDA": "HIGH"}

        report = format_signal_report(conviction, sentiment, commission, quality)

        assert "SIGNAL QUALITY" in report
        assert "CONVICTION CURVES" in report
        assert "NEWS SENTIMENT" in report
        assert "COMMISSION TRAJECTORY" in report
        assert "NVDA" in report
        assert "HIGH" in report
        assert "RISING" in report
        assert "54.7%" in report

    def test_high_commission_warning(self):
        commission = CommissionTrajectory(total=30.0, daily_avg=3.0, annualized_pct=75.6, active_days=10, total_days=14)
        report = format_signal_report({}, {}, commission, {})
        assert "REDUCE TRADING FREQUENCY" in report

    def test_low_commission_no_warning(self):
        commission = CommissionTrajectory(total=2.0, daily_avg=0.5, annualized_pct=12.6, active_days=4, total_days=14)
        report = format_signal_report({}, {}, commission, {})
        assert "REDUCE TRADING FREQUENCY" not in report
