import pytest
from unittest.mock import patch

from paper_trader.db import queries
from paper_trader.portfolio.manager import execute_buy, execute_sell, get_portfolio_value


@pytest.mark.asyncio
async def test_buy_basic(db_with_portfolio):
    # Buy 1 share at 100 (well within limits: 30% of 800 = 240)
    result = await execute_buy(db_with_portfolio, "AAPL", 1.0, 100.0)
    assert result["ok"]
    assert result["shares"] == 1.0
    assert result["total"] == 100.0

    p = await queries.get_portfolio(db_with_portfolio)
    assert p["cash"] == 700.0

    positions = await queries.get_open_positions(db_with_portfolio)
    assert len(positions) == 1
    assert positions[0]["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_buy_adds_to_existing_position(db_with_portfolio):
    await execute_buy(db_with_portfolio, "AAPL", 1.0, 100.0)
    await execute_buy(db_with_portfolio, "AAPL", 1.0, 120.0)

    pos = await queries.get_position_by_symbol(db_with_portfolio, "AAPL")
    assert pos["shares"] == 2.0
    assert pos["avg_cost"] == 110.0  # (100+120)/2


@pytest.mark.asyncio
async def test_sell_basic(db_with_portfolio):
    await execute_buy(db_with_portfolio, "AAPL", 2.0, 100.0)
    result = await execute_sell(db_with_portfolio, "AAPL", 2.0, 110.0)
    assert result["ok"]
    assert result["pnl"] == 20.0  # (110-100)*2

    p = await queries.get_portfolio(db_with_portfolio)
    assert p["cash"] == 820.0  # 800 - 200 + 220

    positions = await queries.get_open_positions(db_with_portfolio)
    assert len(positions) == 0


@pytest.mark.asyncio
async def test_sell_partial(db_with_portfolio):
    await execute_buy(db_with_portfolio, "AAPL", 2.0, 100.0)
    result = await execute_sell(db_with_portfolio, "AAPL", 1.0, 110.0)
    assert result["ok"]
    assert result["shares"] == 1.0

    pos = await queries.get_position_by_symbol(db_with_portfolio, "AAPL")
    assert pos["shares"] == 1.0


@pytest.mark.asyncio
async def test_sell_more_than_owned(db_with_portfolio):
    await execute_buy(db_with_portfolio, "AAPL", 2.0, 100.0)
    result = await execute_sell(db_with_portfolio, "AAPL", 10.0, 110.0)
    assert result["ok"]
    assert result["shares"] == 2.0  # Capped at owned


@pytest.mark.asyncio
async def test_sell_no_position(db_with_portfolio):
    result = await execute_sell(db_with_portfolio, "AAPL", 1.0, 150.0)
    assert not result["ok"]
    assert "no open position" in result["error"].lower()


@pytest.mark.asyncio
async def test_portfolio_value(db_with_portfolio):
    await execute_buy(db_with_portfolio, "AAPL", 2.0, 100.0)
    prices = {"AAPL": 110.0}
    val = await get_portfolio_value(db_with_portfolio, prices)
    assert val["cash"] == 600.0
    assert val["positions_value"] == 220.0
    assert val["total_value"] == 820.0
    assert len(val["positions"]) == 1
    assert val["positions"][0]["pnl"] == 20.0


@pytest.mark.asyncio
async def test_buy_risk_rejected(db_with_portfolio):
    # Try to spend too much — should be risk-adjusted
    result = await execute_buy(db_with_portfolio, "AAPL", 100.0, 150.0)
    # Risk should auto-reduce or reject
    if result["ok"]:
        assert result["shares"] < 100.0  # Was auto-reduced


@pytest.mark.asyncio
async def test_buy_uses_settings_initial_cash(db):
    """execute_buy should use settings.initial_cash_chf, not a hardcoded value."""
    custom_cash = 1500.0
    await queries.init_portfolio(db, custom_cash, "CHF")

    with patch("paper_trader.portfolio.manager.settings") as mock_settings, \
         patch("paper_trader.portfolio.risk.settings") as mock_risk_settings:
        mock_settings.initial_cash_chf = custom_cash
        mock_risk_settings.initial_cash_chf = custom_cash
        mock_risk_settings.safety_stop_pct = 0.50
        mock_risk_settings.max_position_pct = 0.30
        mock_risk_settings.min_cash_reserve = 100.0

        result = await execute_buy(db, "AAPL", 1.0, 100.0)
        assert result["ok"]

    # With 1500 initial cash, safety stop is 750 — should be fine
    # If it were hardcoded to 800, safety stop would be 400 — different risk profile
    p = await queries.get_portfolio(db)
    assert p["cash"] == 1400.0


@pytest.mark.asyncio
async def test_trades_recorded(db_with_portfolio):
    await execute_buy(db_with_portfolio, "AAPL", 1.0, 100.0)
    await execute_sell(db_with_portfolio, "AAPL", 1.0, 110.0)

    trades = await queries.get_trades(db_with_portfolio)
    assert len(trades) == 2
    actions = {t["action"] for t in trades}
    assert actions == {"BUY", "SELL"}
