import pytest
from unittest.mock import patch

from paper_trader.db import queries
from paper_trader.portfolio.commission import apply_slippage, calculate_commission
from paper_trader.portfolio.manager import execute_buy, execute_sell, get_portfolio_value


@pytest.mark.asyncio
async def test_buy_basic(db_with_portfolio):
    # Buy 1 share at 100 (well within limits: 25% of 1000 = 250)
    result = await execute_buy(db_with_portfolio, "AAPL", 1.0, 100.0)
    assert result["ok"]
    assert result["shares"] == 1.0
    # Price should be slippage-adjusted (slightly higher than 100)
    assert result["price"] > 100.0
    assert result["commission"] > 0

    cost = result["total"]
    commission = result["commission"]
    p = await queries.get_portfolio(db_with_portfolio)
    assert p["cash"] == pytest.approx(1000.0 - cost - commission, abs=0.01)

    positions = await queries.get_open_positions(db_with_portfolio)
    assert len(positions) == 1
    assert positions[0]["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_buy_adds_to_existing_position(db_with_portfolio):
    # Use small amounts to stay within growth tier's 25% position cap ($250 of $1000)
    r1 = await execute_buy(db_with_portfolio, "AAPL", 1.0, 50.0)
    r2 = await execute_buy(db_with_portfolio, "AAPL", 1.0, 60.0)

    pos = await queries.get_position_by_symbol(db_with_portfolio, "AAPL")
    assert pos["shares"] == 2.0
    # avg_cost is average of slippage-adjusted prices
    expected_avg = (r1["price"] + r2["price"]) / 2
    assert pos["avg_cost"] == pytest.approx(expected_avg, abs=0.01)


@pytest.mark.asyncio
async def test_sell_basic(db_with_portfolio):
    buy_result = await execute_buy(db_with_portfolio, "AAPL", 2.0, 100.0)
    sell_result = await execute_sell(db_with_portfolio, "AAPL", 2.0, 110.0)
    assert sell_result["ok"]
    # P&L uses slippage-adjusted prices
    expected_pnl = (sell_result["price"] - buy_result["price"]) * 2
    assert sell_result["pnl"] == pytest.approx(expected_pnl, abs=0.01)
    assert sell_result["commission"] > 0

    p = await queries.get_portfolio(db_with_portfolio)
    # cash = 1000 - buy_cost - buy_comm + sell_proceeds - sell_comm
    expected = 1000.0 - buy_result["total"] - buy_result["commission"] + sell_result["total"] - sell_result["commission"]
    assert p["cash"] == pytest.approx(expected, abs=0.01)

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
    buy_result = await execute_buy(db_with_portfolio, "AAPL", 2.0, 100.0)
    prices = {"AAPL": 110.0}
    val = await get_portfolio_value(db_with_portfolio, prices)
    expected_cash = 1000.0 - buy_result["total"] - buy_result["commission"]
    assert val["cash"] == pytest.approx(expected_cash, abs=0.01)
    assert val["positions_value"] == 220.0
    assert val["total_value"] == pytest.approx(expected_cash + 220.0, abs=0.01)
    assert len(val["positions"]) == 1
    # P&L = (110 - slipped_buy_price) * 2
    assert val["positions"][0]["pnl"] == pytest.approx((110.0 - buy_result["price"]) * 2, abs=0.01)


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
        mock_settings.stop_loss_pct = 0.07
        mock_risk_settings.initial_cash_chf = custom_cash
        mock_risk_settings.safety_stop_pct = 0.50
        mock_risk_settings.max_position_pct = 0.30
        mock_risk_settings.min_cash_reserve = 100.0
        mock_risk_settings.sector_cap_pct = 0.60

        result = await execute_buy(db, "AAPL", 1.0, 100.0)
        assert result["ok"]

    p = await queries.get_portfolio(db)
    assert p["cash"] == pytest.approx(1500.0 - result["total"] - result["commission"], abs=0.01)


@pytest.mark.asyncio
async def test_buy_usd_with_fx(db_with_portfolio):
    """BUY a USD stock with FX conversion — cost should be slipped_price * shares * rate."""
    result = await execute_buy(db_with_portfolio, "AAPL", 1.0, 100.0, usd_chf_rate=0.88)
    assert result["ok"]
    # Cost in CHF = slipped_price * 0.88 ≈ 100.05 * 0.88 ≈ 88.04
    assert result["total"] == pytest.approx(result["price"] * 0.88, abs=0.01)

    p = await queries.get_portfolio(db_with_portfolio)
    assert p["cash"] == pytest.approx(1000.0 - result["total"] - result["commission"], abs=0.01)


@pytest.mark.asyncio
async def test_buy_chf_stock(db_with_portfolio):
    """BUY a Swiss .SW stock — no FX conversion, cost equals slipped_price * shares."""
    result = await execute_buy(db_with_portfolio, "NESN.SW", 1.0, 100.0, usd_chf_rate=0.88)
    assert result["ok"]
    # CHF stock: cost = slipped_price (no FX), slippage = 10 bps → ~100.10
    assert result["total"] == pytest.approx(result["price"], abs=0.01)
    assert result["price"] > 100.0  # SIX slippage applied

    p = await queries.get_portfolio(db_with_portfolio)
    assert p["cash"] == pytest.approx(1000.0 - result["total"] - result["commission"], abs=0.01)

    # Check currency stored in position
    pos = await queries.get_position_by_symbol(db_with_portfolio, "NESN.SW")
    assert pos["currency"] == "CHF"


@pytest.mark.asyncio
async def test_sell_usd_with_fx(db_with_portfolio):
    """SELL a USD stock — proceeds should be FX-converted to CHF with slippage."""
    await execute_buy(db_with_portfolio, "AAPL", 2.0, 100.0, usd_chf_rate=0.88)
    result = await execute_sell(db_with_portfolio, "AAPL", 2.0, 110.0, usd_chf_rate=0.88)
    assert result["ok"]
    # Gross proceeds: 2 * slipped_sell_price * 0.88
    expected_proceeds = 2 * result["price"] * 0.88
    assert result["total"] == pytest.approx(expected_proceeds, abs=0.01)


@pytest.mark.asyncio
async def test_portfolio_value_with_fx(db_with_portfolio):
    """Portfolio value should convert USD positions to CHF."""
    buy_result = await execute_buy(db_with_portfolio, "AAPL", 2.0, 100.0, usd_chf_rate=0.88)
    prices = {"AAPL": 110.0}
    val = await get_portfolio_value(db_with_portfolio, prices, usd_chf_rate=0.88)
    # positions_value = 2 * 110 * 0.88 = 193.6 CHF
    assert abs(val["positions_value"] - 193.6) < 0.1
    # cash = 1000 - buy_cost - buy_commission
    expected_cash = 1000.0 - buy_result["total"] - buy_result["commission"]
    assert val["cash"] == pytest.approx(expected_cash, abs=0.1)
    assert val["total_value"] == pytest.approx(expected_cash + 193.6, abs=0.1)


@pytest.mark.asyncio
async def test_trades_recorded(db_with_portfolio):
    await execute_buy(db_with_portfolio, "AAPL", 1.0, 100.0)
    await execute_sell(db_with_portfolio, "AAPL", 1.0, 110.0)

    trades = await queries.get_trades(db_with_portfolio)
    assert len(trades) == 2
    actions = {t["action"] for t in trades}
    assert actions == {"BUY", "SELL"}


@pytest.mark.asyncio
async def test_commission_recorded_in_trades(db_with_portfolio):
    """Commission should be recorded in trades table."""
    await execute_buy(db_with_portfolio, "AAPL", 1.0, 100.0)
    trades = await queries.get_trades(db_with_portfolio)
    assert len(trades) == 1
    assert trades[0]["commission_chf"] > 0


@pytest.mark.asyncio
async def test_total_commissions_query(db_with_portfolio):
    """get_total_commissions should sum all commissions."""
    await execute_buy(db_with_portfolio, "AAPL", 1.0, 100.0)
    await execute_sell(db_with_portfolio, "AAPL", 1.0, 110.0)
    total = await queries.get_total_commissions(db_with_portfolio)
    assert total > 0


@pytest.mark.asyncio
async def test_buy_commission_deducted_from_cash(db_with_portfolio):
    """Commission should be deducted from cash on buy."""
    result = await execute_buy(db_with_portfolio, "AAPL", 1.0, 100.0)
    assert result["ok"]
    cost = result["total"]
    commission = result["commission"]

    p = await queries.get_portfolio(db_with_portfolio)
    assert p["cash"] == pytest.approx(1000.0 - cost - commission, abs=0.01)


@pytest.mark.asyncio
async def test_sell_commission_deducted_from_proceeds(db_with_portfolio):
    """Commission should be deducted from proceeds on sell."""
    buy_result = await execute_buy(db_with_portfolio, "AAPL", 2.0, 100.0)
    cash_after_buy = 1000.0 - buy_result["total"] - buy_result["commission"]

    sell_result = await execute_sell(db_with_portfolio, "AAPL", 2.0, 110.0)
    assert sell_result["ok"]
    gross_proceeds = sell_result["total"]
    sell_commission = sell_result["commission"]

    p = await queries.get_portfolio(db_with_portfolio)
    expected = cash_after_buy + gross_proceeds - sell_commission
    assert p["cash"] == pytest.approx(expected, abs=0.01)


@pytest.mark.asyncio
async def test_six_whole_shares_rounding(db_with_portfolio):
    """SIX stocks should be rounded to whole shares."""
    # Request 2.7 shares of NESN.SW — should get floor(2.7) = 2
    result = await execute_buy(db_with_portfolio, "NESN.SW", 2.7, 100.0, usd_chf_rate=0.88)
    assert result["ok"]
    assert result["shares"] == 2.0  # Rounded down to whole


@pytest.mark.asyncio
async def test_slippage_applied_on_buy(db_with_portfolio):
    """BUY should record a slippage-adjusted price higher than the quote."""
    result = await execute_buy(db_with_portfolio, "AAPL", 1.0, 100.0)
    assert result["ok"]
    # US slippage = 5 bps → price should be 100.05
    assert result["price"] == pytest.approx(100.05, abs=0.01)

    # avg_cost in position should also reflect slipped price
    pos = await queries.get_position_by_symbol(db_with_portfolio, "AAPL")
    assert pos["avg_cost"] == pytest.approx(100.05, abs=0.01)


@pytest.mark.asyncio
async def test_slippage_applied_on_sell(db_with_portfolio):
    """SELL should record a slippage-adjusted price lower than the quote."""
    await execute_buy(db_with_portfolio, "AAPL", 2.0, 100.0)
    result = await execute_sell(db_with_portfolio, "AAPL", 2.0, 110.0)
    assert result["ok"]
    # US slippage = 5 bps → sell price should be 109.945
    assert result["price"] == pytest.approx(109.945, abs=0.01)


@pytest.mark.asyncio
async def test_slippage_six_larger_than_us(db_with_portfolio):
    """SIX stocks should have larger slippage than US stocks."""
    us_result = await execute_buy(db_with_portfolio, "AAPL", 1.0, 100.0)
    six_result = await execute_buy(db_with_portfolio, "NESN.SW", 1.0, 100.0)
    # SIX = 10 bps, US = 5 bps → SIX slippage is bigger
    us_slip = us_result["price"] - 100.0
    six_slip = six_result["price"] - 100.0
    assert six_slip > us_slip
