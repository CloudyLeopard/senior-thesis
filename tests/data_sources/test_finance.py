import pytest
import pandas as pd
import yfinance as yf
from kruppe.data_source.finance.yfin import YFinanceData

# Dummy classes to simulate yfinance behavior
class DummyTicker:
    def __init__(self, ticker, session):
        self._ticker = ticker
        self.info = getattr(self, "_info", {})  # set by test via class attribute if available

    def get_income_stmt(self, pretty=True):
        return self._income_df  # set by test

    def get_balance_sheet(self, pretty=True):
        return self._balance_df  # set by test

class DummySector:
    def __init__(self, sector, session):
        self.overview = {"description": "Dummy sector description"}

class DummyIndustry:
    def __init__(self, industry, session):
        self.overview = {"description": "Dummy industry description"}

@pytest.mark.asyncio
async def test_get_company_background_success(monkeypatch):
    # Prepare dummy info with all keys
    info = {
        "longName": "Test Corp",
        "longBusinessSummary": "We do testing.",
        "sector": "Tech",
        "industry": "Software",
        "currentPrice": 123.45,
        "fullTimeEmployees": 1000,
        "country": "USA",
        "state": "CA",
        "city": "San Francisco",
        "zip": "94105"
    }
    # Monkeypatch yfinance objects
    DummyTicker._info = info
    monkeypatch.setattr(yf, "Ticker", DummyTicker)
    monkeypatch.setattr(yf, "Sector", DummySector)
    monkeypatch.setattr(yf, "Industry", DummyIndustry)

    src = YFinanceData()
    result = await src.get_company_background("TEST")
    # Check that each key:value line appears
    for k, v in info.items():
        assert f"{k}: {v}" in result
    # Check that sector_description and industry_description are appended
    assert "sector_description: Dummy sector description" in result
    assert "industry_description: Dummy industry description" in result

@pytest.mark.asyncio
async def test_get_company_background_info_exception(monkeypatch, caplog):
    # DummyTicker.info will raise
    class BadTicker(DummyTicker):
        @property
        def info(self):
            raise RuntimeError("fail")

        @info.setter
        def info(self, value):
            pass
    monkeypatch.setattr(yf, "Ticker", BadTicker)
    src = YFinanceData()
    caplog.set_level("WARNING")
    result = await src.get_company_background("FAIL")
    assert result is None
    assert "Failed to retrieve company information" in caplog.text

@pytest.mark.asyncio
async def test_get_company_income_stmt_empty(monkeypatch, caplog):
    # Income statement empty DataFrame
    DummyTicker._income_df = pd.DataFrame()
    monkeypatch.setattr(yf, "Ticker", DummyTicker)
    src = YFinanceData()
    caplog.set_level("WARNING")
    result = await src.get_company_income_stmt("EMPTY")
    assert result is None
    assert "No income statement data found" in caplog.text

@pytest.mark.asyncio
async def test_get_company_income_stmt_nonempty(monkeypatch):
    # Build a DataFrame with rows and 3 years of data
    # yfinance has reverse ordering, hence the index order below
    df = pd.DataFrame({
        "2021": [10, 20, 30, 40, 50],
        "2022": [11, 21, 31, 41, 51],
        "2023": [12, 22, 32, 42, 52]
    }, index=["EBIT", "EBITDA", "Diluted Average Shares", "Cost", "Revenue",]) 
    DummyTicker._income_df = df
    monkeypatch.setattr(yf, "Ticker", DummyTicker)
    src = YFinanceData()
    # Request last 1 year => keep 2 columns (2021,2022), reverse rows, and pick slices
    result = await src.get_company_income_stmt("OK", years=1)
    # Check index order and contents
    expected_idx = ["Revenue", "Cost", "Diluted Average Shares", "EBIT", "EBITDA"]
    assert list(result.index) == expected_idx
    # Check columns limited to two
    assert list(result.columns) == ["2021", "2022"]

@pytest.mark.asyncio
async def test_get_company_balance_sheet_empty(monkeypatch, caplog):
    DummyTicker._balance_df = pd.DataFrame()
    monkeypatch.setattr(yf, "Ticker", DummyTicker)
    src = YFinanceData()
    caplog.set_level("WARNING")
    result = await src.get_company_balance_sheet("EMPTY")
    assert result is None
    assert "No balance sheet data found" in caplog.text

@pytest.mark.asyncio
async def test_get_company_balance_sheet_nonempty(monkeypatch):
    df = pd.DataFrame({
        "2021": [100, 200, 300],
        "2022": [110, 210, 310],
        "2023": [120, 220, 320]
    }, index=["Assets", "Liabilities", "Equity"])
    DummyTicker._balance_df = df
    monkeypatch.setattr(yf, "Ticker", DummyTicker)
    src = YFinanceData()
    result = await src.get_company_balance_sheet("OK", years=0)
    # years=0 => keep only first column, then reverse rows
    assert list(result.columns) == ["2021"]
    assert list(result.index) == ["Equity", "Liabilities", "Assets"]

@pytest.mark.asyncio
async def test_get_company_news_not_ready():
    src = YFinanceData()
    with pytest.raises(NotImplementedError):
        await src.get_company_news("ANY")