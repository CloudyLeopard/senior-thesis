import pytest
from pydantic.error_wrappers import ValidationError
import pandas as pd

from kruppe.data_source.finance.base_fin import FinancialSource
from kruppe.llm import BaseLLM
from kruppe.functional.finhub import FinHub
from kruppe.models import Response

class DummySource(FinancialSource):
    source: str = "dummy"

    def __init__(self, background=None, income_df=None, balance_df=None, **data):
        super().__init__(**data)
        self._background = background
        self._income_df = income_df
        self._balance_df = balance_df

    async def get_company_background(self, ticker):
        return self._background

    async def get_company_income_stmt(self, ticker, years):
        return self._income_df

    async def get_company_balance_sheet(self, ticker, years):
        return self._balance_df

class DummyLLM(BaseLLM):
    text: str = "text"

    async def async_generate(self, messages):
        return Response(text=self.text)

    def generate():
        raise NotImplementedError
    
    async def async_generate_with_tools():
        raise NotImplementedError
    

def test_init_requires_fin_source_and_llm():
    # Missing both
    with pytest.raises(ValidationError):
        FinHub()
    # Missing llm
    with pytest.raises(ValidationError):
        FinHub(fin_source=DummySource())
    # Missing fin_source
    with pytest.raises(ValidationError):
        FinHub(llm=DummyLLM())

@pytest.mark.asyncio
async def test_get_company_background_success():
    hub = FinHub(
        fin_source=DummySource(background="Acme Corp is a widget maker"),
        llm=DummyLLM()
    )
    result, sources = await hub.get_company_background("ACME")
    assert result == "Acme Corp is a widget maker"
    assert sources == []

@pytest.mark.asyncio
async def test_get_company_background_none():
    hub = FinHub(
        fin_source=DummySource(background=None),
        llm=DummyLLM()
    )
    result, sources = await hub.get_company_background("ZZZZ")
    assert "Failed to find any information about the company ZZZZ" in result
    assert sources == []

@pytest.mark.asyncio
async def test_get_company_income_stmt_success():
    df = pd.DataFrame({"col": [1, 2]})
    hub = FinHub(
        fin_source=DummySource(income_df=df),
        llm=DummyLLM()
    )
    result, sources = await hub.get_company_income_stmt("TEST", years=1)
    assert "col" in result
    assert "1" in result and "2" in result
    assert sources == []

@pytest.mark.asyncio
async def test_get_company_income_stmt_none():
    hub = FinHub(
        fin_source=DummySource(income_df=None),
        llm=DummyLLM()
    )
    result, sources = await hub.get_company_income_stmt("NONE", years=2)
    assert "Failed to find any income statement for NONE" in result
    assert sources == []

@pytest.mark.asyncio
async def test_get_company_balance_sheet_success():
    df = pd.DataFrame({"asset": [100, 200]})
    hub = FinHub(
        fin_source=DummySource(balance_df=df),
        llm=DummyLLM()
    )
    result, sources = await hub.get_company_balance_sheet("BAL", years=0)
    assert "asset" in result
    assert "100" in result and "200" in result
    assert sources == []

@pytest.mark.asyncio
async def test_get_company_balance_sheet_none():
    hub = FinHub(
        fin_source=DummySource(balance_df=None),
        llm=DummyLLM()
    )
    result, sources = await hub.get_company_balance_sheet("NOBAL", years=3)
    assert "Failed to find any balance sheet for NOBAL" in result
    assert sources == []

@pytest.mark.asyncio
async def test_analyze_with_valid_response():
    background = "Some background"
    income_df = pd.DataFrame({"rev": [10]})
    balance_df = pd.DataFrame({"assets": [20]})
    text = "Thoughts... Final Analysis: The company looks solid."
    hub = FinHub(
        fin_source=DummySource(background=background, income_df=income_df, balance_df=balance_df),
        llm=DummyLLM(text=text)
    )
    analysis, sources = await hub.analyze_company_financial_stmts("XYZ", years=1)
    assert analysis == "The company looks solid."
    assert sources == []

@pytest.mark.asyncio
async def test_analyze_without_final_analysis():
    background = "BG"
    income_df = pd.DataFrame({"x": [1]})
    balance_df = pd.DataFrame({"y": [2]})
    hub = FinHub(
        fin_source=DummySource(background=background, income_df=income_df, balance_df=balance_df),
        llm=DummyLLM(text="No delimiter here")
    )
    analysis, sources = await hub.analyze_company_financial_stmts("ABC", years=2)
    assert analysis == "No delimiter here"
    assert sources == []

def test_schema_methods():
    hub = FinHub(
        fin_source=DummySource(),
        llm=DummyLLM()
    )
    bkg = hub.get_company_background_schema
    assert bkg["function"]["name"] == "get_company_background"
    inc = hub.get_company_income_stmt_schema
    assert inc["function"]["name"] == "get_company_income_stmt"
    bal = hub.get_company_balance_sheet_schema
    assert bal["function"]["name"] == "get_company_balance_sheet"
    ana = hub.analyze_company_financial_stmts_schema
    assert ana["function"]["name"] == "analyze_company_financial_stmts"
    for schema, required in [
        (bkg, ["ticker"]),
        (inc, ["ticker", "years"]),
        (bal, ["ticker", "years"]),
        (ana, ["ticker", "years"]),
    ]:
        props = schema["function"]["parameters"]
        assert props["required"] == required