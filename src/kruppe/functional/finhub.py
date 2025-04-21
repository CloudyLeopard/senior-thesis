from typing import Dict, Tuple, List
import pandas as pd
import logging
from datetime import datetime

from pydantic import computed_field

from kruppe.data_source.finance.base_fin import FinancialSource
from kruppe.functional.base_tool import BaseTool
from kruppe.llm import BaseLLM
from kruppe.prompts.functional import ANALYZE_FINANCIALS_SYSTEM, ANALYZE_FINANCIALS_USER, ANALYZE_FINANCIALS_TOOL_DESCRIPTION
from kruppe.models import FinancialDocument

logger = logging.getLogger(__name__)

# TODO: make the second return value not an empty list and actually return some source

class FinHub(BaseTool):
    fin_source: FinancialSource
    llm: BaseLLM

    async def get_company_background(self, ticker: str) -> Tuple[str, List[FinancialDocument]]:
        obs = await self.fin_source.get_company_background(ticker)

        if obs is None:
            obs = f"Failed to find any information about the company {ticker}. Are you sure it exists?"
            return obs, []

        metadata = {
            "ticker": ticker,
            "datasource": self.fin_source.source,
            "title": f"{ticker} Company Background"
        }

        return obs, [FinancialDocument(text=obs, metadata=metadata)]
    
    async def get_company_income_stmt(self, ticker: str, years: int = 0) -> Tuple[str, List[FinancialDocument]]:
        df = await self.fin_source.get_company_income_stmt(ticker, years)
        
        if df is None:
            return f"Failed to find any income statement for {ticker}. Are you sure it exists?", []

        # Convert the DataFrame to a string representation
        df_str = df.to_string()
        obs = f"\n{df_str}"

        metadata = {
            "ticker": ticker,
            "datasource": self.fin_source.source,
            "title": f"{ticker} Income Statement"
        }

        return obs, [FinancialDocument(text=df_str, metadata=metadata)]
    
    async def get_company_balance_sheet(self, ticker: str, years: int = 0) -> Tuple[str, List]:
        df = await self.fin_source.get_company_balance_sheet(ticker, years)
        
        if df is None:
            return f"Failed to find any balance sheet for {ticker}. Are you sure it exists?", []
        
        # Convert the DataFrame to a string representation
        df_str = df.to_string()
        obs = f"\n{df_str}"

        metadata = {
            "ticker": ticker,
            "datasource": self.fin_source.source,
            "title": f"{ticker} Balance Sheet"
        }

        return obs, [FinancialDocument(text=df_str, metadata=metadata)]


    async def analyze_company_financial_stmts(self, ticker: str, years: int = 3) -> Tuple[str, List[FinancialDocument]]:
        background, bkg_sources = await self.get_company_background(ticker)
        income_stmt, inc_sources = await self.get_company_income_stmt(ticker, years)
        balance_sheet, bal_sources = await self.get_company_balance_sheet(ticker, years)

        messages = [
            {"role": "system", "content": ANALYZE_FINANCIALS_SYSTEM},
            {"role": "user", "content": ANALYZE_FINANCIALS_USER.format(
                ticker=ticker,
                firm_background=background,
                income_statement=income_stmt,
                balance_sheet=balance_sheet
            )}
        ]

        analysis_response = await self.llm.async_generate(messages)
        analysis_str = analysis_response.text

        try:
            thought, analysis = analysis_str.split("Final Analysis:", 1)
            thought = thought.strip()
            analysis = analysis.strip()
        except ValueError:
            logger.warning("Failed to parse analysis response. Falling back to raw response.")
            thought = "No specific thought provided."
            analysis = analysis_str.strip()

        sources = bkg_sources + inc_sources + bal_sources
        
        return analysis, sources
    
    # basically there is a lot...

    @computed_field
    @property
    def get_company_background_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": "get_company_background",
                "description": "Retrieve company background information",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "required": [
                        "ticker"
                    ],
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Stock ticker symbol for the company"
                        }
                    },
                    "additionalProperties": False
                }
            }
        }
    
    @computed_field
    @property
    def get_company_income_stmt_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": "get_company_income_stmt",
                "description": "Get the company's income statement.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "required": [
                        "ticker",
                        "years"
                    ],
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Stock ticker symbol of the company."
                        },
                        "years": {
                            "type": "number",
                            "description": "Number of years of data to retrieve (0 for most recent).",
                        }
                    },
                    "additionalProperties": False
                }
            }
        }
    
    @computed_field
    @property
    def get_company_balance_sheet_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": "get_company_balance_sheet",
                "description": "Get the company's balance sheet.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "required": [
                        "ticker",
                        "years"
                    ],
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "The stock ticker symbol of the company"
                        },
                        "years": {
                            "type": "number",
                            "description": "Number of years of data to retrieve (0 for most recent)."
                        }
                    },
                    "additionalProperties": False
                }
            }
        }
    
    @computed_field
    @property
    def analyze_company_financial_stmts_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": "analyze_company_financial_stmts",
                "description": ANALYZE_FINANCIALS_TOOL_DESCRIPTION,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "required": [
                        "ticker",
                        "years"
                    ],
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "The stock ticker symbol of the company being analyzed."
                        },
                        "years": {
                            "type": "number",
                            "description": "Number of years (default is 3) for which the financial data is analyzed."
                        }
                    },
                    "additionalProperties": False
                }
            }
        }