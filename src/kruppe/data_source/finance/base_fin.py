from abc import abstractmethod
import pandas as pd

from kruppe.data_source.base_source import DataSource

class FinancialSource(DataSource):

    @abstractmethod
    async def get_company_background(self, ticker: str) -> str:
        ...
    
    @abstractmethod
    async def get_company_income_stmt(self, ticker: str, years: int = 0) -> pd.DataFrame:
        """
        Get the company's income statement.
        """
        ...
    
    @abstractmethod
    async def get_company_balance_sheet(self, ticker: str, years: int = 0) -> pd.DataFrame:
        """
        Get the company's balance sheet.
        """
        ...