import pandas as pd


from kruppe.data_source.finance.base_fin import FinancialSource

class SECSource(FinancialSource):
    source: str = "SEC Financials"

    async def get_company_background(self, ticker: str) -> str:
        ...
    
    async def get_company_income_stmt(self, ticker: str, years: int = 0) -> pd.DataFrame:
        """
        Get the company's income statement.
        """
        ...

    async def get_company_balance_sheet(self, ticker: str, years: int = 0) -> pd.DataFrame:
        """
        Get the company's balance sheet.
        """
        ...
    
    
   