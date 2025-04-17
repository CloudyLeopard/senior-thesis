import yfinance as yf
import pandas as pd
from typing import List
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter
import logging


from kruppe.common.utils import not_ready
from kruppe.data_source.finance.base_fin import FinancialSource
from kruppe.models import Document
from kruppe.__about__ import version as kruppe_version

# session stuff, according to yfinance documentation
class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
   pass

session = CachedLimiterSession(
   limiter=Limiter(RequestRate(2, Duration.SECOND*5)),  # max 2 requests per 5 seconds
   bucket_class=MemoryQueueBucket,
   backend=SQLiteCache("yfinance.cache"),
)

session.headers['User-agent'] = f'kruppe/{kruppe_version}'

logger = logging.getLogger(__name__)

class YFinanceData(FinancialSource):
    source: str = "yfinance"

    async def get_company_background(self, ticker: str) -> str | None:
        # Retrieve company background information using yfinance
        dat = yf.Ticker(ticker, session=session)

        # Attempt to retrieve company information
        try:
            info = dat.info
        except Exception:
            logger.warning("Failed to retrieve company information for ticker %s", ticker)
            return None
        
        # extract relevant keys i want
        info_keys_to_keep = ["longName", "longBusinessSummary", "sector", "industry", "currentPrice", "fullTimeEmployees", "country", "state", "city", "zip"]

        # list of tuples to maintain order
        info_to_return = [(key, info.get(key)) for key in info_keys_to_keep]

        # Add additional sector and industry information if available
        sectorKey = info.get("sector")
        if sectorKey:
            sector = yf.Sector(sectorKey, session=session)
            info_to_return.append(("sector_description", sector.overview.get('description')))
        
        industryKey = info.get("industry")
        if industryKey:
            industry = yf.Industry(industryKey, session=session)
            info_to_return.append(("industry_description", industry.overview.get('description')))

        # Filter out None values, and format the output
        return "\n".join(f"{key}: {value}" for key, value in info_to_return if value is not None)
    
    async def get_company_income_stmt(self, ticker: str, years: int = 0) -> pd.DataFrame | None:
        """
        Get the company's income statement.
        """
        dat = yf.Ticker(ticker, session=session)

        df_income = dat.get_income_stmt(pretty=True)
        if df_income.empty:
            logger.warning("No income statement data found for ticker %s", ticker)
            return None

        df_income = df_income.iloc[:, :years+1] # limit to the last `years` years of data
        
        # reverse row order - for some rzn yfinance returns top line item at the bottom
        df_income = df_income.iloc[::-1]

        # select only the relevant rows
        # keep everything up to “Diluted Average Shares”
        df_income_1 = df_income.loc[:"Diluted Average Shares"]

        # only add EBIT/EBITDA if they’re not already in df_income_1
        additional = ["EBIT", "EBITDA"]
        to_add = [r for r in additional if r in df_income.index and r not in df_income_1.index]
        df_income_2 = df_income.loc[to_add]

        df_result = pd.concat([df_income_1, df_income_2])

        return df_result

    async def get_company_balance_sheet(self, ticker: str, years: int = 0) -> pd.DataFrame | None:
        """
        Get the company's balance sheet.
        """
        dat = yf.Ticker(ticker, session=session)
        df_balance = dat.get_balance_sheet(pretty=True)
        if df_balance.empty:
            logger.warning("No balance sheet data found for ticker %s", ticker)
            return None
        
        df_balance = df_balance.iloc[:, :years+1]  # limit to the last `years` years of data

        # reverse row order
        df_balance = df_balance.iloc[::-1]
        return df_balance
    
    @not_ready
    async def get_company_news(self, ticker: str, max_results: int = 10) -> List[Document]:
        """
        Get the company's news.
        """
        news = yf.Search(ticker, session=session, max_results=max_results)

        raise NotImplementedError("This method is not implemented yet.")
    
   