import httpx
from typing import List, Dict
import logging
from datetime import datetime, timedelta
import os
from pydantic import Field
import time

from rag.scraper.base_source import BaseDataSource, RequestSourceException
from rag.scraper.utils import WebScraper, HTTPX_CONNECTION_LIMITS
from rag.models import Document

logger = logging.getLogger(__name__)

class NewYorkTimesData(BaseDataSource):
    headers: Dict[str, str]
    sources: str = "New York Times"
    apiKey: str = Field(default_factory=lambda: os.getenv("NYTIMES_API_KEY"))

    def fetch(self, query: str) -> List[Document]:
        raise NotImplementedError()
    
    def async_fetch(self, query: str) -> List[Document]:
        raise NotImplementedError()
    
    async def fetch_news_feed(self, num_results: int = 20) -> List[Document]:
        # sections = ["business", "education", "job market", "technology", "u.s.", "world"]
        sections = ["business", "technology"]
        article_metadata = []
        documents = []

        async with httpx.AsyncClient(timeout=20.0, limits=HTTPX_CONNECTION_LIMITS) as client:
            for section in sections:
                url = f"https://api.nytimes.com/svc/news/v3/content/all/{section}.json?api-key={self.apiKey}&limit={num_results}"
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    data = response.json()
                except httpx.HTTPStatusError as e:
                    msg = e.response.text
                    if e.response.status_code == 429:
                        msg = "New York Times query limit reached"
                    logger.error(
                        "New York Times HTTP Error %d: %s",
                        e.response.status_code,
                        msg,
                    )
                    raise RequestSourceException(msg)
                except httpx.RequestError as e:
                    logger.error("New York Times Failed to fetch documents: %s", e)
                    raise RequestSourceException(e)
                
                if data["num_results"] > 0:
                    article_metadata.extend(data["results"])
                time.sleep(13)
        
        urls = [article["url"] for article in article_metadata]
        client = httpx.AsyncClient(timeout=10.0, headers=self.headers, limits=HTTPX_CONNECTION_LIMITS)
        scraper = WebScraper(async_client=client)
        article_data = await scraper.async_scrape_links(urls)

        for article, meta in zip(article_data, article_metadata):
            if article is None:
                continue
            metadata = self.parse_metadata(
                query=None,
                url=meta["url"],
                title=meta["title"],
                publication_time=meta["published_date"],
                abstract=meta["abstract"],
            ) # NOTE: there are a lot more metadata avaiable
            documents.append(Document(text=article["content"], metadata=metadata))
        
        await client.aclose()
        return documents
    
    async def fetch_archive(self, months: int) -> List[Document]:
        # section_names = ["Business", "Education", "Job Market", "Technology", "U.S.", "World"]
        section_names = ["Business", "Technology"]

        # Get today's date
        today = datetime.today().date()
        start_date = today - timedelta(days=months*30)

        # Generate all year-month pairs from start_date to today
        year_month_pairs = set()
        current_date = start_date

        while current_date <= today:
            year_month_pairs.add((current_date.year, current_date.month))
            # Move to the next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1, day=1)
        
        urls = [
            f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={self.apiKey}"
            for year, month in year_month_pairs
        ]

        async with httpx.AsyncClient(timeout=10.0, limits=HTTPX_CONNECTION_LIMITS) as client:
            article_metadata = []

            for url in urls:
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    data = response.json()
                except httpx.HTTPStatusError as e:
                    msg = e.response.text
                    if e.response.status_code == 429:
                        msg = "New York Times query limit reached"
                    logger.error(
                        "New York Times HTTP Error %d: %s",
                        e.response.status_code,
                        msg,
                    )
                    raise RequestSourceException(msg)
                except httpx.RequestError as e:
                    logger.error("New York Times Failed to fetch documents: %s", e)
                    raise RequestSourceException(e)
                article_metadata.extend([doc for doc in data["response"]['docs'] if doc.get("section_name") in section_names])
                time.sleep(13)

        urls = [article["web_url"] for article in article_metadata]

        client = httpx.AsyncClient(timeout=10.0, headers=self.headers, limits=HTTPX_CONNECTION_LIMITS)
        scraper = WebScraper(async_client=client)
        article_data = await scraper.async_scrape_links(urls)

        documents = []
        for article, meta in zip(article_data, article_metadata):
            if article is None:
                continue
            metadata = self.parse_metadata(
                query=None,
                url=meta["web_url"],
                title=meta["headline"]["main"],
                publication_time=meta["pub_date"],
                abstract=meta["snippet"],
            ) # NOTE: there are a lot more metadata avaiable
            documents.append(Document(text=article["content"], metadata=metadata))

        await client.aclose()
        return documents
        

