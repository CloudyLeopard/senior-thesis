import httpx
from typing import List, Dict
import logging
from datetime import datetime, timedelta
import os
from pydantic import Field
import time
import asyncio

from rag.scraper.base_source import BaseDataSource, RequestSourceException
from rag.scraper.utils import WebScraper, HTTPX_CONNECTION_LIMITS
from rag.models import Document

logger = logging.getLogger(__name__)

class NewYorkTimesData(BaseDataSource):
    headers: Dict[str, str]
    sources: str = "New York Times"
    apiKey: str = Field(default_factory=lambda: os.getenv("NYTIMES_API_KEY"))

    async def _nyt_scraper_helper(self, article_metadata: List[Dict[str, str]], query: str = None) -> List[Document]:
        urls = [article["web_url"] for article in article_metadata]

        client = httpx.AsyncClient(timeout=10.0, headers=self.headers, limits=HTTPX_CONNECTION_LIMITS)
        scraper = WebScraper(async_client=client)
        article_data = await scraper.async_scrape_links(urls)

        documents = []
        for article, meta in zip(article_data, article_metadata):
            if article is None:
                continue
            metadata = self.parse_metadata(
                query=query,
                url=meta["web_url"],
                title=meta["headline"]["main"],
                publication_time=meta["pub_date"],
                abstract=meta["snippet"],
            ) # NOTE: there are a lot more metadata avaiable
            documents.append(Document(text=article["content"], metadata=metadata))

        await client.aclose()
        return documents
    
    def fetch(self, query: str, num_results: int = 20, sort: str = "newest") -> List[Document]:
        return asyncio.run(self.async_fetch(query=query, num_results=num_results, sort=sort))
    
    async def async_fetch(self, query: str, num_results: int = 20, sort: str = "newest") -> List[Document]:
        article_metadata = []
        async with httpx.AsyncClient(timeout=20.0, limits=HTTPX_CONNECTION_LIMITS) as client:
            url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

            NUM_RESULTS_PER_PAGE = 10
            page = 0
            while page * NUM_RESULTS_PER_PAGE < num_results: # Fetch until we have enough articles
                params = {
                    "q": query,
                    "api-key": self.apiKey,
                    "sort": sort, # newest, oldest, relevance
                    "fl": "web_url,headline,pub_date,snippet",
                    "page": page,
                }

                page += 1 # Increment page number

                try:
                    response = await client.get(url, params=params)
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
                
                # Break if there are no more articles to fetch
                if data["response"]["meta"]["hits"] <= data["response"]["meta"]["offset"]:
                    break

                article_metadata.extend(data["response"]["docs"])
                time.sleep(13)

        article_metadata = article_metadata[:num_results]
        logger.debug("Fetched %d articles", len(article_metadata))

        return await self._nyt_scraper_helper(article_metadata, query=query)
    
    async def fetch_news_feed(self, num_results: int = 20) -> List[Document]:
        # sections = ["business", "education", "job market", "technology", "u.s.", "world"]
        sections = ["business", "technology"]
        article_metadata = []

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
        
        return await self._nyt_scraper_helper(article_metadata)
    
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

        return await self._nyt_scraper_helper(article_metadata)
        

