import httpx
from typing import List, Dict, AsyncGenerator, Literal
import logging
from datetime import datetime
import os
from pydantic import Field
import time

from kruppe.data_source.news.base_news import NewsSource
from kruppe.data_source.utils import WebScraper, HTTPX_CONNECTION_LIMITS, RequestSourceException, load_headers
from kruppe.models import Document

logger = logging.getLogger(__name__)

class NewYorkTimesData(NewsSource):
    headers_path: str
    headers: Dict[str, str] = Field(default_factory=lambda data: load_headers(data["headers_path"]))
    apiKey: str = Field(default_factory=lambda: os.getenv("NYTIMES_API_KEY"))
    source: str = "New York Times"
    description: str = "The New York Times covers a broad spectrum of news, including domestic, national, and international events, offering opinion pieces, investigative reports, and reviews, with a focus on providing in-depth, independent journalism."
    shorthand: str= "nyt"

    async def _nyt_scraper_helper(self, article_metadata: List[Dict[str, str]], query: str = None) -> AsyncGenerator[Document, None]:
        logger.info("Scraping %s NYT article links", len(article_metadata))
        urls = [article.get("web_url") or article.get("url") for article in article_metadata]
        meta_dict = {url: article for url, article in zip(urls, article_metadata)}

        client = httpx.AsyncClient(timeout=10.0, headers=self.headers, limits=HTTPX_CONNECTION_LIMITS)
        try:
            scraper = WebScraper(async_client=client)
            async for data in scraper.async_scrape_links(urls):
                if data is None:
                    continue

                original_meta = meta_dict[data["meta"]["url"]]
                metadata = self.parse_metadata(
                    query=query,
                    url=data["meta"]["url"],
                    title=data["meta"]["title"],
                    publication_time=data["meta"]["publication_time"],
                    description=original_meta.get("abstract") or original_meta.get("snippet"),
                    section=original_meta.get("section") or original_meta.get("section_name"),
                    document_type=original_meta.get("item_type") or original_meta.get("document_type")
                )
                yield Document(text=data["content"], metadata=metadata)
        finally:
            await client.aclose()
    
    async def news_search(
        self,
        query: str,
        max_results: int = 20,
        sort: Literal["relevance", "date"]  = "date",
        **kwargs
    ) -> AsyncGenerator[Document, None]:
        
        article_metadata = []
        async with httpx.AsyncClient(timeout=20.0, limits=HTTPX_CONNECTION_LIMITS) as client:
            url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

            NUM_RESULTS_PER_PAGE = 10
            page = 0
            while page * NUM_RESULTS_PER_PAGE < max_results: # Fetch until we have enough articles
                params = {
                    "q": query,
                    "api-key": self.apiKey,
                    "sort": {"date": "newest", "relevance": "relevance"}[sort], # newest, oldest, relevance
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

        article_metadata = article_metadata[:max_results]
        logger.debug("Fetched %d articles", len(article_metadata))

        async for document in self._nyt_scraper_helper(article_metadata, query=query):
            yield document
    
    async def news_recent(
        self,
        days: int = None, # TODO: not implemented
        max_results: int = 20,
        filter: Dict = None, # TODO: not implemented
        **kwargs
    ) -> AsyncGenerator[Document, None]:
        # sections = ["business", "education", "job market", "technology", "u.s.", "world"]
        sections = ["business", "technology"]
        article_metadata = []

        async with httpx.AsyncClient(timeout=20.0, limits=HTTPX_CONNECTION_LIMITS) as client:
            logger.info("Fetching links from newsfeed")
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
                if len(article_metadata) >= max_results:
                    break

                time.sleep(13)
        
        # hard cap
        article_metadata = article_metadata[:max_results]
        async for document in self._nyt_scraper_helper(article_metadata):
            yield document
    
    async def news_archive(
        self,
        start_date: str,
        end_date: str,
        max_results: int = 100, # TODO: not implemented
        filter: Dict = None, # TODO: not implemented,
        **kwargs
    ) -> AsyncGenerator[Document, None]:
        # section_names = ["Business", "Education", "Job Market", "Technology", "U.S.", "World"]
        section_names = ["Business", "Technology"]

        # Get today's date
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # Generate all year-month pairs from start_date to today
        year_month_pairs = set()
        current_date = start_date

        while current_date <= end_date:
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
                if len(article_metadata) >= max_results:
                    break
                time.sleep(13)
        
        # hard cap
        article_metadata = article_metadata[:max_results]

        async for document in self._nyt_scraper_helper(article_metadata):
            yield document
        

