import httpx
from typing import List, Dict, AsyncGenerator, Literal
import logging
from datetime import datetime
import os
from pydantic import Field
import time

from kruppe.data_source.news.base_news import NewsSource
from kruppe.data_source.scraper import WebScraper, RequestSourceException, load_headers
from kruppe.models import Document
from kruppe.common.log import log_io

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

        client = httpx.AsyncClient(headers=self.headers)
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
        
        retries = 3 # NOTE: cheat method for when theres a problem
        article_metadata = []
        async with httpx.AsyncClient() as client:
            url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

            # NUM_RESULTS_PER_PAGE = 10
            page = 0
            while len(article_metadata) < max_results: # Fetch until we have enough articles
                params = {
                    "q": query,
                    "api-key": self.apiKey,
                    "sort": "newest", # only newest matters
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
                    logger.error("New York Times Failed to fetch documents (%d / %d tries): %s", 4-retries, 3, repr(e))
                    retries -= 1
                    if retries > 0:
                        time.sleep(13)
                        continue
                    
                    if len(article_metadata) == 0:
                        raise RequestSourceException("Failed to fetch any articles from New York Times")
                    else:
                        # NOTE: cheat method for when theres a problem
                        logger.warning("Failed to fetch more articles from New York Times, returning what we have")
                        break
                
                # Break if there are no more articles to fetch
                if not data["response"]["docs"]:
                    break

                article_metadata.extend(data["response"]["docs"])

                # Sleep to avoid hitting the rate limit
                time.sleep(13)

        article_metadata = article_metadata[:max_results]
        logger.debug("Fetched %d articles", len(article_metadata))

        async for document in self._nyt_scraper_helper(article_metadata, query=query):
            yield document
    
    @log_io
    async def news_recent(
        self,
        days: int = None, # TODO: not implemented
        max_results: int = 20,
        filter: Dict = None, # TODO: not implemented
        **kwargs
    ) -> AsyncGenerator[Document, None]:
        sections = ["business", "education", "job market", "technology", "u.s.", "world"]
        # sections = ["business", "technology"]
        article_metadata = []

        async with httpx.AsyncClient() as client:
            logger.info("Fetching links from newsfeed")
            for section in sections:
                url = f"https://api.nytimes.com/svc/news/v3/content/all/{section}.json?api-key={self.apiKey}&limit={max_results}"
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
                

                # Sleep to avoid hitting the rate limit
                time.sleep(13)
        
        # hard cap
        article_metadata.sort(key=lambda x: x.get("published_date", ""), reverse=True)
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

        # Get start and end date
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

        async with httpx.AsyncClient() as client:
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
        

