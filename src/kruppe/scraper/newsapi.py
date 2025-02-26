import os
import httpx
from pydantic import Field, field_validator
from typing import List, AsyncGenerator
import logging
import asyncio

from kruppe.scraper.base_source import BaseDataSource, RequestSourceException
from kruppe.scraper.utils import WebScraper, HTTPX_CONNECTION_LIMITS
from kruppe.models import Document

logger = logging.getLogger(__name__)

class NewsAPIData(BaseDataSource):
    source: str = "NewsAPI"
    apiKey: str = Field(default_factory=lambda: os.getenv("NEWS_API_KEY"))
    
    @field_validator("apiKey", mode="after")
    @classmethod
    def validate_env_vars(cls, v):
        if v is None or v == "":
            raise ValueError("News API key must be set")
        return v
    
    async def async_fetch(self, query: str, num_results: int = 10, months: int = None, sort_by = "publishedAt", **kwargs) -> AsyncGenerator[Document, None]:
        params = {"apiKey": self.apiKey}
        params["q"] = query
        if months:
            pass # TODO: doesn't work right now, need premium subscription to use "month"
            # params["from"] = (datetime.now() - timedelta(days=months * 30)).date().isoformat()
        params["language"] = "en"
        params["sortBy"] = sort_by # valid options are: relevancy, popularity, publishedAt.
        params["page"] = 1 # TODO: modify this code so that we can do multiple pages
        params["pageSize"] = num_results

        async with httpx.AsyncClient(timeout=10.0, limits=HTTPX_CONNECTION_LIMITS) as client:
            logger.debug("Fetching documents from NewsAPI API")

            try:
                response = await client.get("https://newsapi.org/v2/everything", params=params)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as e:
                msg = e.response.text
                if e.response.status_code == 429:
                    msg = "NewsAPI query limit reached"
                logger.error(
                    "NewsAPI HTTP Error %d: %s",
                    e.response.status_code,
                    msg,
                )
                raise RequestSourceException(msg)
            except httpx.RequestError as e:
                logger.error("NewsAPI Failed to fetch documents: %s", e)
                raise RequestSourceException(e)
            
            articles = data["articles"]
            total_results = data["totalResults"]
            logger.debug("Fetched %d documents from NewsAPI API", total_results)

            # scrape list of links
            logger.debug("Scraping documents from links")
            scraper = WebScraper(async_client=client)

            logger.debug("Scraping links")
            links = [article["url"] for article in articles]
            meta_dictionary = {link: article for link, article in zip(links, articles)}
            
            async for data in scraper.async_scrape_links(links):
                if data is None:
                    continue

                original_meta = meta_dictionary[data["meta"]["url"]]
                metadata = self.parse_metadata(
                    query=query,
                    source=original_meta["source"]["name"],
                    # title=original_meta["title"],
                    # publication_time=original_meta["publishedAt"],
                    # description=original_meta["description"],
                    **data["meta"]
                )

                print("CONTENT:", data["content"][:10])

                yield Document(text=data["content"], metadata=metadata)

