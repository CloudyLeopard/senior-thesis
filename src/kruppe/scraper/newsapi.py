import os
import httpx
from pydantic import Field, field_validator
from typing import List
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
    
    def fetch(self, query: str, num_results: int = 10, months: int = None, sort_by = "publishedAt", **kwargs) -> List[Document]:
        return asyncio.run(self.async_fetch(query=query, num_results=num_results, months=months, sort_by=sort_by, **kwargs))
    
    async def async_fetch(self, query: str, num_results: int = 10, months: int = None, sort_by = "publishedAt", **kwargs) -> List[Document]:
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
            scraped_data = await scraper.async_scrape_links([article["url"] for article in articles])
        
        logger.debug("Converting data to Document objects")
        documents = []
        for article, data in zip(articles, scraped_data):
            if data is None:
                continue

            metadata = self.parse_metadata(
                query=query,
                url=article["url"],
                source=article["source"]["name"],
                # title=article["title"],
                # publication_time=article["publishedAt"],
                # description=article["description"],
                **data["meta"]
            )
            document = Document(text=data["content"], metadata=metadata)
            documents.append(document)

        logger.debug("Successfully fetched %d documents from NewsAPI API", len(documents))
        return documents    

