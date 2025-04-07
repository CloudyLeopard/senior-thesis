import os
import httpx
from pydantic import Field, field_validator
from typing import AsyncGenerator, Dict, Literal
import logging

from kruppe.data_source.news.base_news import NewsSource
from kruppe.data_source.utils import WebScraper, HTTPX_CONNECTION_LIMITS, RequestSourceException, not_ready
from kruppe.models import Document

logger = logging.getLogger(__name__)

class NewsAPIData(NewsSource):
    source: str = "NewsAPI"
    shorthand: str = "newsapi"
    apiKey: str = Field(default_factory=lambda: os.getenv("NEWS_API_KEY"))
    
    @field_validator("apiKey", mode="after")
    @classmethod
    def validate_env_vars(cls, v):
        if v is None or v == "":
            raise ValueError("News API key must be set")
        return v
    
    async def _parse_newsapi_response(
        self, data: dict, client: httpx.AsyncClient, query: str = None
    ) -> AsyncGenerator[Document, None]:
        """Internal method to parse the response from NewsAPI and yield Document objects."""
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

            yield Document(text=data["content"], metadata=metadata)
    
    async def news_search(
        self, 
        query: str,
        max_results: int = 10,
        sort: Literal["relevance", "date", "popularity"] = "date",
        **kwargs
    ) -> AsyncGenerator[Document, None]:

        params = {"apiKey": self.apiKey}
        params["q"] = query
        params["language"] = "en"
        params["sortBy"] = {"date": "publishedAt", "relevance": "relevancy", "popularity": "popularity"}[sort] # valid options are: relevancy, popularity, publishedAt.
        params["page"] = 1 # TODO: modify this code so that we can do multiple pages
        params["pageSize"] = min(max_results, 100)

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
            
            generator = self._parse_newsapi_response(data, client, query)
            async for doc in generator:
                yield doc

    async def news_recent(
        self,
        days: int = 0,
        max_results: int = 20,
        filter: Dict = None,  # TODO: not implemented
        **kwargs,
    ) -> AsyncGenerator[Document, None]:
        params = {"apiKey": self.apiKey}
        params["country"] = "us"
        params["page"] = 1 # TODO: modify this code so that we can do multiple pages
        params["pageSize"] = min(max_results, 100)

        categories = ["business", "technology"]

        async with httpx.AsyncClient() as client:
            logger.debug("Fetching documents from NewsAPI API")

            for category in categories:
                params["category"] = category

                try:
                    response = await client.get("https://newsapi.org/v2/top-headlines", params=params)
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
            
                generator = self._parse_newsapi_response(data, client)
                async for doc in generator:
                    yield doc
        
    
    @not_ready
    async def news_archive(self):
        raise NotImplementedError