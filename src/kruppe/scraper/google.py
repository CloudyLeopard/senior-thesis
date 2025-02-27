import os
import httpx
from typing import AsyncGenerator
import logging
from pydantic import Field, field_validator

from kruppe.scraper.base_source import (
    BaseDataSource,
    RequestSourceException,
)
from kruppe.scraper.utils import WebScraper, HTTPX_CONNECTION_LIMITS
from kruppe.models import Document

logger = logging.getLogger(__name__)


class GoogleSearchData(BaseDataSource):
    """Wrapper that calls on Google Search JSON API"""

    api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    search_engine_id: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    )
    source: str = "Google Search API"

    @field_validator("api_key", "search_engine_id", mode="after")
    @classmethod
    def validate_env_vars(cls, v):
        if v is None or v == "":
            raise ValueError("Google Search API key and search engine ID must be set")
        return v
    
    async def async_fetch(
        self, query: str, num_results: int = 10, or_terms: str = None, **kwargs
    ) -> AsyncGenerator[Document, None]:
        """
        Async version of fetch. Fetches links from Google Search API, scrapes them, and returns as a list of Documents.
        If document store is set, save documents to document store.

        Args:
            query (str): The main search query to fetch relevant links.
            or_terms (str, optional): Additional search terms to include in the query.
            pages (int, optional): The number of pages of search results to fetch.

        Returns:
            List[Document]: A list of Document objects containing the text and metadata
                            of the scraped links.

        Raises:
            HTTPError: If the request to the Google Search API fails.
            PermissionError: If the Google Search API query limit is reached.
        """
        # get list of links from Google Search API
        links = []
        async with httpx.AsyncClient(
            timeout=10.0, limits=HTTPX_CONNECTION_LIMITS
        ) as client:
            params = {"key": self.api_key, "cx": self.search_engine_id}
            params["q"] = query
            params["orTerms"] = or_terms

            for page in range(num_results // 10 + 1):
                params["start"] = page * 10 + 1

                try:
                    logger.debug(
                        "Async fetching links from Google Search API (page %d)", page
                    )
                    response = await client.get(
                        "https://www.googleapis.com/customsearch/v1", params=params
                    )
                    response.raise_for_status()
                    response_json = response.json()
                except httpx.HTTPStatusError as e:
                    msg = e.response.text
                    if e.response.status_code == 429:
                        msg = "Google Search API query limit reached"
                    logger.error(
                        "Google Search API HTTP Error %d: %s",
                        e.response.status_code,
                        msg,
                    )
                    raise RequestSourceException(msg)
                except httpx.RequestError as e:
                    logger.error("Google Search API Failed to fetch links: %s", e)
                    raise RequestSourceException(e)

                num_results = int(response_json["searchInformation"]["totalResults"])
                raw_results = response_json["items"] if num_results != 0 else []

                logger.debug("Found %d results", num_results)

                # list of websites, where each website is a "title" and a "link"
                links.extend([result["link"] for result in raw_results])

            # scrape list of links
            logger.debug("Initialize Async WebScraper")
            scraper = WebScraper(async_client=client)

            logger.debug("Async scraping links")
            async for data in scraper.async_scrape_links(links):
                if data is None:
                    continue

                metadata = self.parse_metadata(
                    query=query,
                    **data["meta"]
                )

                document = Document(text=data["content"], metadata=metadata)
                yield document