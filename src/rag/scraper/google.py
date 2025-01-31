import os
import httpx
from typing import List
import logging
from pydantic import Field, field_validator

from rag.scraper.base_source import (
    BaseDataSource,
    RequestSourceException,
)
from rag.scraper.utils import WebScraper, HTTPX_CONNECTION_LIMITS
from rag.models import Document

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

    def fetch(
        self, query: str, num_results: int = 10, or_terms: str = None, **kwargs
    ) -> List[Document]:
        """Fetch links from Google Search API, scrape them, and return as a list of Documents.
        If document store is set, save documents to document store

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
        with httpx.Client(timeout=10.0, limits=HTTPX_CONNECTION_LIMITS) as client:
            params = {"key": self.api_key, "cx": self.search_engine_id}
            params["q"] = query
            params["orTerms"] = or_terms

            for page in range(num_results // 10 + 1):
                # not doing asyncio cuz pages is usually really small (< 50)

                # 10 results per page
                params["start"] = page * 10 + 1

                try:
                    logger.debug(
                        "Fetching links from Google Search API (page %d)", page
                    )
                    response = client.get(
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
            logger.debug("Initialize WebScraper")
            scraper = WebScraper(sync_client=client)

            logger.debug("Scraping links")
            scraped_data = scraper.scrape_links(links)

        # create List of Documents
        logger.debug("Converting data to Document objects")
        documents = []
        for link, data in zip(links, scraped_data):
            if data is None:
                # if scraping fails, skip
                continue

            metadata = self.parse_metadata(
                query=query,
                url=link,
                title=data["title"],
                publication_time=data["time"],
            )
            document = Document(text=data["content"], metadata=metadata)
            documents.append(document)

        logger.debug(
            "Successfully fetched %d documents from Google Search API", len(documents)
        )
        return documents

    async def async_fetch(
        self, query: str, num_results: int = 10, or_terms: str = None, **kwargs
    ) -> List[Document]:
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
            scraped_data = await scraper.async_scrape_links(links)

        # create List of Documents
        logger.debug("Converting data to Document objects")
        documents = []
        for link, data in zip(links, scraped_data):
            if data is None:
                # if scraping fails, skip
                continue

            metadata = self.parse_metadata(
                query=query,
                url=link,
                title=data["title"],
                publication_time=data["time"],
            )

            document = Document(text=data["content"], metadata=metadata)
            documents.append(document)

        logger.debug(
            "Successfully async fetched %d documents from Google Search API",
            len(documents),
        )
        return documents
