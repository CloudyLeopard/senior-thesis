from abc import ABC, abstractmethod
import os
import aiohttp
from aiohttp import ClientSession
import requests
from requests.exceptions import HTTPError
from typing import List, Dict
import asyncio

from lexisnexisapi import webservices, credentials


from .util import get_logger
from .scraper import AsyncWebScraper
from .models import Document

class BaseDataSource(ABC):
    """Custom data source class interface"""

    # TODO: separate async from regular data sources

    @abstractmethod
    def fetch(self, query: str) -> List[Document]:
        """Fetch links relevant to the query with the corresponding data source

        Args:
            query: query to retrieve text from

        Returns:
            List of Documents

        Raises:
            ...
        """
        pass

    @abstractmethod
    async def async_fetch(self, query: str, session: ClientSession) -> List[Document]:
        """Async fetch links relevant to the query with the corresponding data source

        Args:
            query: query to retrieve text from

        Returns:
            List of Documents

        Raises:
            ...
        """
        pass


class YFinanceData(BaseDataSource):
    def fetch(self, query: str) -> List[str]:
        return


class LexisNexisData(BaseDataSource):
    def __init__(self):
        self.source = "LexisNexis"

        # credentials stored at `credentials.cred_file_path()`
        self.token = webservices.token()

    def fetch(self, query: str) -> List[Dict[str, str]]:
        """Fetch news articles from LexisNexis API."""

        # see https://dev.lexisnexis.com/dev-portal/documentation/News#/News%20API/get_News for documentation
        search_string = query  # TODO: adjust this

        # TODO: adjust parameter based on documentation
        parameters = {
            "$search": search_string,
            "$expand": "Document",  # A navigation property name which will be included with the current result set.
            "$top": "3",  # Sets the maximum number of results to receive for this request.
            # Filter with two conditions
            "$filter": "Language eq LexisNexis.ServicesApi.Language'English' and year(Date) eq 2023",
            "$select": "WordLength,People,Subject",
            # '$orderby': 'Date asc'
        }

        try:
            data = webservices.call_api(
                access_token=self.token, endpoint="News", params=parameters
            )
        except HTTPError as e:
            self.logger.error("HTTP Error %d: %s", e.response.status_code, str(e))

            if e.response.status_code == 429:
                raise PermissionError("Lexis Nexis query limit reached")
            else:
                raise ValueError(f"Invalid response: HTTP {e.status}")

        # TODO: parse data
        return


class NYTimesData(BaseDataSource):
    pass


class GuardiansData(BaseDataSource):
    pass


class NewsAPIData(BaseDataSource):
    pass


class ProQuestData(BaseDataSource):
    pass


class BingsNewsData(BaseDataSource):
    pass


class GoogleSearchData(BaseDataSource):
    """Wrapper that calls on Google Search JSON API"""

    def __init__(
        self, session: ClientSession = None
    ):
        self.source = "GoogleSearchAPI"

        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        # TODO: error for if the above are not set

        # load aiohttp client session for async request
        self.session = session

    def fetch(self, query: str, pages: 2) -> List[str]:
        return

    async def async_fetch(self, query: str, pages: 2) -> List[Dict[str, str]]:
        """async fetch links from Google Search API and scrape links."""

        # get list of website links from Google
        links = []
        for page in range(pages):
            response_json = await self._request_google_search_api(
                search_query=query, start_page=page
            )

            num_results = int(response_json["searchInformation"]["totalResults"])
            raw_results = response_json["items"] if num_results != 0 else []

            # list of websites, where each website is a "title" and a "link"
            for result in raw_results:
                links.append(result["link"])

        # scrape list of links
        scraper = AsyncWebScraper()
        scraped_data = await scraper.scrape_links(links)

        # create List of Documents
        documents = []
        for link, content in zip(links, scraped_data):
            metadata = {"url": link, "source": self.source}
            document = Document(text=content, metadata=metadata)
            documents.append(document)
        return documents

    async def _request_google_search_api(
        self,
        search_query: str,
        or_terms: str = "",
        start_page: int = 0,
    ):
        """request Google Custom Search API with aiohttp"""

        parameters = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": search_query,
            "orTerms": or_terms,
            "start": start_page * 10 + 1,
        }
        try:
            async with self.session.get(
                "https://www.googleapis.com/customsearch/v1", params=parameters
            ) as resp:
                resp.raise_for_status()
                response_json = await resp.json()
                return response_json
        except aiohttp.ClientResponseError as e:
            self.logger.error("HTTP Error %d: %s", e.status, str(e))
            if e.status == 429:
                raise PermissionError("Google search query limit per day reached")
            else:
                raise ValueError(f"Invalid response: HTTP {e.status}")
        except (aiohttp.ClientError, aiohttp.http_exceptions.HttpProcessingError) as e:
            self.logger.error(f"Client or HTTP processing error: {str(e)}")
            raise ConnectionError("A network or server error occurred")
        except Exception as e:
            self.logger.exception("Unexpected error occurred during the API request")
            raise RuntimeError("An unexpected error occurred") from e
