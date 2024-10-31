from abc import ABC, abstractmethod
import os
from unittest.mock import Base
import aiohttp
from aiohttp import ClientSession
import requests
from requests.exceptions import HTTPError
from typing import List, Dict
import pathlib

from lexisnexisapi import webservices, credentials

from rag.scraper import WebScraper
from rag.document_storages import BaseDocumentStore
from rag.models import Document

class BaseDataSource(ABC):
    """Custom data source class interface"""
    def __init__(self, document_store: BaseDocumentStore = None):
        self.document_store = document_store
        
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

    def process_document(document: Document):
        # TODO: later on, perhaps use LLM on scraped text data
        # to extract information that can be used later. Example: category
        pass


class YFinanceData(BaseDataSource):
    def fetch(self, query: str) -> List[str]:
        return


class LexisNexisData(BaseDataSource):
    def __init__(self, document_store: BaseDocumentStore = None):
        super().__init__(document_store)
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

    def __init__(self, document_store: BaseDocumentStore = None):
        super().__init__(document_store)
        self.source = "GoogleSearchAPI"

        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        # TODO: error for if the above are not set


    def fetch(self, query: str, pages = 1) -> List[str]:
         # get list of website links from Google
        links = []
        for page in range(pages):
            response_json = self._request_google_search_api(
                search_query=query, start_page=page
            )

            num_results = int(response_json["searchInformation"]["totalResults"])
            raw_results = response_json["items"] if num_results != 0 else []

            # list of websites, where each website is a "title" and a "link"
            for result in raw_results:
                links.append(result["link"])

        # scrape list of links
        scraper = WebScraper()
        scraped_data = scraper.scrape_links(links)

        # create List of Documents
        documents = []
        for link, content in zip(links, scraped_data):
            metadata = {
                "url": link, 
                "datasource": self.source,
                "query": query
            }
            document = Document(text=content, metadata=metadata)
            documents.append(document)
        
        # if document store is set, save documents to document store
        if self.document_store:
            self.document_store.save_documents(documents)
        return documents
        

    async def async_fetch(self, query: str, session: ClientSession, pages = 1) -> List[Document]:
        """async fetch links from Google Search API and scrape links."""

        # get list of website links from Google
        links = []
        for page in range(pages):
            response_json = await self._async_request_google_search_api(
                search_query=query, session=session, start_page=page
            )

            num_results = int(response_json["searchInformation"]["totalResults"])
            raw_results = response_json["items"] if num_results != 0 else []

            # list of websites, where each website is a "title" and a "link"
            for result in raw_results:
                links.append(result["link"])

        # scrape list of links
        scraper = WebScraper()
        scraped_data = await scraper.async_scrape_links(session, links)

        # create List of Documents
        documents = []
        for link, content in zip(links, scraped_data):
            metadata = {
                "url": link, 
                "datasource": self.source,
                "query": query
            }
            document = Document(text=content, metadata=metadata)
            documents.append(document)
        return documents
    

    def _request_google_search_api(self, search_query: str, or_terms: str = "", start_page: int = 0):
        """request Google Custom Search API with requests"""

        parameters = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": search_query,
            "orTerms": or_terms,
            "start": start_page * 10 + 1,
        }
        try:
            resp = requests.get("https://www.googleapis.com/customsearch/v1", params=parameters)
            resp.raise_for_status()
            response_json = resp.json()
            return response_json
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                raise PermissionError("Google search query limit per day reached")
            else:
                raise ValueError(f"Invalid response: HTTP {e.response.status_code}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError("A network or server error occurred") from e
        except Exception as e:
            raise RuntimeError("An unexpected error occurred") from e


    async def _async_request_google_search_api(
        self,
        search_query: str,
        session: ClientSession,
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
            async with session.get(
                "https://www.googleapis.com/customsearch/v1", params=parameters
            ) as resp:
                resp.raise_for_status()
                response_json = await resp.json()
                return response_json
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                raise PermissionError("Google search query limit per day reached")
            else:
                raise ValueError(f"Invalid response: HTTP {e}")
        except (aiohttp.ClientError, aiohttp.http_exceptions.HttpProcessingError) as e:
            raise ConnectionError("A network or server error occurred")
        except Exception as e:
            raise RuntimeError("An unexpected error occurred") from e

class DirectoryData(BaseDataSource):
    def __init__(self):
        self.source = "Local Directory"

    def fetch(self, path: str):
        """given path to directory, fetch all .txt files within that directory"""
        dir = pathlib.Path(path)
        if (not dir.is_dir()):
            raise ValueError("Invalid path - must be a directory")
        
        documents = []
        for txt_file in dir.glob("*.txt"):
            txt = txt_file.read_text()
            name = txt_file.name

            doc = Document(text=txt, metadata = {
                "name": name,
                "path": txt_file.as_posix(),
                "datasource": self.source
            })
            documents.append(doc)
        
        return documents
    
    async def async_fetch(self, path: str):
        """N/A. Calls on sync. fetch function"""
        return self.fetch(self, path)
                