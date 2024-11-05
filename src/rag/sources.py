from abc import ABC, abstractmethod
import os
import httpx
from requests.exceptions import HTTPError
from typing import List, Dict
import pathlib

from lexisnexisapi import webservices

from rag.scraper import WebScraper
from rag.document_storages import BaseDocumentStore
from rag.models import Document

class BaseDataSource(ABC):
    """Custom data source class interface"""
    def __init__(self, document_store: BaseDocumentStore = None):
        self.document_store = document_store
        
    @abstractmethod
    def fetch(self) -> List[Document]:
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
    async def async_fetch(self) -> List[Document]:
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
        """
        Initialize LexisNexisData with optional document_store.

        Args:
            document_store: Optional BaseDocumentStore for storing retrieved documents.

        Note:
            The credentials must be set in the file located at
            credentials.cred_file_path(). If not set, a warning will be
            raised.
        """
        super().__init__(document_store)
        self.source = "LexisNexis"

        # TODO: if credentials not set, warn user where to set credentials
        self.token = webservices.token()

    def fetch(self, query: str, num_results = 5) -> List[Document]:
        """
        Fetch documents from Lexis Nexis based on query
        
        see https://dev.lexisnexis.com/dev-portal/documentation/News#/News%20API/get_News for documentation

        Args:
            query: query to retrieve text from
            num_results: number of results to retrieve (default: 5)

        Returns:
            List of Document objects with text and metadata

        Raises:
            ValueError: if response is invalid
            RuntimeError: if Lexis Nexis query limit is reached
        """
        search_string = query  # TODO: adjust this

        # TODO: adjust parameter based on documentation
        parameters = {
            "$search": search_string,
            "$expand": "Document",  # "Document" to get html data
            "$top": str(num_results),  # Sets the maximum number of results to receive for this request.
            # Filter with two conditions
            "$filter": "Language eq LexisNexis.ServicesApi.Language'English' and year(Date) eq 2024",
            "$select": "ResultId, Title, Source",
        }

        try:
            data = webservices.call_api(
                access_token=self.token, endpoint="News", params=parameters
            )
        except HTTPError as e:
            # self.logger.error("HTTP Error %d: %s", e.response.status_code, str(e))

            if e.response.status_code == 429:
                raise RuntimeError("HTTP Error 429: Lexis Nexis query limit reached")
            else:
                raise ValueError(f"Invalid response: HTTP {e}")

        documents = []
        for result in data["value"]:
            html = result["Document"]["Content"]
            try:
                text = WebScraper._scrape_html(html)
            except Exception:
                text = html # fallback to html if scraping fails

            document = Document(
                text=text,
                metadata={
                    # required metadata
                    "query": query,
                    "datasource": self.source,
                    # lexisnexis unique metadata
                    "title": result["Title"],
                    "source": result["Source"]["Name"],
                    "lexisResultId": result["ResultId"],
                    "citation": result["Document"].get("Citation", "")
                },
            )

            documents.append(document)

            # if document store is set, save document to document store
            if self.document_store:
                self.document_store.save_document(document)
        
        return documents
    
    async def async_fetch(self, query: str, num_results = 10) -> List[Document]:
        raise NotImplementedError


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

    def __init__(self, document_store: BaseDocumentStore = None, api_key: str = None, search_engine_id: str = None):
        super().__init__(document_store) # init document store
        self.source = "GoogleSearchAPI"

        self.parameters = {
            "key": api_key or os.getenv("GOOGLE_API_KEY"),
            "cx": search_engine_id or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        }
        # TODO: error for if API key or search engine ID is not set
    
    def fetch(self, query: str, or_terms: str = None, pages = 1) -> List[Document]:
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
        with httpx.Client(timeout=10.0) as client:
            params = self.parameters
            params["q"] = query
            params["orTerms"] = or_terms

            for page in range(pages):
                # not doing asyncio cuz pages is usually really small (< 50)

                # 10 results per page
                params["start"] = page * 10 + 1

                # TODO: put in wrapper for error handling (i.e. 429 error = google search query limit per day reached)
                response = client.get("https://www.googleapis.com/customsearch/v1", params=params)
                response.raise_for_status()
                response_json = response.json()

                num_results = int(response_json["searchInformation"]["totalResults"])
                raw_results = response_json["items"] if num_results != 0 else []

                # list of websites, where each website is a "title" and a "link"
                links.extend([result["link"] for result in raw_results])
        
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
    
    async def async_fetch(self, query: str, or_terms: str = None, pages = 1) -> List[Document]:
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
        async with httpx.AsyncClient(timeout=10.0) as client:
            params = self.parameters
            params["q"] = query
            params["orTerms"] = or_terms

            for page in range(pages):
                params["start"] = page * 10 + 1

                # TODO: put in wrapper for error handling (i.e. 429 error = google search query limit per day reached)
                response = await client.get("https://www.googleapis.com/customsearch/v1", params=params)
                response.raise_for_status()
                response_json = response.json()

                num_results = int(response_json["searchInformation"]["totalResults"])
                raw_results = response_json["items"] if num_results != 0 else []

                # list of websites, where each website is a "title" and a "link"
                links.extend([result["link"] for result in raw_results])
        
        # scrape list of links
        scraper = WebScraper()
        scraped_data = await scraper.async_scrape_links(links)

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


class WikipediaData(BaseDataSource):
    def __init__(self):
        self.source = "Wikipedia"

    def fetch(self, query: str) -> List[Document]:
        """Fetches links from Wikipedia, scrapes them, and returns as a list of Documents.
        If document store is set, save documents to document store.

        Args:
            query (str): The main search query to fetch relevant links.

        Returns:
            List[Document]: A list of Document objects containing the text and metadata 
                            of the scraped links.

        Raises:
            HTTPError: If the request to the Wikipedia API fails.
        """
        # https://pypi.org/project/Wikipedia-API/#description
        raise NotImplementedError
    
    async def async_fetch(self, query: str) -> List[Document]:
        """N/A. Calls on sync. fetch function"""
        return self.fetch(self, query)

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
                