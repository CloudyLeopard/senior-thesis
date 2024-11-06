from abc import ABC, abstractmethod
import os
import httpx
from typing import List, Dict
import pathlib
from bs4 import BeautifulSoup
import asyncio

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

        Returns:
            List[Document]: A list of Document objects containing the text and metadata 
                            of the scraped links.

        Raises:
            HTTPError: If the request to the data source API fails.

        """
        pass

    @abstractmethod
    async def async_fetch(self) -> List[Document]:
        """Async fetch links relevant to the query with the corresponding data source

        Returns:
            List[Document]: A list of Document objects containing the text and metadata 
                            of the scraped links.

        Raises:
            HTTPError: If the request to the data source API fails.
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
        except httpx.HTTPStatusError as e:
            # self.logger.error("HTTP Error %d: %s", e.response.status_code, str(e))
            if e.response.status_code == 429:
                e.msg = "HTTP Error 429: Lexis Nexis query limit reached"
            raise e
        except httpx.RequestError as e:
            print(f"An error occurred while requesting {e.request.url!r}.")
            raise e
        
        documents = []
        for result in data["value"]:
            html = result["Document"]["Content"]
            try:
                data = WebScraper.default_html_parser(html)
                text = data["content"]
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
        for link, data in zip(links, scraped_data):
            if data is None:
                # if scraping fails, skip
                continue

            metadata = {
                "url": link, 
                "title": data["title"],
                "datasource": self.source,
                "query": query
            }
            document = Document(text=data["content"], metadata=metadata)
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
        for link, data in zip(links, scraped_data):
            if data is None:
                # if scraping fails, skip
                continue

            metadata = {
                "url": link, 
                "title": data["title"],
                "datasource": self.source,
                "query": query
            }
            document = Document(text=data["content"], metadata=metadata)
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

class FinancialTimesData(BaseDataSource):
    def __init__(self, headers: Dict[str, str], document_store: BaseDocumentStore = None):
        super().__init__(document_store)
        self.source = "Financial Times"
        self.headers = headers
    
    @staticmethod
    def _parse_search_page(html: str) -> List[str]:
        # get the list of links from the search page's html
        soup = BeautifulSoup(html, "lxml")
        
        # get all search items links
        # NOTE: 1 page = 25 search results
        links = []
        search_divs = soup.find_all('div', class_='search-item')
        for div in search_divs:
            a_tag = div.find('a', class_='js-teaser-heading-link')
            if a_tag:
                links.append(a_tag.get('href'))
        
        return links

    @staticmethod
    def _ft_blog_html_parser(html: str, post_id: str) -> Dict[str, str]:
        """FT has 'blogs', which are a different format than FT's 'articles'.
        Often, multiple blog posts with very different topic can be on the same 
        page. So, we use "post_id" to isolate the one we want, and this post_id
        can be found after the # in the returned urls from search page

        Returns:
            Dict[str, str]: A dictionary containing the content and other metadata of the blog post.
        """
        soup = BeautifulSoup(html, "lxml")
        post_id_element = soup.find(id=post_id)
        title = post_id_element.find('h2').get_text()
        content = "\n".join([p.text.strip() for p in post_id_element.find_all('p')])
        posted_time = post_id_element.find('time').get('datetime')

        return {
            "title": title,
            "content": content,
            "time": posted_time
        }

    def fetch(self, query: str, sort="relevance", pages=1) -> List[Document]:
        """Fetch links from Financial Times, scrapes them, and returns as a list of Documents.
        If document store is set, save documents to document store.

        Args:
            query (str): The main search query to fetch relevant links.
            sort (str, optional): The sort order of the search results. Accepted values are "date" and "relevance". Defaults to "relevance".
            pages (int, optional): The number of search pages to scrape. Defaults to 1.

        Returns:
            List[Document]: A list of Document objects containing the text and metadata 
                            of the scraped links.

        Raises:
            HTTPError: If the request to the Financial Times API fails.
        """

        links = []
        with httpx.Client(timeout=10.0, headers=self.headers) as client:
            # Search FT using query and scrape list of articles
            url = "https://www.ft.com/search"

            for page in range(1, pages+1):
                params = {
                    "q": query,
                    "sort": sort, # date or relevance
                    "page": page,
                    "isFirstView": "false"
                }
                response = client.get(url, params=params)
                response.raise_for_status() # TODO: error handling
                html = response.text
                links.extend(self._parse_search_page(html))
            
            # FT articles has two types: regular articles that can be
            # scraped with default parser, and blogs where we just want to
            # extract the relevant blog portion
            article_links = []
            blog_links = []

            for link in links:
                if link.startswith("https"):
                    article_links.append(link)
                else:
                    blog_links.append(f"https://www.ft.com{link}")

            # NOTE: not using the default parser. The custom parser takes an additional input
            # so i am scraping the link individually, rather than scraping the whole list

            # scrape articles
            scraper = WebScraper(sync_client=client)
            articles_data = scraper.scrape_links(article_links)

            # scrape blogs
            scraper.set_html_parser(self._ft_blog_html_parser)
            blog_data = []
            for link in blog_links:
                post_id = link.split("#")[1]
                blog_data.append(scraper.scrape_link(url=link, post_id=post_id))

        # combine articles and blogs
        documents = []
        for link, article in zip(article_links + blog_links, articles_data + blog_data):
            documents.append(
                Document(
                    text=article["content"],
                    metadata={
                        "datasource": self.source,
                        "query": query,
                        "link": link,
                        "title": article["title"],
                        "time": article["time"]
                    }
                )
            )
        
        # if document store is set, save documents to document store
        if self.document_store:
            self.document_store.save_documents(documents)

        return documents


    async def async_fetch(self, query: str, sort="relevance", pages=1) -> List[Document]:
        """Async version of fetch. Fetches links from Financial Times, scrapes them, and returns as a list of Documents.
        If document store is set, save documents to document store.

        Args:
            query (str): The main search query to fetch relevant links.
            sort (str, optional): The sort order of the search results. Accepted values are "date" and "relevance". Defaults to "relevance".
            pages (int, optional): The number of search pages to scrape. Defaults to 1.

        Returns:
            List[Document]: A list of Document objects containing the text and metadata 
                            of the scraped links.

        Raises:
            HTTPError: If the request to the Financial Times API fails.
        """
        links = []
        client = httpx.AsyncClient(timeout=10.0, headers=self.headers)
        try:
            # Search FT using query and scrape list of articles
            url = "https://www.ft.com/search"

            for page in range(1, pages+1):
                params = {
                    "q": query,
                    "sort": sort, # date or relevance
                    "page": page,
                    "isFirstView": "false"
                }
                response = await client.get(url, params=params)
                response.raise_for_status() # TODO: error handling
                html = response.text
                links.extend(self._parse_search_page(html))
            
            # FT articles has two types: regular articles that can be
            # scraped with default parser, and blogs where we just want to
            # extract the relevant blog portion
            article_links = []
            blog_links = []

            for link in links:
                if link.startswith("https"):
                    article_links.append(link)
                else:
                    blog_links.append(f"https://www.ft.com{link}")

            # scrape articles
            scraper = WebScraper(async_client=client)
            articles_data = await scraper.async_scrape_links(article_links)

            # scrape blogs
            # NOTE: not using the default parser. The custom parser takes an additional input
            # so i am scraping the link individually, rather than scraping the whole list
            scraper.set_html_parser(self._ft_blog_html_parser)

            # NOTE: need to use asyncio.gather to scrape the blog links, since we can't call on 
            # the ascrape_links method while using a custom html parser with custom input
            blog_data = await asyncio.gather(
                *(scraper.async_scrape_link(url=link, post_id=link.split("#")[1]) for link in blog_links)
            )
        finally:
            await client.aclose()

        # combine articles and blogs
        documents = []
        for link, article in zip(article_links + blog_links, articles_data + blog_data):
            documents.append(
                Document(
                    text=article["content"],
                    metadata={
                        "datasource": self.source,
                        "query": query,
                        "link": link,
                        "title": article["title"],
                        "time": article["time"]
                    }
                )
            )
        
        # if document store is set, save documents to document store
        if self.document_store:
            self.document_store.save_documents(documents)

        return documents

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
                