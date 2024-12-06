from abc import ABC, abstractmethod
import os
import httpx
import requests
from typing import List, Dict
import pathlib
from bs4 import BeautifulSoup
import asyncio
import logging
from datetime import datetime, timedelta

from lexisnexisapi import webservices, credentials
import yfinance as yf

from rag.tools.scraper import WebScraper
from rag.document_storages import BaseDocumentStore
from rag.models import Document

logger = logging.getLogger(__name__)
HTTPX_CONNECTION_LIMITS = httpx.Limits(max_keepalive_connections=20, max_connections=100)

class RequestSourceException(Exception):
    pass

# TODO: remove the "document_store" parameter. Keep entering documents into 
# document store separate from the data source class
class BaseDataSource(ABC):
    """Custom data source class interface"""

    def __init__(self, document_store: BaseDocumentStore = None):
        self.document_store = document_store
        self.datasource = "Unknown"

    @abstractmethod
    def fetch(self, query: str, num_results: int = 10, **kwargs) -> List[Document]:
        """Fetch links relevant to the query with the corresponding data source

        Returns:
            List[Document]: A list of Document objects containing the text and metadata
                            of the scraped links.

        Raises:
            RequestSourceException: If the request to the data source API fails.

        """
        pass

    @abstractmethod
    async def async_fetch(self, query: str, num_results: int = 10, **kwargs) -> List[Document]:
        """Async fetch links relevant to the query with the corresponding data source

        Returns:
            List[Document]: A list of Document objects containing the text and metadata
                            of the scraped links.

        Raises:
            RequestSourceException: If the request to the data source API fails.
        """
        pass

    def process_document(document: Document):
        # TODO: later on, perhaps use LLM on scraped text data
        # to extract information that can be used later. Example: category
        pass

    @classmethod
    def parse_metadata(
        cls,
        query: str,
        url: str = None,
        title: str = None,
        publication_time: str = None,
        **kwargs,
    ) -> Dict[str, str]:
        metadata = {
            "query": query or "",
            "datasource": cls.__name__,
            "url": url or "",
            "title": title or "",
            "publication_time": publication_time or "",
        }
        metadata.update(kwargs)
        return metadata


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
        cred = credentials.get_Credentials()
        if (not cred.get("WSAPI_CLIENT_ID")) or (not cred.get("WSAPI_SECRET")):
            logger.error(
                "LexisNexis credentials not set. Please set them in %s",
                credentials.cred_file_path(),
            )
            raise ValueError("LexisNexis credentials not set")
        self.token = webservices.token()

    def fetch(self, query: str, num_results=10, **kwargs) -> List[Document]:
        """
        Fetch documents from Lexis Nexis based on query

        see https://dev.lexisnexis.com/dev-portal/documentation/News#/News%20API/get_News for documentation

        Args:
            query: query to retrieve text from
            num_results: number of results to retrieve (default: 10)

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
            "$top": str(
                num_results
            ),  # Sets the maximum number of results to receive for this request.
            # Filter with two conditions
            "$filter": "Language eq LexisNexis.ServicesApi.Language'English' and year(Date) eq 2024",
            "$select": "ResultId, Title, Source",
        }

        try:
            logger.debug("Fetching documents from Lexis Nexis API")
            data = webservices.call_api(
                access_token=self.token, endpoint="News", params=parameters
            )
        except requests.exceptions.HTTPError as e:
            msg = e.response.reason
            if e.response.status_code == 429:
                msg = "Lexis Nexis query limit reached"
            logger.error("Lexis Nexis HTTP Error %d: %s", e.response.status_code, msg)
            raise RequestSourceException(msg)
        except requests.exceptions.RequestException as e:
            logger.error("Lexis Nexis Failed to fetch documents: %s", e)
            raise RequestSourceException(e)

        logger.debug("Converting data to Document objects")
        documents = []
        for result in data["value"]:
            html = result["Document"]["Content"]
            try:
                data = WebScraper.default_html_parser(html)
                text = data["content"]
            except Exception:
                text = html  # fallback to html if scraping fails

            metadata = self.parse_metadata(
                query=query,
                title=result["Title"],
                source=result["Source"]["Name"],
                lexisResultId=result["ResultId"],
                citation=result["Document"].get("Citation", ""),
            )

            document = Document(text=text, metadata=metadata)

            documents.append(document)

        # if document store is set, save document to document store
        if self.document_store:
            logger.debug("Saving documents to document store")
            self.document_store.save_documents(documents)

        logger.debug(
            "Successfully fetched %d documents from Lexis Nexis API", len(documents)
        )
        return documents

    async def async_fetch(self, query: str, num_results=10, **kwargs) -> List[Document]:
        """Fallback to sync fetch"""
        return self.fetch(query, num_results)


class NYTimesData(BaseDataSource):
    pass


class GuardiansData(BaseDataSource):
    pass


class NewsAPIData(BaseDataSource):
    
    def __init__(self, document_store: BaseDocumentStore = None, api_key: str = None):
        super().__init__(document_store)
        self.source = "NewsAPI"
        
        self.parameters = {
            "apiKey": api_key or os.getenv("NEWS_API_KEY"),
        }
    
    def fetch(self, query: str, num_results: int = 10, months: int = None, sort_by = "publishedAt", **kwargs) -> List[Document]:
        """Fetch with NewsAPI. Maximum 100 top results per fetch."""
        params = self.parameters
        params["q"] = query
        if months: # NOTE: doesn't work right now, need premium subscription to use "month"
            pass
            # params["from"] = (datetime.now() - timedelta(days=months * 30)).date().isoformat()
        params["language"] = "en"
        params["sortBy"] = sort_by # valid options are: relevancy, popularity, publishedAt.
        params["page"] = 1 # TODO: modify this code so that we can do multiple pages
        params["pageSize"] = num_results

        with httpx.Client(timeout=20.0, limits=HTTPX_CONNECTION_LIMITS) as client:
            logger.debug("Fetching documents from NewsAPI API")

            try:
                response = client.get("https://newsapi.org/v2/everything", params=params)
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
            num_results = data["totalResults"]
            logger.debug("Fetched %d documents from NewsAPI API", num_results)

            # scrape list of links
            logger.debug("Scraping documents from links")
            scraper = WebScraper(sync_client=client)

            logger.debug("Scraping links")
            scraped_data = scraper.scrape_links([article["url"] for article in articles])
        
        logger.debug("Converting data to Document objects")
        documents = []
        for article, data in zip(articles, scraped_data):
            if data is None:
                continue

            metadata = self.parse_metadata(
                query=query,
                url=article["url"],
                title=article["title"],
                publication_time=article["publishedAt"],
                source=article["source"]["name"],
                description=article["description"],
            )
            document = Document(text=data["content"], metadata=metadata)
            documents.append(document)
        
        # if document store is set, save documents to document store
        if self.document_store:
            logger.debug("Saving documents to document store")
            self.document_store.save_documents(documents)

        logger.debug("Successfully fetched %d documents from NewsAPI API", len(documents))
        return documents
    
    async def async_fetch(self, query: str, num_results: int = 10, months: int = None, sort_by = "publishedAt", **kwargs) -> List[Document]:
        params = self.parameters
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
            num_results = data["totalResults"]
            logger.debug("Fetched %d documents from NewsAPI API", num_results)

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
                title=article["title"],
                publication_time=article["publishedAt"],
                source=article["source"]["name"],
                description=article["description"],
            )
            document = Document(text=data["content"], metadata=metadata)
            documents.append(document)
        
        # if document store is set, save documents to document store
        if self.document_store:
            logger.debug("Saving documents to document store")
            self.document_store.save_documents(documents)

        logger.debug("Successfully fetched %d documents from NewsAPI API", len(documents))
        return documents    

class ProQuestData(BaseDataSource):
    pass


class BingsNewsData(BaseDataSource):
    pass


class GoogleSearchData(BaseDataSource):
    """Wrapper that calls on Google Search JSON API"""

    def __init__(
        self,
        document_store: BaseDocumentStore = None,
        api_key: str = None,
        search_engine_id: str = None,
    ):
        super().__init__(document_store)  # init document store
        self.source = "GoogleSearchAPI"

        self.parameters = {
            "key": api_key or os.getenv("GOOGLE_API_KEY"),
            "cx": search_engine_id or os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
        }

        if not self.parameters["key"] or not self.parameters["cx"]:
            logger.error("Google Search API key and search engine ID must be set")
            raise ValueError("Google Search API key and search engine ID must be set")

        logger.debug("Google Search API key and search engine ID set")

    def fetch(self, query: str, num_results: int = 10, or_terms: str = None, **kwargs) -> List[Document]:
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
            params = self.parameters
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

        # if document store is set, save documents to document store
        if self.document_store:
            logger.debug("Saving documents to document store")
            self.document_store.save_documents(documents)

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
        async with httpx.AsyncClient(timeout=10.0, limits=HTTPX_CONNECTION_LIMITS) as client:
            params = self.parameters
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

        # if document store is set, save documents to document store
        if self.document_store:
            logger.debug("Saving documents to document store")
            self.document_store.save_documents(documents)

        logger.debug(
            "Successfully async fetched %d documents from Google Search API",
            len(documents),
        )
        return documents


class WikipediaData(BaseDataSource):
    def __init__(self):
        self.source = "Wikipedia"

    def fetch(self, query: str, num_results: int = 10, **kwargs) -> List[Document]:
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

    async def async_fetch(self, query: str, num_results: int = 10, **kwargs) -> List[Document]:
        """N/A. Calls on sync. fetch function"""
        return self.fetch(self, query)


class FinancialTimesData(BaseDataSource):
    def __init__(
        self, headers: Dict[str, str], document_store: BaseDocumentStore = None
    ):
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
        search_divs = soup.find_all("div", class_="search-item")
        for div in search_divs:
            a_tag = div.find("a", class_="js-teaser-heading-link")
            if a_tag:
                links.append(a_tag.get("href"))

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
        if "post" not in post_id:
            post_id = f"post-{post_id}"
        post_id_element = soup.find(id=post_id)
        if not post_id_element:
            # if cannot find post_id in website, return None
            return None

        title = post_id_element.find("h2").get_text()
        content = "\n".join([p.text.strip() for p in post_id_element.find_all("p")])
        posted_time = post_id_element.find("time").get("datetime")

        return {"title": title, "content": content, "time": posted_time}

    def fetch(self, query: str, num_results: int = 25, sort="relevance", months: int = None, **kwargs) -> List[Document]:
        """Fetch links from Financial Times, scrapes them, and returns as a list of Documents.
        If document store is set, save documents to document store.

        Args:
            query (str): The main search query to fetch relevant links.
            num_results(int, optional): The number of search results to scrape. Defaults to 25 (or 1 page)
            sort (str, optional): The sort order of the search results. Accepted values are "date" and "relevance". Defaults to "relevance".

        Returns:
            List[Document]: A list of Document objects containing the text and metadata
                            of the scraped links.

        Raises:
            HTTPError: If the request to the Financial Times API fails.
        """

        links = []
        with httpx.Client(timeout=10.0, headers=self.headers, limits=HTTPX_CONNECTION_LIMITS) as client:
            # Search FT using query and scrape list of articles
            url = "https://www.ft.com/search"

            pages = num_results // 25 + 1
            for page in range(1, pages+1):
                params = {
                    "q": query,
                    "sort": sort,  # date or relevance
                    "page": page,
                    "isFirstView": "false",
                }

                # TODO: ft times have better date filters. Check FT search page
                # and implement them at a later time.
                if months:
                    params["from"] = (datetime.now() - timedelta(days=months * 30)).date().isoformat()

                # parse search page
                logger.debug(
                    "Scraping Financial Times for query: %s (sort: %s, page: %d)",
                    query,
                    sort,
                    page,
                )
                try:
                    response = client.get(url, params=params)
                    response.raise_for_status()  # TODO: error handling
                    html = response.text
                    links.extend(self._parse_search_page(html))
                except httpx.HTTPStatusError as e:
                    logger.error(
                        "Financial Times HTTP Error %d: %s", e.response.status_code, e.response.text
                    )
                    raise RequestSourceException(e.response.text)
                except httpx.RequestError as e:
                    logger.error("Error fetching Financial Times search page: %s", e)
                    raise RequestSourceException(e)

            # FT articles has two types: regular articles that can be
            # scraped with default parser, and blogs where we just want to
            # extract the relevant blog portion
            article_links = []
            blog_links = []

            for link in links:
                if not link.startswith("https"):
                    link = f"https://www.ft.com{link}"
                if "#" in link:
                    blog_links.append(link)
                else:
                    article_links.append(link)

            # NOTE: not using the default parser. The custom parser takes an additional input
            # so i am scraping the link individually, rather than scraping the whole list

            # scrape articles
            logger.debug("Initialize WebScraper")
            scraper = WebScraper(sync_client=client)

            logger.debug("Scraping %d Financial Times articles", len(links))
            articles_data = scraper.scrape_links(article_links)

            # scrape blogs
            scraper.set_html_parser(self._ft_blog_html_parser)
            blog_data = []
            for link in blog_links:
                post_id = link.split("#")[1]
                blog_data.append(scraper.scrape_link(url=link, post_id=post_id))

        # combine articles and blogs
        logger.debug("Converting data to Document Objects")
        documents = []
        for link, article in zip(article_links + blog_links, articles_data + blog_data):
            if article is None:
                continue
                
            metadata = self.parse_metadata(
                query=query,
                url=link,
                title=article["title"],
                publication_time=article["time"],
            )
            documents.append(Document(text=article["content"], metadata=metadata))

        # if document store is set, save documents to document store
        if self.document_store:
            logger.debug("Saving %d documents to document store", len(documents))
            self.document_store.save_documents(documents)

        logger.debug(
            "Successfully fetched %d documents from Financial Times", len(documents)
        )
        return documents

    async def async_fetch(
        self, query: str, num_results: int = 25, sort="relevance", months: int = None, **kwargs
    ) -> List[Document]:
        """Async version of fetch. Fetches links from Financial Times, scrapes them, and returns as a list of Documents.
        If document store is set, save documents to document store.

        Args:
            query (str): The main search query to fetch relevant links.
            num_results(int, optional): The number of search results to scrape. Defaults to 25 (or 1 page)
            sort (str, optional): The sort order of the search results. Accepted values are "date" and "relevance". Defaults to "relevance".

        Returns:
            List[Document]: A list of Document objects containing the text and metadata
                            of the scraped links.

        Raises:
            HTTPError: If the request to the Financial Times API fails.
        """
        links = []
        client = httpx.AsyncClient(timeout=10.0, headers=self.headers, limits=HTTPX_CONNECTION_LIMITS)
        try:
            # Search FT using query and scrape list of articles
            url = "https://www.ft.com/search"

            search_requests = []
            pages = num_results // 25 + 1
            for page in range(1, pages + 1):
                params = {
                    "q": query,
                    "sort": sort,  # date or relevance
                    "page": page,
                    "isFirstView": "false",
                }

                if months:
                    params["from"] = (datetime.now() - timedelta(days=months * 30)).date().isoformat()
                
                search_requests.append(client.build_request("GET", url, params=params))
            
            async def send_requests_for_links(search_request):
                try:
                    response = await client.send(search_request)
                    response.raise_for_status()  # TODO: error handling
                    html = response.text
                    links.extend(self._parse_search_page(html))
                except httpx.HTTPStatusError as e:
                    logger.error(
                        "Financial Times HTTP Error %d: %s", e.response.status_code, e.response.text
                    )
                    raise RequestSourceException(e.response.text)
                except httpx.RequestError as e:
                    logger.error("Error fetching Financial Times search page: %s", e)
                    raise RequestSourceException(e)
            
            await asyncio.gather(*map(send_requests_for_links, search_requests))

            logger.info("Fetched %d links from Financial Times on query %s", len(links), query)
            # FT articles has two types: regular articles that can be
            # scraped with default parser, and blogs where we just want to
            # extract the relevant blog portion
            article_links = []
            blog_links = []

            for link in links:
                if not link.startswith("https"):
                    link = f"https://www.ft.com{link}"
                if "#" in link:
                    blog_links.append(link)
                else:
                    article_links.append(link)

            # scrape articles
            logger.debug("Initialize Async WebScraper")
            scraper = WebScraper(async_client=client)

            logger.debug("Async scraping %d Financial Times articles", len(links))
            articles_data = await scraper.async_scrape_links(article_links)

            # scrape blogs
            # NOTE: not using the default parser. The custom parser takes an additional input
            # so i am scraping the link individually, rather than scraping the whole list
            scraper.set_html_parser(self._ft_blog_html_parser)

            # NOTE: need to use asyncio.gather to scrape the blog links, since we can't call on
            # the ascrape_links method while using a custom html parser with custom input
            blog_data = await asyncio.gather(
                *(
                    scraper.async_scrape_link(url=link, post_id=link.split("#")[1])
                    for link in blog_links
                )
            )
        finally:
            await client.aclose()

        # combine articles and blogs
        logger.debug("Converting data to Document Objects")
        documents = []
        for link, article in zip(article_links + blog_links, articles_data + blog_data):
            if article is None:
                # if both selenium scrape and httpx scrape fails, article will return None
                # in this case, we can't parse the metadata, so we skip
                continue
            metadata = self.parse_metadata(
                query=query,
                url=link,
                title=article["title"],
                publication_time=article["time"],
            )
            documents.append(Document(text=article["content"], metadata=metadata))

        # if document store is set, save documents to document store
        if self.document_store:
            logger.info("Saving %d documents to document store", len(documents))
            self.document_store.save_documents(documents)

        logger.info(
            "Successfully async fetched %d documents from Financial Times",
            len(documents),
        )
        return documents
    

class DirectoryData(BaseDataSource):
    def __init__(self, path: str):
        self.source = "Local Directory"
        dir = pathlib.Path(path)
        if not dir.is_dir():
            raise ValueError("Invalid path - must be a directory")
        self.dir = dir

    def fetch(self, query: str, num_results: int = None, **kwargs) -> List[Document]:
        """given path to data folder, fetch text files in 
        subdirectory with name matching query"""
        subdir = self.dir / query
        if not subdir.is_dir():
            raise ValueError(f"Subdirectory {subdir} does not exist")
        
        documents = []
        for file_path in subdir.rglob("*"):
            if file_path.suffix == ".txt":
                txt = file_path.read_text()

                metadata = self.parse_metadata(
                    query="NA",
                    name=file_path.name,
                    path=file_path.as_posix(),
                )
                documents.append(Document(text=txt, metadata=metadata))
            if file_path.suffix == ".pdf":
                pdf_text, pdf_meta = self.simple_pdf_parser(file_path.as_posix())
                metadata = self.parse_metadata(
                    query="query",
                    name=file_path.name,
                    path=file_path.as_posix(),
                    publication_time = pdf_meta.get("creation_date"),
                    title = pdf_meta.get("title")
                )
                documents.append(Document(text=pdf_text, metadata=metadata))

        return documents

    async def async_fetch(self, query: str, num_results: int = None, **kwargs) -> List[Document]:
        """N/A. Calls on sync. fetch function"""
        return self.fetch(query, num_results)

    @staticmethod
    def simple_pdf_parser(pdf_path: str) -> tuple[str, Dict[str, str]]:        
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        pdf_pages = []
        for page in reader.pages:
            # extract text
            extracted_text = page.extract_text()

            # process text
            processed_lines = []
            for line in extracted_text.split("\n"):
                line = line.strip()
                if not line:
                    # line is empty
                    continue
                processed_lines.append(line)

            pdf_pages.append("\n".join(processed_lines))
        
        pdf_text = "\n".join(pdf_pages)

        pdf_meta = reader.metadata or {}
        return pdf_text, pdf_meta


if __name__ == "__main__":
    # import json
    # with open(".ft-headers.json", "r")  as f:
    #     headers = json.load(f)
    # ft_source = FinancialTimesData(headers=headers)
    # documents = ft_source.fetch("uk inflation")
    # for doc in documents:
    #     print (doc)
    #     print('-'*20)

    lexis_source = LexisNexisData()
    documents = lexis_source.fetch("uk inflation")
    for doc in documents:
        print(doc)
        print("-" * 20)
