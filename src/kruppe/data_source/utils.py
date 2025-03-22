import asyncio
import httpx
from selenium import webdriver
from bs4 import BeautifulSoup
import time
from typing import List, Dict, AsyncGenerator
import logging
from pathlib import Path
from tqdm import tqdm
import stamina
import json
from functools import wraps

from kruppe.utils import log_io
from kruppe.data_source.news.base_news import NewsSource
from kruppe.models import Document

MAX_CONNECTIONS = 200
HTTPX_CONNECTION_LIMITS = httpx.Limits(max_keepalive_connections=20, max_connections=MAX_CONNECTIONS)
DEFAULT_HTTPX_TIMEOUT = httpx.Timeout(10.0, connect=60.0, pool=60.0) # yea i have no idea but this fixes PoolTimeout and ConnectTimeout error

# TODO: timeout error

logger = logging.getLogger(__name__)

class RequestSourceException(Exception):
    pass

def not_ready(func):
    func._not_ready = True
    @wraps(func)
    def wrapper(*args, **kwargs):
        raise NotImplementedError(f"{func.__name__} is not implemented")
    return wrapper

def is_method_ready(obj, method_name: str):
    method = getattr(obj, method_name)
    return not getattr(method, "_not_ready", False)

async def combine_async_generators(async_gens):
    queue = asyncio.Queue()

    async def producer(agen):
        async for item in agen:
            await queue.put(item)
        await queue.put(None)

    tasks = [asyncio.create_task(producer(agen)) for agen in async_gens]
    finished = 0
    total = len(async_gens)
    
    while finished < total:
        item = await queue.get()
        if item is None:
            finished += 1
        else:
            yield item
    
    await asyncio.gather(*tasks)

def retry_on_httpx_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        logger.warning("HTTP ERROR %d: %s", exc.response.status_code, exc.response.text)
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    
    elif isinstance(exc, httpx.PoolTimeout):
        # see my notes on "error debugging" on how to deal with this
        logger.warning("POOL TIMEOUT")
    else:
        print("ERROR", repr(exc)) # to see what error it is
    return isinstance(exc, httpx.HTTPError)

def load_headers(header_path: str):
    # Resolve the path to header.json which is outside of the src folder.
    return json.loads(Path(header_path).read_text())

class WebScraper:
    """Scrapes list of websites using aiohttp or requests, and fallsback to selenium if aiohttp fails"""

    def __init__(
        self,
        headless=True,
        html_parser: callable = None,
        sync_client: httpx.AsyncClient = None, # NOTE: this is not used
        async_client: httpx.AsyncClient = None,
    ):
        self.headless = headless
        self.html_parser = html_parser or self.default_html_parser
        self.sync_client = sync_client # NOTE: this is not used
        self.async_client = async_client

        # set timeout all to the same cuz im lazy im gonna fix this later
        if async_client:
            self.async_client.timeout = DEFAULT_HTTPX_TIMEOUT

    @staticmethod
    def default_html_parser(html: str, url: str, **kwargs) -> Dict[str, str]:
        """Scrape html and return a dict with different information scraped from the site"""

        if html is None:
            return None

        # parse html
        soup = BeautifulSoup(html, "lxml")

        # scrape title
        title_tag = soup.find("meta", property="og:title")
        title = title_tag.get("content") if title_tag else None

        # scrape description
        description_tag = soup.find("meta", property="og:description")
        description = description_tag.get("content") if description_tag else None

        # scrape article content
        text = "\n".join([p.get_text(" ", strip=True) for p in soup.find_all("p")])

        # time
        time_tag = soup.find("meta", property="article:published_time")
        publication_time = time_tag.get("content") if time_tag else None

        # document type
        document_type_tag = soup.find("meta", property="og:type")
        document_type = document_type_tag.get("content") if document_type_tag else None

        # tags
        tags_tag = soup.find("meta", property="article:tag")
        tags = tags_tag.get("content") if tags_tag else None

        # author
        author_tag = soup.find("meta", property="article:author")
        author = author_tag.get("content") if author_tag else None

        # putting all the scraped information into a dict
        scraped_data = {
            "content": text,
            "meta": {
                "url": url,
                "title": title,
                "description": description,
                "publication_time": publication_time,
                "document_type": document_type,
                "tags": tags,
                "author": author,
            }
        }

        return scraped_data

    def set_html_parser(self, html_parser: callable = None):
        """
        Set a custom HTML parser for the WebScraper instance.

        Args:
            html_parser (callable, optional): A callable that takes an HTML string
            and returns a dictionary with parsed information. If None, the default
            HTML parser will be used.
        """
        if html_parser is not None:
            # Ensure the custom HTML parser has "html" as one of its arguments
            parser_args = html_parser.__code__.co_varnames
            if "html" not in parser_args:
                raise ValueError("The custom HTML parser must accept 'html' as an argument.")
            
            self.html_parser = html_parser
        else:
            self.html_parser = self.default_html_parser

    def _create_selenium_driver(self):
        """Create and configure the Selenium WebDriver."""
        chrome_options = webdriver.ChromeOptions()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # should not need to set driver instance for Selenium version >= 4.6
        chrome_service = webdriver.ChromeService()

        driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

        return driver

    def _scrape_with_selenium(self, driver, url: str, **kwargs) -> str | None:
        """Fallback to Selenium to scrape the page."""
        logging.debug("Attempting to scrape %s with Selenium", url)

        try:
            driver.get(url)
            time.sleep(2)  # Give the page time to load
            page_content = driver.page_source
            logging.info("Successfully scraped %s with Selenium", url)

            return page_content
        except Exception:
            logging.warning("Failed to scrape %s with Selenium", url)
            return None

    

    @stamina.retry(on=retry_on_httpx_error, attempts=3)
    async def _async_scrape_with_httpx(self, client: httpx.AsyncClient, url: str, **kwargs) -> str:
        r = await client.get(url)
        r.raise_for_status()

        # if the link is a pdf, return None
        if r.headers.get("Content-Type") == "application/pdf":
            logger.warning("Url %s is a pdf. Skipping.", url)
            return None

        html = r.text
        return html


    @log_io
    async def async_scrape_link(
        self,
        url: str,
        driver: webdriver = None,
        headers: Dict[str, str] = None,
        **kwargs,
    ) -> Dict[str, str] | None:
        """Async function for scraping a single link"""
        logger.info("Async scraping %s", url)

        # init async client
        if self.async_client is None:
            client = httpx.AsyncClient(headers=headers)
            logging.info("Inititalized new async client")
        else:
            client = self.async_client
            logging.info("Using entered async client")
        
        # scrape
        html = None
        try:
            # try with httpx first (faster)
            html = await self._async_scrape_with_httpx(client, url, **kwargs)
            
        except httpx.HTTPError as exc:
            logger.warning("Error occurred while scraping %s: %s", url, repr(exc))

            # if httpx doesn't work, fallback to selenium if it exists
            if driver:
                logger.info("Attempting to scrape %s with Selenium", url)
                html = self._scrape_with_selenium(driver, url, **kwargs)
            else:
                return None
        finally:
            # close async client
            if self.async_client is None:
                await client.aclose()
                logging.debug("Closed async client")

            # if scraping returns a non-empty document
            if html:
                kwargs["url"] = url
                data = self.html_parser(html, **kwargs)

                if data is None:
                    return None
                else:
                    # this is for if i scrape through html but
                    # even if html has text, i didn't scrape out anything useful
                    if "content" in data and not data["content"]:
                        return None
                    return data


            

    async def async_scrape_links(
        self,
        links: List[str],
        headers: Dict[str, str] = None,
        selenium_fallback: bool = True,
        progress_bar: bool = False,
    ) -> AsyncGenerator[Dict[str, str], None]:
        """Scrape multiple links and returns a list of dicts (scraped info). Returns None if neither method works.

        Args:
            links: List of URLs to scrape
            Additional arguments to pass to the html_parser function
        """

        logger.debug("Async scraping %d links", len(links))

        logger.debug("Creating Selenium driver")
        driver = self._create_selenium_driver() if selenium_fallback else None
        if self.async_client is None:
            client = httpx.AsyncClient(headers=headers, limits=HTTPX_CONNECTION_LIMITS)
        else:
            client = self.async_client

        if progress_bar: 
            pbar = tqdm(total=len(links), desc="Async scraping links")
        try:

            # # NOTE: I do NOT know if I need to do things in batches. 
            # # I think httpx handles it for me, but past experience tells me now.
            # results = []
            # for i in range(0, len(links), HTTPX_CONNECTION_LIMITS.max_connections):
            #     tasks = [
            #         self.async_scrape_link(url=link, driver=driver)
            #         for link in links[i : i + HTTPX_CONNECTION_LIMITS.max_connections]
            #     ]
            #     results.extend(await asyncio.gather(*tasks))
            #     pbar.update(len(tasks))
            # return results

            tasks = [self.async_scrape_link(url=link, driver=driver) for link in links]
            for completed in asyncio.as_completed(tasks):
                if progress_bar:
                    pbar.update(1)
                data = await completed
                if data is not None:
                    yield data

        finally:
            if self.async_client is None:
                await client.aclose()

            if driver:
                driver.quit()
                logger.debug("Closed Selenium driver")
            
            if progress_bar:
                pbar.close()



class NewsArticleSearcher:
    def __init__(self, sources: List[NewsSource]):
        """Chooses which sources to use for fetching documents"""
        self.sources = sources  # TODO: do some kind of "setting" here to determine which sources to use
        self.documents = []

    def search(self, query: str, num_results: int = 10, **kwargs) -> List[Document]:
        documents = []
        for source in self.sources:
            try:
                fetched_documents = source.fetch(
                    query, num_results=num_results, **kwargs
                )
                documents.extend(fetched_documents[:num_results])
            except RequestSourceException as e:
                logging.error(
                    "Error occurred while fetching documents from %s: %s",
                    source.__class__.__name__,
                    str(e),
                )

        logging.info("Fetched %d documents from sources", len(documents))
        self.documents.extend(documents)
        return documents

    async def async_search(
        self, query: str, num_results: int = 10, **kwargs
    ) -> List[Document]:
        async def _async_fetch(async_fetch: callable):
            try:
                return await async_fetch(query, num_results=num_results, **kwargs)
            except RequestSourceException as e:
                logging.error(
                    "Error occurred while fetching documents from %s: %s",
                    async_fetch.__name__,
                    str(e),
                )
                return []

        documents = []
        results = await asyncio.gather(
            *[_async_fetch(source.async_fetch) for source in self.sources]
        )
        for result in results:
            documents.extend(result[:num_results])
        self.documents.extend(documents)
        return documents

    def export_documents(self, directory: str, create_dir: bool) -> List[Document]:
        """Export documents into directory as txt files, with a json file
        containing each document's metadata"""

        dir_path = Path(directory)
        if not dir_path.exists():
            if create_dir:
                dir_path.mkdir(parents=True)
            else:
                raise ValueError(
                    "Directory does not exist. Set create_dir=True to create it."
                )

        metadata = {}
        for i, document in enumerate(self.documents):
            file_path = dir_path / f"{i}.txt"
            file_path.write_text(document.text)
            metadata[i] = document.metadata

        import json

        metadata_path = dir_path / "metadata.json"
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=4)
        return self.documents


# Example usage
if __name__ == "__main__":
    scraper = WebScraper()
    print(scraper.random())
