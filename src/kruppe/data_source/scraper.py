import asyncio
import httpx
from selenium import webdriver
from bs4 import BeautifulSoup
import time
from typing import List, Dict, AsyncGenerator
import logging
from pathlib import Path
import stamina
import json

from kruppe.common.log import log_io

logger = logging.getLogger(__name__)

HTTPX_CONNECTION_LIMITS = httpx.Limits(max_keepalive_connections=20, max_connections=300)
HTTPX_TIMEOUT = httpx.Timeout(5.0, connect=10.0, pool=None) # yea i have no idea but this fixes PoolTimeout and ConnectTimeout error

semaphore = asyncio.Semaphore(225) # Limit concurrent requests to avoid overwhelming the server
# Note: best combination i've found with httpx request is 225 semaphore, 300 max_connections

class RequestSourceException(Exception):
    pass

def retry_on_httpx_error(exc: Exception) -> bool:
    # acceptable client errors
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code == 429 or exc.response.status_code >= 500:
            logger.warning(f"RETRY ON HTTP ERROR {exc.response.status_code}")
            return True  # Retry on 429 and 5xx errors
        elif exc.response.status_code == 408:
            logger.warning(f"RETRY ON HTTP TIMEOUT ERROR {exc.response.status_code}")
            return True
        else:
            # use wrapper function to log the error
            return False
    
    # network error
    if isinstance(exc, httpx.NetworkError):
        logger.warning(f"RETRY ON CONNECT ERROR: {repr(exc)}")
        # This includes connection errors like DNS failure, refused connection, etc.
        return True  # Retry on connection errors
    
    if isinstance(exc, httpx.PoolTimeout):
        logger.warning(f"RETRY ON POOL TIMEOUT ERROR: {repr(exc)}")
        # see my notes on "error debugging" on how to deal with this Pool Connect Error
        # PoolTimeout indicates that the connection pool is exhausted (waited for too long before sending request)
        return True  # Retry on pool timeout errors
    elif isinstance(exc, httpx.TimeoutException):
        # on all other timeout errors, including connect timeout and read timeout
        # do not retry
        logger.warning(f"TIMEOUT ERROR {type(exc).__name__}: {repr(exc)}", )
        return False  # Retry on timeout errors
    
    logger.error(f"ERROR {repr(exc)}") # to see what error it is
    return False

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
        # also set all httpx connection limit to
        if async_client:
            self.async_client.timeout = HTTPX_TIMEOUT
            self.async_client.limits = HTTPX_CONNECTION_LIMITS

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
        logger.debug("[Selenium] Attempting to scrape %s ", url)

        try:
            driver.get(url)
            time.sleep(2)  # Give the page time to load
            page_content = driver.page_source
            logger.debug("[Selenium] Successfully scraped %s", url)

            return page_content
        except Exception:
            logger.warning("[Selenium] Failed to scrape %s", url)
            return None

    

    @stamina.retry(on=retry_on_httpx_error, attempts=3)
    async def _async_scrape_with_httpx(self, client: httpx.AsyncClient, url: str, **kwargs) -> str:
        
        async with semaphore:  # Use semaphore to limit concurrent requests
            logger.debug("[httpx] Attempting to scrape %s", url)
            r = await client.get(url)

        
        r.raise_for_status()

        # if the link is a pdf, return None
        if r.headers.get("Content-Type") == "application/pdf":
            logger.warning(f"[httpx] Url {url} is a pdf. Skipping.")
            return None

        html = r.text
        logger.debug("[httpx] Successfully scraped %s", url)
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

        # init async client
        if self.async_client is None:
            client = httpx.AsyncClient(headers=headers)
            logger.debug("Inititalized new async httpx client in async_scrape_link")
        else:
            client = self.async_client
        
        # scrape
        html = None
        try:
            # try with httpx first (faster)
            html = await self._async_scrape_with_httpx(client, url, **kwargs)
            
        except httpx.HTTPError as exc:
            logger.warning("[httpx] Failed to scrape %s : %s", url, repr(exc))

            # if httpx doesn't work, fallback to selenium if it exists
            if driver:
                html = self._scrape_with_selenium(driver, url, **kwargs)
            else:
                return None
        except Exception as exc:
            # NOTE: this is a catch-all for any other exceptions that may occur
            # could be dangerous
            logger.error("[httpx] Unexpected error while scraping %s: %s", url, repr(exc))
            return None
        finally:
            # close async client
            if self.async_client is None:
                await client.aclose()
                logger.debug("Closed async client")

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
        selenium_fallback: bool = False, # turn this one on if you want to use selenium, but will be slow
        progress_bar: bool = False,
    ) -> AsyncGenerator[Dict[str, str], None]:
        """Scrape multiple links and returns a list of dicts (scraped info). Returns None if neither method works.

        Args:
            links: List of URLs to scrape
            Additional arguments to pass to the html_parser function
        """

        logger.debug("Async scraping %d links", len(links))
        
        if selenium_fallback:
            logger.debug("Creating Selenium driver in async_scrape_links")
        driver = self._create_selenium_driver() if selenium_fallback else None
        
        if self.async_client is None:
            logger.debug("Initializing new async httpx client in async_scrape_links")
            client = httpx.AsyncClient(headers=headers, limits=HTTPX_CONNECTION_LIMITS)
        else:
            client = self.async_client

        try:
            coros = [self.async_scrape_link(url=link, driver=driver) for link in links]
            async for completed in asyncio.as_completed(coros):
                data = await completed
                if data is not None:
                    yield data

        finally:
            if self.async_client is None:
                await client.aclose()
                logger.debug("Closed async httpx client in async_scrape_links")

            if driver:
                driver.quit()
                logger.debug("Closed Selenium driver in async_scrape_links")



# Example usage
if __name__ == "__main__":
    scraper = WebScraper()
    print(scraper.random())
