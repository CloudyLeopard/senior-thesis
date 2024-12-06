import asyncio
import httpx
from selenium import webdriver
from bs4 import BeautifulSoup
import time
from typing import List, Dict
import logging

# TODO: timeout error

logger = logging.getLogger(__name__)

class WebScraper:
    """Scrapes list of websites using aiohttp or requests, and fallsback to selenium if aiohttp fails"""

    def __init__(
        self,
        headless=True,
        html_parser: callable = None,
        sync_client: httpx.AsyncClient = None,
        async_client: httpx.AsyncClient = None,
    ):
        self.headless = headless
        self.html_parser = html_parser or self.default_html_parser
        self.sync_client = sync_client
        self.async_client = async_client

    @staticmethod
    def default_html_parser(html: str) -> Dict[str, str]:
        """Scrape html and return a dict with different information scraped from the site"""

        # parse html
        soup = BeautifulSoup(html, "lxml")

        # scrape title

        title = soup.title.get_text(" ", strip=True) if soup.title else ""

        # scrape article content
        text = "\n".join([p.get_text(" ", strip=True) for p in soup.find_all("p")])

        # time
        time = soup.find("time")
        if time:
            # get the "datetime" from the time attribute.
            # if DNE, fall back to just the text shown
            time = time.get("datetime") or time.get_text()

        # putting all the scraped information into a dict
        scraped_data = {"title": title, "content": text, "time": time or ""}

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

    def _scrape_with_selenium(self, driver, url: str):
        """Fallback to Selenium to scrape the page."""
        logging.debug("Attempting to scrape %s with Selenium", url)
        try:
            driver.get(url)
            time.sleep(2)  # Give the page time to load
            page_content = driver.page_source
            logging.info("Successfully scraped %s with Selenium", url)
            return page_content
        except Exception as e:
            logging.error("Error occurred while scraping %s with Selenium: %s", url, str(e))
            return None

    async def async_scrape_link(
        self,
        url: str,
        driver: webdriver = None,
        headers: Dict[str, str] = None,
        retries: int = 3,
        backoff: int = 0.5,
        **kwargs,
    ) -> Dict[str, str] | None:
        """Async function for scraping a single link"""
        logger.debug("Async scraping %s", url)

        if self.async_client is None:
            client = httpx.AsyncClient(headers=headers)
            logging.debug("Inititalized async client")
        else:
            client = self.async_client
        for attempt in range(retries):
            try:
                r = await client.get(url)
                r.raise_for_status()
                if r.headers.get("Content-Type") == "application/pdf":
                    # if the link is a pdf, return None
                    logger.warning("Url %s is a pdf. Skipping.", url)
                    return None

                html = r.text
                logger.info("Successfully async scraped %s", url)
                return self.html_parser(html, **kwargs)
            except httpx.HTTPError as exc:
                if attempt < retries - 1:
                    await asyncio.sleep(backoff * (2 ** attempt))
                    continue

                logger.warning("Async httpx request failed %s after %d attempts", exc.request.url, retries)
                if driver:
                    html = self._scrape_with_selenium(driver, url)
                    if html is None:
                        return None
                    return self.html_parser(html, **kwargs)
                else:
                    logger.warning("No Selenium driver provided. Skipping.")
                    return None
            except httpx.ConnectTimeout as exc:
                logger.error("Timeout error while scraping %s", exc.request.url)
                return None
            except Exception as exc:
                logger.error("Unexpected error while scraping %s: %s", url, str(exc))
                return None
            finally:
                if self.async_client is None:
                    await client.aclose()
                    logging.debug("Closed async client")

    async def async_scrape_links(
        self, links: List[str], headers: Dict[str, str] = None, selenium_fallback: bool = True
    ) -> List[Dict[str, str]] | None:
        """Scrape multiple links and returns a list of dicts (scraped info). Returns None if neither method works.

        Args:
            links: List of URLs to scrape
            Additional arguments to pass to the html_parser function
        """

        logger.debug("Async scraping %d links", len(links))

        logger.debug("Creating Selenium driver")
        driver = self._create_selenium_driver() if selenium_fallback else None
        if self.async_client is None:
            client = httpx.AsyncClient(headers=headers)
        else:
            client = self.async_client

        try:
            return await asyncio.gather(
                *[self.async_scrape_link(url=link, driver=driver) for link in links]
            )
        finally:
            if self.async_client is None:
                await client.aclose()
            
            if driver:
                driver.quit()
                logger.debug("Closed Selenium driver")

    def scrape_link(
        self,
        url: str,
        driver: webdriver = None,
        headers: Dict[str, str] = None,
        retries: int = 3,
        backoff: int = 0.5,
        **kwargs,
    ):
        """Try requests, fallback to Selenium if requests fail."""
        logger.debug("Scraping %s", url)

        if self.sync_client is None:
            client = httpx.Client(headers=headers)
        else:
            client = self.sync_client

        for attempt in range(retries):
            try:
                r = client.get(url)
                r.raise_for_status()
                if r.headers.get("Content-Type") == "application/pdf":
                    # if the link is a pdf, return None
                    logger.warning("Url %s is a pdf. Skipping.", url)
                    return None

                html = r.text
                logger.info("Successfully scraped %s", url)
                return self.html_parser(html, **kwargs)
            except httpx.HTTPError as exc:
                if attempt < retries - 1:
                    time.sleep(backoff * (2 ** attempt))
                    continue

                logger.warning("httpx request failed %s after %d attempts", exc.request.url, retries)
                if driver:
                    html = self._scrape_with_selenium(driver, url)
                    if html is None:
                        return None
                    return self.html_parser(html, **kwargs)
                else:
                    logger.warning("No Selenium driver provided. Skipping.")
                    return None
            except httpx.ConnectTimeout as exc:
                logger.error("Timeout error while scraping %s", exc.request.url)
                return None
            except Exception as exc:
                logger.error("Unexpected error while scraping %s: %s", url, str(exc))
                return None
            finally:
                if self.sync_client is None:
                    client.close()

    def scrape_links(
        self, links: List[str], headers: Dict[str, str] = None, selenium_fallback: bool = True,
    ) -> List[Dict[str, str]]:
        """Scrape multiple links and returns a list of dicts (scraped info). Returns None if neither method works."""
        logger.debug("Scraping %d links", len(links))

        if self.sync_client is None:
            client = httpx.Client(headers=headers)
        else:
            client = self.sync_client


        driver = self._create_selenium_driver() if selenium_fallback else None
        try:
            return [self.scrape_link(url=link, driver=driver) for link in links]
        finally:
            if self.sync_client is None:
                client.close()
            if driver:
                driver.quit()
                logger.debug("Closed Selenium driver")


# Example usage
if __name__ == "__main__":
    scraper = WebScraper()
    print(scraper.random())
