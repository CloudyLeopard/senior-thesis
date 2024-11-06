import asyncio
import httpx
from selenium import webdriver
from bs4 import BeautifulSoup
import time

from typing import List, Dict

# TODO: timeout error


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
        if soup.title:
            title = soup.title.get_text(" ", strip=True)

        # scrape article content
        text = "\n".join([p.get_text(" ", strip=True) for p in soup.find_all("p")])

        # time
        time = soup.find("time")
        if time:
            # get the "datetime" from the time attribute.
            # if DNE, fall back to just the text shown
            time = time.get("datetime") or time.get_text()

        # putting all the scraped information into a dict
        scraped_data = {"title": title or "", "content": text, "time": time or ""}

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
        print(url)
        """Fallback to Selenium to scrape the page."""

        try:
            driver.get(url)
            time.sleep(2)  # Give the page time to load
            page_content = driver.page_source
            return page_content
        except Exception as e:
            print(f"Error occurred while scraping {url} with Selenium: {e}")
            return None

    async def async_scrape_link(
        self,
        url: str,
        driver: webdriver = None,
        headers: Dict[str, str] = None,
        *args,
        **kwargs,
    ) -> Dict[str, str] | None:
        """Async function for scraping a single link"""

        if self.async_client is None:
            client = httpx.AsyncClient(headers=headers)
        else:
            client = self.async_client

        r = await client.get(url)
        try:
            r.raise_for_status()
            if r.headers.get("Content-Type") == "application/pdf":
                # if the link is a pdf, return None
                return None

            html = r.text
            return self.html_parser(html, *args, **kwargs)
        except httpx.HTTPError as exc:
            if driver:
                print(f"Error while requesting {exc.request.url!r}.")
                print(f"Falling back to Selenium for {url}")
                html = self._scrape_with_selenium(driver, url)
                if html is None:
                    return None
                return self.html_parser(html, *args, **kwargs)
            else:
                print(f"Error while requesting {exc.request.url!r}.")
                print(f"No Selenium driver provided. Skipping {url}.")
                return None
        finally:
            if self.async_client is None:
                await client.aclose()

    async def async_scrape_links(
        self, links: List[str], headers: Dict[str, str] = None
    ) -> List[Dict[str, str]] | None:
        """Scrape multiple links and returns a list of dicts (scraped info). Returns None if neither method works.

        Args:
            links: List of URLs to scrape
            Additional arguments to pass to the html_parser function
        """

        driver = self._create_selenium_driver()
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
            driver.quit()

    def scrape_link(
        self,
        url: str,
        driver: webdriver = None,
        headers: Dict[str, str] = None,
        *args,
        **kwargs,
    ):
        """Try requests, fallback to Selenium if requests fail."""
        if self.sync_client is None:
            client = httpx.Client(headers=headers)
        else:
            client = self.sync_client

        r = client.get(url)
        try:
            r.raise_for_status()
            if r.headers.get("Content-Type") == "application/pdf":
                # if the link is a pdf, return None
                return None

            html = r.text
            return self.html_parser(html, *args, **kwargs)
        except httpx.HTTPError as exc:
            if driver:
                print(f"Error while requesting {exc.request.url!r}.")
                print(f"Falling back to Selenium for {url}")
                html = self._scrape_with_selenium(driver, url)
                if html is None:
                    return None
                return self.html_parser(html, *args, **kwargs)
            else:
                print(f"Error while requesting {exc.request.url!r}.")
                print(f"No Selenium driver provided. Skipping {url}.")
                return None
        finally:
            if self.sync_client is None:
                client.close()

    def scrape_links(
        self, links: List[str], headers: Dict[str, str] = None
    ) -> List[Dict[str, str]]:
        """Scrape multiple links and returns a list of dicts (scraped info). Returns None if neither method works."""

        if self.sync_client is None:
            client = httpx.Client(headers=headers)
        else:
            client = self.sync_client

        driver = self._create_selenium_driver()
        try:
            return [self.scrape_link(url=link, driver=driver) for link in links]
        finally:
            if self.sync_client is None:
                client.close()
            driver.quit()


# Example usage
if __name__ == "__main__":

    async def main():
        scraper = WebScraper()

        # Links to scrape
        links = [
            "https://en.wikipedia.org/wiki/Cleveland_Guardians",
            "https://python.langchain.com/docs/concepts/#prompt-templates",
        ]

        # Run the scraping
        scraped_data = await scraper.scrape_links(links)

        # Output the results
        for i, data in enumerate(scraped_data):
            if data:
                print(f"Content for {links[i]}:\n{data[:500]}...\n")

    # Run the async event loop
    asyncio.run(main())
