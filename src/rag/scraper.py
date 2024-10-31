import asyncio
import httpx
from selenium import webdriver
from bs4 import BeautifulSoup
import time

from typing import List

# TODO: timeout error

class WebScraper:
    """Scrapes list of websites using aiohttp or requests, and fallsback to selenium if aiohttp fails"""

    def __init__(self, headless=True):
        self.headless = headless
    
    @staticmethod
    def _scrape_html(html: str) -> str:
        """scrape with bs4, get all p tags content"""
        soup = BeautifulSoup(html, "lxml")
        text = "\n".join([p.get_text(" ", strip=True) for p in soup.find_all("p")])

        return text
    
    def _create_selenium_driver(self):
        """Create and configure the Selenium WebDriver."""
        chrome_options = webdriver.ChromeOptions()
        if self.headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')

        # should not need to set driver instance for Selenium version >= 4.6
        chrome_service = webdriver.ChromeService()

        driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
        
        return driver

    def _scrape_with_selenium(self, driver, url: str):
        """Fallback to Selenium to scrape the page."""
        
        try:
            driver.get(url)
            time.sleep(2)  # Give the page time to load
            page_content = driver.page_source
            return page_content
        except Exception as e:
            print(f"Error occurred while scraping {url} with Selenium: {e}")
            return None
    
    
    async def async_scrape_links(self, links: List[str]) -> List[str]:
        async def _scrape_link(client: httpx.AsyncClient, driver, url: str) -> str:
            """internal helper function for scraping a single link"""
            r = await client.get(url)
            try:
                r.raise_for_status()
                html = r.text
                return self._scrape_html(html)
            except httpx.HTTPError as exc:
                print(f"Error while requesting {exc.request.url!r}.")
                print(f"Falling back to Selenium for {url}")
                html = self._scrape_with_selenium(driver, url)
                return self._scrape_html(html)
        
        driver = self._create_selenium_driver()
        try:
            async with httpx.AsyncClient() as client:
                tasks = [_scrape_link(client, driver, link) for link in links]
                return await asyncio.gather(*tasks)
        finally:
            driver.quit()
            

    
    def scrape_links(self, links: List[str]) -> List[str]:
        """Scrape multiple links and return a list of strings. Returns None if neither method works."""
        
        def _scrape_link(client: httpx.Client, driver, url: str) -> str:
            """Try requests, fallback to Selenium if requests fail."""
            r = client.get(url)
            try:
                r.raise_for_status()
                html = r.text
                return self._scrape_html(html)
            except httpx.HTTPError as exc:
                print(f"Error while requesting {exc.request.url!r}.")
                print(f"Falling back to Selenium for {url}")
                html = self._scrape_with_selenium(driver, url)
                return self._scrape_html(html)
        
        driver = self._create_selenium_driver()
        try:
             with httpx.Client() as client:
                return [_scrape_link(client, driver, link) for link in links]
        finally:
            driver.quit()


# Example usage
if __name__ == "__main__":
    async def main():        
        scraper = WebScraper()
        
        # Links to scrape
        links = [
            "https://en.wikipedia.org/wiki/Cleveland_Guardians",
            "https://python.langchain.com/docs/concepts/#prompt-templates"
        ]
        
        # Run the scraping
        scraped_data = await scraper.scrape_links(links)
        
        # Output the results
        for i, data in enumerate(scraped_data):
            if data:
                print(f"Content for {links[i]}:\n{data[:500]}...\n")
    
    # Run the async event loop
    asyncio.run(main())
