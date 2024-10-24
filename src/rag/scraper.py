import asyncio
import requests
import aiohttp
from aiohttp import ClientError, ClientSession
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
    
    def _scrape_with_requests(self, url: str):
        """Scrape the page using requests."""
        try:
            response = requests.get(url, timeout=10)  # Set a timeout to avoid hanging
            if response.status_code == 200:
                return self._scrape_html(response.text)
            else:
                raise requests.exceptions.RequestException(f"Failed to fetch {url} with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error occurred while scraping {url} with requests: {e}")
            return None

    def _scrape_with_selenium(self, driver, url: str):
        """Fallback to Selenium to scrape the page."""
        
        try:
            driver.get(url)
            time.sleep(2)  # Give the page time to load
            page_content = driver.page_source
            return self._scrape_html(page_content)
        except Exception as e:
            print(f"Error occurred while scraping {url} with Selenium: {e}")
            return None
    
    async def _scrape_with_aiohttp(self, session: ClientSession, url: str):
        """Scrape the page using aiohttp."""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    return self._scrape_html(html)
                else:
                    raise ClientError(f"Failed to fetch {url} with status {response.status}")
        except ClientError as e:
            print(f"Error occurred while scraping {url} with aiohttp: {e}")
            return None

    async def async_scrape_links(self, session: ClientSession, links: List[str]):
        """Scrape multiple links asynchronously, and returns list of strings. Returns None if neither works."""
        
        async def _scrape_link(session: ClientSession, driver, url: str):
            """Helper method, try aiohttp, fallback to Selenium if aiohttp fails."""
            page_content = await self._scrape_with_aiohttp(session, url)
            if page_content is None:
                print(f"Falling back to Selenium for {url}")
                page_content = self._scrape_with_selenium(driver, url)
            return page_content
        
        driver = self._create_selenium_driver() # init selenium driver
        try:
            tasks = []
            for link in links:
                task = _scrape_link(session, driver, link)
                tasks.append(task)
            return await asyncio.gather(*tasks)
        finally:
            driver.quit() # quit selenium driver
    
    def scrape_links(self, links: List[str]):
        """Scrape multiple links and return a list of strings. Returns None if neither method works."""
        
        def _scrape_link(driver, url: str):
            """Try requests, fallback to Selenium if requests fail."""
            page_content = self._scrape_with_requests(url)
            if page_content is None:
                print(f"Falling back to Selenium for {url}")
                page_content = self._scrape_with_selenium(driver, url)
            return page_content

        driver = self._create_selenium_driver()  # Initialize Selenium driver
        results = []
        try:
            for link in links:
                result = _scrape_link(driver, link)
                results.append(result)
            return results
        finally:
            driver.quit()  # Quit Selenium driver


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
