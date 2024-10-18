import asyncio
import aiohttp
from aiohttp import ClientError, ClientSession
from selenium import webdriver
from bs4 import BeautifulSoup
import time

from typing import List
class AsyncWebScraper:
    """Scrapes list of websites using aiohttp, and fallsback to selenium if aiohttp fails"""

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

    def _scrape_with_selenium(self, url: str):
        """Fallback to Selenium to scrape the page."""
        driver = self._create_selenium_driver() # only initialize selenium when needed
        try:
            driver.get(url)
            time.sleep(2)  # Give the page time to load
            page_content = driver.page_source
            return self._scrape_html(page_content)
        except Exception as e:
            print(f"Error occurred while scraping {url} with Selenium: {e}")
            return None
        finally:
            driver.quit()

    async def _scrape_link(self, session: ClientSession, url: str):
        """Try aiohttp, fallback to Selenium if aiohttp fails."""
        page_content = await self._scrape_with_aiohttp(session, url)
        if page_content is None:
            print(f"Falling back to Selenium for {url}")
            page_content = self._scrape_with_selenium(url)
        return page_content

    async def scrape_links(self, links: List[str]):
        """Scrape multiple links asynchronously, and returns list of strings. Returns None if neither works."""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for link in links:
                task = self._scrape_link(session, link)
                tasks.append(task)
            return await asyncio.gather(*tasks)

# Example usage
if __name__ == "__main__":
    async def main():        
        scraper = AsyncWebScraper()
        
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
