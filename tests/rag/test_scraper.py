import pytest
from aiohttp import ClientSession

from rag.scraper import WebScraper

# Arrange
@pytest.fixture
def scraper():
    return WebScraper()

@pytest.mark.asyncio(loop_scope="session")
async def test_scrape(scraper, session):
    """test overall scraper"""
    urls = [
        "https://en.wikipedia.org/wiki/Cleveland_Guardians",
        "https://python.langchain.com/docs/concepts/#prompt-templates"
    ]

    # test async version
    data = await scraper.async_scrape_links(session, urls)

    assert len(urls) == len(data)
    assert any(data) # text is not empty

    # test nonasync version
    data = scraper.scrape_links(urls)
    assert len(urls) == len(data)
    assert any(data)

@pytest.mark.asyncio(loop_scope="session")
async def test_scrape_aiohttp(scraper, session):
    """test scraping with aiohttp"""

    url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"

    data = await scraper._scrape_with_aiohttp(session, url)
    assert data

def test_scrape_selenium(scraper):
    """test scraping with selenium"""

    url = "https://en.wikipedia.org/wiki/Vector_database"

    driver = scraper._create_selenium_driver()
    data = scraper._scrape_with_selenium(driver, url)
    driver.quit()
    assert data


def test_scrape_requests(scraper):
    """test scraping with selenium"""

    url = "https://en.wikipedia.org/wiki/Vector_database"

    data = scraper._scrape_with_requests(url)
    assert data
