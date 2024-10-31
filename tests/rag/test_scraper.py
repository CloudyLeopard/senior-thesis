import pytest

from rag.scraper import WebScraper

# Arrange
@pytest.fixture
def scraper():
    return WebScraper()

@pytest.mark.asyncio(loop_scope="session")
async def test_scrape(scraper):
    """test overall scraper"""
    urls = [
        "https://en.wikipedia.org/wiki/Cleveland_Guardians",
        "https://python.langchain.com/docs/concepts/#prompt-templates"
    ]

    # test async version
    data = await scraper.async_scrape_links(urls)

    assert len(urls) == len(data)
    assert all(data) # text is not empty

    # test nonasync version
    data = scraper.scrape_links(urls)
    assert len(urls) == len(data)
    assert all(data)

def test_scrape_selenium(scraper):
    """test scraping with selenium"""

    url = "https://en.wikipedia.org/wiki/Vector_database"

    driver = scraper._create_selenium_driver()
    data = scraper._scrape_with_selenium(driver, url)
    driver.quit()
    assert data
