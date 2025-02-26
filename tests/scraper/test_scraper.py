import pytest
import pytest_asyncio

from kruppe.scraper.utils import WebScraper
import httpx
import asyncio


@pytest_asyncio.fixture(params=[True, False])
async def scraper(request):
    """test scraper with inputted clients and not inputted clients"""
    if request.param:
        async_client = httpx.AsyncClient()
    else:
        async_client = None

    yield WebScraper(async_client = async_client)

    if request.param:
        await async_client.aclose()

@pytest.fixture
def urls():
    return [
        "https://en.wikipedia.org/wiki/Cleveland_Guardians",
        "https://python.langchain.com/docs/concepts/#prompt-templates",
    ]


@pytest.mark.asyncio(loop_scope="session")
async def test_async_scrape_links(scraper, urls):
    """test overall scraper"""
    # test async version
    async for data in scraper.async_scrape_links(urls):
        assert data
        assert "content" in data
        assert "meta" in data
        assert len(data["meta"]) > 0

@pytest.mark.asyncio(loop_scope="session")
async def test_async_scrape_link(scraper, urls):
    data = await asyncio.gather(*(scraper.async_scrape_link(url) for url in urls))
    
    assert len(urls) == len(data)
    assert all(data)


@pytest.mark.asyncio(loop_scope="session")
async def test_scrape_custom_html_parser(scraper):
    """test custom html parser"""
    url = "https://en.wikipedia.org/wiki/Vector_database"
    def custom_parser(html: str, random_str: str):
        return {"title": "some title", "text": html, "random_str": random_str}
    
    scraper.set_html_parser(custom_parser)
    
    # test async
    data = await scraper.async_scrape_link(url, random_str = "hahahahaha")
    assert data["title"] == "some title"
    assert data["random_str"] == "hahahahaha"
    assert data["text"] and isinstance(data["text"], str) # this should contain the scraped html
    scraper.set_html_parser() # reset scraper to default


def test_scrape_selenium(scraper):
    """test scraping with selenium"""

    url = "https://en.wikipedia.org/wiki/Vector_database"

    driver = scraper._create_selenium_driver()
    data = scraper._scrape_with_selenium(driver, url)
    driver.quit()
    assert "content" in data
    assert "meta" in data
    assert len(data["meta"]) > 0
