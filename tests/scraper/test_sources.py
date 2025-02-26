import pytest
from uuid import UUID

from kruppe.scraper import (
    GoogleSearchData,
    LexisNexisData,
    DirectoryData,
    FinancialTimesData,
    NewsAPIData,
    RequestSourceException,
    NewYorkTimesData
)
from kruppe.models import Document
import json

@pytest.fixture(
    params=[
        # YFinanceData,
        # "lexis",
        # NYTimesData,
        # GuardiansData,
        # NewsAPIData,
        # ProQuestData,
        # BingsNewsData,
        # "newsapi",
        # pytest.param("google", marks=pytest.mark.xfail(reason="Google Cloud Project for JSON Search API is currently disabled")),
        "financial times",
        # "nytimes"
    ]
)
def source(request):
    name = request.param

    if name == "google":
        return GoogleSearchData()
    elif name == "newsapi":
        return NewsAPIData()
    elif name == "lexis":
        return LexisNexisData()
    elif name == "financial times":
        with open(".ft-headers.json") as f:
            headers = json.load(f)
        return FinancialTimesData(headers=headers)
    elif name == "nytimes":
        with open(".nyt-headers.json") as f:
            headers = json.load(f)
        return NewYorkTimesData(headers=headers)


@pytest.mark.slow
# @pytest.mark.flaky(retries=2)
@pytest.mark.asyncio(loop_scope="session")
async def test_fetch_async(source, query):
    query = query.text
    try:
        documents = await source.async_fetch(query)
    except NotImplementedError:
        pytest.skip(f"async_fetch not implemented for {source}")

    assert len(documents) > 0
    assert any(text for text in [document.text for document in documents])
    for document in documents:
        assert isinstance(document, Document)
        assert len(document.metadata) > 0
        assert document.metadata.get("datasource") == source.__class__.__name__
        assert document.metadata.get("query") == query
        assert "url" in document.metadata
        assert "title" in document.metadata
        assert "publication_time" in document.metadata
        assert document.uuid and isinstance(document.uuid, UUID)


def test_fetch_directory():
    source = DirectoryData(path="tests/data/1")
    documents = source.fetch()

    assert len(documents) > 0
    for document in documents:
        assert isinstance(document, Document)
        assert len(document.text) > 0
        assert len(document.metadata) > 0
        assert document.metadata.get("datasource") == source.__class__.__name__
        assert document.uuid and isinstance(document.uuid, UUID)

@pytest.mark.asyncio(loop_scope="session")
async def test_ft_news_feed(caplog):
    caplog.set_level("DEBUG")
    with open(".ft-headers.json") as f:
        headers = json.load(f)
    source = FinancialTimesData(headers=headers)
    links = await source.fetch_news_feed(days=2)

    assert len(links) > 0

@pytest.mark.asyncio(loop_scope="session")
async def test_ft_scrape_links():
    with open(".ft-headers.json") as f:
        headers = json.load(f)
    source = FinancialTimesData(headers=headers)

    links = ["https://www.ft.com/content/d50a9332-4c89-11e7-a3f4-c742b9791d43",
             "https://www.ft.com/content/b3fcb6de-8456-11e7-a4ce-15b2513cb3ff",]

    documents = await source.async_scrape_links(links)

    assert len(documents) > 0
    for document in documents:
        assert isinstance(document, Document)
        assert len(document.text) > 0
        assert len(document.metadata) > 0
        assert document.metadata.get("datasource") == source.__class__.__name__
        assert document.uuid and isinstance(document.uuid, UUID)

@pytest.mark.slow
@pytest.mark.asyncio(loop_scope="session")
async def test_nyt_fetch_news_feed(caplog):
    caplog.set_level("DEBUG")
    with open(".nyt-headers.json") as f:
        headers = json.load(f)
    source = NewYorkTimesData(headers=headers)

    documents = await source.fetch_news_feed(num_results = 21)

    assert len(documents) > 0
    for document in documents:
        assert isinstance(document, Document)
        assert len(document.text) > 0
        assert len(document.metadata) > 0
        assert document.metadata.get("datasource") == source.__class__.__name__
        assert document.uuid and isinstance(document.uuid, UUID)
    
@pytest.mark.slow
@pytest.mark.asyncio(loop_scope="session")
async def test_nyt_fetch_archive():
    with open(".nyt-headers.json") as f:
        headers = json.load(f)
    source = NewYorkTimesData(headers=headers)

    documents = await source.fetch_archive(months = 1)

    assert len(documents) > 0
    for document in documents:
        assert isinstance(document, Document)
        assert len(document.text) > 0
        assert len(document.metadata) > 0
        assert document.metadata.get("datasource") == source.__class__.__name__
        assert document.uuid and isinstance(document.uuid, UUID)