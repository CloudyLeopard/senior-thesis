import pytest
from uuid import UUID
import time
import logging
import pandas as pd

from kruppe.data_source.news.nyt import NewYorkTimesData
from kruppe.data_source.news.ft import FinancialTimesData
from kruppe.data_source.news.newsapi import NewsAPIData
from kruppe.functional.newshub import NewsHub
from kruppe.models import Document

logger = logging.getLogger(__name__)

@pytest.fixture()
def newshub():
    newshub = NewsHub(news_sources=[
        NewYorkTimesData(headers_path=".nyt-headers.json"),
        FinancialTimesData(headers_path=".ft-headers.json"),
        NewsAPIData()
    ])
    
    return newshub


@pytest.mark.slow
# @pytest.mark.flaky(retries=2)
@pytest.mark.asyncio(loop_scope="session")
async def test_news_search(newshub, query, caplog):
    caplog.set_level("WARNING")

    query = query.text

    start_time = time.time()

    df, sources = await newshub.news_search(query, max_results=20)

    assert isinstance(df, pd.DataFrame)

    logger.debug(df)
    assert "title" in df.columns
    assert "description" in df.columns
    assert "publication_time" in df.columns

    assert len(sources) > 0
    assert len(df) == len(sources)
    size = len(sources)

    for document in sources:
        assert document.text
        assert isinstance(document, Document)
        assert len(document.metadata) > 0
        assert document.metadata.get("query") == query
        assert "url" in document.metadata
        assert "title" in document.metadata
        assert "publication_time" in document.metadata
        assert document.id and isinstance(document.id, UUID)
    

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\033[34mInfo on [NewsHub.news_search]:\n- Elapsed time: {elapsed_time:.2f} seconds\n- Number of documents: {size}\033[0m")


@pytest.mark.asyncio(loop_scope="session")
async def test_news_recent(newshub, caplog):
    caplog.set_level("WARNING")

    start_time = time.time()

    df, sources = await newshub.news_recent(days=1, max_results=10)

    assert isinstance(df, pd.DataFrame)

    logger.debug(df)
    assert "title" in df.columns
    assert "description" in df.columns
    assert "publication_time" in df.columns

    assert len(sources) > 0
    assert len(df) == len(sources)
    size = len(sources)

    for document in sources:
        assert document.text
        assert isinstance(document, Document)
        assert len(document.metadata) > 0
        assert "url" in document.metadata
        assert "title" in document.metadata
        assert "publication_time" in document.metadata
        assert document.id and isinstance(document.id, UUID)
    

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\033[34mInfo on [NewsHub.news_recent]:\n- Elapsed time: {elapsed_time:.2f} seconds\n- Number of documents: {size}\033[0m")

    
@pytest.mark.slow
@pytest.mark.asyncio(loop_scope="session")
async def test_news_archive(newshub, caplog):
    caplog.set_level("WARNING")

    start_time = time.time()

    df, sources = await newshub.news_archive(start_date="2024-01-01", end_date="2024-01-02", max_results=20)

    assert isinstance(df, pd.DataFrame)

    logger.debug(df)
    assert "title" in df.columns
    assert "description" in df.columns
    assert "publication_time" in df.columns

    assert len(sources) > 0
    assert len(df) == len(sources)
    size = len(sources)

    for document in sources:
        assert document.text
        assert isinstance(document, Document)
        assert len(document.metadata) > 0
        assert "url" in document.metadata
        assert "title" in document.metadata
        assert "publication_time" in document.metadata
        assert document.id and isinstance(document.id, UUID)
    

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\033[34mInfo on [NewsHub.news_archive]:\n- Elapsed time: {elapsed_time:.2f} seconds\n- Number of documents: {size}\033[0m")
