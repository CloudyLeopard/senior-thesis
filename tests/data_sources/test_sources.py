import pytest
from uuid import UUID
import time
import logging

from kruppe.common.utils import is_method_ready
from kruppe.data_source.news.base_news import NewsSource
from kruppe.data_source.directory import DirectoryData
from kruppe.data_source.news.nyt import NewYorkTimesData
from kruppe.data_source.news.ft import FinancialTimesData
from kruppe.data_source.news.newsapi import NewsAPIData
from kruppe.models import Document

logger = logging.getLogger(__name__)

@pytest.fixture(
    params=[
        NewYorkTimesData(headers_path=".nyt-headers.json"),
        # FinancialTimesData(headers_path=".ft-headers.json"),
        # NewsAPIData(),
    ],
    ids=lambda data: data.shorthand
)
def source(request):
    instance = request.param
    
    return instance


@pytest.mark.slow
# @pytest.mark.flaky(retries=2)
@pytest.mark.asyncio(loop_scope="session")
async def test_news_search(source: NewsSource, query, caplog):
    caplog.set_level("WARNING")

    if not is_method_ready(source, "news_search"):
        pytest.skip(f"{source.__class__.__name__} news_search not ready")
    
    query = query.text

    start_time = time.time()

    size = 0
    async for document in source.news_search(query, max_results=20):
        size += 1
        assert document.text
        assert isinstance(document, Document)
        assert len(document.metadata) > 0
        assert document.metadata.get("query") == query
        assert "url" in document.metadata
        assert "title" in document.metadata
        assert "publication_time" in document.metadata
        assert document.id and isinstance(document.id, UUID)
    
    assert size > 0

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\033[34mInfo on [{source.__class__.__name__}.news_search]:\n- Elapsed time: {elapsed_time:.2f} seconds\n- Number of documents: {size}\033[0m")


@pytest.mark.asyncio(loop_scope="session")
async def test_news_recent(source: NewsSource, caplog):
    caplog.set_level("WARNING")

    if not is_method_ready(source, "news_recent"):
        pytest.skip(f"{source.__class__.__name__} news_recent not ready")
    
    start_time = time.time()

    size = 0    
    async for document in source.news_recent(days=1, max_results=20):
        size += 1
        assert isinstance(document, Document)
        assert len(document.text) > 0
        assert len(document.metadata) > 0
        assert "title" in document.metadata
        assert "description" in document.metadata
        assert document.id and isinstance(document.id, UUID)
    
    assert size > 0

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\033[34mInfo on [{source.__class__.__name__}.news_recent]:\n- Elapsed time: {elapsed_time:.2f} seconds\n- Number of documents: {size}\033[0m")


    
@pytest.mark.slow
@pytest.mark.asyncio(loop_scope="session")
async def test_news_archive(source: NewsSource, caplog):
    caplog.set_level("WARNING")

    if not is_method_ready(source, "news_archive"):
        pytest.skip(f"{source.__class__.__name__} news_archive not ready")

    start_time = time.time()

    size = 0
    async for document in source.news_archive(start_date="2024-01-01", end_date="2024-01-02", max_results=20):
        size += 1
        assert isinstance(document, Document)
        assert len(document.text) > 0
        assert len(document.metadata) > 0
        assert document.id and isinstance(document.id, UUID)
    
    assert size > 0

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\033[34mInfo on [{source.__class__.__name__}.news_archive]:\n- Elapsed time: {elapsed_time:.2f} seconds\n- Number of documents: {size}\033[0m")

def test_fetch_directory():
    source = DirectoryData(path="tests/data/1")

    size = 0
    for document in source.fetch():
        size += 1
        assert isinstance(document, Document)
        assert len(document.text) > 0
        assert len(document.metadata) > 0
        assert document.metadata.get("datasource") == source.__class__.__name__
        assert document.id and isinstance(document.id, UUID)
    assert size > 0