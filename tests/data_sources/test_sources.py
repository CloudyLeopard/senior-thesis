import pytest
from uuid import UUID

from kruppe.data_source.utils import is_method_ready
from kruppe.data_source.directory import DirectoryData
from kruppe.data_source.news.nyt import NewYorkTimesData
from kruppe.data_source.news.ft import FinancialTimesData
from kruppe.models import Document
import json

@pytest.fixture(
    params=[
        NewYorkTimesData(headers_path=".nyt-headers.json"),
        FinancialTimesData(headers_path=".ft-headers.json"),
    ]
)
def source(request):
    instance = request.param
    
    return instance


@pytest.mark.slow
# @pytest.mark.flaky(retries=2)
@pytest.mark.asyncio(loop_scope="session")
async def test_news_search(source, query):
    if not is_method_ready(source, "news_search"):
        pytest.skip(f"{source.__class__.__name__} news_search not ready")
    
    query = query.text

    size = 0
    async for document in source.news_search(query, num_results=10):
        size += 1
        assert document.text
        assert isinstance(document, Document)
        assert len(document.metadata) > 0
        assert document.metadata.get("datasource") == source.__class__.__name__
        assert document.metadata.get("query") == query
        assert "url" in document.metadata
        assert "title" in document.metadata
        assert "publication_time" in document.metadata
        assert document.uuid and isinstance(document.uuid, UUID)
    
    assert size > 0

@pytest.mark.asyncio(loop_scope="session")
async def test_news_recent(source):
    if not is_method_ready(source, "news_recent"):
        pytest.skip(f"{source.__class__.__name__} news_recent not ready")
    
    size = 0
    async for document in source.news_recent(days=1, num_results=10):
        size += 1
        assert isinstance(document, Document)
        assert len(document.text) > 0
        assert len(document.metadata) > 0
        assert "title" in document.metadata
        assert "description" in document.metadata
        assert document.metadata.get("datasource") == source.__class__.__name__
        assert document.uuid and isinstance(document.uuid, UUID)
    
    assert size > 0

    
@pytest.mark.slow
@pytest.mark.asyncio(loop_scope="session")
async def test_news_archive(source):
    if not is_method_ready(source, "news_archive"):
        pytest.skip(f"{source.__class__.__name__} news_archive not ready")

    size = 0
    async for document in source.news_archive(start_date="2024-01-01", end_date="2024-01-02", num_results=10):
        size += 1
        assert isinstance(document, Document)
        assert len(document.text) > 0
        assert len(document.metadata) > 0
        assert document.metadata.get("datasource") == source.__class__.__name__
        assert document.uuid and isinstance(document.uuid, UUID)
    
    assert size > 0

def test_fetch_directory():
    source = DirectoryData(path="tests/data/1")

    size = 0
    for document in source.fetch():
        size += 1
        assert isinstance(document, Document)
        assert len(document.text) > 0
        assert len(document.metadata) > 0
        assert document.metadata.get("datasource") == source.__class__.__name__
        assert document.uuid and isinstance(document.uuid, UUID)
    assert size > 0