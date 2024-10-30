import pytest
from uuid import UUID
import pytest_asyncio
import os

from rag.sources import GoogleSearchData, DirectoryData
from rag.models import Document

@pytest.fixture(params=[
    # YFinanceData,
    # LexisNexisData,
    # NYTimesData,
    # GuardiansData,
    # NewsAPIData,
    # ProQuestData,
    # BingsNewsData,
    GoogleSearchData,
])
def source(request):
    return request.param()

@pytest.mark.slow
@pytest.mark.flaky(retries=2)
def test_fetch(source, query):
    documents = source.fetch(query)

    assert len(documents) > 0
    
    for document in documents:
        assert isinstance(document, Document)
        assert len(document.text) > 0 # may report error for some sources that return empty text
        assert len(document.metadata) > 0
        assert document.metadata["datasource"] == source.source
        assert document.metadata["query"] == query
        assert document.uuid and isinstance(document.uuid, UUID)
    

@pytest.mark.slow
@pytest.mark.flaky(retries=2)
@pytest.mark.asyncio(loop_scope="session")
async def test_fetch_async(source, session, query):
    documents = await source.async_fetch(query, session)

    assert len(documents) > 0
    for document in documents:
        assert isinstance(document, Document)
        assert len(document.text) > 0
        assert len(document.metadata) > 0
        assert document.metadata["datasource"] == source.source
        assert document.metadata["query"] == query
        assert document.uuid and isinstance(document.uuid, UUID)

def test_fetch_directory():
    source = DirectoryData()
    documents = source.fetch("tests/rag/data")

    assert len(documents) > 0
    for document in documents:
        assert isinstance(document, Document)
        assert len(document.text) > 0
        assert len(document.metadata) > 0
        assert document.metadata["datasource"] == source.source
        assert document.uuid and isinstance(document.uuid, UUID)