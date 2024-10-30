import pytest
import pytest_asyncio

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
def test_fetch(source, query):
    documents = source.fetch(query)

    assert len(documents) > 0
    assert all(isinstance(document, Document) for document in documents)

@pytest.mark.slow
@pytest.mark.asyncio(loop_scope="session")
async def test_fetch_async(source, session, query):
    documents = await source.async_fetch(query, session)

    assert len(documents) > 0
    assert all(isinstance(document, Document) for document in documents)

def test_fetch_directory():
    source = DirectoryData()
    documents = source.fetch("tests/rag/data")

    assert len(documents) > 0
    assert all(isinstance(document, Document) for document in documents)