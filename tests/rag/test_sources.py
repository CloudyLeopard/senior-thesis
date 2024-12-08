import pytest
from uuid import UUID

from rag.tools.sources import (
    GoogleSearchData,
    LexisNexisData,
    DirectoryData,
    FinancialTimesData,
    NewsAPIData,
    RequestSourceException
)
from rag.models import Document
import json


@pytest.fixture(
    params=[
        # YFinanceData,
        "lexis",
        # NYTimesData,
        # GuardiansData,
        # NewsAPIData,
        # ProQuestData,
        # BingsNewsData,
        "newsapi",
        pytest.param("google", marks=pytest.mark.xfail(reason="Google Cloud Project for JSON Search API is currently disabled")),
        "financial times",
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


@pytest.mark.slow
# @pytest.mark.flaky(retries=2)
def test_fetch(source, query):
    try:
        documents = source.fetch(query)
    except NotImplementedError:
        pytest.skip(f"async_fetch not implemented for {source}")
    except RequestSourceException as e:
        pytest.xfail(str(e))
        raise e
    except Exception as e:
        pytest.xfail("Failed to capture exception.", str(e))
        raise e

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


@pytest.mark.slow
# @pytest.mark.flaky(retries=2)
@pytest.mark.asyncio(loop_scope="session")
async def test_fetch_async(source, query):
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
    source = DirectoryData("tests/rag/data/1")
    documents = source.fetch()

    assert len(documents) > 0
    for document in documents:
        assert isinstance(document, Document)
        assert len(document.text) > 0
        assert len(document.metadata) > 0
        assert document.metadata.get("datasource") == source.__class__.__name__
        assert document.uuid and isinstance(document.uuid, UUID)
