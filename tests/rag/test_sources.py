import pytest
from uuid import UUID
from httpx import HTTPStatusError

from rag.tools.sources import (
    GoogleSearchData,
    LexisNexisData,
    DirectoryData,
    FinancialTimesData,
    NewsAPIData
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
        "newsapi"
        "google",
        "financial times",
    ]
)
def source(request):
    name = request.param

    if name == "google":
        pytest.xfail(
            reason="Google Cloud Project for JSON Search API is currently disabled"
        )
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
    except HTTPStatusError as e:
        # TODO: this doesn't work. at least, im not actually catching the status code
        if e.response.status_code == 429:
            pytest.xfail(f"{source.source}Rate limit exceeded")
        else:
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
    source = DirectoryData()
    documents = source.fetch("tests/rag/data/1")

    assert len(documents) > 0
    for document in documents:
        assert isinstance(document, Document)
        assert len(document.text) > 0
        assert len(document.metadata) > 0
        assert document.metadata.get("datasource") == source.__class__.__name__
        assert document.uuid and isinstance(document.uuid, UUID)
