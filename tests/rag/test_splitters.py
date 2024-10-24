import pytest

from rag.text_splitters import RecursiveTextSplitter
from rag.models import Document

pytestmark = pytest.mark.parametrize("cls", [
    RecursiveTextSplitter,
])

@pytest.fixture
def text_splitter(cls):
    return cls(chunk_size=128, chunk_overlap=16)

def test_splitter(text_splitter, documents):
    chunked_documents = text_splitter.split_documents(documents)

    assert len(chunked_documents) > len(documents)
    assert all([isinstance(doc, Document) for doc in documents])
    assert len(chunked_documents[0].text) < len(documents[0].text)
    assert chunked_documents[0].metadata == documents[0].metadata