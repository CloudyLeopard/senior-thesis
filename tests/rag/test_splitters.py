import pytest

from rag.text_splitters import RecursiveTextSplitter
from rag.models import Document

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 64

@pytest.fixture(params=[
    RecursiveTextSplitter,
])
def text_splitter(request):
    return request.param(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

def test_splitter(text_splitter, documents):
    # test chunk multiple documents at once
    chunked_documents = text_splitter.split_documents(documents)

    assert len(chunked_documents) > len(documents)
    assert all([isinstance(doc, Document) for doc in documents])
    assert len(chunked_documents[0].text) < len(documents[0].text)

    # test chunk specifics
    chunked_documents = text_splitter.split_documents([documents[0]])

    assert len(chunked_documents) > 1
    for chunk in chunked_documents:
        assert 0 < len(chunk.text) <= CHUNK_SIZE
        assert chunk.metadata == documents[0].metadata
        assert chunk.db_id == documents[0].db_id
        assert chunk.uuid == documents[0].uuid
    
    assert len(set([chunk.text for chunk in chunked_documents])) == len(chunked_documents)
    print("Number of chunks: ", len(chunked_documents))

    _text_hashes = set()
    documents = [doc for doc in chunked_documents if (doc_hash := hash(doc)) not in _text_hashes and not _text_hashes.add(doc_hash)]
    print("Number of unique chunks: ", len(_text_hashes))
    assert len(chunked_documents) == len(documents)