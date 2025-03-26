from numpy import isin
from kruppe.functional.rag.text_splitters import RecursiveTextSplitter, ContextualTextSplitter
from kruppe.models import Chunk
from uuid import UUID
import pytest

CHUNK_SIZE = 128
CHUNK_OVERLAP = 8

def test_splitter(documents, text_splitter):
    text_splitter = RecursiveTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    # test chunk multiple documents at once
    chunked_documents = text_splitter.split_documents(documents)

    assert len(chunked_documents) > len(documents)
    assert all(isinstance(chunk, Chunk) for chunk in chunked_documents)
    assert len(chunked_documents[0].text) < len(documents[0].text)
    assert all(isinstance(chunk.id, UUID) for chunk in chunked_documents)

    # test chunk specifics
    chunked_documents = text_splitter.split_documents([documents[0]])

    assert len(chunked_documents) > 1
    for i in range(len(chunked_documents)):
        chunk = chunked_documents[i]
        assert 0 < len(chunk.text) <= CHUNK_SIZE
        assert chunk.metadata == documents[0].metadata
        assert chunk.document_id == documents[0].id
        
        # test for prev_chunk_id and next_chunk_id
        if (i > 0):
            assert isinstance(chunk.prev_chunk_id, UUID)
            assert chunk.prev_chunk_id == chunked_documents[i-1].id
        if (i < len(chunked_documents) - 1):
            assert isinstance(chunk.next_chunk_id, UUID)
            assert chunk.next_chunk_id == chunked_documents[i+1].id

@pytest.mark.asyncio
async def test_contextual_splitter(documents, llm):
    text_splitter = ContextualTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, llm=llm)

    # skipping testing chunking multiple documents to save cost lol

    # test chunk specifics
    chunked_documents = await text_splitter.async_split_documents([documents[0]])

    assert len(chunked_documents) > 1
    for i in range(len(chunked_documents)):
        chunk = chunked_documents[i]
        
        # extract context and chunked text
        approx_split = chunk.text.split("-TEXT-") # -TEXT- separates context from chunked text
        assert len(approx_split) == 2
        context_str = approx_split[0]
        chunk_str = approx_split[1]

        # tests
        assert len(context_str) > 0
        assert 0 < len(chunk_str) <= CHUNK_SIZE
        assert chunk.metadata == documents[0].metadata
        assert chunk.document_id == documents[0].id
        
        # test for prev_chunk_id and next_chunk_id
        if (i > 0):
            assert isinstance(chunk.prev_chunk_id, UUID)
            assert chunk.prev_chunk_id == chunked_documents[i-1].id
        if (i < len(chunked_documents) - 1):
            assert isinstance(chunk.next_chunk_id, UUID)
            assert chunk.next_chunk_id == chunked_documents[i+1].id