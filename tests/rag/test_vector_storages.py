import pytest
from uuid import UUID
import os

from rag.models import Document
from rag.vectorstore import (
    MilvusVectorStore,
    ChromaVectorStore,
    InMemoryVectorStore
)

@pytest.fixture(
    params=[
        MilvusVectorStore,
        ChromaVectorStore,
        InMemoryVectorStore,
        ],
    scope="module",
)
def vector_storage(request, embedding_model, text_splitter, documents2):
    storage_class = request.param

    if storage_class == MilvusVectorStore:
        uri = os.getenv("ZILLIZ_URI")
        token = os.getenv("ZILLIZ_TOKEN")
        vector_storage = storage_class(
            embedding_model=embedding_model,
            collection_name="ind_test",
            uri=uri,
            token=token,
            reset_collection=True,
        )
    elif storage_class == ChromaVectorStore:
        vector_storage = storage_class(
            embedding_model=embedding_model,
            collection_name="ind_test",
        )
    elif storage_class == InMemoryVectorStore:
        vector_storage = storage_class(
            embedding_model=embedding_model,
        )

    chunked_documents = text_splitter.split_documents(documents2)
    vector_storage.insert_documents(chunked_documents)

    yield vector_storage

    # vector_storage.close()

@pytest.mark.asyncio(loop_scope="session")
async def test_insert_remove_vectorstore(vector_storage, text_splitter, documents):
    chunked_documents = text_splitter.split_documents(documents)

    # # test insert documents
    # insert_res = vector_storage.insert_documents(chunked_documents)
    # assert len(insert_res) == len(chunked_documents)

    # # test remove documents
    # remove_res = vector_storage.remove_documents(insert_res)
    # assert remove_res == len(insert_res)

    # test insert documents async
    insert_res = await vector_storage.async_insert_documents(chunked_documents)
    assert len(insert_res) == len(chunked_documents)

    # test remove documents async
    remove_res = vector_storage.remove_documents(insert_res)
    assert remove_res == len(insert_res)


@pytest.mark.asyncio(loop_scope="session")
async def test_search_vectorstore(vector_storage, embedding_model, query):
    top_k = 3

    query_vector = embedding_model.embed([query])[0]

    # test search vector
    retrieved_docs = vector_storage.search(query_vector, top_k)
    assert len(retrieved_docs) == top_k

    for doc in retrieved_docs:
        assert isinstance(doc, Document)
        assert len(doc.text) > 0
        assert len(doc.metadata) > 0
        assert doc.uuid and isinstance(doc.uuid, UUID)

    # test search vector async
    retrieved_docs = await vector_storage.async_search(query_vector, top_k)
    assert len(retrieved_docs) == top_k

    for doc in retrieved_docs:
        assert isinstance(doc, Document)
        assert len(doc.text) > 0
        assert len(doc.metadata) > 0
        assert doc.uuid and isinstance(doc.uuid, UUID)
