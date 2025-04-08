import pytest
from uuid import UUID

from kruppe.models import Chunk
from kruppe.functional.rag.vectorstore.in_memory import InMemoryVectorStore
from kruppe.functional.rag.vectorstore.milvus import MilvusVectorStore
from kruppe.functional.rag.vectorstore.chroma import ChromaVectorStore

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
        pytest.xfail("MilvusVectorStore is currently not working - did not fix post refactor (Document)")

        vector_storage = storage_class(
            embedding_model=embedding_model,
            collection_name="ind_test",
            reset_collection=True,
        )
    elif storage_class == ChromaVectorStore:
        vector_storage = storage_class(
            embedding_model=embedding_model,
            collection_name="ind_test",
        )
    elif storage_class == InMemoryVectorStore:
        pytest.xfail("InMemoryVectorStore is currently not working - did not fix post refactor (Document)")
        vector_storage = storage_class(
            embedding_model=embedding_model,
        )
    documents = documents2
    chunked_documents = text_splitter.split_documents(documents)
    vector_storage.insert_documents(chunked_documents)

    yield vector_storage

    # vector_storage.close()

@pytest.mark.asyncio(loop_scope="session")
async def test_insert_remove_vectorstore(vector_storage, text_splitter, documents):
    chunked_documents = text_splitter.split_documents(documents)

    # test insert documents async
    insert_res = await vector_storage.async_insert_documents(chunked_documents)
    assert len(insert_res) == len(chunked_documents)

    # test remove documents
    remove_res = vector_storage.remove_documents(insert_res)
    assert remove_res == len(insert_res)


@pytest.mark.asyncio(loop_scope="session")
async def test_search_vectorstore(vector_storage, embedding_model, query, documents3):
    top_k = 3

    query_vector = embedding_model.embed([query])[0]

    # test search vector
    retrieved_docs = vector_storage.search(query_vector, top_k)
    assert len(retrieved_docs) == top_k

    prev_score = 1.0 # score ranges from 0 to 1
    for doc in retrieved_docs:
        assert isinstance(doc, Chunk)
        assert len(doc.text) > 0
        assert len(doc.metadata) > 0
        assert doc.id and isinstance(doc.id, UUID)

        assert doc.score >= 0.0
        assert doc.score <= prev_score
        prev_score = doc.score
   


    # test search vector async
    retrieved_docs = await vector_storage.async_search(query_vector, top_k)
    assert len(retrieved_docs) == top_k

    prev_score = 1.0 # score ranges from 0 to 1
    for doc in retrieved_docs:
        assert isinstance(doc, Chunk)
        assert len(doc.text) > 0
        assert len(doc.metadata) > 0
        assert doc.id and isinstance(doc.id, UUID)

        assert doc.score >= 0.0
        assert doc.score <= prev_score
        prev_score = doc.score
    

    # test with filters
    vector_storage.insert_documents(documents3)

    filter = {'title': {'$eq': 'title B'}}
    retrieved_docs = vector_storage.search(query_vector, top_k, filter)
    assert len(retrieved_docs) == 1
    assert retrieved_docs[0].metadata['title'] == 'title B'
    assert retrieved_docs[0].metadata['description'] == 'description B'

    # test with time filters
    start_time = 1612155600 # 2021-02-01 00:00:00
    end_time = 1617249600 # 2021-04-01 00:00:00
    filter_start = {'publication_time': {'$gt': start_time}}
    filter_end = {'publication_time': {'$lt': end_time}}
    filter = {'$and': [filter_start, filter_end]}
    retrieved_docs = vector_storage.search(query_vector, top_k, filter)
    assert len(retrieved_docs) == 2
    metadata = [doc.metadata for doc in retrieved_docs]
    ids = [doc.id for doc in retrieved_docs]
    assert documents3[1].metadata in metadata
    assert documents3[2].metadata in metadata
    assert documents3[1].id in ids
    assert documents3[2].id in ids