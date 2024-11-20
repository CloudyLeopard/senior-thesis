import pytest
from uuid import UUID
import os

from rag.models import Document
from rag.vector_storages import MilvusVectorStorage, ChromaVectorStorage



@pytest.fixture(params=[
    MilvusVectorStorage,
    ChromaVectorStorage,
])
def vector_storage(request, embedding_model, text_splitter, documents2):
    storage_class = request.param

    if storage_class == MilvusVectorStorage:
        uri = os.getenv("ZILLIZ_URI")
        token = os.getenv("ZILLIZ_TOKEN")
        vector_storage = storage_class(
            embedding_model=embedding_model,
            collection_name="ind_test",
            uri=uri,
            token=token,
            reset_collection=True
        )
    elif storage_class == ChromaVectorStorage:
        vector_storage = storage_class(
            embedding_model=embedding_model,
            collection_name="ind_test",
        )
    
    chunked_documents = text_splitter.split_documents(documents2)
    vector_storage.insert_documents(chunked_documents)

    yield vector_storage

    vector_storage.close()

def test_insert_remove_vectorstore(vector_storage, text_splitter, documents):
    chunked_documents = text_splitter.split_documents(documents)
    
    # test insert documents
    insert_res = vector_storage.insert_documents(chunked_documents)
    assert len(insert_res) == len(chunked_documents)

    # test remove documents
    remove_res = vector_storage.remove_documents(insert_res)
    assert remove_res == len(insert_res)

def test_search_vectorstore(vector_storage, query):
    # test search vector
    top_k=3
    retrieved_docs = vector_storage.similarity_search(query, top_k)
    assert len(retrieved_docs) == top_k

    for doc in retrieved_docs:
        assert isinstance(doc, Document)
        assert len(doc.text) > 0
        assert len(doc.metadata) > 0
        assert "datasource" in doc.metadata
        assert doc.uuid and isinstance(doc.uuid, UUID)