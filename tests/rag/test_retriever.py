import pytest

from rag.retriever import *
from rag.models import Document

@pytest.fixture(
    params=[SimpleRetriever,]
)
def retriever(request, vector_storage, document_storage, embedding_model):
    retriever = request.param(
        vector_storage=vector_storage,
        document_storage=document_storage,
        embedding_model=embedding_model,
    )
    return retriever


def test_retrieve(retriever, query):
    top_k = 3

    retrieved_documents = retriever.retrieve(query, top_k)
    assert 0 < len(retrieved_documents) <= top_k
    assert all([isinstance(doc, Document) for doc in retrieved_documents])

    top_doc = retrieved_documents[0]
    assert top_doc.text and top_doc.metadata  # test that text and metadata is not empty
