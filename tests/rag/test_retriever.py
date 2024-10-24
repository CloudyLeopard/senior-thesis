import pytest

from rag.retriever import *
from rag.models import Document

pytestmark = pytest.mark.parametrize("cls", [
    SimpleRetriever,
    AsyncSimpleRetriever
])

# TODO populate the vector database to actually test

@pytest.mark.parametrize("retriever_cls", [
    SimpleRetriever,
])
def test_retrieve(retriever_cls, vector_storage, document_storage, embedding_model):
    retriever = retriever_cls(vector_storage, document_storage, embedding_model)

    prompt = "What is Tesla's stock price?"
    top_k = 3

    retrieved_documents = retriever.retrieve(prompt, top_k)
    assert 0 < len(retrieved_documents) <= top_k
    assert all([isinstance(doc, Document) for doc in retrieved_documents])
    
    top_doc = retrieved_documents[0]
    assert top_doc.text and top_doc.metadata # test that text and metadata is not empty
