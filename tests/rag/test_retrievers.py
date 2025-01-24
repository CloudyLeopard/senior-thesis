import pytest

from rag.retriever.simple_retriever import SimpleRetriever
from rag.index.vectorstore_index import VectorStoreIndex
from rag.vectorstore.in_memory import InMemoryVectorStore

@pytest.fixture(scope="module")
def index (embedding_model, documents):
    vectorstore = InMemoryVectorStore(embedding_model=embedding_model)
    index = VectorStoreIndex(embedder=embedding_model, vectorstore=vectorstore)

    # test insert documents
    index.add_documents(documents)

    yield index

@pytest.mark.asyncio
async def test_simple_retriever(index, embedding_model, query):
    retriever = SimpleRetriever(embedder = embedding_model, index = index)

    top_k = 3
    relevant_documents = retriever.retrieve(query, top_k=top_k)
    assert len(relevant_documents) == top_k
    assert relevant_documents[0].text != ""

    # test retrieve documents async
    relevant_documents = await retriever.async_retrieve(query, top_k=top_k)
    assert len(relevant_documents) == top_k
    assert relevant_documents[0].text != ""