import pytest

from rag.index.vectorstore_index import VectorStoreIndex
from rag.vectorstore import InMemoryVectorStore

@pytest.mark.asyncio
async def test_vectorstore_index(embedding_model, documents, documents2, query):
    vectorstore = InMemoryVectorStore(embedding_model=embedding_model)
    index = VectorStoreIndex(embedder=embedding_model, vectorstore=vectorstore)

    # test insert documents
    index.add_documents(documents)
    assert vectorstore.size() >= 0

    # test retrieve documents
    relevant_documents = index.query(query, top_k=3)
    assert len(relevant_documents) == 3
    assert relevant_documents[0].text != ""

    # test insert documents async
    await index.async_add_documents(documents2)
    assert vectorstore.size() >= 0 

    # test retrieve documents async
    relevant_documents = await index.async_query(query, top_k=3)
    assert len(relevant_documents) == 3
    assert relevant_documents[0].text != ""