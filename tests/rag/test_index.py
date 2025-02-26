import pytest

from kruppe.rag.index.vectorstore_index import VectorStoreIndex
from kruppe.rag.index.contextual_index import ContextualVectorStoreIndex
from kruppe.rag.vectorstore.in_memory import InMemoryVectorStore
from kruppe.models import Response

@pytest.mark.asyncio
async def test_vectorstore_index(embedding_model, llm, documents, documents2, query):
    vectorstore = InMemoryVectorStore(embedding_model=embedding_model)
    index = VectorStoreIndex(llm=llm, vectorstore=vectorstore)

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

    # test generate async
    response = await index.async_generate(query)
    assert isinstance(response, Response)
    assert response.text
    assert len(response.sources) > 0


@pytest.mark.costly
@pytest.mark.asyncio
async def test_contextual_index(embedding_model, llm, documents2, query):
    vectorstore = InMemoryVectorStore(embedding_model=embedding_model)
    index = ContextualVectorStoreIndex(vectorstore=vectorstore, llm=llm)

    # test insert documents async
    await index.async_add_documents(documents2)
    assert vectorstore.size() >= 0 

    # test retrieve documents async
    relevant_documents = await index.async_query(query, top_k=3)
    assert len(relevant_documents) == 3
    assert relevant_documents[0].text != ""

    # test generate async
    response = await index.async_generate(query)
    assert isinstance(response, Response)
    assert response.text
    assert len(response.sources) > 0