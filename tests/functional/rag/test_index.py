import pytest

from kruppe.functional.rag.index.vectorstore_index import VectorStoreIndex
from kruppe.functional.rag.index.contextual_index import ContextualVectorStoreIndex
from kruppe.models import Response

@pytest.mark.asyncio
async def test_vectorstore_index(vector_storage, llm, text_splitter, documents, query):
    index = VectorStoreIndex(
        vectorstore=vector_storage,
        llm=llm,
        text_splitter=text_splitter,
    )

    # test insert documents async
    size_before = vector_storage.size()
    await index.async_add_documents(documents)
    assert vector_storage.size() > size_before

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
async def test_contextual_index(vector_storage, llm, text_splitter, documents, query):
    index = ContextualVectorStoreIndex(
        vectorstore=vector_storage,
        llm=llm,
        text_splitter=text_splitter,
    )

    # test insert documents async
    size_before = vector_storage.size()
    await index.async_add_documents(documents)
    assert vector_storage.size() >= size_before

    # test retrieve documents async
    relevant_documents = await index.async_query(query, top_k=3)
    assert len(relevant_documents) == 3
    assert relevant_documents[0].text != ""

    # test generate async
    response = await index.async_generate(query)
    assert isinstance(response, Response)
    assert response.text
    assert len(response.sources) > 0