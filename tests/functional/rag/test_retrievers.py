import pytest
import time

from kruppe.functional.rag.retriever.fusion_retriever import QueryFusionRetriever
from kruppe.functional.rag.retriever.simple_retriever import SimpleRetriever
from kruppe.functional.rag.index.vectorstore_index import VectorStoreIndex

@pytest.mark.asyncio
async def test_simple_retriever(vector_storage, llm, query):
    index = VectorStoreIndex(llm=llm, vectorstore=vector_storage)

    top_k = 3
    retriever = SimpleRetriever(index=index, top_k=top_k)

    relevant_documents = retriever.retrieve(query)
    assert len(relevant_documents) == top_k
    assert relevant_documents[0].text != ""

    # test retrieve documents async
    start_time = time.time()

    relevant_documents = await retriever.async_retrieve(query)
    assert len(relevant_documents) == top_k
    assert relevant_documents[0].text != ""

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\033[34mSimple Retriever: Elapsed time-{elapsed_time:.2f} seconds; Number of documents-{len(relevant_documents)}\033[0m")

@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["simple", "rrf"])
async def test_fusion_retriever(vector_storage, llm, mode, query):
    index1 = VectorStoreIndex(llm=llm, vectorstore=vector_storage)
    index2 = VectorStoreIndex(llm=llm, vectorstore=vector_storage)
    retriever1 = SimpleRetriever(index=index1, top_k=3)
    retriever2 = SimpleRetriever(index=index2, top_k=3)

    top_k = 5
    n_queries = 3

    fusion_retriever = QueryFusionRetriever(
        retrievers=[retriever1, retriever2],
        llm=llm,
        num_queries=n_queries,
        top_k=top_k,
        mode=mode,
    )

    relevant_documents = fusion_retriever.retrieve(query)
    # check for duplicates
    ids = [doc.id for doc in relevant_documents]
    assert len(set(ids)) == len(ids)
    texts = [doc.text for doc in relevant_documents]
    assert len(set(texts)) == len(texts)
    # check for correctness
    assert 0 < len(relevant_documents) <= top_k

    # test async
    start_time = time.time()

    relevant_documents = await fusion_retriever.async_retrieve(query)
    # check for duplicates
    ids = [doc.id for doc in relevant_documents]
    assert len(set(ids)) == len(ids)
    texts = [doc.text for doc in relevant_documents]
    assert len(set(texts)) == len(texts)
    # check for correctness
    assert 0 < len(relevant_documents) <= top_k
    assert relevant_documents[0].text != ""

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\033[34mFusion Retriever [{mode}]: Elapsed time-{elapsed_time:.2f} seconds; Number of documents-{len(relevant_documents)}\033[0m")
