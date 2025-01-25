import asyncio
import json

from rag.llm import BaseLLM, BaseEmbeddingModel, OpenAILLM, OpenAIEmbeddingModel
from rag.scraper.ft import FinancialTimesData
from rag.vectorstore.in_memory import InMemoryVectorStore
from rag.index.base_index import BaseIndex
from rag.index.vectorstore_index import VectorStoreIndex
from rag.retriever.simple_retriever import SimpleRetriever
from rag.retriever.base_retriever import BaseRetriever
from rag.prompts import RAGPromptFormatter, SimplePromptFormatter
from rag.models import Query
from rag.document_store import AsyncMongoDBStore

'''
 # collect documents
    with open("./.ft-headers.json") as f:
        headers = json.load(f)
    ft_scraper = FinancialTimesData(headers=headers)
    documents = await ft_scraper.async_fetch(query.text)


'''

async def build_vectorstore_index(embedding_model: BaseEmbeddingModel,
                                  save_path: str = None) -> BaseIndex:
    # TODO: add document collection logic here
    # ...
    print("Downloading documents...")
    document_store = await AsyncMongoDBStore.create(db_name="FinancialNews", collection_name="2025-01-23")
    documents = await document_store.get_all_documents()

    # insert documents into index
    print("Indexing documents...")
    vectorstore = InMemoryVectorStore(embedding_model=embedding_model)
    index = VectorStoreIndex(embedder=embedding_model, vectorstore=vectorstore)
    await index.async_add_documents(documents)

    if save_path:
        print("Saving vectorstore...")
        vectorstore.save_pickle(save_path)

    return index

async def ask_simple_llm(query: str, llm: BaseLLM):
    prompt_formatter = SimplePromptFormatter()
    messages = prompt_formatter.format_messages(user_prompt=query)
    response = await llm.async_generate(messages)

    return response

async def ask_simple_rag(
    query: str, index: BaseIndex, embedding_model: BaseEmbeddingModel, llm: BaseLLM
):
    # define query
    query = Query(text=query, metadata={})

    # define retriever and retrieve documents
    retriever = SimpleRetriever(embedder=embedding_model, index=index)
    relevant_documents = await retriever.async_retrieve(query)

    # answer query
    prompt_formatter = RAGPromptFormatter()
    prompt_formatter.add_documents(relevant_documents)
    messages = prompt_formatter.format_messages(user_prompt=query.text)
    response = await llm.async_generate(messages)

    return {
        "response": response,
        "contexts": relevant_documents,
    }


if __name__ == "__main__":
    asyncio.run(build_vectorstore_index(embedding_model=OpenAIEmbeddingModel(),
                                        save_path = "/Users/danielliu/Workspace/fin-rag/src/rag/tmp/vectorstore.pickle"))
    # vs = InMemoryVectorStore.load_pickle("/Users/danielliu/Workspace/fin-rag/src/rag/tmp/vectorstore.pickle")
    # print(vs.embedding_model)
    # print(vs._embeddings_matrix.shape)
    # print(len(vs.documents))
