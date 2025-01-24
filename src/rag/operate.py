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

'''
 # collect documents
    with open("./.ft-headers.json") as f:
        headers = json.load(f)
    ft_scraper = FinancialTimesData(headers=headers)
    documents = await ft_scraper.async_fetch(query.text)


'''

async def build_vectorstore_index(embedding_model: BaseEmbeddingModel) -> BaseIndex:
    # TODO: add document collection logic here
    # ...
    documents = []

    # insert documents into index
    vectorstore = InMemoryVectorStore(embedding_model=embedding_model)
    index = VectorStoreIndex(embedder=embedding_model, vectorstore=vectorstore)
    await index.async_add_documents(documents)

    return index

async def ask_simple_llm(query: str, llm: BaseLLM):
    llm = llm()

    prompt_formatter = SimplePromptFormatter()
    messages = prompt_formatter.format_messages(user_prompt=query)
    response = await llm.async_generate(messages)

    return response

async def ask_simple_rag(
    query: str, index: BaseIndex, embedding_model: BaseEmbeddingModel, llm: BaseLLM
):
    # define query
    query = Query(text=query, metadata={})

    # define llm
    llm = llm()

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


# if __name__ == "__main__":
#     asyncio.run(())
