import asyncio

from rag.llm import OpenAILLM, OpenAIEmbeddingModel
from rag.scraper.ft import FinancialTimesData
from rag.vectorstore.in_memory import InMemoryVectorStore
from rag.index.vectorstore_index import VectorStoreIndex
from rag.retriever.simple_retriever import SimpleRetriever
from rag.prompts import RAGPromptFormatter
from rag.models import Query

async def main():
    # define query
    query = Query(text="What is the capital of France?", metadata={})

    # define llm
    llm = OpenAILLM()
    embedding_model = OpenAIEmbeddingModel()

    # collect documents
    with open("./.ft-headers.json") as f:
        headers = f.read()
    ft_scraper = FinancialTimesData(headers=headers)
    documents = await ft_scraper.async_fetch(query.text)

    # insert documents into index
    vectorstore = InMemoryVectorStore(embedding_model = embedding_model)
    index = VectorStoreIndex(embedder=embedding_model, vectorstore=vectorstore)

    await index.async_add_documents(documents)

    # define retriever and retrieve documents
    retriever = SimpleRetriever(embedder=embedding_model, index=index)

    relevant_documents = await retriever.async_retrieve(query)

    # answer query
    prompt_formatter = RAGPromptFormatter()
    prompt_formatter.add_documents(relevant_documents)
    messages = prompt_formatter.format_messages(user_prompt=query.text)
    response = await llm.async_generate(messages)

    print(response)

if __name__ == "__main__":
    asyncio.run(main())