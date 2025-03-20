import asyncio
from tqdm import tqdm
import os

from kruppe.llm import BaseLLM, BaseEmbeddingModel, NYUOpenAIEmbeddingModel, OpenAIEmbeddingModel
from kruppe.data_source.ft import FinancialTimesData
from kruppe.data_source.utils import WebScraper
from kruppe.functional.rag.vectorstore.base_store import BaseVectorStore
from kruppe.functional.rag.vectorstore.in_memory import InMemoryVectorStore

# from rag.vectorstore.milvus import MilvusVectorStore
from kruppe.functional.rag.vectorstore.chroma import ChromaVectorStore
from kruppe.functional.rag.index.base_index import BaseIndex
from kruppe.functional.rag.index.vectorstore_index import VectorStoreIndex
from kruppe.functional.rag.retriever.simple_retriever import SimpleRetriever
from kruppe.prompt_formatter import RAGPromptFormatter, SimplePromptFormatter
from kruppe.models import Query, Document
from kruppe.functional.document_store import AsyncMongoDBStore

"""
 # collect documents
    with open("./.ft-headers.json") as f:
        headers = json.load(f)
    ft_scraper = FinancialTimesData(headers=headers)
    documents = await ft_scraper.async_fetch(query.text)


"""


async def built_ft_vectorstore_index(
    embedding_model: BaseEmbeddingModel, vectorstore: BaseVectorStore
) -> BaseIndex:
    dir = "/Volumes/Lexar/Daniel Liu/ft"

    num_documents = 10000

    # collect documents and insert documents into index
    index = VectorStoreIndex(vectorstore=vectorstore)

    print(f"Collecting documents from {dir} and indexing into {vectorstore.__class__.__name__}...")
    with open(f"{dir}/ft_scrape_info.txt", "r") as f:
        # 0 is id, 1 is url, 2 is scraped_at
        headers = f.readline().split("\t")

        line = f.readline()
        pbar = tqdm(total=num_documents, desc=f"Indexing ft html into {vectorstore.__class__.__name__}")
        batched_documents = []
        while line:
            parts = line.split("\t")

            if os.path.exists(f"{dir}/{parts[0]}.html"):
                with open(f"{dir}/{parts[0]}.html", "r") as f_html:
                    html = f_html.read()
                    if html: # some files are empty
                        data = WebScraper.default_html_parser(html)
                        metadata = FinancialTimesData.parse_metadata(
                            query=None,
                            url=parts[1],
                            title=data["title"],
                            publication_time=data["time"],
                        )

                        batched_documents.append(
                            Document(text=data["content"], metadata=metadata)
                        )

            if len(batched_documents) == 1000:
                await index.async_add_documents(batched_documents)
                batched_documents = []
            line = f.readline()
            pbar.update(1)

        if batched_documents:
            await index.async_add_documents(batched_documents)
        pbar.close()
        print("DONE.")

    return index


async def build_index_from_mongo_db(
        index: BaseIndex, collection_name: str, db_name="FinancialNews"
) -> BaseIndex:
    # TODO: add document collection logic here
    # ...
    print("Downloading documents...")
    document_store = await AsyncMongoDBStore.create(
        db_name=db_name, collection_name=collection_name
    )
    documents = await document_store.get_all_documents()

    # insert documents into index
    print("Indexing documents...")
    await index.async_add_documents(documents)

    print("DONE.")

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
    embedding_model = OpenAIEmbeddingModel()
    vectorstore = ChromaVectorStore(
        embedding_model=embedding_model,
        collection_name="financial-times",
        persist_path="/tmp/ft_chroma_vectorstore",
    )
    asyncio.run(built_ft_vectorstore_index(embedding_model=embedding_model, vectorstore=vectorstore))

    # save_path = "/Users/danielliu/Workspace/fin-rag/src/rag/tmp/vectorstore.pickle"
    # vectorstore.save_pickle(save_path)
    # vs = InMemoryVectorStore.load_pickle("/Users/danielliu/Workspace/fin-rag/src/rag/tmp/vectorstore.pickle")
    # print(vs.embedding_model)
    # print(vs._embeddings_matrix.shape)
    # print(len(vs.documents))
