from pymilvus import MilvusClient
from typing import List, Dict
from aiohttp import ClientSession
import asyncio


from .sources import BaseDataSource
from .models import Document
from .embeddings import OpenAIEmbeddingModel
from .text_splitters import RecursiveTextSplitter
from .vector_storages import MilvusVectorStorage
from .retriever import SimpleRetriever
from .generator import OpenAILLM
from .document_storages import MongoDBStore


class DataSourceManager:
    """Manager class to fetch text data given query for list of sources"""

    def __init__(self):
        self.sources: List[BaseDataSource] = []
        self.document_storage = MongoDBStore(uri="mongodb://localhost:27017")
        self.vector_storage = MilvusVectorStorage(uri="http://localhost:19530")
        self.text_splitter = RecursiveTextSplitter()
        self.embedding_model = OpenAIEmbeddingModel()
        self.documents: List[Document] = []

    def choose_source(self, source: str):
        pass

    def add_source(self, source: str):
        pass

    def fetch_documents(self, query: str) -> List[Document]:
        ...
        # store document detail into document storage, and set db id
        for document in combined_results:
            db_id = self.document_storage.save_document(document)
            document.set_db_id(db_id)
        ...

    async def async_fetch_documents(self, query: str) -> List[Document]:
        async with ClientSession() as session:
            tasks = []
            for source in self.sources:
                task = asyncio.create_task(
                    source.async_fetch(query=query, session=session)
                )
                tasks.append(task)
            fetched_results = await asyncio.gather(*tasks)

            # flatten list of lists into single list
            for result in fetched_results:
                for document in result:
                    db_id = self.document_storage.save_document(document)
                    document.set_db_id(db_id)

                    # add document to list
                    self.documents.append(document)

    def split_documents(self):
        """split documents, embed them, and store into vector storage"""
        self.documents = self.text_splitter.split_documents(self.documents)

    def index_documents(self):
        embeddings = self.embedding_model.embed(
            [document.text for document in self.documents]
        )
        self.vector_storage.index_documents(embeddings=embeddings, documents=self.documents)


class RAGCoordinator:
    def __init__(self):

        # # Use Zilliz (Cloud) Vectorstorage
        # milvus_client = MilvusClient(
        #     uri=os.getenv("ZILLIZ_URI"),
        #     token=os.getenv("ZILLIZ_TOKEN")
        # )
        embedding_model = OpenAIEmbeddingModel()
        vector_storage = MilvusVectorStorage(uri="http://localhost:19530")

        self.retriever = SimpleRetriever(
            vector_storage=vector_storage, embedding_model=embedding_model
        )
        self.generator = OpenAILLM()

    def simple_run(self, query: str):
        """run rag pipeline using only similarity search"""
        contexts = self.retriever.retrieve(prompt=query)

        return self.generator.generate(prompt=query, contexts=contexts)
