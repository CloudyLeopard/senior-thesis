from typing import List
import logging
from pydantic import Field

from rag.llm import BaseEmbeddingModel
from rag.index.base_index import BaseIndex
from rag.models import Document, Query
from rag.vectorstore.base_store import BaseVectorStore
from rag.text_splitters import BaseTextSplitter, RecursiveTextSplitter

logger = logging.getLogger(__name__)

class VectorStoreIndex(BaseIndex):

    vectorstore: BaseVectorStore
    text_splitter: BaseTextSplitter = Field(default_factory=RecursiveTextSplitter)
    embedder: BaseEmbeddingModel = None

    def model_post_init(self, __context):
        self.embedder = self.vectorstore.embedding_model

    def add_documents(self, documents: List[Document]):
        # split documents into chunks
        chunked_documents = self.text_splitter.split_documents(documents)
        
        # insert documents into vector store
        self.vectorstore.insert_documents(chunked_documents)
    
    async def async_add_documents(self, documents: List[Document]):
        # split documents into chunks
        chunked_documents = self.text_splitter.split_documents(documents)        
        # insert documents into vector store
        await self.vectorstore.async_insert_documents(chunked_documents)

    def query(self, query: Query, top_k: int = 3) -> List[Document]:
        # embed query
        query_vector = self.embedder.embed([query])[0]

        # search vector store
        relevant_documents = self.vectorstore.search(vector=query_vector, top_k=top_k)

        return relevant_documents

    async def async_query(self, query: Query, top_k: int = 3) -> List[Document]:
        # embed query
        query_vector = await self.embedder.async_embed([query])
        query_vector = query_vector[0]

        # search vector store
        relevant_documents = await self.vectorstore.async_search(vector=query_vector, top_k=top_k)

        return relevant_documents