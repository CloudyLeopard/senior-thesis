from typing import List
from pydantic import BaseModel

from rag.llm import BaseEmbeddingModel
from rag.index.base_index import BaseIndex
from rag.models import Document, Query
from rag.vector_store.base_store import BaseVectorStore
from rag.text_splitters import BaseTextSplitter, RecursiveTextSplitter

class VectorStoreIndex(BaseIndex, BaseModel):

    embedder: BaseEmbeddingModel
    vector_store: BaseVectorStore
    text_splitter: BaseTextSplitter = RecursiveTextSplitter()

    def add_documents(self, documents: List[Document]):
        # split documents into chunks
        chunked_documents = self.text_splitter.split_documents(documents)
        
        # insert documents into vector store
        self.vector_store.insert_documents(chunked_documents)
    
    async def async_add_documents(self, documents: List[Document]):
        # split documents into chunks
        chunked_documents = self.text_splitter.split_documents(documents)
        
        # insert documents into vector store
        await self.vector_store.async_insert_documents(chunked_documents)

    def query(self, query: Query, top_k: int = 3) -> List[Document]:
        # embed query
        query_vector = self.embedder.embed([query])[0]

        # search vector store
        relevant_documents = self.vector_store.search(vector=query_vector, top_k=top_k)

        return relevant_documents

    async def async_query(self, query: Query, top_k: int = 3) -> List[Document]:
        # embed query
        query_vector = await self.embedder.async_embed([query])[0]

        # search vector store
        relevant_documents = await self.vector_store.async_search(vector=query_vector, top_k=top_k)

        return relevant_documents