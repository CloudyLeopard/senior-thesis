from typing import List, Dict, Any
import logging
from pydantic import PrivateAttr
import asyncio

from kruppe.llm import BaseEmbeddingModel
from kruppe.functional.rag.index.base_index import BaseIndex
from kruppe.models import Chunk, Document, Query, Response
from kruppe.functional.rag.vectorstore.base_store import BaseVectorStore
from kruppe.prompt_formatter import RAGPromptFormatter

logger = logging.getLogger(__name__)

class VectorStoreIndex(BaseIndex):

    vectorstore: BaseVectorStore
    _embedder: BaseEmbeddingModel = PrivateAttr(default=None) 

    def model_post_init(self, __context):
        self._embedder = self.vectorstore.embedding_model

    def add_documents(self, documents: List[Document]):
        # split documents into chunks
        chunked_documents = self.text_splitter.split_documents(documents)
        
        # insert documents into vector store
        self.vectorstore.insert_documents(chunked_documents)
    
    async def async_add_documents(self, documents: List[Document]):
        # split documents into chunks
        chunked_documents = await self.text_splitter.async_split_documents(documents)        
        # insert documents into vector store
        await self.vectorstore.async_insert_documents(chunked_documents)

    def query(self, query: Query, top_k: int = 3, filter: Dict[str, Any] = None) -> List[Chunk]:
        # embed query
        query_vector = self._embedder.embed([query])[0]

        # search vector store
        ret_chunks = self.vectorstore.search(
            vector=query_vector,
            top_k=top_k,
            filter=filter
        )

        return ret_chunks

    async def async_query(self, query: Query, top_k: int = 3, filter: Dict[str, Any] = None) -> List[Chunk]:
        # embed query
        query_vector = await self._embedder.async_embed([query])
        query_vector = query_vector[0]

        # search vector store
        ret_chunks = await self.vectorstore.async_search(
            vector=query_vector,
            top_k=top_k,
            filter=filter
        )

        return ret_chunks
    
    async def async_generate(self, query: Query, top_k: int = 3, filter: Dict[str, Any] = None) -> Response:
        # retrieve relevant documents
        ret_chunks = await self.async_query(query, top_k=top_k, filter=filter)

        # format rag prompt
        prompt_formatter = RAGPromptFormatter()
        prompt_formatter.add_documents(ret_chunks)
        messages = prompt_formatter.format_messages(user_prompt=query)

        # generate response and add sources
        response = await self.llm.async_generate(messages)
        response.sources = ret_chunks

        return response
    
    def generate(self, query: Query, top_k: int = 3, filter: Dict[str, Any] = None) -> Response:
        return asyncio.run(self.async_generate(query, top_k=top_k, filter=filter))