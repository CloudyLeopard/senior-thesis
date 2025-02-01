# retriever/custom_retriever.py

from kruppe.rag.retriever.base_retriever import BaseRetriever
from kruppe.models import Query, Document
from typing import List

class SimpleRetriever(BaseRetriever):
    """
    A custom retriever that performs vector similarity search on a vector store.
    """
    def retrieve(self, query: Query, top_k: int = 5) -> List[Document]:
        return self.index.query(query, top_k)

    async def async_retrieve(self, query: Query, top_k: int = 5) -> List[Document]:
        return await self.index.async_query(query, top_k)