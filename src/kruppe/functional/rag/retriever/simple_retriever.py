# retriever/custom_retriever.py

from kruppe.functional.rag.retriever.base_retriever import BaseRetriever
from kruppe.models import Query, Document
from kruppe.functional.rag.index.base_index import BaseIndex
from typing import List

class SimpleRetriever(BaseRetriever):
    """
    A custom retriever that performs vector similarity search on a vector store.
    """
    index: BaseIndex

    def retrieve(self, query: Query) -> List[Document]:
        return self.index.query(query, self.top_k)

    async def async_retrieve(self, query: Query) -> List[Document]:
        return await self.index.async_query(query, self.top_k)