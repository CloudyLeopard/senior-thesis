from kruppe.functional.rag.retriever.base_retriever import BaseRetriever
from kruppe.models import Query, Document
from kruppe.functional.rag.index.base_index import BaseIndex
from typing import List, Dict, Any

class SimpleRetriever(BaseRetriever):
    """
    A custom retriever that performs vector similarity search on a vector store.
    """
    index: BaseIndex

    def retrieve(self, query: Query, filter: Dict[str, Any] = None) -> List[Document]:
        return self.index.query(query, self.top_k, filter)

    async def async_retrieve(self, query: Query, filter: Dict[str, Any] = None) -> List[Document]:
        return await self.index.async_query(query, self.top_k, filter)