from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict
from typing import List

from kruppe.llm import BaseEmbeddingModel
from kruppe.rag.index.base_index import BaseIndex
from kruppe.models import Document, Query

class BaseRetriever(ABC, BaseModel):
    """retrieves query from index"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedder: BaseEmbeddingModel
    index: BaseIndex

    @abstractmethod
    def retrieve(self, query: Query, top_k: int = 5) -> List[Document]:
        """Retrieve documents based on the query."""
        pass

    @abstractmethod
    async def async_retrieve(self, query: Query, top_k: int = 5) -> List[Document]:
        """Retrieve documents based on the query asynchronously."""
        pass