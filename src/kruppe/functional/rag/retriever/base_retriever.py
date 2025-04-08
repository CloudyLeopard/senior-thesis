from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict
from typing import List

from kruppe.models import Document, Query

class BaseRetriever(ABC, BaseModel):
    """retrieves query from index"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    top_k: int = 10

    @abstractmethod
    def retrieve(self, query: Query) -> List[Document]:
        """Retrieve documents based on the query."""
        pass

    @abstractmethod
    async def async_retrieve(self, query: Query) -> List[Document]:
        """Retrieve documents based on the query asynchronously."""
        pass