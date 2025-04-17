from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any

from kruppe.models import Chunk, Query

class BaseRetriever(ABC, BaseModel):
    """retrieves query from index"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    top_k: int = 10

    @abstractmethod
    def retrieve(self, query: Query | str, filter: Dict[str, Any] = None) -> List[Chunk]:
        """Retrieve documents based on the query."""
        pass

    @abstractmethod
    async def async_retrieve(self, query: Query | str, filter: Dict[str, Any] = None) -> List[Chunk]:
        """Retrieve documents based on the query asynchronously."""
        pass