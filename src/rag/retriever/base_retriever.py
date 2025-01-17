from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List

from rag.llm import BaseEmbeddingModel
from rag.index.base_index import BaseIndex
from rag.models import Document, Query

class BaseRetriever(ABC, BaseModel):
    """retrieves query from index"""

    embedder: BaseEmbeddingModel
    index: BaseIndex

    @abstractmethod
    def retrieve(self, query: Query, top_k: int = 5) -> List[Document]:
        """Retrieve documents based on the query."""
        pass

    @abstractmethod
    def async_retrieve(self, query: Query, top_k: int = 5) -> List[Document]:
        """Retrieve documents based on the query asynchronously."""
        pass