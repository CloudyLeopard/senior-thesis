from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field

from kruppe.models import Document, Chunk, Query
from kruppe.functional.rag.text_splitters import BaseTextSplitter, RecursiveTextSplitter


class BaseIndex(ABC, BaseModel):
    text_splitter: BaseTextSplitter = Field(default_factory=RecursiveTextSplitter)

    model_config = ConfigDict(arbitrary_types_allowed=True)
        
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the index."""
        pass

    @abstractmethod
    async def async_add_documents(self, documents: List[Document]) -> None:
        """Add documents to the index asynchronously."""
        pass

    @abstractmethod
    def query(self, query: Query | str, top_k: int = 3, filter: Dict[str, Any] = None) -> List[Chunk]:
        """Query the index."""
        pass

    @abstractmethod
    async def async_query(self, query: Query | str, top_k: int = 3, filter: Dict[str, Any] = None) -> List[Chunk]:
        """Query the index asynchronously."""
        pass

    def as_retriever(self, top_k: int = 10):
        from kruppe.functional.rag.retriever.simple_retriever import SimpleRetriever

        return SimpleRetriever(index=self, top_k=top_k)