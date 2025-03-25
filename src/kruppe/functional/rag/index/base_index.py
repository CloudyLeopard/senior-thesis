from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field

from kruppe.llm import BaseLLM
from kruppe.models import Document, Chunk, Query, Response
from kruppe.functional.rag.text_splitters import BaseTextSplitter, RecursiveTextSplitter


class BaseIndex(ABC, BaseModel):
    llm: BaseLLM
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
    def query(self, query: Query, top_k: int = 3, filter: Dict[str, Any] = None) -> List[Chunk]:
        """Query the index."""
        pass

    @abstractmethod
    async def async_query(self, query: Query, top_k: int = 3, filter: Dict[str, Any] = None) -> List[Chunk]:
        """Query the index asynchronously."""
        pass

    @abstractmethod
    def generate(self, query: Query, top_k: int = 3, filter: Dict[str, Any] = None) -> Response:
        """Generate a response based on the query."""
        pass

    @abstractmethod
    async def async_generate(self, query: Query, top_k: int = 3, filter: Dict[str, Any] = None) -> Response:
        """Generate a response based on the query asynchronously."""
        pass