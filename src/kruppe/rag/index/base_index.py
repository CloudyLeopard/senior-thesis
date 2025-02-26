from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel, ConfigDict, Field

from kruppe.llm import BaseLLM
from kruppe.models import Document, Query, Response
from kruppe.rag.text_splitters import BaseTextSplitter, RecursiveTextSplitter


class BaseIndex(ABC, BaseModel):
    llm: BaseLLM
    text_splitter: BaseTextSplitter = Field(default_factory=RecursiveTextSplitter)

    model_config = ConfigDict(arbitrary_types_allowed=True)
        
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the index."""
        pass

    @abstractmethod
    async def async_add_documents(self, documents: List[Document]):
        """Add documents to the index asynchronously."""
        pass

    @abstractmethod
    def query(self, query: Query) -> List[Document]:
        """Query the index."""
        pass

    @abstractmethod
    async def async_query(self, query: Query) -> List[Document]:
        """Query the index asynchronously."""
        pass

    @abstractmethod
    def generate(self, query: Query) -> Response:
        """Generate a response based on the query."""
        pass

    @abstractmethod
    async def async_generate(self, query: Query) -> Response:
        """Generate a response based on the query asynchronously."""
        pass