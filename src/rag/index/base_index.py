from abc import ABC, abstractmethod
from typing import List

from rag.models import Document, Query

class BaseIndex(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Document]):
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