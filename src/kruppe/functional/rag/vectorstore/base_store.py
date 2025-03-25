from abc import ABC, abstractmethod
from pydantic import BaseModel, PrivateAttr, ConfigDict
from typing import List, Dict, Any

from kruppe.models import Document, Chunk
from kruppe.llm import BaseEmbeddingModel


class BaseVectorStore(ABC, BaseModel):
    """Base class for vector storage"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedding_model: BaseEmbeddingModel
    _text_hashes: set[str] = PrivateAttr(default_factory=set)

    def __contains__(self, item) -> bool:
        return hash(item) in self._text_hashes

    def size(self) -> int:
        return len(self._text_hashes)

    @abstractmethod
    def insert_documents(self, documents: List[Document]) -> List[int]:
        """insert documents, embed them, return list of ids"""
        # self._texts_hashes.update(hash(document.text) for document in documents)
        pass

    @abstractmethod
    async def async_insert_documents(self, documents: List[Document]) -> List[int]:
        """insert documents, embed them, return list of ids"""
        pass

    @abstractmethod
    def search(self, vector: List[float], top_k: int = 3, filter: Dict[str, Any] = None) -> List[Chunk]:
        """given vector, return top_k relevant results"""
        pass

    @abstractmethod
    async def async_search(self, vector: List[float], top_k: int = 3, filter: Dict[str, Any] = None) -> List[Chunk]:
        """given query, asynchronously return top_k relevant results"""
        pass

    @abstractmethod
    def remove_documents(self, ids: List[int]) -> int:
        """remove documents by their ids"""
        pass

    # @abstractmethod
    # def clear(self) -> None:
    #     self._texts_hashes = set()
    
    # def close(self):
    #     """close the vector storage"""
    #     pass
