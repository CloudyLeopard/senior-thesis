from abc import ABC, abstractmethod
from pydantic import BaseModel, PrivateAttr
from typing import List

from rag.models import Document
from rag.embeddings import BaseEmbeddingModel


class BaseVectorStore(ABC, BaseModel):
    """Base class for vector storage"""

    embedding_model: BaseEmbeddingModel
    text_hashes: set[str] = PrivateAttr(default_factory=set)

    def __contains__(self, item) -> bool:
        return hash(item) in self.texts_hashes

    def __len__(self) -> int:
        return len(self.texts_hashes)

    @abstractmethod
    def insert_documents(self, documents: List[Document]) -> List[int]:
        """insert documents, embed them, return list of ids"""
        # self.texts_hashes.update(hash(document.text) for document in documents)
        pass

    @abstractmethod
    async def async_insert_documents(self, documents: List[Document]) -> List[int]:
        """insert documents, embed them, return list of ids"""
        pass

    @abstractmethod
    def search(self, vector: List[float], top_k: int = 3) -> List[Document]:
        """given vector, return top_k relevant results"""
        pass

    @abstractmethod
    async def async_search(self, vector: List[float], top_k: int = 3) -> List[Document]:
        """given query, asynchronously return top_k relevant results"""
        pass

    @abstractmethod
    def remove_documents(self, ids: List[int]) -> int:
        """remove documents by their ids"""
        pass

    # @abstractmethod
    # def clear(self) -> None:
    #     self.texts_hashes = set()
    
    # def close(self):
    #     """close the vector storage"""
    #     pass
