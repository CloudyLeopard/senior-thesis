from abc import ABC, abstractmethod
from typing import List

from rag.database import VectorDatabase
from rag.embeddings import BaseEmbeddingModel, BaseAsyncEmbeddingModel
from rag.models import Document


class BaseRetriever(ABC):
    """Retriever interface"""
    def __init__(
        self,
        vector_database: VectorDatabase,
        embedding_model: BaseEmbeddingModel,
    ):
        """init vector storage and embedding models"""
        self.vector_database = vector_database
        self.embedding_model = embedding_model

    @abstractmethod
    def retrieve(self, prompt: str, top_k=3) -> List[Document]:
        pass


class SimpleRetriever(BaseRetriever):
    """Retriever class using only similarity search"""

    def retrieve(self, prompt: str, top_k=3) -> List[Document]:
        """return list of documents using retrieval based only on similarity search"""

        return self.vector_database.retrieve_documents(prompt, top_k)