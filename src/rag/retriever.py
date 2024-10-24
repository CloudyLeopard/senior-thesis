from abc import ABC, abstractmethod
from typing import List

from rag.vector_storages import BaseVectorStorage
from rag.document_storages import BaseDocumentStore
from rag.embeddings import BaseEmbeddingModel, BaseAsyncEmbeddingModel
from rag.models import Document


class BaseRetriever(ABC):
    """Retriever interface"""

    @abstractmethod
    def retrieve(self, prompt: str, top_k=3) -> List[Document]:
        pass


class SimpleRetriever(BaseRetriever):
    """Retriever class using only similarity search"""

    def __init__(
        self,
        vector_storage: BaseVectorStorage,
        document_storage: BaseDocumentStore,
        embedding_model: BaseEmbeddingModel,
    ):
        """init vector storage and embedding models"""
        self.vector_storage = vector_storage
        self.document_storage = document_storage
        self.embedding_model = embedding_model

    def retrieve(self, prompt: str, top_k=3) -> List[Document]:
        """return list of documents using retrieval based only on similarity search"""

        embeddings = self.embedding_model.embed([prompt])

        retrieved_data = self.vector_storage.search_vectors(embeddings, top_k=top_k)
        retrieved_data = retrieved_data[0] # only one prompt, not a list of prompts
        documents = [
            self.document_storage.get_document(data["db_id"]) for data in retrieved_data
        ]

        return documents


# ----- ASYNC ------


class BaseAsyncRetriever(ABC):
    """Retriever interface"""

    @abstractmethod
    async def retrieve(self, prompt: str, top_k=3) -> List[Document]:
        pass


class AsyncSimpleRetriever(BaseAsyncRetriever):
    """Async retriever class using only similarity search"""

    def __init__(
        self, vector_storage: BaseVectorStorage, embedding_model=BaseAsyncEmbeddingModel
    ):
        """init vector storage and embedding models"""
        self.vector_storage = vector_storage
        self.embedding_model = embedding_model

    async def retrieve(self, prompt: str, top_k=3) -> List[Document]:
        """return list of documents using retrieval based only on similarity search"""

        embeddings = await self.embedding_model.embed([prompt])

        # TODO: async retrieval
        retrieved_data = self.vector_storage.search_vector(embeddings[0], top_k=top_k)
        documents = [
            self.document_storage.get_document(data["db_id"]) for data in retrieved_data
        ]

        return documents