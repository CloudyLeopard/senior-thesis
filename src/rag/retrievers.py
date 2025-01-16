from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from typing import List
import logging

from rag.document_store import BaseDocumentStore
from rag.vector_storages import BaseVectorStorage
from rag.models import Document

logger = logging.getLogger(__name__)


class BaseRetriever(BaseModel, ABC):
    """Retriever interface"""
    index: BaseVectorStorage = Field(..., description="The index used for similarity search")

    @abstractmethod
    def retrieve(self, prompt: str, top_k=3) -> List[Document]:
        pass

    @abstractmethod
    async def async_retrieve(self, prompt: str, top_k=3) -> List[Document]:
        pass


class SimpleRetriever(BaseRetriever):
    """Retriever class using only similarity search"""

    def retrieve(self, prompt: str, top_k=3) -> List[Document]:
        """return list of documents using retrieval based only on similarity search"""

        return self.index.similarity_search(prompt, top_k)

    async def async_retrieve(self, prompt: str, top_k=3) -> List[Document]:
        raise await self.index.async_similarity_search(prompt, top_k)

class DocumentRetriever(BaseRetriever):
    """Retriever class using similarity search and document storage"""
    def __init__(
        self,
        vector_storage: BaseVectorStorage,
        document_store: BaseDocumentStore
    ):
        super().__init__(vector_storage=vector_storage)
        self.document_store = document_store

    def retrieve(self, prompt: str, top_k=3) -> List[Document]:
        """
        Retrieve a list of documents based on a similarity search and update their metadata.

        This method performs a similarity search using the provided prompt and retrieves 
        a list of documents. For each document, the corresponding data is fetched from 
        the document storage using its UUID, and the document's metadata and database ID 
        are updated.

        Args:
            prompt (str): The input query to perform the similarity search.
            top_k (int, optional): The number of top documents to retrieve. Defaults to 3.

        Returns:
            List[Document]: A list of documents with updated metadata retrieved from 
            the document storage.
        """
        retrieved_documents = self.vector_storage.similarity_search(prompt, top_k)
        for document in retrieved_documents:
            # retrieve corresponding data in document storage
            db_document = self.document_store.get_document_by_uuid(document.uuid)

            # if document exists in document storage, set metadata
            # this should always be true
            if db_document:
                document.metadata = db_document.metadata
                document.db_id = db_document.db_id
            else:
                logger.warning("No record of document %s found in document storage", document.uuid)
        
        logger.info("Retrieved %d documents", len(retrieved_documents))
        return retrieved_documents

    async def async_retrieve(self, prompt: str, top_k=3) -> List[Document]:
        raise NotImplementedError
    
        # TODO: implement async similarity search.
        # TODO: check if the passed in document store is async or not (since pymongo and motor are separate)
        retrieved_documents = await self.vector_storage.async_similarity_search(prompt, top_k)
        for document in retrieved_documents:
            # retrieve corresponding data in document storage
            db_document = await self.document_store.get_document_by_uuid(document.uuid)

            # if document exists in document storage, set metadata
            # this should always be true
            if db_document:
                document.metadata = db_document.metadata
                document.db_id = db_document.db_id
            else:
                print("No DB_ID found")
        

        return retrieved_documents
