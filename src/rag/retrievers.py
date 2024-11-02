from abc import ABC, abstractmethod
from typing import List


from rag.document_storages import BaseDocumentStore
from rag.vector_storages import BaseVectorStorage
from rag.models import Document


class BaseRetriever(ABC):
    """Retriever interface"""
    def __init__(
        self,
        vector_storage: BaseVectorStorage,
    ):
        """A retriever retrieves documents using the vector storage. Certain retrieveres
        will also use a document storage to get the original data/metadata"""
        self.vector_storage = vector_storage

    @abstractmethod
    def retrieve(self, prompt: str, top_k=3) -> List[Document]:
        pass


class SimpleRetriever(BaseRetriever):
    """Retriever class using only similarity search"""
    def __init__(
        self,
        vector_storage: BaseVectorStorage,
    ):
        super().__init__(vector_storage=vector_storage)

    def retrieve(self, prompt: str, top_k=3) -> List[Document]:
        """return list of documents using retrieval based only on similarity search"""

        return self.vector_storage.similarity_search(prompt, top_k)

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
        """return list of documents using retrieval based on similarity search and document storage"""

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
                print("No DB_ID found")
        

        return retrieved_documents