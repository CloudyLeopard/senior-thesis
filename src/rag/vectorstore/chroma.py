from uuid import UUID
from typing import List
from pydantic import Field
from typing import Optional
import chromadb
import logging

from rag.vectorstore.base_store import BaseVectorStore
from rag.models import Document

logger = logging.getLogger(__name__)

class ChromaVectorStore(BaseVectorStore):
    """Custom Vector storage class for using Chroma"""

    collection_name: str = Field(default="financial_context")
    persist_directory: Optional[str] = None
    client: Optional[chromadb.ClientAPI] = None
    collection: Optional[chromadb.Collection] = None

    def model_post_init(self, __context):
        if self.client is None:
            if self.persist_directory:
                self.client = chromadb.PersistentClient(path=self.persist_directory)
            else:
                self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(self.collection_name)

    def insert_documents(self, documents: List[Document]):
        # remove duplicate documents
        documents = [doc for doc in documents if (doc_hash := hash(doc)) not in self._text_hashes and not self._text_hashes.add(doc_hash)]

        data = {
            "ids": [],
            "metadatas": [],
            "documents": [],
            "embeddings": self.embedding_model.embed(
                [document.text for document in documents]
            ),
        }

        for document in documents:
            data["ids"].append(str(hash(document)))

            metadata = document.metadata
            metadata["uuid"] = str(document.uuid)
            data["metadatas"].append(document.metadata)

            data["documents"].append(document.text)

        # insert data into chromadb       
        logger.info("Inserting documents into ChromaDB")
        self.collection.upsert(
            ids=data["ids"],
            metadatas=data["metadatas"],
            documents=data["documents"],
            embeddings=data["embeddings"],
        )

        return data["ids"]
    
    async def async_insert_documents(self, documents):
        """Falls back to synchronous insert_documents"""
        return self.insert_documents(documents)

    def search(self, vector: List[float], top_k: int = 3):
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
        )

        documents = [
            Document(
                text=document,
                uuid=UUID(metadata.pop("uuid")),
                metadata=metadata,
            )
            for document, doc_id, metadata in zip(
                results["documents"][0], results["ids"][0], results["metadatas"][0]
            )
        ]

        return documents
    
    async def async_search(self, vector: List[float], top_k=3):
        """Falls back to synchronous search"""
        return self.search(vector, top_k)

    def remove_documents(self, ids):
        # TODO: update text hashes

        self.collection.delete(ids=ids)
        return len(ids) # cheating here cuz chroma deletes doesn't return anything

    # def clear(self):
    #     super().clear()
    #     return self.collection.delete()

    
