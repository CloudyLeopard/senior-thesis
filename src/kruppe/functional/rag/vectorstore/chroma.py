from uuid import UUID
from typing import List, Dict, Any
from pydantic import Field
from typing import Optional
import chromadb
import logging

from kruppe.functional.rag.vectorstore.base_store import BaseVectorStore
from kruppe.models import Document, Chunk

logger = logging.getLogger(__name__)

class ChromaVectorStore(BaseVectorStore):
    """Custom Vector storage class for using Chroma"""

    collection_name: str = Field(default="financial_context")
    persist_path: Optional[str] = None
    client: Optional[chromadb.ClientAPI] = None
    collection: Optional[chromadb.Collection] = None

    def model_post_init(self, __context):
        if self.client is None:
            if self.persist_path:
                self.client = chromadb.PersistentClient(path=self.persist_path)
            else:
                self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(self.collection_name)

    def insert_documents(self, documents: List[Document]):
        data = {
            "ids": [],
            "metadatas": [],
            "documents": [],
            "embeddings": self.embedding_model.embed(
                [document.text for document in documents]
            ),
        }

        for document in documents:
            # unique id for each document/chunk
            data["ids"].append(str(document.id))

           # metadata for each document/chunk
            metadata = document.metadata
            if isinstance(document, Chunk):
                metadata["document_id"] = str(document.document_id)
                metadata["prev_chunk_id"] = str(document.prev_chunk_id)
                metadata["next_chunk_id"] = str(document.next_chunk_id)
            data["metadatas"].append(document.metadata)

            # add text for each document/chunk
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
        embeddings = await self.embedding_model.async_embed(
            [document.text for document in documents]
        )

        # insert data into chromadb
        data = {
            "ids": [],
            "metadatas": [],
            "documents": [],
            "embeddings": embeddings,
        }

        for document in documents:
            # unique id for each document/chunk
            data["ids"].append(str(document.id))

            # metadata for each document/chunk
            metadata = document.metadata
            if isinstance(document, Chunk):
                metadata["document_id"] = str(document.document_id)
                metadata["prev_chunk_id"] = str(document.prev_chunk_id)
                metadata["next_chunk_id"] = str(document.next_chunk_id)
            data["metadatas"].append(document.metadata)

            # add text for each document/chunk
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

    def search(self, vector: List[float], top_k: int = 3, filter: Dict[str, Any] = None) -> List[Chunk]:
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=filter
        )

        retrieved_docs = [] # chunks, really - im treating docs like chunks
        for res_text, res_id, res_metadata in zip(results["documents"][0], results["ids"][0], results["metadatas"][0]):
            document_id = res_metadata.pop("document_id", res_id) # if no document_id, then chunk_id is document_id
            prev_chunk_id = res_metadata.pop("prev_chunk_id", None)
            next_chunk_id = res_metadata.pop("next_chunk_id", None)

            retrieved_chunk = Chunk(
                text=res_text,
                id=UUID(res_id),
                metadata=res_metadata,
                document_id=UUID(document_id) ,
                prev_chunk_id=UUID(prev_chunk_id) if prev_chunk_id else None,
                next_chunk_id=UUID(next_chunk_id) if next_chunk_id else None,
            )

            retrieved_docs.append(retrieved_chunk)

        return retrieved_docs
    
    async def async_search(self, vector: List[float], top_k=3, filter: Dict[str, Any] = None):
        """Falls back to synchronous search"""
        return self.search(vector=vector, top_k=top_k, filter=filter)

    def remove_documents(self, ids):
        # TODO: update text hashes

        self.collection.delete(ids=ids)
        return len(ids) # cheating here cuz chroma deletes doesn't return anything

    # def clear(self):
    #     super().clear()
    #     return self.collection.delete()

    
