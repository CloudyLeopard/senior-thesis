import numpy as np
from uuid import UUID
from typing import List, Dict, Any, Tuple
from pydantic import Field, PrivateAttr, model_validator
from typing import Optional
import chromadb
from chromadb.utils.batch_utils import create_batches
import logging

from kruppe.functional.rag.vectorstore.base_store import BaseVectorStore
from kruppe.models import Document, Chunk

logger = logging.getLogger(__name__)

def distance_to_similarity(distances: List[float]) -> List[float]:
    return (1 / (1 + np.array(distances))).tolist()

def get_chroma_collection_names(client: chromadb.ClientAPI) -> List[str]:
    return [collection.name for collection in client.list_collections()]

class ChromaVectorStore(BaseVectorStore):
    """Custom Vector storage class for using Chroma"""

    collection_name: str = Field(default="financial_context")
    persist_path: Optional[str] = None
    client: Optional[chromadb.ClientAPI] = None
    _collection: chromadb.Collection = PrivateAttr()

    @model_validator(mode="after")
    def validate_client_and_collection(self):     
        # three options for clients   
        if self.client is None:
            if self.persist_path:
                # option 1: persistent client
                self.client = chromadb.PersistentClient(path=self.persist_path)
            else:
                # option 2: in-memory client
                self.client = chromadb.Client()
            # option 3: client passed in

        self._collection = self.client.get_or_create_collection(
            self.collection_name,
            metadata={"hnsw:space": "l2"},
        )
        return self

    def size(self) -> int:
        return self._collection.count()
    
    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self._collection = self.client.get_or_create_collection(self.collection_name)

    def insert_documents(self, documents: List[Document]):
        if len(documents) == 0:
            logger.warning("No documents to insert into ChromaDB")
            return []
        
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
                metadata["prev_chunk_id"] = str(document.prev_chunk_id) if document.prev_chunk_id else ""
                metadata["next_chunk_id"] = str(document.next_chunk_id) if document.next_chunk_id else ""

            data["metadatas"].append(document.metadata)

            # add text for each document/chunk
            data["documents"].append(document.text)

        # insert data into chromadb
        batches = create_batches(api=self.client, ids=data["ids"], metadatas=data["metadatas"], documents=data["documents"], embeddings=data["embeddings"])
        for batch in batches:
            self._collection.upsert(
                ids=batch[0],
                documents=batch[3],
                embeddings=batch[1],
                metadatas=batch[2],
            )
        
        logger.info("Inserted %d documents/chunks into ChromaDB", len(data["documents"]))

        return data["ids"]
    
    async def async_insert_documents(self, documents):
        if len(documents) == 0:
            logger.warning("No documents to insert into ChromaDB")
            return []
        
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
                # need to handle None values for prev_chunk_id and next_chunk_id... 
                # AHHHH this screwed me over for a while
                metadata["prev_chunk_id"] = str(document.prev_chunk_id) if document.prev_chunk_id else ""
                metadata["next_chunk_id"] = str(document.next_chunk_id) if document.next_chunk_id else ""

            data["metadatas"].append(document.metadata)

            # add text for each document/chunk
            data["documents"].append(document.text)

        # insert data into chromadb
        batches = create_batches(api=self.client, ids=data["ids"], metadatas=data["metadatas"], documents=data["documents"], embeddings=data["embeddings"])
        for batch in batches:
            self._collection.upsert(
                ids=batch[0],
                documents=batch[3],
                embeddings=batch[1],
                metadatas=batch[2],
            )
        
        logger.info("Inserted %d documents/chunks into ChromaDB", len(data["documents"]))

        return data["ids"]

    def search(self, vector: List[float], top_k: int = 3, filter: Dict[str, Any] = None) -> Tuple[List[Chunk], List[int]]:
        if filter == {}: 
            # apparently chroma doesn't like empty filters
            filter = None

        results = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=filter
        )

        retrieved_docs = [] # chunks, really - im treating docs like chunks
        similarity_scores = distance_to_similarity(results["distances"][0]) # NOTE: chroma by default returns L2 norm or euclidean distance
        for res_text, res_id, res_metadata, res_score in zip(results["documents"][0], results["ids"][0], results["metadatas"][0], similarity_scores):
            document_id = res_metadata.pop("document_id", res_id) # if no document_id, then chunk_id is document_id
            
            # get prev_chunk_id and next_chunk_id if they exist
            # NOTE: either the data doesn't have a "prev_chunk_id" or "next_chunk_id" field
            # OR it is the first/last chunk, and its field's value is ""
            prev_chunk_id = res_metadata.pop("prev_chunk_id", None)
            next_chunk_id = res_metadata.pop("next_chunk_id", None)

            # create a Chunk object for each chunked text
            retrieved_chunk = Chunk(
                text=res_text,
                id=UUID(res_id),
                metadata=res_metadata,
                document_id=UUID(document_id),
                score=res_score,
                prev_chunk_id=UUID(prev_chunk_id) if prev_chunk_id else None,
                next_chunk_id=UUID(next_chunk_id) if next_chunk_id else None,
            )

            retrieved_docs.append(retrieved_chunk)

        return retrieved_docs
    
    async def async_search(self, vector: List[float], top_k=3, filter: Dict[str, Any] = None) -> List[Chunk]:
        """Falls back to synchronous search"""
        return self.search(vector=vector, top_k=top_k, filter=filter)

    def remove_documents(self, ids):
        # TODO: update text hashes

        self._collection.delete(ids=ids)
        return len(ids) # cheating here cuz chroma deletes doesn't return anything

    # def clear(self):
    #     super().clear()
    #     return self._collection.delete()

    
