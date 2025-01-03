import asyncio
import numpy as np
from abc import ABC, abstractmethod
from pymilvus import MilvusClient, DataType
from uuid import UUID
from typing import List, Any
import os
import chromadb
import logging

from rag.models import Document, OPENAI_TEXT_EMBEDDING_SMALL_DIM
from rag.embeddings import BaseEmbeddingModel
from rag.document_storages import MONGODB_OBJECTID_DIM

logger = logging.getLogger(__name__)

class BaseVectorStorage(ABC):
    """Custom VectorStorage Class Interface. A vectorstorage is required to have a
    'text' field and a 'vector' field."""

    def __init__(self, embedding_model: BaseEmbeddingModel):
        self.embedding_model = embedding_model
        self.texts_hashes = set()

    def __contains__(self, item) -> bool:
        return hash(item) in self.texts_hashes

    def __len__(self) -> int:
        return len(self.texts_hashes)

    @abstractmethod
    def insert_documents(self, documents: List[Document]) -> List[int]:
        """insert documents, embed them, return list of ids"""
        self.texts_hashes.update(hash(document.text) for document in documents)

    @abstractmethod
    def similarity_search(self, query: str, top_k: int) -> List[Document]:
        """given query, return top_k relevant results"""
        pass

    @abstractmethod
    def remove_documents(self, ids: List[int]) -> int:
        """remove documents by their ids"""
        # self.texts_hashes.difference_update(ids)
        # TODO: this doesn't work lol
        pass

    @abstractmethod
    async def async_insert_documents(self, documents: List[Document]) -> List[int]:
        """insert documents, embed them, return list of ids"""
        self.texts_hashes.update(hash(document.text) for document in documents)

    @abstractmethod
    async def async_similarity_search(self, query: str, top_k: int) -> List[Document]:
        """given query, asynchronously return top_k relevant results"""
        pass

    @abstractmethod
    def clear(self) -> None:
        self.texts_hashes = set()

    def as_retriever(self):
        from rag.retrievers import SimpleRetriever

        return SimpleRetriever(index=self)
    
    def close(self):
        """close the vector storage"""
        pass

class MilvusVectorStorage(BaseVectorStorage):
    """Custom Vector storage class for using Milvus"""

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        collection_name="financial_context",
        uri: str = "",
        token: str = "",
        reset_collection: bool = False,
    ):
        """
        Initialize Milvus Vector Storage.

        Args:
        - embedding_model (BaseEmbeddingModel): The embedding model to use for vectorizing text.
        - collection_name (str): The name of the Milvus collection to store the documents in. Defaults to "financial_context".
        - uri (str): The uri of the Milvus server. Defaults to the value of the ZILLIZ_URI environment variable.
        - token (str): The token to use for authentication. Defaults to the value of the ZILLIZ_TOKEN environment variable.
        - reset_collection (bool): Whether to reset the collection before inserting documents. Defaults to False.

        If the collection does not exist, it is created with the appropriate schema.
        """
        super().__init__(embedding_model)

        self.client = MilvusClient(
            uri=uri or os.getenv("ZILLIZ_URI"),
            token=token or os.getenv("ZILLIZ_TOKEN"),
        )
        self.collection_name = collection_name

        # if collection does not exist, create schema
        if reset_collection:
            self.client.drop_collection(
                self.collection_name
            )  # dev mode, reset every time
        if not self.client.has_collection(self.collection_name):
            self._create_schema_and_collection()

    def _create_schema_and_collection(self):
        # create fields
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=OPENAI_TEXT_EMBEDDING_SMALL_DIM,
        )
        schema.add_field(
            field_name="text", datatype=DataType.VARCHAR, max_length=2048
        )  # stores chunked text

        schema.add_field(
            field_name="db_id",
            datatype=DataType.VARCHAR,
            max_length=MONGODB_OBJECTID_DIM,
        )  # mongo db id length = 24 byte

        schema.add_field(
            field_name="uuid",
            datatype=DataType.VARCHAR,
            max_length=128,  # uuid is 128 bit
        )  # stores uuid

        schema.add_field(
            field_name="datasource", datatype=DataType.VARCHAR, max_length=128
        )

        # create index
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="id",
        )
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="IP",
        )

        # create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def clear(self):
        """delete all documents in milvus"""
        super().clear()
        self.client.delete(collection_name=self.collection_name, filter="id >= 0")

    def close(self):
        """exit milvus client. not necessary"""
        self.client.close()

    def reset_collection(self):
        self.client.drop_collection(self.collection_name)
        self._create_schema_and_collection()

    def insert_documents(self, documents: List[Document]) -> List[int]:
        """Embed documents with embeddings, index and store into vector storage

        Args:
            documents: List of Documents that will be indexed
        """
        # remove duplicate documents
        documents = [doc for doc in documents if (doc_hash := hash(doc)) not in self.texts_hashes and not self.texts_hashes.add(doc_hash)]

        data = []
        for document in documents:
            # required fields
            entry = {
                "text": document.text,
                "vector": self.embedding_model.embed([document.text])[0],
                "uuid": str(document.uuid),
            }

            # optional field
            entry["db_id"] = document.db_id
            entry["datasource"] = document.metadata.get("datasource", "")
            data.append(entry)

        # insert data into milvus database
        res = self.client.insert(collection_name=self.collection_name, data=data)
        return res["ids"]  # returns id of inserted vector

    def remove_documents(self, ids: List[Any]) -> int:
        """delete document based on primary id in milvus. returns deleted count"""
        super().remove_documents(ids)
        res = self.client.delete(
            collection_name=self.collection_name,
            filter=f"id in [{','.join([str(id) for id in ids])}]",
        )
        return res["delete_count"]

    def _search_vector_storage(self, vectors: List[List[float]], top_k: int):
        """Search milvus and get top_k relevant results given list of vectors.
        Returns List[List[Dict]]"""

        # output_fields determines which metadata to extract
        retrieved_data = self.client.search(
            collection_name=self.collection_name,
            data=vectors,
            limit=top_k,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["id", "text", "uuid", "db_id", "datasource"],
            # filter=f'source == "{source}"' # filter by metadata
        )

        return retrieved_data

    def similarity_search(self, query: str, top_k=3):
        """given query, return top_k relevant results"""
        embedding = self.embedding_model.embed([query])[0]

        return self.similarity_search_by_vector(embedding, top_k)

    def similarity_search_by_vector(
        self, vector: List[float], top_k: int = 3
    ) -> List[Document]:
        """return top_k relevant results based on vector

        Args:
            vector(List[float]): vector to retrieve
            top_k(int): top k results to retrieve
        Returns:
            return list of most relevant document with metadata stored in
            vector storage. Not connected to document storage.
        """

        retrieved_data = self._search_vector_storage(vectors=[vector], top_k=top_k)
        top_results = retrieved_data[0]

        top_documents = []
        for result in top_results:
            entity = result["entity"]
            text = entity.pop("text")
            uuid = entity.pop("uuid")
            db_id = entity.pop("db_id")

            doc = Document(
                text=text,
                uuid=UUID(uuid),
                metadata=entity,  # whatever is left is part of metadata
                db_id=db_id,
            )

            top_documents.append(doc)

        return top_documents

    async def async_similarity_search(self, query: str, top_k=3):
        """Milvus does not support async search, so this falls back to
        synchronous similarity_search"""
        return self.similarity_search(query, top_k)

    async def async_insert_documents(self, documents):
        """Milvus does not support async insert, so this falls back to
        synchronous insert_documents"""
        return self.insert_documents(documents)


class ChromaVectorStorage(BaseVectorStorage):
    """Custom Vector storage class for using Chroma"""

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        collection_name: str = "financial_context",
        persist_directory: str = None,
        client: chromadb.Client = None,
    ):
        super().__init__(embedding_model)

        if client is not None:
            self.client = client
        else:
            if persist_directory:
                self.client = chromadb.PersistentClient(path=persist_directory)
            else:
                self.client = chromadb.Client()

        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(collection_name)

    def insert_documents(self, documents: List[Document]):
        # remove duplicate documents
        documents = [doc for doc in documents if (doc_hash := hash(doc)) not in self.texts_hashes and not self.texts_hashes.add(doc_hash)]

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

    def similarity_search(self, query: str, top_k=3):
        embeddings = self.embedding_model.embed([query])

        results = self.collection.query(
            query_embeddings=embeddings,
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

    def remove_documents(self, ids):
        super().remove_documents(ids)

        self.collection.delete(ids=ids)
        return len(ids) # cheating here cuz chroma deletes doesn't return anything

    def clear(self):
        super().clear()
        return self.collection.delete()

    async def async_insert_documents(self, documents):
        """Not yet implemented. Falls back to synchronous insert_documents"""
        return self.insert_documents(documents)

    async def async_similarity_search(self, query: str, top_k=3):
        """Not yet implemented. Falls back to synchronous similarity_search"""
        return self.similarity_search(query, top_k)

def cosine_similarity(a: np.ndarray, b: np.ndarray):
    """calculates cosine similarity between a and b, both 2d array of shape (n or m, embedding_dim)
    returns a 2d array of shape (n, m) where similarity[i][j] = cosine similarity between a[i] and b[j]
    """
    norm_product = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.dot(a, b.T) / norm_product
class NumPyVectorStorage(BaseVectorStorage):
    def __init__(self, embedding_model: BaseEmbeddingModel):
        super().__init__(embedding_model)
        self.documents: List[Document] = []
        self._embeddings_matrix: np.ndarray = None
    
    async def async_insert_documents(self, documents: List[Document]):
        # remove duplicate documents
        documents = [doc for doc in documents if (doc_hash := hash(doc)) not in self.texts_hashes and not self.texts_hashes.add(doc_hash)]

        texts = [document.text for document in documents]
        embeddings = await self.embedding_model.async_embed(texts)
        if self._embeddings_matrix is None:
            self._embeddings_matrix = np.array(embeddings)
            ids = list(range(len(documents)))
        else:
            self._embeddings_matrix = np.concatenate((self._embeddings_matrix, np.array(embeddings)), axis=0)
            ids = list(range(len(self.documents), len(documents) + len(self.documents)))
        
        self.documents.extend(documents)
        return ids

    def insert_documents(self, documents: List[Document]):
        return asyncio.run(self.async_insert_documents(documents))
    
    async def async_similarity_search(self, query: str, top_k: int = 3) -> List[Document]:
        top_k = min(top_k, len(self.documents))
        if (top_k == 0):
            return []

        query_embeddings = await self.embedding_model.async_embed([query])
        np_query = np.array(query_embeddings[0]) # this is a 1D array

        similarity_scores = cosine_similarity(np_query.reshape(1, -1), self._embeddings_matrix)[0]
        similarity_scores = np.nan_to_num(similarity_scores, nan=-np.inf) # treat nan values as -inf
        sorted_indices = np.argsort(similarity_scores)[::-1] # sort in descending order
        top_indices = sorted_indices[:top_k] # get top k indices

        documents = [self.documents[index] for index in top_indices]
        return documents

    def similarity_search(self, query: str, top_k: int = 3):
        return asyncio.run(self.async_similarity_search(query, top_k))
        
    def remove_documents(self, ids: List[int]):
        mask = np.ones(len(self.documents), dtype=bool)
        mask[ids] = False
        original_size = len(self.documents)
        self.documents = [doc for doc, keep in zip(self.documents, mask) if keep]
        self._embeddings_matrix = self._embeddings_matrix[mask]
        return original_size - len(self.documents)
    
    def clear(self):
        super().clear()
        self.documents = []
        self._embeddings_matrix = None
