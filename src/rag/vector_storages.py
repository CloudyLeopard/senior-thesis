from abc import ABC, abstractmethod
from pymilvus import MilvusClient, DataType
from uuid import UUID
import os
from typing import List, Dict, Any

from rag.models import Document, OPENAI_TEXT_EMBEDDING_SMALL_DIM

from rag.embeddings import BaseEmbeddingModel
from rag.document_storages import MONGODB_OBJECTID_DIM


class BaseVectorStorage(ABC):
    """Custom VectorStorage Class Interface. A vectorstorage is required to have a
    'text' field and a 'vector' field."""

    def __init__(self, embedding_model: BaseEmbeddingModel):
        self.embedding_model = embedding_model

    @abstractmethod
    def insert_documents(
        self, documents: List[Document]
    ) -> List[int]:
        """insert documents, embed them, return list of ids"""
        pass

    @abstractmethod
    def similarity_search(self, query: str, top_k: int):
        """given query, return top_k relevant results"""
        pass

    @abstractmethod
    def similarity_search_by_vector(
        self, vector: List[float], top_k: int = 3
    ) -> List[Document]:
        pass


class MilvusVectorStorage(BaseVectorStorage):
    """Custom Vector storage class for using Milvus"""

    def __init__(self, embedding_model: BaseEmbeddingModel, uri: str, token="", collection_name="financial_context"):
        """init Milvus Vectorstorage Client

        Args:
            uri(str): MilvusClient uri
            token(str): token to access zilliz
            collection_name(str): name of collection
        """
        self.embedding_model = embedding_model
        self.client = MilvusClient(uri=uri, token=token)
        self.collection_name = collection_name

        # if collection does not exist, create schema
        self.client.drop_collection(self.collection_name) # dev mode, reset every time
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

    def close(self):
        """exit milvus client. not necessary"""
        self.client.close()

    def insert_documents(
        self, documents: List[Document]
    ) -> List[int]:
        """Embed documents with embeddings, index and store into vector storage

        Args:
            documents: List of Documents that will be indexed
        """
        data = []
        for document in documents:
            # required fields
            entry = {
                "text": document.text,
                "vector": self.embedding_model.embed([document.text])[0],
                "uuid": str(document.uuid)
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
        res = self.client.delete(
            collection_name=self.collection_name,
            filter=f"id in [{','.join([str(id) for id in ids])}]",
        )
        return res["delete_count"]

    def _search_vector_storage (self, vectors: List[List[float]], top_k: int):
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
                metadata=entity, # whatever is left is part of metadata
                db_id=db_id
            )
            
            top_documents.append(doc)
        
        return top_documents



