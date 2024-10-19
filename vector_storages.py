from abc import ABC, abstractmethod
from pymilvus import MilvusClient, DataType
import os
from typing import List, Dict

from .embeddings import AsyncOpenAIEmbeddingModel, OpenAIEmbeddingModel
from .models import Document, OPENAI_TEXT_EMBEDDING_SMALL_DIM
from .document_storages import MONGODB_OBJECTID_DIM

class BaseVectorStorage(ABC):
    """Custom VectorStorage Class Interface"""

    @abstractmethod
    def index_documents(self, embeddings: List[float], documents: List[Document]):
        pass

    @abstractmethod
    def search_vector(self, vector: List[float], top_k: int):
        pass


class MilvusVectorStorage(BaseVectorStorage):
    """Custom Vector storage class for using Milvus"""
    def __init__(self, uri: str, token = "", collection_name="financial_context"):
        """init Milvus Vectorstorage Client
        
        Args:
            uri(str): MilvusClient uri
            token(str): token to access zilliz
            collection_name(str): name of collection
        """
        self.client = MilvusClient(
            uri=uri,
            token=token
        )
        self.collection_name = collection_name

        # if collection does not exist, create schema
        if not self.client.has_collection(self.collection_name):
            self._create_schema()

        
    def _create_schema(self):
        # create fields
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=OPENAI_TEXT_EMBEDDING_SMALL_DIM)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_field(field_name="db_id", datatype=DataType.VARCHAR, max_length=MONGODB_OBJECTID_DIM) # mongo db id length = 24 byte

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
            collection_name=self.collection_name, schema=schema, index_params=index_params
        )

    def exit_client(self):
        """exit milvus client. not necessary"""
        self.client.close()

    def index_documents(self, embeddings: List[float], documents: List[Document]):
        """Embed documents with embeddings, index and store into vector storage
        
        Args:
            documents: List of Documents that will be indexed
            embeddings: List of embeddings that matches the documents
        """
        for document, embedding in zip(documents, embeddings):
            data = {
                "text": document.text,
                "vector": embedding,
                "db_id": document.db_id
            }
            
            # insert data into milvus database
            self.client.insert(collection_name=self.collection_name, data=data)
    
    def search_vector(self, vector: List[float], top_k: int = 3) -> List[str]:
        """return relevant results based on vector
        
        Args:
            vector(List[float]): vector to retrieve
            top_k(int): top k results to retrieve
        Returns:
            List of dictionaries with fields text, id, and db_id
            """
        retrieved_data = self.client.search(
            collection_name=self.collection_name,
            data=vector,
            limit=top_k,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["id", "text", "db_id"],
            # filter=f'source == "{source}"' # filter by metadata
        )


        # NOTE: the retrieved data is really stored in retrieved_data[0]
        # TODO: return more than just list of text - return also metadata
        # return list of contexts
        return retrieved_data[0]

