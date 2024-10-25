from abc import ABC, abstractmethod
from pymilvus import MilvusClient, DataType
import os
from typing import List, Dict, Any

from .models import Document, OPENAI_TEXT_EMBEDDING_SMALL_DIM
from .document_storages import MONGODB_OBJECTID_DIM

class BaseVectorStorage(ABC):
    """Custom VectorStorage Class Interface"""

    @abstractmethod
    def insert_documents(self, embeddings: List[List[float]], documents: List[Document]):
        pass

    @abstractmethod
    def search_vectors(self, vectors: List[List[float]], top_k: int = 3) -> List[List[Dict]]:
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
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2048) # stores chunked text
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

    def close(self):
        """exit milvus client. not necessary"""
        self.client.close()

    def insert_documents(self, embeddings: List[List[float]], documents: List[Document]):
        """Embed documents with embeddings, index and store into vector storage
        
        Args:
            documents: List of Documents that will be indexed
            embeddings: List of embeddings that matches the documents
        """
        data = []
        for document, embedding in zip(documents, embeddings):
            data.append({
                "text": document.text,
                "vector": embedding,
                "db_id": document.db_id
            })
            
        # insert data into milvus database
        res = self.client.insert(collection_name=self.collection_name, data=data)
        return res["ids"] # returns id of inserted vector
    
    def remove_documents(self, ids: List[Any]):
        """delete document based on primary id in milvus. returns deleted count"""
        res = self.client.delete(
            collection_name=self.collection_name,
            filter = f"id in [{','.join([str(id) for id in ids])}]"
        )
        print(res)
        return res
    
    def search_vectors(self, vectors: List[List[float]], top_k: int = 3) -> List[List[Dict]]:
        """return relevant results based on vector
        
        Args:
            vector(List[float]): vector to retrieve
            top_k(int): top k results to retrieve
        Returns:
            List of List of dictionaries with fields text, id, and db_id.
            Outer list corresponds to the vectors. Inner list coresponds to the result and rank
            """
        retrieved_data = self.client.search(
            collection_name=self.collection_name,
            data=vectors,
            limit=top_k,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["id", "text", "db_id"],
            # filter=f'source == "{source}"' # filter by metadata
        )


        # NOTE: the retrieved data is really stored in retrieved_data[0]
        # return list of contexts
        return retrieved_data
