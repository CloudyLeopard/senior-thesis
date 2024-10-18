from pymilvus import MilvusClient, DataType
import os
from typing import List

from .embeddings import AsyncOpenAIEmbeddingModel, OpenAIEmbeddingModel
from .models import Document, OPENAI_TEXT_EMBEDDING_SMALL_DIM

class VectorStorage:
    """Custom VectorStorage Class Interface"""
    def __init__(self):
        pass

    def index_documents(self, embeddings: List[float], documents: List[Document]):
        pass


class MilvusVectorStorage:
    """Custom Vector storage class for using Milvus"""
    def __init__(self, client: MilvusClient, dimension=OPENAI_TEXT_EMBEDDING_SMALL_DIM):
        # # Use Zilliz (Cloud) Vectorstorage
        # self.client = MilvusClient(
        #     uri=os.getenv("ZILLIZ_URI"),
        #     token=os.getenv("ZILLIZ_TOKEN")
        # )
        # self.client = MilvusClient(
        #     uri = "http://localhost:19530"
        # )
        self.client = client
        self.dimension = dimension
        self.collection_name = "financial_context"

        # if collection does not exist, create schema
        if not self.client.has_collection(self.collection_name):
            self._create_schema()

        
    def _create_schema(self):
        # create fields
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2048)
        # TODO: add more metadata fields, or connect with a different database

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

    def index_documents(self, embeddings: List[float], documents: List[Document]):
        """Embed documents with embeddings, index and store into vector storage
        
        Args:
            documents: List of Documents that will be indexed
            embeddings: List of embeddings that matches the documents
        """
        for document, embedding in zip(documents, embeddings):
            data = {
                "text": document.text,
                "vector": embedding
            }
            
            # insert data into milvus database
            self.client.insert(collection_name=self.collection_name, data=data)
    
    def search_vector(self, vector: List[float], top_k: int = 3) -> List[str]:
        retrieved_data = self.client.search(
            collection_name=self.collection_name,
            data=vector,
            limit=top_k,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["id", "text", "title", "snippet", "link"],
            # filter=f'source == "{source}"' # filter by metadata
        )


        # NOTE: the retrieved data is really stored in retrieved_data[0]
        # TODO: return more than just list of text - return also metadata
        # return list of contexts
        return [data["entity"]["text"]for data in retrieved_data[0]]

