from pydantic import Optional, field_validator
from pymilvus import MilvusClient, DataType
from uuid import UUID
from typing import List, Any
import os
import logging


from rag.vector_store.base_store import BaseVectorStore
from rag.models import Document, OPENAI_TEXT_EMBEDDING_SMALL_DIM
from rag.document_store import MONGODB_OBJECTID_DIM

logger = logging.getLogger(__name__)

class MilvusVectorStore(BaseVectorStore):
    """Custom Vector storage class for using Milvus"""

    collection_name: str = "financial_context"
    uri: Optional[str] = None
    token: Optional[str] = None
    reset_collection: bool = False

    # Validator to load default values from environment variables
    @field_validator("uri", "token", mode="before")
    def validate_env_vars(cls, value, field_name):
        if value is None:
            env_var = os.getenv(f"ZILLIZ_{field_name.upper()}")
            if env_var:
                return env_var
        return value
    
    # Post-initialization logic
    def model_post_init(self):
        self.client = MilvusClient(
            uri=self.uri,
            token=self.token,
        )

        if self.reset_collection:
            self.client.drop_collection(self.collection_name)
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

    async def async_insert_documents(self, documents):
        """Fall back to synchronous insert_documents"""
        return self.insert_documents(documents)

    def search(self, vector: List[float], top_k: int = 3) -> List[Document]:
        retrieved_data = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            limit=top_k,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["id", "text", "uuid", "db_id", "datasource"],
            # filter=f'source == "{source}"' # filter by metadata
        )
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

    async def async_search(self, query: str, top_k=3):
        """Milvus does not support async search, so this falls back to
        synchronous similarity_search"""
        return self.search(query, top_k)

    
    def remove_documents(self, ids: List[Any]) -> int:
        """delete document based on primary id in milvus. returns deleted count"""
        # TODO: update text hashes

        res = self.client.delete(
            collection_name=self.collection_name,
            filter=f"id in [{','.join([str(id) for id in ids])}]",
        )
        return res["delete_count"]
    
    # def clear(self):
    #     """delete all documents in milvus"""
    #     super().clear()
    #     self.client.delete(collection_name=self.collection_name, filter="id >= 0")

    # def close(self):
    #     """exit milvus client. not necessary"""
    #     self.client.close()