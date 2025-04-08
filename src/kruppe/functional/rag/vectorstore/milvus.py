from calendar import c
from pydantic import field_validator, Field
from pymilvus import MilvusClient, DataType
from uuid import UUID
from typing import List, Any, Optional, Dict
import os
import logging


from kruppe.functional.rag.vectorstore.base_store import BaseVectorStore
from kruppe.models import Document, OPENAI_TEXT_EMBEDDING_SMALL_DIM
# from rag.document_store import MONGODB_OBJECTID_DIM

logger = logging.getLogger(__name__)

class MilvusVectorStore(BaseVectorStore):
    """Custom Vector storage class for using Milvus"""
    # TODO: enable dynamic field
    # https://milvus.io/docs/enable-dynamic-field.md

    collection_name: str = "financial_context"
    uri: str = Field(default_factory = lambda x: os.getenv("ZILLIZ_URI"))
    token: str = Field(default_factory = lambda x: os.getenv("ZILLIZ_TOKEN"))
    reset_collection: bool = False
    client: Optional[MilvusClient] = None
    
    # Post-initialization logic
    def model_post_init(self, __context):
        if not self.client:
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

        # schema.add_field(
        #     field_name="db_id",
        #     datatype=DataType.VARCHAR,
        #     max_length=MONGODB_OBJECTID_DIM,
        # )  # mongo db id length = 24 byte

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

        data = []
        for document in documents:
            # required fields
            entry = {
                "text": document.text,
                "vector": self.embedding_model.embed([document.text])[0],
                "uuid": str(document.uuid),
            }

            # optional field
            # entry["db_id"] = document.db_id
            entry["datasource"] = document.metadata.get("datasource", "")
            data.append(entry)

        # insert data into milvus database
        res = self.client.insert(collection_name=self.collection_name, data=data)
        return res["ids"]  # returns id of inserted vector

    async def async_insert_documents(self, documents):
        """Fall back to synchronous insert_documents"""
        return self.insert_documents(documents)

    def search(self, vector: List[float], top_k: int = 3, filter: Dict[str, Any] = None) -> List[Document]:
        
        milvus_filter = convert_filter_to_milvus(filter) if filter else ""

        retrieved_data = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            limit=top_k,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["id", "text", "uuid", "datasource"], # + ["db_id"],
            filter=milvus_filter
        )
        top_results = retrieved_data[0]

        top_documents = []
        for result in top_results:
            entity = result["entity"]
            text = entity.pop("text")
            uuid = entity.pop("uuid")
            # db_id = entity.pop("db_id")

            doc = Document(
                text=text,
                uuid=UUID(uuid),
                metadata=entity,  # whatever is left is part of metadata
                # db_id=db_id,
            )

            top_documents.append(doc)

        return top_documents

    def async_search(self, vector: List[float], top_k: int = 3, filter: Dict[str, Any] = None) -> List[Document]:
        """falls back to synchronous similarity_search"""
        # TODO: use async client for async search
        return self.search(vector=vector, top_k=top_k, filter=filter)

    
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


def convert_filter_to_milvus(filter_rule: dict) -> str:
    """
    Converts a metadata filter from a Pinecone/Chroma style (JSON-like) format
    into a Milvus boolean expression string.

    Example:
        Input: {
            "age": {"$gte": 18},
            "country": "USA",
            "$or": [
                {"score": {"$gt": 80}},
                {"verified": True}
            ]
        }
        Output:
            "age >= 18 and country == 'USA' and ((score > 80) or (verified == True))"

    Supported operators:
      - Equality: Direct values or "$eq"
      - "$ne"  -> "!="
      - "$gt"  -> ">"
      - "$gte" -> ">="
      - "$lt"  -> "<"
      - "$lte" -> "<="
      - "$in"  -> "in" (expects a list)
      - "$nin" -> "not in" (expects a list)
      - Logical operators: "$and", "$or", "$not"
    """
    def format_value(val):
        """Format a value for Milvus expression (e.g., add quotes for strings)."""
        if isinstance(val, str):
            return f"'{val}'"
        elif isinstance(val, list):
            # Recursively format each element in the list
            return "[" + ", ".join(format_value(v) for v in val) + "]"
        else:
            return str(val)

    def parse_operator(field: str, op_dict: dict) -> str:
        """Parse a dictionary of operators for a given field."""
        expressions = []
        for operator, value in op_dict.items():
            if operator == "$eq":
                expressions.append(f"{field} == {format_value(value)}")
            elif operator == "$ne":
                expressions.append(f"{field} != {format_value(value)}")
            elif operator == "$gt":
                expressions.append(f"{field} > {format_value(value)}")
            elif operator == "$gte":
                expressions.append(f"{field} >= {format_value(value)}")
            elif operator == "$lt":
                expressions.append(f"{field} < {format_value(value)}")
            elif operator == "$lte":
                expressions.append(f"{field} <= {format_value(value)}")
            elif operator == "$in":
                expressions.append(f"{field} in {format_value(value)}")
            elif operator == "$nin":
                expressions.append(f"{field} not in {format_value(value)}")
            else:
                raise ValueError(f"Unsupported operator: {operator}")
        return " and ".join(expressions)

    def parse_filter(filt: dict) -> str:
        """Recursively parse the filter dictionary."""
        expressions = []
        for key, value in filt.items():
            if key == "$and":
                # Expect value to be a list of filter dicts
                sub_exprs = [f"({parse_filter(sub)})" for sub in value]
                expressions.append(" and ".join(sub_exprs))
            elif key == "$or":
                sub_exprs = [f"({parse_filter(sub)})" for sub in value]
                expressions.append(" or ".join(sub_exprs))
            elif key == "$not":
                expressions.append(f"not ({parse_filter(value)})")
            else:
                # key is assumed to be a field name
                if isinstance(value, dict):
                    # Use the operator parser for multiple operators on the same field
                    expressions.append(parse_operator(key, value))
                else:
                    # Direct equality check
                    expressions.append(f"{key} == {format_value(value)}")
        return " and ".join(expressions)

    return parse_filter(filter_rule)


# Example usage:
if __name__ == "__main__":
    # Example filter (Pinecone/Chroma style)
    pinecone_filter = {
        "age": {"$gte": 18},
        "country": "USA",
        "$or": [
            {"score": {"$gt": 80}},
            {"verified": True}
        ]
    }

    milvus_expression = convert_filter_to_milvus(pinecone_filter)
    print("Milvus filter expression:")
    print(milvus_expression)