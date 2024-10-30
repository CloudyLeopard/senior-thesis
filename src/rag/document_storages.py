from abc import ABC, abstractmethod
from pymongo import MongoClient
from bson import ObjectId
from typing import List
from uuid import UUID

from rag.models import Document

MONGODB_OBJECTID_DIM = 24


class BaseDocumentStore(ABC):
    @abstractmethod
    def save_document(self, document: Document) -> str:
        """save Document into database, return id"""
        ...

    @abstractmethod
    def get_document(self, document_id: str) -> Document | None:
        """given id, fetch Document from database"""
        ...


class MongoDBStore(BaseDocumentStore):
    def __init__(
        self,
        uri,
        db_name="financeContextDB",
    ):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db["documents"]

    def close(self):
        self.client.close()

    def save_document(self, document: Document) -> str:
        """save Document into mongodb collection with metadata, return object id"""
        data = {
            "text": document.text,
            "metadata": document.metadata,
            "uuid": document.uuid
        }
        result = self.collection.insert_one(data)
        return str(result.inserted_id)

    def get_document(self, db_id: str) -> Document | None:
        result = self.collection.find_one({"_id": ObjectId(db_id)})
        if result:  # if found, parse into Document class
            return Document(
                text=result["text"],
                metadata=result["metadata"],
                id=result["uuid"],
                db_id=str(result["_id"])
            )
        else:
            return None
        return result
    
    def get_document_by_uuid(self, uuid: UUID) -> Document | None:
        result = self.collection.findone({"uuid": uuid})
        if result:  # if found, parse into Document class
            return Document(
                text=result["text"],
                metadata=result["metadata"],
                id=result["uuid"],
                db_id=str(result["_id"])
            )
        else:
            return None
    
    def search_document(self, regex: str) -> List[Document] | None:
        cursor = self.collection.find({"text": {"$regex":regex}})
        documents = []        
        for result in cursor:
            documents.append(Document(text=result["text"], metadata=result["metadata"]))
        return documents


    def remove_document(self, db_id: str) -> bool:
        result = self.collection.delete_one({"_id": ObjectId(db_id)})
        return result.acknowledged
