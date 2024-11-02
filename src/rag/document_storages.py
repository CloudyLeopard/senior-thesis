from abc import ABC, abstractmethod
from pymongo import MongoClient
from bson import ObjectId
from typing import List
from uuid import UUID
import os

import urllib

from rag.models import Document

MONGODB_OBJECTID_DIM = 24


class BaseDocumentStore(ABC):
    @abstractmethod
    def save_document(self, document: Document) -> str:
        """save Document into database, return id"""
        ...
    
    @abstractmethod
    def save_documents(self, document: List[Document]) -> List[str]:
        """save Documents into database, return list of ids"""
        ...


    @abstractmethod
    def get_document(self, db_id: str) -> Document | None:
        """given database id, fetch Document from database"""
        ...


class MongoDBStore(BaseDocumentStore):
    def __init__(
        self,
        db_name="financeContextDB",
        uri=None,
    ):
        """
        Initialize MongoDBStore with optional uri and db_name.

        Args:
            db_name (str, optional): The name of the MongoDB database to use. Defaults to "financeContextDB".
            uri (str, optional): The uri of the MongoDB server. Defaults to the value of the MONGODB_URI environment variable.

        Note:
            The MONGODB_URI environment variable must be set if the uri is not provided.
        """
        # set uri (if not provided)
        uri = uri or os.getenv("MONGODB_URI")
        if not uri:
            raise ValueError("uri must be set")
        
        self.client = MongoClient(
            uri,
            uuidRepresentation='standard', # uuidRepresentation is needed to use uuid
        ) 
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

        # set db_id in passed document
        document.set_db_id(str(result.inserted_id))
        return str(result.inserted_id)
    
    def save_documents(self, documents: List[Document]) -> List[str]:
        data = []
        for document in documents:
            data.append({
                "text": document.text,
                "metadata": document.metadata,
                "uuid": document.uuid
            })
        result = self.collection.insert_many(data)

        # set db_id in passed document
        for i in range(len(documents)):
            documents[i].set_db_id(str(result.inserted_ids[i]))

        return [str(doc_id) for doc_id in result.inserted_ids]

    def get_document(self, db_id: str) -> Document | None:
        result = self.collection.find_one({"_id": ObjectId(db_id)})
        if result:  # if found, parse into Document class
            return Document(
                text=result["text"],
                metadata=result["metadata"],
                uuid=result["uuid"],
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
                uuid=result["uuid"],
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
