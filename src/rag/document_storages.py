from abc import ABC, abstractmethod
from pymongo import MongoClient
from bson import ObjectId
from typing import List, Optional
from uuid import UUID
import os
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio


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

    @abstractmethod
    def get_document_by_uuid(self, uuid: UUID) -> Document | None:
        """given uuid, fetch Document from database"""
        ...


class MongoDBStore(BaseDocumentStore):
    def __init__(
        self,
        db_name="financeContextDB",
        uri=None,
        reset_db: bool = False
    ):
        """
        Initialize MongoDBStore with optional uri and db_name.

        Args:
            db_name (str, optional): The name of the MongoDB database to use. Defaults to "financeContextDB".
            uri (str, optional): The uri of the MongoDB server. Defaults to the value of the MONGODB_URI environment variable.
            reset_db (bool, optional): Whether to reset the database before inserting documents. Defaults to False.
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

        if reset_db:
            self.db.drop_collection("documents")
            self.db.create_collection("documents")

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
        result = self.collection.find_one({"uuid": uuid})

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

class AsyncMongoDBStore(BaseDocumentStore):
    def __init__(
        self,
        db_name: str = "financeContextDB",
        uri: Optional[str] = None,
    ):
        """
        Initialize instance variables but don't create connections.
        Async initialization is handled by create().
        """
        self.uri = uri or os.getenv("MONGODB_URI")
        if not self.uri:
            raise ValueError("uri must be set")
        
        self.db_name = db_name
        # Don't create the client in __init__
        self.client = None
        self.db = None
        self.collection = None

    @classmethod
    async def create(
        cls,
        db_name: str = "financeContextDB",
        uri: Optional[str] = None,
        reset_db: bool = False
    ) -> "AsyncMongoDBStore":
        """
        Factory method to create and initialize a new AsyncMongoDBStore instance.

        Args:
            db_name: The name of the MongoDB database to use
            uri: The MongoDB connection URI
            reset_db: Whether to reset the database during initialization

        Returns:
            An initialized AsyncMongoDBStore instance

        Usage:
            store = await AsyncMongoDBStore.create(reset_db=True)
        """
        self = cls(db_name, uri)
        
        # Create client with the current event loop
        self.client = AsyncIOMotorClient(
            self.uri,
            uuidRepresentation='standard',
        )
        # the magic line that fixes the "attached to a different event loop" error
        self.client.get_io_loop = asyncio.get_event_loop 

        self.db = self.client[self.db_name]
        self.collection = self.db["documents"]

        # Perform async initialization if needed
        if reset_db:
            await self.db.drop_collection("documents")
            await self.db.create_collection("documents")

        return self

    async def close(self) -> None:
        """Close the database connection."""
        self.client.close()

    async def save_document(self, document: Document) -> str:
        """
        Save a single document to MongoDB.

        Args:
            document: Document instance to save

        Returns:
            str: The database ID of the saved document
        """
        data = {
            "text": document.text,
            "metadata": document.metadata,
            "uuid": document.uuid
        }
        result = await self.collection.insert_one(data)

        # set db_id in passed document
        document.set_db_id(str(result.inserted_id))
        return str(result.inserted_id)
    
    async def save_documents(self, documents: List[Document]) -> List[str]:
        """
        Save multiple documents to MongoDB.

        Args:
            documents: List of Document instances to save

        Returns:
            List[str]: List of database IDs for the saved documents
        """
        data = [
            {
                "text": doc.text,
                "metadata": doc.metadata,
                "uuid": doc.uuid
            }
            for doc in documents
        ]
        
        result = await self.collection.insert_many(data)

        # set db_id in passed documents
        for doc, doc_id in zip(documents, result.inserted_ids):
            doc.set_db_id(str(doc_id))

        return [str(doc_id) for doc_id in result.inserted_ids]

    async def get_document(self, db_id: str) -> Optional[Document]:
        """
        Retrieve a document by its database ID.

        Args:
            db_id: The database ID of the document to retrieve

        Returns:
            Document instance if found, None otherwise
        """
        result = await self.collection.find_one({"_id": ObjectId(db_id)})
        if result:
            return Document(
                text=result["text"],
                metadata=result["metadata"],
                uuid=result["uuid"],
                db_id=str(result["_id"])
            )
        return None
    
    async def get_document_by_uuid(self, uuid: UUID) -> Optional[Document]:
        """
        Retrieve a document by its UUID.

        Args:
            uuid: The UUID of the document to retrieve

        Returns:
            Document instance if found, None otherwise
        """
        result = await self.collection.find_one({"uuid": uuid})
        if result:
            return Document(
                text=result["text"],
                metadata=result["metadata"],
                uuid=result["uuid"],
                db_id=str(result["_id"])
            )
        return None
    
    async def search_document(self, regex: str) -> List[Document]:
        """
        Search for documents matching the regex pattern.

        Args:
            regex: Regular expression pattern to match against document text

        Returns:
            List of matching Document instances
        """
        cursor = self.collection.find({"text": {"$regex": regex}})
        documents = []
        async for result in cursor:
            documents.append(Document(
                text=result["text"],
                metadata=result["metadata"],
                uuid=result["uuid"],
                db_id=str(result["_id"])
            ))
        return documents

    async def remove_document(self, db_id: str) -> bool:
        """
        Remove a document by its database ID.

        Args:
            db_id: The database ID of the document to remove

        Returns:
            bool: True if document was removed, False otherwise
        """
        result = await self.collection.delete_one({"_id": ObjectId(db_id)})
        return result.acknowledged

    async def __aenter__(self) -> "AsyncMongoDBStore":
        """Support for async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ensure proper cleanup when used as context manager."""
        await self.close()