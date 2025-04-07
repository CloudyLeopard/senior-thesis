from calendar import c
from pydantic import Field, computed_field, model_validator, PrivateAttr
from bson import ObjectId
from typing import List, Optional, Any, Dict
from uuid import UUID
import os
import asyncio
import logging
from datetime import datetime

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, BulkWriteError, OperationFailure, WriteError
from pymongo.collection import Collection
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from kruppe.functional.docstore.base_docstore import BaseDocumentStore
from kruppe.models import Document

logger = logging.getLogger(__name__)

MONGODB_OBJECTID_DIM = 24

class MongoDBStore(BaseDocumentStore):

    uri: str = Field(default_factory=lambda: os.getenv("MONGODB_URI"))
    use_async: bool = Field(default=False)
    client: MongoClient = Field(default_factory = lambda data: MongoClient(data['uri'], uuidRepresentation='standard'))
    _collection: Collection = PrivateAttr()
    aclient: AsyncIOMotorClient = Field(default_factory = lambda data: AsyncIOMotorClient(data['uri'], uuidRepresentation='standard'))
    _acollection: AsyncIOMotorCollection = PrivateAttr()

    @model_validator(mode="after")
    def validate_mongodb_collection(self):
        db = self.client[self.db_name] # will create database if DNE

        # init collection if DNE
        try:
            db.validate_collection(self.collection_name)

            # ehhh imma open both the sync and async collection
            # screw dealing with closing them
            self._collection = db[self.collection_name]
            self._acollection = self.aclient[self.db_name][self.collection_name]

            # if not self.use_async:
            #     # init sync collection
            #     self._collection = db[self.collection_name] # will create collection if DNE (when you firsrt store a document)
            # else:
            #     # init async collection
            #     self._acollection = self.aclient[self.db_name][self.collection_name]
            #     self.client.close() # close sync version
        except OperationFailure as e:
            logger.error("Collection %s is invalid. Please use classmethod `create_db`", self.collection_name)
            raise e
        
        # check if collection has unique index on uuid
        # and at least one other unique index
        # if not, raise ValueError

        has_unique_index = False
        has_uuid_index = False
        for index_names, index_info in self._collection.index_information().items():
            if index_info.get("unique"):
                if index_info["key"][0][0] == "uuid":
                    has_uuid_index = True
                else:
                    has_unique_index = True
                
                if has_uuid_index and has_unique_index:
                    break

        if not has_uuid_index:
            logger.warning("Collection %s does not have a unique index on uuid", self.collection_name)
            raise ValueError("Collection does not have a unique index on uuid")
        if not has_unique_index:
            logger.warning("Collection %s does not have a unique index on any other field", self.collection_name)
            raise ValueError("Collection does not have a unique index on any other field")

        return self
    
    @classmethod
    def create_db(
        cls,
        db_name: str,
        collection_name: str,
        unique_indices: List[List[str]] = None,
        reset_db: bool = False,
        uri: str = None) -> "MongoDBStore":

        uri = uri or os.getenv("MONGODB_URI")
        if not uri:
            raise ValueError("uri must be set")
        
        client = MongoClient(uri, uuidRepresentation='standard')
        db = client[db_name] # will create database if DNE

        try: 
            if reset_db:
                db.drop_collection(collection_name)
            
            collection = db[collection_name]
            db.validate_collection(collection_name)
        except OperationFailure:
            
            # if db doesnt exist, create unique indices
            collection.create_index("uuid", unique=True)
            for unique_index in unique_indices:
                collection.create_index(
                    [(index, 1) for index in unique_index],
                    unique=True
                )
        
        return cls(
            db_name=db_name,
            collection_name=collection_name,
            uri=uri,
            use_async=False
        )
    
    @classmethod
    async def acreate_db(
        cls,
        db_name: str,
        collection_name: str,
        unique_indices: List[List[str]] = None,
        reset_db: bool = False,
        uri: str = None
    ) -> "MongoDBStore":

        uri = uri or os.getenv("MONGODB_URI")
        if not uri:
            raise ValueError("uri must be set")
        
        aclient = AsyncIOMotorClient(uri, uuidRepresentation='standard')
        db = aclient[db_name]

        try:
            if reset_db:
                await db.drop_collection(collection_name)
            
            collection = db[collection_name]
            await db.validate_collection(collection_name)
        except OperationFailure:
            collection = await db.create_collection(collection_name)

            # create unique indices if db is new
            await collection.create_index("uuid", unique=True)
            await asyncio.gather(
                *[collection.create_index(
                    [(index, 1) for index in unique_index],
                    unique=True
                ) for unique_index in unique_indices]
            )
        
        return cls(
            db_name=db_name,
            collection_name=collection_name,
            uri=uri,
            use_async=True
        )
    
    @computed_field
    @property
    def document_count(self) -> int:
        return self._collection.estimated_document_count()

    def _parse_mongo_result(self, result: Dict[str, Any]) -> Document:
        """Parse MongoDB result into Document class"""
        res_text = result.pop("text")
        res_id = result.pop("uuid")
        result.pop("_id")
        return Document(text=res_text, metadata=result, id=res_id)

    def close(self):
        # if self.use_async:
        #     raise ValueError("Using async client, please use `aclose`")
        
        self.client.close()
    
    async def aclose(self):
        # if not self.use_async:
        #     raise ValueError("Using sync client, please use `close`")
        
        self.aclient.close()
    
    def clear_collection(self, drop_index: bool = False) -> int:
        # if self.use_async:
        #     raise ValueError("Using async client, please use `aclear_collection`")
        
        result = self._collection.delete_many({})
        if drop_index:
            self._collection.drop_indexes()
        
        return result.deleted_count
    
    async def aclear_collection(self, drop_index: bool = False) -> int:
        # if not self.use_async:
        #     raise ValueError("Using sync client, please use `clear_collection`")
        
        result = await self._acollection.delete_many({})
        if drop_index:
            await self._acollection.drop_indexes()
        
        return result.deleted_count

    def save_document(self, document: Document) -> Document | None:
        """save Document into mongodb collection with metadata, return number of documents saved"""
        # if self.use_async:
        #     raise ValueError("Using async client, please use `asave_document`")
        
        data = {
            "text": document.text,
            "uuid": document.id,
        }
        if document.metadata:
            data.update(document.metadata)
        
        try:
            result = self._collection.insert_one(data)
            return document
        except DuplicateKeyError as dke:
            logger.warning(f"Duplicate document (uuid={document.id}): {dke}")
            return None

    async def asave_document(self, document: Document) -> Document | None:
        """save Document into mongodb collection with metadata, return number of documents saved"""
        # if not self.use_async:
        #     raise ValueError("Using sync client, please use `save_document`")
        
        data = {
            "text": document.text,
            "uuid": document.id,
        }
        if document.metadata:
            data.update(document.metadata)

        try:
            result = await self._acollection.insert_one(data)
            return document
        except DuplicateKeyError as dke:
            logger.warning(f"Duplicate document (uuid={document.id}): {dke}")
            return None

    def save_documents(self, documents: List[Document]) -> List[Document]:
        """Saves multiple Documents into mongodb collection with metadata, unordered.
        Returns all the documents that were saved in a list, unordered. If a document
        fails to save, it will not be included in the return list.

        Args:
            documents (List[Document]): List of Documents to save

        Returns:
            List[Document, None]: List of Documents that were saved
        """
        # if self.use_async:
        #     raise ValueError("Using async client, please use `asave_documents`")
        
        datas = []
        for document in documents:
            data = {
                "text": document.text,
                "uuid": document.id,
            }
            if document.metadata:
                data.update(document.metadata)
            
            datas.append(data)
        logger.debug("Attempting to insert %d documents", len(documents))

        try:
            # insert_many with ordered=False will insert documents in parallel
            # and will not stop even if one document fails
            result = self._collection.insert_many(datas, ordered=False)
            return documents
        except BulkWriteError as bwe:
            # documentation: https://pymongo.readthedocs.io/en/stable/examples/bulk.html
            details = bwe.details
            logger.warning(f"Partial success with inserted count: {details['nInserted']}")

            err_indices = []
            for we in details['writeErrors']:
                err_indices.append(we['index']) # pymongo is kind enough to tell us the positioin fo the failed document(s)
                logger.warning(f"Write error: {we}")
            
            # remove failed documents from return list
            return [documents[i] for i in range(len(documents)) if i not in err_indices]

    async def asave_documents(self, documents: List[Document]) -> List[Document]:
        """save Documents into mongodb collection with metadata, return number of documents saved"""
        # if not self.use_async:
        #     raise ValueError("Using sync client, please use `save_documents`")
        
        datas = []
        for document in documents:
            data = {
                "text": document.text,
                "uuid": document.id,
            }
            if document.metadata:
                data.update(document.metadata)
            
            datas.append(data)
        logger.debug("Attempting to insert %d documents", len(documents))

        try:
            # insert_many with ordered=False will insert documents in parallel
            # and will not stop even if one document fails
            result = await self._acollection.insert_many(datas, ordered=False)
            return documents
        except BulkWriteError as bwe:
            # documentation: https://pymongo.readthedocs.io/en/stable/examples/bulk.html
            details = bwe.details
            logger.warning(f"Partial success with inserted count: {details['nInserted']}")

            err_indices = []
            for we in details['writeErrors']:
                err_indices.append(we['index']) # pymongo is kind enough to tell us the positioin fo the failed document(s)
                logger.warning(f"Write error: {we}")
            
            # remove failed documents from return list
            return [documents[i] for i in range(len(documents)) if i not in err_indices]

    def get_document(self, uuid: UUID = None, db_id: str = None) -> Document | None:
        """Retrieve Document from mongodb collection by either db_id or uuid.
        If both are provided, db_id will be used. If neither are provided, raise ValueError.

        Args:
            db_id (str, optional): MongoDB's Object ID of Document. Defaults to None.
            uuid (UUID, optional): UUID of Document. Defaults to None.

        Returns:
            Document | None: Document if found, None if none are found.
        """
        # if self.use_async:
        #     raise ValueError("Using async client, please use `aget_document`")
        
        # Fetch from MongoDB Collection
        if db_id:
            result = self._collection.find_one({"_id": ObjectId(db_id)})
        elif uuid:
            result = self._collection.find_one({"uuid": uuid})
        else:
            raise ValueError("Either db_id or uuid must be provided.")
        
        # Parse into Document class if found
        if result:
            return self._parse_mongo_result(result)
        else:
            return None

    async def aget_document(self, uuid: UUID = None, db_id: str = None) -> Document | None:
        """Retrieve Document from mongodb collection by either db_id or uuid.
        If both are provided, db_id will be used. If neither are provided, raise ValueError.

        Args:
            db_id (str, optional): MongoDB's Object ID of Document. Defaults to None.
            uuid (UUID, optional): UUID of Document. Defaults to None.

        Returns:
            Document | None: Document if found, None if none are found.
        """
        # if not self.use_async:
        #     raise ValueError("Using sync client, please use `get_document`")
        
        # Fetch from MongoDB Collection
        if db_id:
            result = await self._acollection.find_one({"_id": ObjectId(db_id)})
        elif uuid:
            result = await self._acollection.find_one({"uuid": uuid})
        else:
            raise ValueError("Either db_id or uuid must be provided.")
        
        # Parse into Document class if found
        if result:
            return self._parse_mongo_result(result)
        else:
            return None

    def get_all_documents(self) -> List[Document]:
        # if self.use_async:
        #     raise ValueError("Using async client, please use `aget_all_documents`")
        
        documents = []        

        # Fetch all documents from MongoDB Collection
        cursor = self._collection.find({})
        for result in cursor:
            document = self._parse_mongo_result(result)
            documents.append(document)
        
        return documents
    
    async def aget_all_documents(self) -> List[Document]:
        # if not self.use_async:
        #     raise ValueError("Using sync client, please use `get_all_documents`")
        
        documents = []

        # Fetch all documents from MongoDB Collection
        cursor = self._acollection.find({})
        async for result in cursor:
            document = self._parse_mongo_result(result)
            documents.append(document)
        
        return documents

    def search_documents(self, filter: Dict[str, Any]) -> List[Document] | None:
        # if self.use_async:
        #     raise ValueError("Using async client, please use `asearch_documents`")
        
        documents = []        
        
        cursor = self._collection.find(filter)
        for result in cursor:
            document = self._parse_mongo_result(result)
            documents.append(document)

        return documents
    
    async def asearch_documents(self, filter: Dict[str, Any]) -> List[Document] | None:
        # if not self.use_async:
        #     raise ValueError("Using sync client, please use `search_documents`")
        
        documents = []        

        cursor = self._acollection.find(filter)
        async for result in cursor:
            document = self._parse_mongo_result(result)
            documents.append(document)

        return documents


    def remove_document(self, uuid: UUID = None, db_id: str = None) -> bool:
        """Remove Document from mongodb collection by either db_id or uuid.
        If both are provided, db_id will be used. If neither are provided, raise ValueError.
        """

        if db_id:
            result = self._collection.delete_one({"_id": ObjectId(db_id)})
        elif uuid:
            result = self._collection.delete_one({"uuid": uuid})
        else:
            raise ValueError("Either db_id or uuid must be provided.")
        
        return result.acknowledged

    async def aremove_document(self, uuid: UUID = None, db_id: str = None) -> bool:
        """Remove Document from mongodb collection by either db_id or uuid.
        If both are provided, db_id will be used. If neither are provided, raise ValueError.
        """

        # if not self.use_async:
        #     raise ValueError("Using sync client, please use `remove_document`")

        if db_id:
            result = await self._acollection.delete_one({"_id": ObjectId(db_id)})
        elif uuid:
            result = await self._acollection.delete_one({"uuid": uuid})
        else:
            raise ValueError("Either db_id or uuid must be provided.")
        
        return result.acknowledged

