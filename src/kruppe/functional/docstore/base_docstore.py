from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any
from uuid import UUID

from kruppe.models import Document

class BaseDocumentStore(BaseModel, ABC):
    db_name: str
    collection_name: str
    model_config = ConfigDict(arbitrary_types_allowed=True)
    

    @abstractmethod
    def clear_collection(self) -> int:
        """clear all documents in the collection, return number of documents cleared"""
        ...
    
    async def aclear_collection(self) -> int:
        ...

    @abstractmethod
    def save_document(self, document: Document) -> Document | None:
        """save Document into database, return the saved Document"""
        ...
    
    async def asave_document(self, document: Document) -> Document | None:
        ...

    @abstractmethod
    def save_documents(self, document: List[Document]) -> List[Document]:
        """save Documents into database, return list of saved Documents"""
        ...
    
    async def asave_documents(self, document: List[Document]) -> List[Document]:
        ...

    @abstractmethod
    def get_document(self, uuid: UUID = None, db_id: str = None) -> Document | None:
        """given database id, fetch Document from database"""
        ...
    
    async def aget_document(self, uuid: UUID = None, db_id: str = None) -> Document | None:
        ...

    @abstractmethod
    def search_documents(self, filter: Dict[str, Any]) -> List[Document]:
        ...
    
    async def asearch_documents(self, filter: Dict[str, Any]) -> List[Document]:
        ...

    @abstractmethod
    def remove_document(self, uuid: UUID = None, db_id: str = None) -> bool:
        """delete Document from database, return True if successful"""
        ...

    async def aremove_document(self, uuid: UUID = None, db_id: str = None) -> bool:
        ...