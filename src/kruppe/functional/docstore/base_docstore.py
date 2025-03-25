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
        ...
    
    async def aclear_collection(self) -> int:
        ...

    @abstractmethod
    def save_document(self, document: Document) -> int:
        """save Document into database, return number of documents saved"""
        ...
    
    async def asave_document(self, document: Document) -> int:
        ...

    @abstractmethod
    def save_documents(self, document: List[Document]) -> int:
        """save Documents into database, return number of documents saved"""
        ...
    
    async def asave_documents(self, document: List[Document]) -> int:
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
    def remove_document(self, uuid: UUID = None, db_id: str = None) -> int:
        """delete Document from database"""
        ...

    async def aremove_document(self, uuid: UUID = None, db_id: str = None) -> int:
        ...