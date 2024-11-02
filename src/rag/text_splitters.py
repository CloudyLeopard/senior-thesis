from abc import ABC, abstractmethod
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import Document


class BaseTextSplitter(ABC):
    """Custom Text Splitter interface"""

    @abstractmethod
    def split_documents(documents: List[Document]) -> List[Document]:
        pass


class RecursiveTextSplitter(BaseTextSplitter):
    """langchain recursive character text splitter"""

    def __init__(self, chunk_size=1024, chunk_overlap=32):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """split documents using langchain's recursive character text splitter"""
        new_documents = []
        for document in documents:
            chunks = self.text_splitter.split_text(document.text)
            # need to make sure all chunks have the same metadata, uuid, and db_id as the original document
            chunk_documents = [
                Document(text=chunk, metadata=document.metadata, uuid=document.uuid, db_id=document.db_id)
                for chunk in chunks
            ]
            new_documents.extend(chunk_documents)

        return new_documents
