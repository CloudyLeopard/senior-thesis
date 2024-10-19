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

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=32
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """split documents using langchain's recursive character text splitter"""
        new_documents = []
        for document in documents:
            chunks = self.text_splitter.split_text(document.text)
            chunk_documents = [
                Document(text=chunk, metadata=document.metadata, db_id=document.db_id)
                for chunk in chunks
            ]
            new_documents.extend(chunk_documents)

        return new_documents
