from abc import ABC, abstractmethod
from typing import List, Dict
import logging
from pydantic import BaseModel, Field

from rag.models import Document

logger = logging.getLogger(__name__)

class RequestSourceException(Exception):
    pass

# TODO: remove the "document_store" parameter. Keep entering documents into 
# document store separate from the data source class
class BaseDataSource(ABC, BaseModel):
    """Custom data source class interface"""
    source: str = Field(default="Unknown")

    @abstractmethod
    def fetch(self, query: str, num_results: int = 10, **kwargs) -> List[Document]:
        """Fetch links relevant to the query with the corresponding data source

        Returns:
            List[Document]: A list of Document objects containing the text and metadata
                            of the scraped links.

        Raises:
            RequestSourceException: If the request to the data source API fails.

        """
        pass

    @abstractmethod
    async def async_fetch(self, query: str, num_results: int = 10, **kwargs) -> List[Document]:
        """Async fetch links relevant to the query with the corresponding data source

        Returns:
            List[Document]: A list of Document objects containing the text and metadata
                            of the scraped links.

        Raises:
            RequestSourceException: If the request to the data source API fails.
        """
        pass

    def process_document(document: Document):
        # TODO: later on, perhaps use LLM on scraped text data
        # to extract information that can be used later. Example: category
        pass

    @classmethod
    def parse_metadata(
        cls,
        query: str,
        url: str = None,
        title: str = None,
        publication_time: str = None,
        **kwargs,
    ) -> Dict[str, str]:
        metadata = {
            "query": query or "",
            "datasource": cls.__name__,
            "url": url or "",
            "title": title or "",
            "publication_time": publication_time or "",
        }
        metadata.update(kwargs)
        return metadata


