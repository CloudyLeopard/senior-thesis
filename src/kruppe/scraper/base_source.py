from abc import ABC, abstractmethod
from typing import List, Dict, AsyncGenerator, Generator
import logging
from pydantic import BaseModel, Field
import asyncio
import threading

from kruppe.models import Document

logger = logging.getLogger(__name__)

class RequestSourceException(Exception):
    pass

# TODO: remove the "document_store" parameter. Keep entering documents into 
# document store separate from the data source class
class BaseDataSource(ABC, BaseModel):
    """Custom data source class interface"""
    source: str = Field(default="Unknown")

    def fetch(self, query: str, num_results: int = 10, **kwargs) -> Generator[Document, None, None]:
        """Fetch links relevant to the query with the corresponding data source

        Returns:
            Generator[Document]: A list of Document objects containing the text and metadata
                            of the scraped links.

        Raises:
            RequestSourceException: If the request to the data source API fails.

        """
        # Well this is all ChatGPT code i have no idea if this works but oh well
        queue = asyncio.Queue()

        async def async_consumer():
            async for item in self.async_fetch(query=query, num_results=num_results, **kwargs):
                await queue.put(item)
            await queue.put(None)  # Sentinel to indicate completion

        def run_async():
            asyncio.run(async_consumer())

        # Run the async function in a separate thread
        thread = threading.Thread(target=run_async)
        thread.start()

        while True:
            item = queue.get()  # Blocks until an item is available
            if item is None:
                break
            yield item

    @abstractmethod
    async def async_fetch(self, query: str, num_results: int = 10, **kwargs) -> AsyncGenerator[Document, None]:
        """Async fetch links relevant to the query with the corresponding data source

        Returns:
            AsyncGenerator[Document]: A list of Document objects containing the text and metadata
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
        description: str = None,
        publication_time: str = None,
        **kwargs,
    ) -> Dict[str, str]:
        metadata = {
            "query": query or "",
            "datasource": cls.__name__,
            "url": url or "",
            "title": title or "",
            "description": description or "",
            "publication_time": publication_time or "",
        }
        metadata.update(kwargs)
        return metadata


