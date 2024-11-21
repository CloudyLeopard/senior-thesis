from typing import List
import logging
import asyncio

from rag.tools.sources import BaseDataSource, RequestSourceException
from rag.document_storages import BaseDocumentStore
from rag.models import Document

logger = logging.getLogger(__name__)

class NewsArticleSearcher:
    def __init__(
        self,
        document_store: BaseDocumentStore,
        sources: List[BaseDataSource]
    ):
        self.document_store = document_store
        self.sources = sources # TODO: do some kind of "setting" here to determine which sources to use
    
    def search(self, query: str) -> List[Document]:
        documents = []
        for source in self.sources:
            try:
                fetched_documents = source.fetch(query)
                documents.extend(fetched_documents)
            except RequestSourceException as e:
                logging.error("Error occurred while fetching documents from %s: %s", source.__class__.__name__, str(e))
        
        logging.info("Fetched %d documents from sources", len(documents))
        return documents
    
    def store(self, documents: List[Document]):
        self.document_store.save_documents(documents)
    
    async def async_search(self, query: str) -> List[Document]:

        async def _async_fetch(async_fetch: callable, query: str):
            try:
                return await async_fetch(query)
            except RequestSourceException as e:
                logging.error("Error occurred while fetching documents from %s: %s", async_fetch.__name__, str(e))
                return []
        
        documents = []
        results = await asyncio.gather(*[_async_fetch(source, source.async_fetch, query) for source in self.sources])
        for result in results:
            documents.extend(result)
        return documents
    
    def search_and_store(self, query: str):
        documents = self.search(query)
        self.store(documents)
    
    async def async_search_and_store(self, query: str):
        documents = await self.async_search(query)
        self.store(documents)