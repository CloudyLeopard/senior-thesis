from typing import List
import logging
import asyncio
from pathlib import Path

from rag.tools.sources import BaseDataSource, RequestSourceException
from rag.models import Document

logger = logging.getLogger(__name__)

class NewsArticleSearcher:
    def __init__(
        self,
        sources: List[BaseDataSource]
    ):
        """Chooses which sources to use for fetching documents """
        self.sources = sources # TODO: do some kind of "setting" here to determine which sources to use
        self.documents = []
    def search(self, query: str, num_results: int = 10, **kwargs) -> List[Document]:
        documents = []
        for source in self.sources:
            try:
                fetched_documents = source.fetch(query, num_results = num_results, **kwargs)
                documents.extend(fetched_documents)
            except RequestSourceException as e:
                logging.error("Error occurred while fetching documents from %s: %s", source.__class__.__name__, str(e))
        
        logging.info("Fetched %d documents from sources", len(documents))
        self.documents.extend(documents)
        return documents

    
    async def async_search(self, query: str, num_results: int = 10, **kwargs) -> List[Document]:

        async def _async_fetch(async_fetch: callable):
            try:
                return await async_fetch(query, num_results=num_results, **kwargs)
            except RequestSourceException as e:
                logging.error("Error occurred while fetching documents from %s: %s", async_fetch.__name__, str(e))
                return []
        
        documents = []
        results = await asyncio.gather(*[_async_fetch(source.async_fetch) for source in self.sources])
        for result in results:
            documents.extend(result)
        self.documents.extend(documents)
        return documents

    def export_documents(self, directory: str, create_dir: bool) -> List[Document]:
        """Export documents into directory as txt files, with a json file
        containing each document's metadata"""

        dir_path = Path(directory)
        if not dir_path.exists():
            if create_dir:
                dir_path.mkdir(parents=True)
            else:
                raise ValueError("Directory does not exist. Set create_dir=True to create it.")

        metadata = {}
        for i, document in enumerate(self.documents):
            file_path = dir_path / f"{i}.txt"
            file_path.write_text(document.text)
            metadata[i] = document.metadata
        
        import json
        metadata_path = dir_path / "metadata.json"
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=4)
        return self.documents