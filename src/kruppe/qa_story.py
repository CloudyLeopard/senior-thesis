from pydantic import BaseModel
from typing import List

from kruppe.functional.rag.index.base_index import BaseIndex
from kruppe.models import Document, Query, Response

class QAStories(BaseModel):
    index: BaseIndex

    async def run_pipeline(self, query: Query) -> Document:
        ...
    
    async def generate_possible_stories(self, query: Query) -> List[Response]:
        """Given a list of documents retrieved from the query, generate a list of possible stories.
        that can be created from the documents.

        Args:
            query (Query): user query

        Returns:
            List[str]: list of possible stories
        """
        ...
    
    async def differentiate_documents(self, stories: List[Response], documents: List[Document]):
        """Given a set of documents retrieved from the query, differentiate them into different categories.
        Each category will also have a summary associated with it.

        Args:
            stories (List[Response]): stories generated from the documents
            documents (List[Document]): list of documents retrieved from the query
        """
        # NOTE: do we use "user query" here? or use something else?
        # NOTE: do we run a reranker after differentiating them?
        ...
