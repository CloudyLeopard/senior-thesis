from textwrap import dedent
from typing import AsyncGenerator, Dict, Literal, List
from pydantic import model_validator

from kruppe.data_source.news.base_news import NewsSource
from kruppe.data_source.utils import is_method_ready
from kruppe.models import Document


class NewsHub(NewsSource):
    source: str = "News Hub"
    shorthand: str = "newshub"
    news_sources: List[NewsSource]
    description: str = "A news aggregator"

    @model_validator(mode="after")
    def set_description(self):
        if not self.news_sources:
            raise ValueError("At least one news source must be provided")
        
        self.description = dedent(f"""\
            Newshub is a news aggregator that combines multiple news sources into one platform, providing a comprehensive view of the latest news from various outlets. Contains the following sources: {", ".join([source.source for source in self.news_sources])}."
        """)
        return self

    
    async def news_search(
        self,
        query: str,
        max_results: int = 10,
        sort: Literal["relevance", "date"] = "date",
        **kwargs,
    ) -> AsyncGenerator[Document, None]:
        for news_source in self.news_sources:
            if not is_method_ready(news_source, "news_search"):
                continue

            async for doc in news_source.news_search(query, max_results, sort, **kwargs):
                yield doc
    
    async def news_recent(
        self,
        days: int = 0,
        max_results: int = None,
        filter: Dict = None, # TODO: not implemented
        **kwargs
    ) -> AsyncGenerator[Document, None]:
        for news_source in self.news_sources:
            if not is_method_ready(news_source, "news_recent"):
                continue
            
            async for doc in news_source.news_recent(days, max_results, filter, **kwargs):
                yield doc
            
    
    async def news_archive(
        self,
        start_date,
        end_date,
        max_results = 100, 
        filter = None,
        **kwargs
    ) -> AsyncGenerator[Document, None] :
        
        for news_source in self.news_sources:
            if not is_method_ready(news_source, "news_archive"):
                continue

            async_generator = news_source.news_archive(start_date, end_date, max_results, filter, **kwargs)

            async for doc in async_generator:
                yield doc