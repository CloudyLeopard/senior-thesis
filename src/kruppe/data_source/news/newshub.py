from typing import Dict, Literal, List
from kruppe.data_source.news.base_news import NewsSource
from kruppe.data_source.utils import not_ready, combine_async_generators, is_method_ready


class NewsHub(NewsSource):
    source = "Newshub"
    shorthand = "hub"
    description = "A compilation of different news sources"
    news_sources: List[NewsSource]
    
    @not_ready
    async def news_search(
        self,
        query: str,
        max_results: int = 10,
        sort: Literal["relevance", "date"] = None,
        **kwargs,
    ):
        async_generators = []
        for news_source in self.news_sources:
            if not is_method_ready(news_source, "news_search"):
                continue

            async_generators.append(news_source.news_search(query, max_results, sort, **kwargs))
        combined_generator = combine_async_generators(async_generators)

        async for doc in combined_generator:
            yield doc
    
    @not_ready
    async def new_recent(
        self,
        days: int = 0,
        max_results: int = None,
        filter: Dict = None, # TODO: not implemented
        **kwargs
    ):
        async_generators = []
        for news_source in self.news_sources:
            if not is_method_ready(news_source, "news_recent"):
                continue

            async_generators.append(news_source.news_recent(days, max_results, filter, **kwargs))
        combined_generator = combine_async_generators(async_generators)

        async for doc in combined_generator:
            yield doc
    
    @not_ready
    async def news_archive(
        self,
        start_date,
        end_date,
        max_results = 100, 
        filter = None,
        **kwargs
    ):
        
        async_generators = []
        for news_source in self.news_sources:
            if not is_method_ready(news_source, "news_archive"):
                continue

            async_generators.append(news_source.news_archive(start_date, end_date, max_results, filter, **kwargs))
        combined_generator = combine_async_generators(async_generators)

        async for doc in combined_generator:
            yield doc
