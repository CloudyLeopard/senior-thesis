from typing import Dict, Literal
from kruppe.data_source.news.base_news import NewsSource
from kruppe.data_source.utils import not_ready


class NewsHub(NewsSource):
    source = "Newshub"
    shorthand = "hub"
    description = "A compilation of different news sources"
    
    @not_ready
    async def news_search(
        self,
        query: str,
        num_results: int = 10,
        sort: Literal["relevance", "date"] = None,
        **kwargs,
    ):
        raise NotImplementedError
    
    @not_ready
    async def new_recent(
        self,
        days: int = 0,
        num_results: int = None,
        filter: Dict = None, # TODO: not implemented
        **kwargs
    ):
        raise NotImplementedError
    
    @not_ready
    async def news_archive(
        self,
        start_date,
        end_date,
        num_results = 100, 
        filter = None,
        **kwargs
    ):
        raise NotImplementedError
