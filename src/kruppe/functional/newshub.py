from datetime import datetime
from typing import Dict, Literal, List, Tuple
from pydantic import model_validator, computed_field
import pandas as pd
import logging

from kruppe.functional.base_tool import BaseTool
from kruppe.data_source.news.base_news import NewsSource
from kruppe.common.utils import is_method_ready, convert_to_datetime, combine_async_generators
from kruppe.models import Document
from kruppe.prompts.functional import (
    NEWS_SEARCH_TOOL_DESCRIPTION,
    NEWS_RECENT_TOOL_DESCRIPTION,
    NEWS_ARCHIVE_TOOL_DESCRIPTION,
)


logger = logging.getLogger(__name__)


class NewsHub(BaseTool):
    news_sources: List[NewsSource]
    source: str = "NewsHub" # TODO: remove later

    @model_validator(mode="after")
    def set_source_info(self):
        if not self.news_sources:
            raise ValueError("At least one news source must be provided")

        return self

    async def news_search(
        self,
        query: str,
        max_results: int = 10,
        sort: Literal["relevance", "date"] = "date",
        **kwargs,
    ) -> Tuple[pd.DataFrame, List[Document]]:
        
        # collect async gens
        async_gens = []
        for news_source in self.news_sources:
            if not is_method_ready(news_source, "news_search"):
                continue
                
            async_gens.append(news_source.news_search(
                query, max_results, sort, **kwargs
            ))
        
        df, docs = await self._gen_result_helper(async_gens)
        return df, docs

    async def news_recent(
        self,
        days: int = 0,
        max_results: int = 10,
        filter: Dict = None,  # TODO: not implemented
        **kwargs,
    ) -> Tuple[pd.DataFrame, List[Document]]:
        # collect async gens
        async_gens = []
        for news_source in self.news_sources:
            if not is_method_ready(news_source, "news_recent"):
                continue
                
            async_gens.append(news_source.news_recent(
                days, max_results, filter, **kwargs
            ))
        
        df, docs = await self._gen_result_helper(async_gens)
        return df, docs


    async def news_archive(
        self, start_date, end_date, max_results=10, filter=None, **kwargs
    ) -> Tuple[pd.DataFrame, List[Document]]:
        # collect async gens
        async_gens = []
        for news_source in self.news_sources:
            if not is_method_ready(news_source, "news_archive"):
                continue
                
            async_gens.append(news_source.news_archive(
                start_date, end_date, max_results, filter, **kwargs
            ))
        
        df, docs = await self._gen_result_helper(async_gens)
        return df, docs
    
    async def _gen_result_helper(self, async_gens) -> Tuple[pd.DataFrame, List[Document]]:
         # combine async gens into one
        combined_gen = combine_async_generators(async_gens)

        # collect results
        docs = []
        news = []

        async for doc in combined_gen:
            title = doc.metadata.get("title")
            description = doc.metadata.get("title")
            publication_time = doc.metadata.get("publication_time")

            try:
                publication_dt = convert_to_datetime(publication_time) 
            except ValueError as e:
                publication_dt = datetime.now()  # Fallback to now if conversion fails
                logger.debug(f"Failed to convert publication time '{publication_time}' to datetime: {e}")

            news.append({
                "title": title,
                "description": description,
                "publication_time": publication_dt.date(),
            })

            docs.append(doc)
        
        news.sort(key=lambda x: x["publication_time"])
        df = pd.DataFrame(news)

        return df, docs


    @computed_field
    @property
    def news_search_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": "news_search",
                "description": NEWS_SEARCH_TOOL_DESCRIPTION,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "required": [
                        "query",
                        "max_results",
                        "sort"
                    ],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query for news articles."
                        },
                        "max_results": {
                            "type": "number",
                            "description": "The maximum number of results to return (default is 10)."
                        },
                        "sort": {
                            "type": "string",
                            "description": "The sorting method for the results, either 'relevance' or 'date'.",
                            "enum": [
                                "relevance",
                                "date"
                            ]
                        }
                    },
                    "additionalProperties": False
                }
            }
        }

    @computed_field
    @property
    def news_recent_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": "news_search",
                "description": NEWS_RECENT_TOOL_DESCRIPTION,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "required": [
                        "days",
                        "max_results",
                    ],
                    "properties": {
                        "days": {
                            "type": "number",
                            "description": "The number of days to look back for news articles (default is 0, which means today)."
                        },
                        "max_results": {
                            "type": "number",
                            "description": "The maximum number of results to return (default is 10)."
                        },
                    },
                    "additionalProperties": False
                }
            }
        }
    

    @computed_field
    @property
    def news_archive_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": "news_search",
                "description": NEWS_ARCHIVE_TOOL_DESCRIPTION,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "required": [
                        "start_date",
                        "end_date",
                        "max_results",
                    ],
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Date (YYYY-MM-DD) to start the search from (inclusive)."
                        },
                        "end_date": {
                            "type": "string",
                            "description": "Date (YYYY-MM-DD) to end the search at (inclusive)."
                        },
                        "max_results": {
                            "type": "number",
                            "description": "The maximum number of results to return (default is 10)."
                        },
                    },
                    "additionalProperties": False
                }
            }
        }