from typing import AsyncGenerator, List, Literal, Dict
from pydantic import computed_field

from kruppe.data_source.base_source import DataSource
from kruppe.models import Document


class NewsSource(DataSource):
    async def news_search(
        self,
        query: str,
        max_results: int = 20,
        sort: Literal["relevance", "date"] = None,
        **kwargs,
    ) -> AsyncGenerator[Document, None]: ...

    async def news_recent(
        self,
        days: int = 0,
        max_results: int = 20,
        filter: Dict = None,  # TODO: not implemented
        **kwargs,
    ) -> AsyncGenerator[Document, None]: ...

    async def news_archive(
        self,
        start_date: str,
        end_date: str,
        max_results: int = 20,
        filter: Dict = None,  # TODO: not implemented
        **kwargs,
    ) -> AsyncGenerator[Document, None]: ...

    @computed_field
    @property
    def news_search_schema(self) -> Dict:
        return {
            "type": "function",
            "name": "news_search",
            "description": f"Search {self.source} for news articles with a query. Use news_search to search for a specific topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": f"Search query to {self.source}",
                    },
                    # "max_results": {
                    #     "type": "number",
                    #     "description": "Number of top results to return",
                    # },
                    "sort": {
                        "type": ["string", "null"],
                        "enum": ["relevance", "date"],
                        "description": "How to sort results. Pass null if not needed.",
                    },
                },
                "required": ["query", "sort_by"],
                "additionalProperties": False,
            },
        }

    @computed_field
    @property
    def news_recent_schema(self) -> Dict:
        return {
            "type": "function",
            "name": "news_recent",
            "description": f"Get the most recent news published from {self.source}. Use news_recent to get a general background of recent news.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "number",
                        "description": "Number of days to go back. 0 means articles published today, 1 means articles published from yesterday to today, etc.",
                    },
                    # "max_results": {
                    #     "type": "max_results",
                    #     "description": "Number of results to return.",
                    # },
                    "filter": {
                        "type": "object",
                        "properties": {
                            "include": {
                                "type": ["string", "null"],
                                "description": "Keywords to include, separated by space",
                            },
                            "exclude": {
                                "type": ["string", "null"],
                                "description": "Keywords to exclude, separated by space",
                            },
                        },
                        "required": ["include", "exclude"],
                        "additionalProperties": False,
                    },
                },
                "required": ["days", "filter"],
                "additionalProperties": False,
            },
        }

    @computed_field
    @property
    def news_archive_schema(self) -> Dict:
        return {
            "type": "function",
            "name": "news_archive",
            "description": f"Get archived news articles published from {self.source} between a certain time period. Use news_archive to get historical news.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Starting publication date, formatted as YYYY-MM-DD",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Ending publication date, formatted as YYYY-MM-DD",
                    },
                    # "max_results": {
                    #     "type": "max_results",
                    #     "description": "Number of results to return.",
                    # },
                    "filter": {
                        "type": "object",
                        "properties": {
                            "include": {
                                "type": ["string", "null"],
                                "description": "Keywords to include, separated by space",
                            },
                            "exclude": {
                                "type": ["string", "null"],
                                "description": "Keywords to exclude, separated by space",
                            },
                        },
                        "required": ["include", "exclude"],
                        "additionalProperties": False,
                    },
                },
                "required": ["start_date", "end_date", "filter"],
                "additionalProperties": False,
            },
        }
