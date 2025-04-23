from typing import AsyncGenerator, List, Literal, Dict
from abc import abstractmethod

from kruppe.data_source.base_source import DataSource
from kruppe.models import Document


class NewsSource(DataSource):
    @abstractmethod
    async def news_search(
        self,
        query: str,
        max_results: int = 20,
        sort: Literal["relevance", "date"] = "date",
        **kwargs,
    ) -> AsyncGenerator[Document, None]: ...

    @abstractmethod
    async def news_recent(
        self,
        days: int = 0,
        max_results: int = 20,
        keywords: str = None,
        **kwargs,
    ) -> AsyncGenerator[Document, None]: ...

    @abstractmethod
    async def news_archive(
        self,
        start_date: str,
        end_date: str,
        max_results: int = 20,
        keywords: str = None,
        **kwargs,
    ) -> AsyncGenerator[Document, None]: ...