from abc import ABC, abstractmethod
from typing import Dict, AsyncGenerator, Generator, Literal
import logging
from fastapi.background import P
from pydantic import BaseModel, Field, computed_field
import asyncio
import threading
from textwrap import dedent
from datetime import datetime

from kruppe.models import Document

logger = logging.getLogger(__name__)

class DataSource(ABC, BaseModel):
    source: str
    description: str = None # TODO: make required later

    def get_description(self) -> str:
        return dedent(f"""\
            Data source name: {self.source}
            Data source description: {self.description}
            """
        )
    
    def get_schema(self, method_name: str) -> Dict:
        return getattr(self, f"{method_name}_schema")
    
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
   
class FinancialSource(DataSource):
    ...

class ForumSource(DataSource):
    ...
