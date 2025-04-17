from abc import ABC
from typing import Dict
import logging
from pydantic import BaseModel
from textwrap import dedent


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
   


class ForumSource(DataSource):
    ...
