from abc import ABC, abstractmethod
from typing import List
from finrag.models.query import Query
from finrag.models.result import Result

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: Query, results: List[Result]) -> List[Result]:
        """Rerank retrieved documents."""
        pass