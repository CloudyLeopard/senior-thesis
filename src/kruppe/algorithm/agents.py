from abc import ABC, abstractmethod
from pydantic import BaseModel
from kruppe.llm import BaseLLM


class Researcher(BaseModel, ABC):
    llm: BaseLLM
    system_message: str


    @abstractmethod
    async def execute(self):
        raise NotImplementedError


