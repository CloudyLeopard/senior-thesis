from abc import ABC, abstractmethod
from pydantic import BaseModel
from kruppe.llm import BaseLLM


class Researcher(BaseModel, ABC):
    llm: BaseLLM
    # system_message: str

    @abstractmethod
    async def execute(self):
        raise NotImplementedError


class Lead(BaseModel):
    observation: str # observation that led to the lead/hypothesis, though i don't really use this
    lead: str # working lead
    hypothesis: str # working hypothesis
    
