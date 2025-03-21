from typing import List
from pydantic import BaseModel

from kruppe.llm import BaseLLM
from kruppe.models import Document

class LLMHelper(BaseModel):
    llm: BaseLLM

    async def summarize_documents(self, documents: List[Document]):
        raise NotImplementedError
    
