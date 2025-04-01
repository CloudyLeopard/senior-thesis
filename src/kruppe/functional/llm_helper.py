from typing import List, Dict
from pydantic import BaseModel

from kruppe.llm import OpenAILLM
from kruppe.models import Document

class LLMSummarize(OpenAILLM):

    async def async_generate(
        self, messages: List[Dict], max_tokens=2000
    ):
        
        response = await self.super().async_generate(
            messages=messages,
            max_tokens=max_tokens,
        )

        # TODO: "summarize" the response
        summarized_response = ...

        return summarized_response
    
