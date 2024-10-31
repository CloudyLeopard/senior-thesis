from abc import ABC
from typing import List, Dict
import os
from openai import OpenAI, AsyncOpenAI

from rag.models import Document
from rag.prompts import RAG_PROMPT_STANDARD, RAG_SYSTEM_STANDARD

class BaseLLM(ABC):
    """Custom generator interface"""
    def _synthesize_prompt(self, query: str, contexts: List[Document]) -> List[Dict]:
        """merge contexts and prompt, returns messages for openai api call"""
        prompt = RAG_PROMPT_STANDARD.format(query=query, context="\n\n".join(contexts))

        # format into openai accepted format
        messages = [
            {"role": "system", "content": RAG_SYSTEM_STANDARD},
            {"role": "user", "content": prompt},
        ]
        return messages


class OpenAILLM(BaseLLM):
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI(os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def generate(
        self, messages: List[Dict], max_tokens=2000
    ) -> str:
        """returns openai response based on given prompt and contexts"""

        completion = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens
        )

        # TODO: add logger to track openai response, token usage here

        return completion[0].choices[0].message.content


class AsyncOpenAILLM(BaseLLM):
    def __init__(self, model="gpt-4o-mini"):
        self.client = AsyncOpenAI(os.getenv("OPENAI_API_KEY"))
        self.model = model

    async def async_generate(
        self, messages: List[Dict], max_tokens=2000
    ) -> str:
        """returns openai response based on given prompt and contexts"""

        completion = await self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens
        )

        # TODO: add logger to track openai response, token usage here

        return completion[0].choices[0].message.content

    
