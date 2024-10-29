from abc import ABC, abstractmethod
from typing import List, Dict
import os
from openai import OpenAI, AsyncOpenAI

from rag.models import Document
from rag.prompts import RAG_PROMPT_STANDARD, RAG_SYSTEM_STANDARD

class BaseLLM:
    """Custom generator interface"""
    def _synthesize_prompt(self, query: str, contexts: List[str]) -> List[Dict]:
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
        self, query: str, contexts: List[Document], max_tokens=2000
    ) -> str:
        """returns openai response based on given prompt and contexts"""

        # NOTE: currently, does not use Document's metadata
        messages = self._synthesize_prompt(query=query, contexts=[document.text for document in contexts])

        completion = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens
        )

        # TODO: add logger to track openai response, token usage here

        return completion[0].choices[0].message.content


class AsyncOpenAILLM(BaseLLM):
    def __init__(self, model="gpt-4o-mini", max_tokens=2000):
        self.client = AsyncOpenAI(os.getenv("OPENAI_API_KEY"))
        self.model = model

    async def async_generate(
        self, query: str, contexts: List[Document], max_tokens=2000
    ) -> str:
        """returns openai response based on given prompt and contexts"""

        # NOTE: currently, does not use Document's metadata
        messages = self._synthesize_prompt(query=query, contexts=[document.text for document in contexts])

        completion = await self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens
        )

        # TODO: add logger to track openai response, token usage here

        return completion[0].choices[0].message.content

    
