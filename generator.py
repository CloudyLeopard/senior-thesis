from typing import List, Dict
import os
from openai import OpenAI, AsyncOpenAI

from .models import Document
from .prompts import RAG_PROMPT_STANDARD, RAG_SYSTEM_STANDARD


class BaseGenerator:
    """Custom generator interface"""
    def _synthesize_prompt(self, prompt: str, contexts: List[str]) -> List[Dict]:
        """merge contexts and prompt, returns messages for openai api call"""
        prompt = RAG_PROMPT_STANDARD.format(query=prompt, context="\n\n".join(contexts))

        # format into openai accepted format
        messages = [
            {"role": "system", "content": RAG_SYSTEM_STANDARD},
            {"role": "user", "content": prompt},
        ]
        return messages


class OpenAIGenerator(BaseGenerator):
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI(os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def generate(
        self, prompt: str, contexts: List[Document], max_tokens=2000
    ) -> str:
        """returns openai response based on given prompt and contexts"""

        # NOTE: currently, does not use Document's metadata
        messages = self._synthesize_prompt(prompt=prompt, contexts=[document.text for document in contexts])

        completion = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens
        )

        # TODO: add logger to track openai response, token usage here

        return completion[0].choices[0].message.content


class AsyncOpenAIGenerator(BaseGenerator):
    def __init__(self, model="gpt-4o-mini", max_tokens=2000):
        self.client = AsyncOpenAI(os.getenv("OPENAI_API_KEY"))
        self.model = model

    async def async_generate(
        self, prompt: str, contexts: List[Document], max_tokens=2000
    ) -> str:
        """returns openai response based on given prompt and contexts"""

        # NOTE: currently, does not use Document's metadata
        messages = self._synthesize_prompt(prompt=prompt, contexts=[document.text for document in contexts])

        completion = await self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens
        )

        # TODO: add logger to track openai response, token usage here

        return completion[0].choices[0].message.content

    
