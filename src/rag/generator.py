from abc import ABC, abstractmethod
from typing import List, Dict
import os
from openai import OpenAI, AsyncOpenAI
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class BaseLLM(ABC):
    """Custom generator interface"""
    @abstractmethod
    def generate(self) -> str:
        pass

    @abstractmethod
    async def async_generate(self) -> str:
        pass

class OpenAILLM(BaseLLM):
    def __init__(self, model="gpt-4o-mini", api_key:str = None, keep_history: bool = False):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.sync_client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.keep_history = keep_history
        self.messages = []
    
    def generate(
        self, messages: List[Dict], max_tokens=2000
    ) -> str:
        """returns openai response based on given messages"""

        # if we want to keep history, add messages to history
        if self.keep_history:
            self.messages.extend(messages)
            messages = self.messages

        completion = self.sync_client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens
        )

        if self.keep_history:
            self.messages.append(completion.choices[0].message)

        # TODO: add logger to track openai response, token usage here
        # https://platform.openai.com/docs/api-reference/introduction
        total_tokens = completion.usage.total_tokens

        logger.info("Total tokens used: %d", total_tokens)
        logger.info("Completion: %s", completion.choices[0].message.content)

        return completion.choices[0].message.content

    async def async_generate(
        self, messages: List[Dict], max_tokens=2000
    ) -> str:
        """returns openai response based on given messages"""

        if self.keep_history:
            self.messages.extend(messages)
            messages = self.messages

        completion = await self.async_client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens
        )

        if self.keep_history:
            self.messages.append(completion.choices[0].message)

        # TODO: add logger to track openai response, token usage here
        total_tokens = completion.usage.total_tokens

        logger.info("Total tokens used: %d", total_tokens)
        logger.info("Completion: %s", completion.choices[0].message.content)

        return completion.choices[0].message.content

    
