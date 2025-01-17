from abc import ABC, abstractmethod
from typing import List, Dict
import os
from openai import OpenAI, AsyncOpenAI
import logging

from rag.models import Embeddable

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
        self.session_token_usage = 0
        self.input_token_usage = 0
        self.output_token_usage = 0
    
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
        self.session_token_usage += total_tokens
        self.input_token_usage += completion.usage.prompt_tokens
        self.output_token_usage += completion.usage.completion_tokens

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
        self.session_token_usage += total_tokens
        self.input_token_usage += completion.usage.prompt_tokens
        self.output_token_usage += completion.usage.completion_tokens

        logger.info("Total tokens used: %d", total_tokens)
        logger.info("Completion: %s", completion.choices[0].message.content)

        return completion.choices[0].message.content

    def price(self):
        if self.model == "gpt-4o":
            return (self.input_token_usage * 2.5 + self.output_token_usage * 10) / 1_000_000
        elif self.model == "gpt-4o-mini":
            return (self.input_token_usage * 0.15 + self.output_token_usage * 0.075) / 1_000_000
        else:
            return 0

class BaseEmbeddingModel(ABC):
    "Custom embedding model interface"

    @abstractmethod
    def embed(self, text: List[str]) -> List[List[float]]:
        """embeds a list of strings, and returns a list of embeddings"""
        pass

    @abstractmethod
    def async_embed(self, text: List[str]) -> List[List[float]]:
        """embeds a list of strings asynchronously, and returns a list of embeddings"""
        pass

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-3-small", api_key:str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.sync_client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model = model

    # TODO: work on "retry" when encountered error
    def embed(self, text: List[str] | List[Embeddable]) -> List[List[float]]:
        if isinstance(text[0], Embeddable):
            text = [x.text for x in text]

        embeddings = self.sync_client.embeddings.create(
            input=text, model=self.model
        )

        return [x.embedding for x in embeddings.data]

    async def async_embed(self, text: List[str] | List[Embeddable]) -> List[List[float]]:
        if isinstance(text[0], Embeddable):
            text = [x.text for x in text]
            
        embeddings = await self.async_client.embeddings.create(
            input=text, model=self.model
        )

        return [x.embedding for x in embeddings.data]