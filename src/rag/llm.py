from abc import ABC, abstractmethod
from typing import List, Dict
import os
import asyncio
from openai import OpenAI, AsyncOpenAI
import logging
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict

from rag.models import Embeddable

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class BaseLLM(ABC, BaseModel):
    """Custom generator interface"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def generate(self) -> str:
        pass

    @abstractmethod
    async def async_generate(self) -> str:
        pass

class OpenAILLM(BaseLLM):
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    model: str = "gpt-4o-mini"
    keep_history: bool = False
    messages: List[Dict] = []
    _session_token_usage: int = PrivateAttr(default=0)
    _input_token_usage: int = PrivateAttr(default=0)
    _output_token_usage: int = PrivateAttr(default=0)
    
    sync_client: OpenAI = Field(default_factory=lambda: OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    async_client: AsyncOpenAI = Field(default_factory=lambda: AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    
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
        self._session_token_usage += total_tokens
        self._input_token_usage += completion.usage.prompt_tokens
        self._output_token_usage += completion.usage.completion_tokens

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
        self._session_token_usage += total_tokens
        self._input_token_usage += completion.usage.prompt_tokens
        self._output_token_usage += completion.usage.completion_tokens

        logger.info("Total tokens used: %d", total_tokens)
        logger.info("Completion: %s", completion.choices[0].message.content)

        return completion.choices[0].message.content

    def price(self):
        if self.model == "gpt-4o":
            return (self._input_token_usage * 2.5 + self._output_token_usage * 10) / 1_000_000
        elif self.model == "gpt-4o-mini":
            return (self._input_token_usage * 0.15 + self._output_token_usage * 0.075) / 1_000_000
        else:
            return 0

class BaseEmbeddingModel(ABC, BaseModel):
    "Custom embedding model interface"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def embed(self, text: List[str]) -> List[List[float]]:
        """embeds a list of strings, and returns a list of embeddings"""
        pass

    @abstractmethod
    def async_embed(self, text: List[str]) -> List[List[float]]:
        """embeds a list of strings asynchronously, and returns a list of embeddings"""
        pass

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    model: str = Field(default="text-embedding-3-small")
    api_key: str = Field(default_factory=lambda x: os.getenv("OPENAI_API_KEY"))
    sync_client: OpenAI = None
    async_client: AsyncOpenAI = None

    def model_post_init(self, __context):
        if self.sync_client is None:
            self.sync_client = OpenAI(api_key=self.api_key)
        if self.async_client is None:
            self.async_client = AsyncOpenAI(api_key=self.api_key)
        

    # TODO: work on "retry" when encountered error
    def embed(self, text: List[str] | List[Embeddable]) -> List[List[float]]:
        if isinstance(text[0], Embeddable):
            text = [x.text for x in text]

        embeddings = []
        for i in range(0, len(text), 100):
            result = self.sync_client.embeddings.create(input=text[i : i + 100], model=self.model)
            embeddings.extend([x.embedding for x in result.data])

        return result
    
    async def async_embed(self, text: List[str] | List[Embeddable]) -> List[List[float]]:
        if isinstance(text[0], Embeddable):
            text = [x.text for x in text]
            
        # batch requests so that we don't have a list that is too large
        tasks = []
        for i in range(0, len(text), 100):
            tasks.append(self.async_client.embeddings.create(input=text[i : i + 100], model=self.model))
        results = await asyncio.gather(*tasks)
        return [x.embedding for result in results for x in result.data]