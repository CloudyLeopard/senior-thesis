from abc import ABC, abstractmethod
import os
from typing import List

from openai import OpenAI, AsyncOpenAI


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
    def embed(self, text: List[str]) -> List[List[float]]:
        embeddings = self.client.embeddings.create(
            input=text, model=self.model
        )

        return [x.embedding for x in embeddings.data]

    async def async_embed(self, text: List[str]) -> List[List[float]]:
        embeddings = await self.client.embeddings.create(
            input=text, model=self.model
        )

        return [x.embedding for x in embeddings.data]