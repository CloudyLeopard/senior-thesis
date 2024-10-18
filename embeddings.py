import os
from typing import List

from openai import OpenAI, AsyncOpenAI


class EmbeddingModel:
    "Custom embedding model interface"
    def embed(self, text: List[str]) -> List[float]:
        pass


class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, model="text-embedding-3-small"):
        self.client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
        self.model = model

    # TODO: work on "retry" when encountered error
    def embed(self, text: List[str]) -> List[float]:
        embeddings = self.client.embeddings.create(
            input=text, model=self.model
        )

        return [x.embedding for x in embeddings.data]

class AsyncOpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, model="text-embedding-3-small"):
        self.client = AsyncOpenAI(api_key = os.getenv("OPENAI_API_KEY"))
        self.model = model

    async def embed(self, text: List[str]) -> List[float]:
        embeddings = await self.client.embeddings.create(
            input=text, model=self.model
        )

        return [x.embedding for x in embeddings.data]