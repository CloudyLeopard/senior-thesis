import pytest

from rag.embeddings import OpenAIEmbeddingModel, AsyncOpenAIEmbeddingModel


def test_openai_embedding(texts):
    model = OpenAIEmbeddingModel()
    embeddings = model.embed(texts)

    assert len(embeddings) == len(texts)
    assert len(embeddings[0]) > 0

@pytest.mark.asyncio
async def test_async_openai_embedding(texts):
    model = AsyncOpenAIEmbeddingModel()
    embeddings = await model.embed(texts)

    assert len(embeddings) == len(texts)
    assert len(embeddings[0]) > 0
