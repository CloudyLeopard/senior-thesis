import pytest

from rag.llm import (
    OpenAIEmbeddingModel,
    NYUOpenAIEmbeddingModel,
    OpenAILLM,
    NYUOpenAILLM,
)
from rag.models import Response


@pytest.fixture(params=[OpenAIEmbeddingModel, NYUOpenAIEmbeddingModel])
def embedding_model(request):
    return request.param()

@pytest.fixture(params=[OpenAILLM, NYUOpenAILLM])
def llm(request):
    return request.param()


@pytest.mark.asyncio
async def test_async_openai_embedding(embedding_model, documents):
    texts = [doc.text for doc in documents]
    embeddings = await embedding_model.async_embed(texts)

    assert len(embeddings) == len(texts)
    assert len(embeddings[0]) == 1536

@pytest.mark.asyncio
async def test_llm(llm):
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    response = await llm.async_generate(message)

    assert isinstance(response, Response)
    assert response is not None
    assert len(response.text) > 0
    