import pytest

from rag.llm import OpenAIEmbeddingModel



def test_openai_embedding(documents):
    model = OpenAIEmbeddingModel(model="text-embedding-3-small")
    texts = [doc.text for doc in documents]
    embeddings = model.embed(texts)

    assert len(embeddings) == len(texts)
    assert len(embeddings[0]) == 1536

@pytest.mark.asyncio
async def test_async_openai_embedding(documents):
    model = OpenAIEmbeddingModel(model="text-embedding-3-small")
    texts = [doc.text for doc in documents]
    embeddings = await model.async_embed(texts)

    assert len(embeddings) == len(texts)
    assert len(embeddings[0]) == 1536
