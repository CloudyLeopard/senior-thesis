import pytest

from rag.embeddings import OpenAIEmbeddingModel



def test_openai_embedding(documents):
    model = OpenAIEmbeddingModel()
    texts = [doc.text for doc in documents]
    embeddings = model.embed(texts)

    assert len(embeddings) == len(texts)
    assert len(embeddings[0]) > 0

@pytest.mark.asyncio
async def test_async_openai_embedding(documents):
    model = OpenAIEmbeddingModel()
    texts = [doc.text for doc in documents]
    embeddings = await model.async_embed(texts)

    assert len(embeddings) == len(texts)
    assert len(embeddings[0]) > 0
