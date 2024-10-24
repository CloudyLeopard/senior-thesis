import pytest
import os

from rag.embeddings import OpenAIEmbeddingModel
from rag.vector_storages import MilvusVectorStorage

@pytest.fixture
def vector_storage():
    uri = os.getenv("ZILLIZ_URI")
    token = os.getenv("ZILLIZ_TOKEN")

    return MilvusVectorStorage(uri, token)

