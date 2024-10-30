import pytest
import pytest_asyncio
from aiohttp import ClientSession
import os

from rag.sources import DirectoryData
from rag.text_splitters import RecursiveTextSplitter
from rag.vector_storages import MilvusVectorStorage
from rag.document_storages import MongoDBStore
from rag.embeddings import OpenAIEmbeddingModel, AsyncOpenAIEmbeddingModel


@pytest_asyncio.fixture(scope="session")
async def session():
    client = ClientSession()
    yield client
    await client.close()

@pytest.fixture(scope="module")
def documents():
    source = DirectoryData()
    documents = source.fetch("tests/rag/data")
    return documents

@pytest.fixture
def query():
    return "What is Apple's performance?"

@pytest.fixture(scope="module")
def document_storage():
    uri = os.getenv("MONGODB_URI")
    db_name = "test"
    doc_storage = MongoDBStore(uri=uri, db_name=db_name)
    
    yield doc_storage

    doc_storage.close()


@pytest.fixture(scope="module")
def vector_storage():
    uri = os.getenv("ZILLIZ_URI")
    token = os.getenv("ZILLIZ_TOKEN")

    # uri = "tests/rag/milvus_test.db" # this doesn't work for some reason
    vector_storage = MilvusVectorStorage(uri, token)

    yield vector_storage

    vector_storage.close()

@pytest.fixture(scope="module")
def text_splitter():
    return RecursiveTextSplitter()



@pytest.fixture(scope="module")
def embedding_model():
    return OpenAIEmbeddingModel()

