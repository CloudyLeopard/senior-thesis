import pytest
import pytest_asyncio
from aiohttp import ClientSession
import os

from rag.tools.sources import DirectoryData
from rag.text_splitters import RecursiveTextSplitter
from rag.vector_storages import MilvusVectorStorage
from rag.document_storages import MongoDBStore, AsyncMongoDBStore
from rag.embeddings import OpenAIEmbeddingModel

@pytest_asyncio.fixture(scope="session")
async def session():
    client = ClientSession()
    yield client
    await client.close()


@pytest.fixture(scope="session")
def documents():
    source = DirectoryData()
    documents = source.fetch("tests/rag/data/1")
    return documents


@pytest.fixture(scope="session")
def documents2():
    source = DirectoryData()
    documents = source.fetch("tests/rag/data/2")
    return documents


@pytest.fixture
def query():
    return "What is Apple's performance?"


@pytest.fixture(scope="module")
def document_storage(documents2):
    uri = os.getenv("MONGODB_URI")
    db_name = "test"
    doc_storage = MongoDBStore(uri=uri, db_name=db_name, reset_db=True)

    doc_storage.save_documents(documents2)

    yield doc_storage

    doc_storage.close()

@pytest_asyncio.fixture(scope="module")
async def async_document_storage(documents2):
    uri = os.getenv("MONGODB_URI")
    db_name = "test_motor"
    doc_storage = await AsyncMongoDBStore.create(uri=uri, db_name=db_name, reset_db=True)

    await doc_storage.save_documents(documents2)

    yield doc_storage

    await doc_storage.close()


@pytest.fixture(scope="module")
def vector_storage(embedding_model, text_splitter, documents2):
    uri = os.getenv("ZILLIZ_URI")
    token = os.getenv("ZILLIZ_TOKEN")

    # uri = "tests/rag/milvus_test.db" # this doesn't work for some reason
    vector_storage = MilvusVectorStorage(
        embedding_model,
        collection_name="test",
        uri=uri,
        token=token,
        reset_collection=True,
    )

    chunked_documents = text_splitter.split_documents(documents2)
    vector_storage.insert_documents(chunked_documents)

    yield vector_storage

    vector_storage.close()


@pytest.fixture(scope="module")
def text_splitter():
    return RecursiveTextSplitter()


@pytest.fixture(scope="module")
def embedding_model():
    return OpenAIEmbeddingModel()
