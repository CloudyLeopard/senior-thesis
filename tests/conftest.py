import pytest
import pytest_asyncio
from aiohttp import ClientSession
import os

from rag.sources import DirectoryData
from rag.text_splitters import RecursiveTextSplitter
from rag.vector_storages import MilvusVectorStorage
from rag.document_storages import MongoDBStore
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
    doc_storage = MongoDBStore(uri=uri, db_name=db_name)

    ids = doc_storage.save_documents(documents2)

    yield doc_storage
    
    for id in ids:
        res = doc_storage.remove_document(id)

    doc_storage.close()


@pytest.fixture(scope="module")
def vector_storage(embedding_model, text_splitter, documents2):
    uri = os.getenv("ZILLIZ_URI")
    token = os.getenv("ZILLIZ_TOKEN")

    # uri = "tests/rag/milvus_test.db" # this doesn't work for some reason
    vector_storage = MilvusVectorStorage(embedding_model, uri=uri, token=token)

    chunked_documents = text_splitter.split_documents(documents2)
    ids = vector_storage.insert_documents(chunked_documents)

    yield vector_storage

    vector_storage.remove_documents(ids)

    vector_storage.close()


@pytest.fixture(scope="module")
def text_splitter():
    return RecursiveTextSplitter()


@pytest.fixture(scope="module")
def embedding_model():
    return OpenAIEmbeddingModel()
