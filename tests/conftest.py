import pytest
import pytest_asyncio
import os
import nest_asyncio

from rag.scraper import DirectoryData
from rag.text_splitters import RecursiveTextSplitter
from rag.vectorstore.in_memory import InMemoryVectorStore
from rag.document_store import MongoDBStore, AsyncMongoDBStore
from rag.llm import OpenAIEmbeddingModel
from rag.models import Query

nest_asyncio.apply()

@pytest.fixture(scope="session")
def documents():
    source = DirectoryData(path="tests/rag/data/1")
    documents = source.fetch()
    return documents


@pytest.fixture(scope="session")
def documents2():
    source = DirectoryData(path="tests/rag/data/2")
    documents = source.fetch()
    return documents


@pytest.fixture(scope="session")
def query():
    return Query(text="What is Apple's performance?", metadata={})


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
def text_splitter():
    return RecursiveTextSplitter()


@pytest.fixture(scope="module")
def embedding_model():
    return OpenAIEmbeddingModel()

@pytest.fixture(scope="module")
def vector_storage(embedding_model, text_splitter, documents2):
    vector_storage = InMemoryVectorStore(embedding_model=embedding_model)

    chunked_documents = text_splitter.split_documents(documents2)
    vector_storage.insert_documents(chunked_documents)

    yield vector_storage