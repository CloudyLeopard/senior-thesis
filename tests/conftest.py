import pytest
import pytest_asyncio
import os
import nest_asyncio

from kruppe.scraper import DirectoryData
from kruppe.rag.text_splitters import RecursiveTextSplitter
from kruppe.rag.vectorstore.in_memory import InMemoryVectorStore
from kruppe.document_store import MongoDBStore, AsyncMongoDBStore
from kruppe.llm import OpenAIEmbeddingModel, OpenAILLM
# from kruppe.llm import NYUOpenAIEmbeddingModel, NYUOpenAILLM
from kruppe.models import Query

nest_asyncio.apply()

@pytest.mark.asyncio
async def sanity_check():
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get("https://httpbin.org/get")
        assert response.status_code == 200
        assert response.json()["url"] == "https://httpbin.org/get"
    
    with httpx.Client() as client:
        response = client.get("https://httpbin.org/get")
        assert response.status_code == 200
        assert response.json()["url"] == "https://httpbin.org/get"

@pytest.fixture(scope="session")
def documents():
    source = DirectoryData(path="tests/data/1")
    documents = source.fetch()
    return documents


@pytest.fixture(scope="session")
def documents2():
    source = DirectoryData(path="tests/data/2")
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
    # return OpenAIEmbeddingModel()
    return OpenAIEmbeddingModel()

@pytest.fixture(scope="module")
def llm():
    # return OpenAILLM()
    return OpenAILLM()

@pytest.fixture(scope="module")
def vector_storage(embedding_model, text_splitter, documents2):
    vector_storage = InMemoryVectorStore(embedding_model=embedding_model)

    chunked_documents = text_splitter.split_documents(documents2)
    vector_storage.insert_documents(chunked_documents)

    yield vector_storage