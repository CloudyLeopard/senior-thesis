import pytest
import pytest_asyncio
from aiohttp import ClientSession
import os
import time

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
def document_storage(documents):
    uri = os.getenv("MONGODB_URI")
    db_name = "test"
    doc_storage = MongoDBStore(uri=uri, db_name=db_name)
    
    yield doc_storage

    doc_storage.close()


@pytest.fixture(scope="module")
def vector_storage(embedding_model, text_splitter, document_storage, documents):
    uri = os.getenv("ZILLIZ_URI")
    token = os.getenv("ZILLIZ_TOKEN")

    # uri = "tests/rag/milvus_test.db" # this doesn't work for some reason
    vector_storage = MilvusVectorStorage(uri, token)

    # insert data
    # insert documents
    ids = [document_storage.save_document(doc) for doc in documents]
    # print(ids)
    assert len(ids) > 0 # making sure docs are inserted

    for doc, id in zip(documents, ids):
        doc.db_id = id
    
    chunked_documents = text_splitter.split_documents(documents)
    embeddings = embedding_model.embed([doc.text for doc in chunked_documents])
    insert_res = vector_storage.insert_documents(embeddings, chunked_documents)
    time.sleep(3) # make sure data is inserted
    yield vector_storage

    # removing documents to reset database
    for id in ids:
        res = document_storage.remove_document(id)
        assert res # making sure docs are deleted
    vector_storage.remove_documents(insert_res)
    vector_storage.close()

@pytest.fixture(scope="module")
def text_splitter():
    return RecursiveTextSplitter()



@pytest.fixture(scope="module")
def embedding_model():
    return OpenAIEmbeddingModel()

