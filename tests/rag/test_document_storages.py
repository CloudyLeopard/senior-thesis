import pytest
import os

from rag.document_storages import MongoDBStore
from rag.models import Document

@pytest.fixture(scope="class")
def document_storage():
    uri = os.getenv("MONGODB_URI")
    db_name = "test"
    return MongoDBStore(uri=uri, db_name=db_name)

def test_document_storage(document_storage, documents):
    # test saving documents
    ids = [document_storage.save_document(doc) for doc in documents]
    
    assert len(ids) == len(documents), "Insert document error"
    assert all(ids), "Insert document error"

    # test fetching documents
    for i in range(len(ids)):
        fetched_res = document_storage.get_document(ids[i])
        assert isinstance(fetched_res, Document), "Get document error"
        assert fetched_res.text == documents[i].text, "Get document error"
        assert fetched_res.metadata == documents[i].metadata, "Get document error"

    # test removing documents
    for id in ids:
        res = document_storage.remove_document(id)
        assert res, "Remove document error" # this should be true

    
