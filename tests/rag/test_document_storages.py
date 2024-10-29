import pytest
import os

from rag.document_storages import MongoDBStore
from rag.models import Document

def test_document_storage(document_storage, documents):
    # insert documents
    ids = [document_storage.save_document(doc) for doc in documents]
    assert len(ids) == len(documents)

    # test getting documents
    for i in range(len(ids)):
        res = document_storage.get_document(id[i])
        assert res.text == documents[i].text
        assert res.metadata == documents[i].metadata

    # test fetching documents
    for doc in documents:
        regex = f"^{doc.text[:20]}"
        fetched_res = document_storage.search_document(regex)
        assert len(fetched_res) > 0
        assert isinstance(fetched_res[0], Document), "Get document error"
        assert fetched_res[0].text == doc.text, "Get document error"
        assert fetched_res[0].metadata == doc.metadata, "Get document error"
    
    # removing documents to reset database
    for id in ids:
        res = document_storage.remove_document(id)
        assert res # making sure docs are deleted
    