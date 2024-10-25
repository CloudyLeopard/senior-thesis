import pytest
import os

from rag.document_storages import MongoDBStore
from rag.models import Document

def test_document_storage(document_storage, documents):

    # NOTE: "inserting" and "removing" document is assumed correct. It's used in conftest
    # test fetching documents
    for doc in documents:
        regex = f"^{doc.text[:20]}"
        fetched_res = document_storage.search_document(regex)
        assert len(fetched_res) > 0
        assert isinstance(fetched_res[0], Document), "Get document error"
        assert fetched_res[0].text == doc.text, "Get document error"
        assert fetched_res[0].metadata == doc.metadata, "Get document error"
    
