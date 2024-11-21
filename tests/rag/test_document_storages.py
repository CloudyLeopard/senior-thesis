
import pytest
from rag.models import Document

# TODO: loop through document storages using pytest parametrize
def test_document_storage(document_storage, documents):
    # insert documents
    ids = document_storage.save_documents(documents)
    assert len(ids) == len(documents)
    
    # test getting documents
    for i in range(len(ids)):
        res = document_storage.get_document(ids[i])
        assert res.text == documents[i].text
        assert res.metadata == documents[i].metadata
        assert res.db_id == documents[i].db_id

    # test fetching documents
    for doc in documents:
        regex = f"^{doc.text[:20]}"
        fetched_res = document_storage.search_documents(regex)
        assert len(fetched_res) > 0
        assert isinstance(fetched_res[0], Document), "Get document error"
        assert fetched_res[0].text == doc.text, "Get document error"
        assert fetched_res[0].metadata == doc.metadata, "Get document error"
    
    # removing documents to reset database
    for id in ids:
        res = document_storage.remove_document(id)
        assert res # making sure docs are deleted


@pytest.mark.asyncio
async def test_async_document_storage(async_document_storage, documents):
    # insert documents
    ids = await async_document_storage.save_documents(documents)
    assert len(ids) == len(documents)
    
    # test getting documents
    for i in range(len(ids)):
        res = await async_document_storage.get_document(ids[i])
        assert res.text == documents[i].text
        assert res.metadata == documents[i].metadata
        assert res.db_id == documents[i].db_id

    # test fetching documents
    for doc in documents:
        regex = f"^{doc.text[:20]}"
        fetched_res = await async_document_storage.search_documents(regex)
        assert len(fetched_res) > 0
        assert isinstance(fetched_res[0], Document), "Get document error"
        assert fetched_res[0].text == doc.text, "Get document error"
        assert fetched_res[0].metadata == doc.metadata, "Get document error"
    
    # removing documents to reset database
    for id in ids:
        res = await async_document_storage.remove_document(id)
        assert res # making sure docs are deleted