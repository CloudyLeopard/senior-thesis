import pytest
from uuid import uuid4

from kruppe.models import Document
from kruppe.functional.docstore.mongo_store import MongoDBStore

# TODO: loop through document storages using pytest parametrize
def test_document_storage(documents, documents3):
    inserted_docs_count = 0

    unique_indices = [["datasource", "title"]]

    docstore = MongoDBStore.create_db(
        db_name="test",
        collection_name="test_solo",
        reset_db=True,
        unique_indices=unique_indices
    )
    
    # insert document
    num_saved = docstore.save_document(documents[0])
    assert num_saved == 1
    inserted_docs_count += num_saved

    num_saved = docstore.save_documents(documents[1:])
    assert num_saved == len(documents) - 1
    inserted_docs_count += num_saved
    
    # test getting documents using uuid
    uuids = [doc.id for doc in documents]
    for i in range(len(uuids)):
        res = docstore.get_document(uuid=uuids[i])
        assert res.text == documents[i].text
        assert res.metadata == documents[i].metadata
        assert res.id == documents[i].id
    
    # insert some customized data
    num_saved = docstore.save_documents(documents3)
    assert num_saved == len(documents3)
    inserted_docs_count += num_saved

    # test get all documents
    ret_docs = docstore.get_all_documents()
    assert len(ret_docs) == inserted_docs_count

    # test with filters
    filter = {'title': {'$eq': 'title B'}}
    retrieved_docs = docstore.search_documents(filter)
    assert len(retrieved_docs) == 1
    assert retrieved_docs[0].metadata['title'] == 'title B'
    assert retrieved_docs[0].metadata['description'] == 'description B'

    # test with time filters
    start_time = 1612155600 # 2021-02-01 00:00:00
    end_time = 1617249600 # 2021-04-01 00:00:00
    filter_start = {'publication_time': {'$gt': start_time}}
    filter_end = {'publication_time': {'$lt': end_time}}
    filter = {'$and': [filter_start, filter_end]}
    retrieved_docs = docstore.search_documents(filter)
    assert len(retrieved_docs) == 2
    metadata = [doc.metadata for doc in retrieved_docs]
    uuids = [doc.id for doc in retrieved_docs]
    assert documents3[1].metadata in metadata
    assert documents3[2].metadata in metadata
    assert documents3[1].id in uuids
    assert documents3[2].id in uuids
    
    # test duplicate insert (on uuid)
    num_saved = docstore.save_document(documents3[0])
    assert num_saved == 0
    inserted_docs_count += num_saved
    assert len(docstore.get_all_documents()) == inserted_docs_count

    # test duplicate bulk insert
    num_saved = docstore.save_documents(documents3)
    assert num_saved == 0
    assert len(docstore.get_all_documents()) == inserted_docs_count

    # test duplicate insert (on unique index)
    # # document C except the combination of datasource and title is unique
    dupe_doc_ok = documents3[2].model_copy(deep=True)
    dupe_doc_ok.id = uuid4()
    dupe_doc_ok.metadata['datasource'] = "datasource C"
    num_saved = docstore.save_document(dupe_doc_ok)
    assert num_saved == 1
    inserted_docs_count += num_saved
    assert len(docstore.get_all_documents()) == inserted_docs_count

    # # document C but this time its a full duplicate
    dupe_doc_bad = documents3[2].model_copy(deep=True)
    dupe_doc_bad.id = uuid4()
    num_saved = docstore.save_document(dupe_doc_bad)
    assert num_saved == 0
    inserted_docs_count += num_saved
    assert len(docstore.get_all_documents()) == inserted_docs_count

    # test remove 
    del_doc_1 = documents3[0]
    num_rem = docstore.remove_document(uuid=del_doc_1.id)
    assert num_rem == 1
    inserted_docs_count -= num_rem
    assert len(docstore.get_all_documents()) == inserted_docs_count
    assert docstore.get_document(uuid=del_doc_1.id) is None

    # test remove by uuid
    del_doc_2 = docstore.get_all_documents()[0]
    num_rem = docstore.remove_document(uuid=del_doc_2.id)
    assert num_rem == 1
    inserted_docs_count -= num_rem
    assert len(docstore.get_all_documents()) == inserted_docs_count
    assert docstore.get_document(uuid=del_doc_2.id) is None

    # test remove all
    num_rem = docstore.clear_collection(drop_index=True)
    assert num_rem == inserted_docs_count
    assert docstore.get_all_documents() == []

    docstore.close()


@pytest.mark.asyncio(loop_scope="module")
async def test_async_document_storage(documents, documents3):
    inserted_docs_count = 0

    unique_indices = [["datasource", "title"]]

    docstore = await MongoDBStore.acreate_db(
        db_name="test",
        collection_name="test_solo",
        reset_db=True,
        unique_indices=unique_indices
    )
    
    # insert document
    num_saved = await docstore.asave_document(documents[0])
    assert num_saved == 1
    inserted_docs_count += num_saved

    num_saved = await docstore.asave_documents(documents[1:])
    assert num_saved == len(documents) - 1
    inserted_docs_count += num_saved
    
    # test getting documents using uuid
    uuids = [doc.id for doc in documents]
    for i in range(len(uuids)):
        res = await docstore.aget_document(uuid=uuids[i])
        assert res.text == documents[i].text
        assert res.metadata == documents[i].metadata
        assert res.id == documents[i].id
    
    # insert some customized data
    num_saved = await docstore.asave_documents(documents3)
    assert num_saved == len(documents3)
    inserted_docs_count += num_saved

    # test get all documents
    ret_docs = await docstore.aget_all_documents()
    assert len(ret_docs) == inserted_docs_count

    # test with filters
    filter = {'title': {'$eq': 'title B'}}
    retrieved_docs = await docstore.asearch_documents(filter)
    assert len(retrieved_docs) == 1
    assert retrieved_docs[0].metadata['title'] == 'title B'
    assert retrieved_docs[0].metadata['description'] == 'description B'

    # test with time filters
    start_time = 1612155600
    end_time = 1617249600
    filter_start = {'publication_time': {'$gt': start_time}}
    filter_end = {'publication_time': {'$lt': end_time}}
    filter = {'$and': [filter_start, filter_end]}
    retrieved_docs = await docstore.asearch_documents(filter)
    assert len(retrieved_docs) == 2
    metadata = [doc.metadata for doc in retrieved_docs]
    uuids = [doc.id for doc in retrieved_docs]
    assert documents3[1].metadata in metadata
    assert documents3[2].metadata in metadata
    assert documents3[1].id in uuids
    assert documents3[2].id in uuids

    # test duplicate insert (on uuid)
    num_saved = await docstore.asave_document(documents3[0])
    assert num_saved == 0
    inserted_docs_count += num_saved
    all_docs = await docstore.aget_all_documents()
    assert len(all_docs) == inserted_docs_count

    # test duplicate bulk insert
    num_saved = await docstore.asave_documents(documents3)
    assert num_saved == 0
    inserted_docs_count += num_saved
    all_docs = await docstore.aget_all_documents()
    assert len(all_docs) == inserted_docs_count

    # test duplicate insert (on unique index)
    # # document C except the combination of datasource and title is unique
    dupe_doc_ok = documents3[2].model_copy(deep=True)
    dupe_doc_ok.id = uuid4()
    dupe_doc_ok.metadata['datasource'] = "datasource C"
    num_saved = await docstore.asave_document(dupe_doc_ok)
    assert num_saved == 1
    inserted_docs_count += num_saved
    all_docs = await docstore.aget_all_documents()
    assert len(all_docs) == inserted_docs_count

    # # document C but this time its a full duplicate
    dupe_doc_bad = documents3[2].model_copy(deep=True)
    dupe_doc_bad.id = uuid4()
    num_saved = await docstore.asave_document(dupe_doc_bad)
    assert num_saved == 0
    inserted_docs_count += num_saved
    all_docs = await docstore.aget_all_documents()
    assert len(all_docs) == inserted_docs_count

    # test remove
    del_doc_1 = documents3[0]
    num_rem = await docstore.aremove_document(del_doc_1.id)
    assert num_rem == 1
    inserted_docs_count -= num_rem
    all_docs = await docstore.aget_all_documents()
    assert len(all_docs) == inserted_docs_count
    ret_doc = await docstore.aget_document(uuid=del_doc_1.id)
    assert ret_doc is None

    # test remove by uuid
    all_docs = await docstore.aget_all_documents()
    del_doc_2 = all_docs[0]
    num_rem = await docstore.aremove_document(uuid=del_doc_2.id)
    assert num_rem == 1
    inserted_docs_count -= num_rem
    all_docs = await docstore.aget_all_documents()
    assert len(all_docs) == inserted_docs_count
    ret_doc = await docstore.aget_document(uuid=del_doc_2.id)
    assert ret_doc is None

    # test remove all
    num_rem = await docstore.aclear_collection(drop_index=True)
    assert num_rem == inserted_docs_count
    all_docs = await docstore.aget_all_documents()
    assert all_docs == []

    await docstore.aclose()