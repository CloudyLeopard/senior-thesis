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
    saved_doc = docstore.save_document(documents[0])
    assert saved_doc == documents[0]
    inserted_docs_count += 1

    saved_docs = docstore.save_documents(documents[1:])
    assert len(saved_docs) == len(documents) - 1
    inserted_docs_count += len(saved_docs)
    
    # test getting documents using uuid
    uuids = [doc.id for doc in documents]
    for i in range(len(uuids)):
        res = docstore.get_document(uuid=uuids[i])
        assert res.text == documents[i].text
        assert res.metadata == documents[i].metadata
        assert res.id == documents[i].id
    
    # insert some customized data
    saved_docs = docstore.save_documents(documents3)
    assert len(saved_docs) == len(documents3)
    inserted_docs_count += len(saved_docs)

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
    saved_doc = docstore.save_document(documents3[0])
    assert saved_doc is None
    inserted_docs_count += 0
    assert len(docstore.get_all_documents()) == inserted_docs_count

    # test duplicate bulk insert
    saved_docs = docstore.save_documents(documents3)
    assert len(saved_docs) == 0
    inserted_docs_count += len(saved_docs)
    assert len(docstore.get_all_documents()) == inserted_docs_count

    # test duplicate insert (on unique index)
    # # document C except the combination of datasource and title is unique
    dupe_doc_ok = documents3[2].model_copy(deep=True)
    dupe_doc_ok.id = uuid4()
    dupe_doc_ok.metadata['datasource'] = "datasource C"
    saved_doc = docstore.save_document(dupe_doc_ok)
    assert saved_doc == dupe_doc_ok
    inserted_docs_count += 1
    assert len(docstore.get_all_documents()) == inserted_docs_count

    # # document C but this time its a full duplicate
    dupe_doc_bad = documents3[2].model_copy(deep=True)
    dupe_doc_bad.id = uuid4()
    saved_doc = docstore.save_document(dupe_doc_bad)
    assert saved_doc is None
    inserted_docs_count += 0
    assert len(docstore.get_all_documents()) == inserted_docs_count

    # test remove 
    del_doc_1 = documents3[0]
    rem_doc_bool = docstore.remove_document(uuid=del_doc_1.id)
    assert rem_doc_bool
    inserted_docs_count -= 1
    assert len(docstore.get_all_documents()) == inserted_docs_count
    assert docstore.get_document(uuid=del_doc_1.id) is None

    # test remove by uuid
    del_doc_2 = docstore.get_all_documents()[0]
    rem_doc_bool = docstore.remove_document(uuid=del_doc_2.id)
    assert rem_doc_bool
    inserted_docs_count -= 1
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
    saved_doc = await docstore.asave_document(documents[0])
    assert saved_doc == documents[0]
    inserted_docs_count += 1

    saved_docs = await docstore.asave_documents(documents[1:])
    assert len(saved_docs) == len(documents) - 1
    inserted_docs_count += len(saved_docs)
    
    # test getting documents using uuid
    uuids = [doc.id for doc in documents]
    for i in range(len(uuids)):
        res = await docstore.aget_document(uuid=uuids[i])
        assert res.text == documents[i].text
        assert res.metadata == documents[i].metadata
        assert res.id == documents[i].id
    
    # insert some customized data
    saved_docs= await docstore.asave_documents(documents3)
    assert len(saved_docs) == len(documents3)
    inserted_docs_count += len(saved_docs)

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
    saved_doc = await docstore.asave_document(documents3[0])
    assert saved_doc is None
    inserted_docs_count += 0
    all_docs = await docstore.aget_all_documents()
    assert len(all_docs) == inserted_docs_count

    # test duplicate bulk insert
    saved_docs = await docstore.asave_documents(documents3)
    assert saved_docs == []
    inserted_docs_count += len(saved_docs)
    all_docs = await docstore.aget_all_documents()
    assert len(all_docs) == inserted_docs_count

    # test duplicate insert (on unique index)
    # # document C except the combination of datasource and title is unique
    dupe_doc_ok = documents3[2].model_copy(deep=True)
    dupe_doc_ok.id = uuid4()
    dupe_doc_ok.metadata['datasource'] = "datasource C"
    saved_doc = await docstore.asave_document(dupe_doc_ok)
    assert saved_doc == dupe_doc_ok
    inserted_docs_count += 1
    all_docs = await docstore.aget_all_documents()
    assert len(all_docs) == inserted_docs_count

    # # document C but this time its a full duplicate
    dupe_doc_bad = documents3[2].model_copy(deep=True)
    dupe_doc_bad.id = uuid4()
    saved_doc = await docstore.asave_document(dupe_doc_bad)
    assert saved_doc is None
    inserted_docs_count += 0
    all_docs = await docstore.aget_all_documents()
    assert len(all_docs) == inserted_docs_count

    # test remove
    del_doc_1 = documents3[0]
    rem_doc_bool = await docstore.aremove_document(del_doc_1.id)
    assert rem_doc_bool
    inserted_docs_count -= 1
    all_docs = await docstore.aget_all_documents()
    assert len(all_docs) == inserted_docs_count
    ret_doc = await docstore.aget_document(uuid=del_doc_1.id)
    assert ret_doc is None

    # test remove by uuid
    all_docs = await docstore.aget_all_documents()
    del_doc_2 = all_docs[0]
    rem_doc_bool = await docstore.aremove_document(uuid=del_doc_2.id)
    assert rem_doc_bool
    inserted_docs_count -= 1
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