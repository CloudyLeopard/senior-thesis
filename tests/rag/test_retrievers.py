from uuid import UUID

from rag.retrievers import SimpleRetriever, DocumentRetriever


def test_simple_retriever_retrieve(vector_storage, query):
    retriever = SimpleRetriever(vector_storage=vector_storage)
    
    retrieved_documents = retriever.retrieve(prompt=query, top_k=3)
    assert len(retrieved_documents) == 3
    for retrieved_doc in retrieved_documents:
        assert retrieved_doc.text != ""
        assert retrieved_doc.uuid and isinstance(retrieved_doc.uuid, UUID)


def test_document_retriever_retrieve(vector_storage, document_storage, query):
    retriever = DocumentRetriever(vector_storage=vector_storage, document_store=document_storage)

    retrieved_documents = retriever.retrieve(prompt=query, top_k=3)
    assert len(retrieved_documents) == 3
    for retrieved_doc in retrieved_documents:
        assert retrieved_doc.text != ""
        assert retrieved_doc.metadata
        assert retrieved_doc.db_id
        assert retrieved_doc.uuid and isinstance(retrieved_doc.uuid, UUID)


