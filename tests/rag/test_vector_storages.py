
def test_vector_storage(vector_storage, embedding_model, text_splitter, documents):
    chunked_documents = text_splitter.split_documents(documents)
    embeddings = embedding_model.embed([doc.text for doc in chunked_documents])
    
    # test insert documents
    insert_res = vector_storage.insert_documents(embeddings, chunked_documents)
    assert len(insert_res) == len(chunked_documents)

    # test search vector
    retrieved_docs = vector_storage.search_vectors(embeddings)
    assert len(retrieved_docs) == len(embeddings)
    for i in range(retrieved_docs):
        assert retrieved_docs[i][0].text == chunked_documents.text
    
    # test remove documents
    remove_res = vector_storage.remove_document(insert_res)
    assert remove_res == len(insert_res)