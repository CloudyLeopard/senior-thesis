

def test_vector_storage(vector_storage, embedding_model, documents):
    embeddings = embedding_model.embed([doc.text for doc in documents])
    
    search_res = vector_storage.search_vectors(embeddings)
    assert len(search_res) == len(embeddings)
    assert search_res[0][0]["entity"]["text"][:50] == documents[0].text[:50]