import time

from rag.embeddings import OpenAIEmbeddingModel




def test_vector_storage(vector_storage, documents):
    embedding_model = OpenAIEmbeddingModel()
    embeddings = embedding_model.embed([doc.text for doc in documents])
    
    # test insert
    insert_res = vector_storage.insert_documents(embeddings, documents)
    assert len(insert_res) == len(documents)
    assert insert_res[0] > 0

    time.sleep(3) # make sure data is inserted before searching for it

    # test search
    try:
        search_res = vector_storage.search_vectors(embeddings)
        assert len(search_res) == len(embeddings)
        print(type(search_res))
        print(search_res)
        assert search_res[0][0]["id"] == insert_res[0]
        assert search_res[0][0]["entity"]["text"] == documents[0].text
        assert search_res[1][0]["id"] == insert_res[1]

    # test delete
    finally:
        remove_res = vector_storage.remove_documents(insert_res)
        assert remove_res == len(documents)
