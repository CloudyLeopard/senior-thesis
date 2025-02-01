from kruppe.rag.text_splitters import RecursiveTextSplitter, ContextualTextSplitter
from kruppe.models import Chunk

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 64

def test_recursive_splitter(documents):
    text_splitter = RecursiveTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    # test chunk multiple documents at once
    chunked_documents = text_splitter.split_documents(documents)

    assert len(chunked_documents) > len(documents)
    assert all([isinstance(chunk, Chunk) for chunk in chunked_documents])
    assert len(chunked_documents[0].text) < len(documents[0].text)

    # test chunk specifics
    chunked_documents = text_splitter.split_documents([documents[0]])

    assert len(chunked_documents) > 1
    for chunk in chunked_documents:
        assert 0 < len(chunk.text) <= CHUNK_SIZE
        assert chunk.metadata == documents[0].metadata
        assert chunk.db_id == documents[0].db_id
        assert chunk.uuid == documents[0].uuid
    
    assert len(set([chunk.text for chunk in chunked_documents])) == len(chunked_documents)
    print("Number of chunks: ", len(chunked_documents))

    _text_hashes = set()
    documents = [doc for doc in chunked_documents if (doc_hash := hash(doc)) not in _text_hashes and not _text_hashes.add(doc_hash)]
    print("Number of unique chunks: ", len(_text_hashes))
    assert len(chunked_documents) == len(documents)

def test_contextual_splitter(documents, llm):
    text_splitter = ContextualTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, llm=llm)

    # # test chunk multiple documents at once
    # chunked_documents = text_splitter.split_documents(documents)

    # assert len(chunked_documents) > len(documents)
    # assert all([isinstance(chunk, Chunk) for chunk in chunked_documents])
    # assert len(chunked_documents[0].text) < len(documents[0].text)

    # test chunk specifics
    chunked_documents = text_splitter.split_documents([documents[0]])

    assert len(chunked_documents) > 1
    for chunk in chunked_documents:
        assert len(chunk.text) > 0 # not comparing to CHUNK_SIZE because chunk.text also includes context
        assert chunk.metadata == documents[0].metadata
        assert chunk.db_id == documents[0].db_id
        assert chunk.uuid == documents[0].uuid
        assert len(chunk.context) > 0
        assert len(chunk.original_text) == len(chunk.text) - len(chunk.context) # apparently after adding the space, no need to add +1