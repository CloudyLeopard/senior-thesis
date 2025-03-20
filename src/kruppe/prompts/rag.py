from textwrap import dedent

RAG_STANDARD_USER = dedent(
    """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    
    Question: {query}
    Context: {context}
    """
)

RAG_STANDARD_SYSTEM = dedent(
    """
    You are a helpful assistant that answers a query using the given contexts
    """
)

SPLITTER_CONTEXTUALIZE_USER = dedent(
    """
    <document> 
    {WHOLE_DOCUMENT}
    </document> 
    Here is the chunk we want to situate within the whole document 
    <chunk> 
    {CHUNK_CONTENT}
    </chunk> 
    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
    """
)