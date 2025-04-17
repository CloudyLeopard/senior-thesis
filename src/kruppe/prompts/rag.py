from textwrap import dedent

STANDARD_SYSTEM = "You are a helpful assistant."

RAG_STANDARD_USER = dedent(
    """\
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    
    Question: {query}
    Context: {context}
    """
)

RAG_STANDARD_SYSTEM = dedent(
    """\
    You are a helpful assistant that answers a query using the given contexts
    """
)

SPLITTER_CONTEXTUALIZE_SYSTEM = dedent(
    """\
    You are a helpful assistant that contextualizes chunks of text within a larger document.
    """
)

SPLITTER_CONTEXTUALIZE_USER = dedent(
    """\
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

FUSION_GENERATE_QUERIES_SYSTEM = dedent(
    """\
    You are a helpful assistant that generates queries for a vector storage system to retrieve relevant documents.
    """
)

FUSION_GENERATE_QUERIES_USER = dedent(
    """\
    -Instruction-
    Given a search query, generate {n} query/queries that are semantically similar to the original query. 
    The generated queries will be used to retrieve relevant documents from a vector storage system.
    The generated {n} query/queries should be diverse and cover different aspects of the original query.
    Phrase them as statements instead of questions. The generated queries should be in the format of a list of queries, each separated by from each other by a single new line character.

    -Output Structure-
    [query 1]
    [query 2]
    ...

    -Input-
    Search query:
    {query}

    -Output-
    """
)