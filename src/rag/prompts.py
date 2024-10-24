RAG_PROMPT_STANDARD = '''You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.\
 If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\
 \nQuestion: {query}
 \nContext: {context}
'''

RAG_SYSTEM_STANDARD = '''You are a helpful assistant that answers a query using the given query'''