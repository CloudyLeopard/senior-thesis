
from rag.sources import GoogleSearchData, LexisNexisData, FinancialTimesData
from rag.text_splitters import RecursiveTextSplitter
from rag.embeddings import OpenAIEmbeddingModel
from rag.document_storages import MongoDBStore
from rag.vector_storages import MilvusVectorStorage
from rag.retrievers import DocumentRetriever
from rag.prompts import RAGPromptFormatter
from rag.generator import OpenAILLM
import argparse

# TODO: scripts for async versions (much faster)
def rag_load_data(query: str, verbose: bool = False):
    """given query, scrape relevant news articles, store in document store and vector store"""

    # loading in all the api key thru environment variables
    text_splitter = RecursiveTextSplitter()
    document_store = MongoDBStore()
    vector_store = MilvusVectorStorage(embedding_model=OpenAIEmbeddingModel())

    if verbose:
        print("Initialized document store (MongoDB) and vector store (Milvus)")

    total_documents = []

    # scrape news articles
    # Google
    google_data = GoogleSearchData(document_store=document_store)
    if verbose:
        print("Attempting to load news articles from google")
    try:
        documents = google_data.fetch(query) # returns list of Documents
        total_documents.extend(documents)
        if verbose:
            print("Finished loading news articles from google")
    except Exception:
        print("Could not load news articles from google")

    # Lexis Nexis
    lexis_data = LexisNexisData(document_store=document_store)
    if verbose:
        print("Attempting to load Lexis Nexis data")
    try:
        documents = lexis_data.fetch(query)
        total_documents.extend(documents)
        if verbose:
            print("Finished loading Lexis Nexis data")
    except Exception:
        print("Could not load Lexis Nexis data")
    
    # Financial Times
    ft_data = FinancialTimesData(document_store=document_store)
    if verbose:
        print("Attempting to load Financial Times data")
    try:
        documents = ft_data.fetch(query)
        total_documents.extend(documents)
        if verbose:
            print("Finished loading Financial Times data")
    except Exception:
        print("Could not load Financial Times data")

    if verbose:
        print("Attempting to insert documents into vector store")

    chunked_documents = text_splitter.split_documents(total_documents) # split documents into chunks
    vector_store.insert_documents(chunked_documents) # insert documents into vector store
    if verbose:
        print("Finished inserting documents into vector store")

    # close connections
    document_store.close()
    vector_store.close()

    print("Number of documents found and loaded: ", len(total_documents))

def rag_query(query: str, verbose: bool = False):
    """given query, retrieve relevant documents and generate response"""

    # loading in all the api key thru environment variables
    vector_store = MilvusVectorStorage(embedding_model=OpenAIEmbeddingModel())
    document_store = MongoDBStore()
    retriever = DocumentRetriever(vector_storage=vector_store, document_store=document_store)
    print("Initialized retriever using vector store (Milvus) and document store (MongoDB)")

    prompt_formatter = RAGPromptFormatter()
    llm = OpenAILLM()

    documents = retriever.retrieve(prompt=query, top_k=3)
    prompt_formatter.add_documents(documents)
    messages = prompt_formatter.format_messages(user_prompt=query)

    print("Generating response...")
    response = llm.generate(messages)

    # close connections
    document_store.close()
    vector_store.close()
    
    print("Query: ", query)

    if verbose:
        print("-- Messages --")
        for message in messages:
            print(message["role"], ": ", message["content"])
        print("-- End Messages --")
    print("Response: ", response)


def main():
    parser = argparse.ArgumentParser(description="Load data for RAG.")
    # make load and query mutually exclusive (user can't do both at once)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--load', action='store_true', help="Load data for RAG. WARNING: must have either --load or --query")
    group.add_argument('--query', action='store_true', help="Query data for RAG. WARNING: must have either --load or --query")
    # optional verbose mode
    parser.add_argument('-v', action='store_true', help="Verbose mode.")
    # required query string
    parser.add_argument("query", type=str, help="The query for sources to retrieve data for.")
    args = parser.parse_args()
    
    if args.load:
        rag_load_data(args.query, verbose=args.v)
    elif args.query:
        rag_query(args.query, verbose=args.v)
    else:
        print("Must have either --load or --query")

if __name__ == "__main__":
    # main_load()
    main()