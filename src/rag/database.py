from typing import List

from rag.embeddings import BaseEmbeddingModel
from rag.document_storages import BaseDocumentStore
from rag.vector_storages import BaseVectorStorage
from rag.text_splitters import BaseTextSplitter
from rag.retrievers import *
from rag.models import Document

class VectorDatabase:
    """Class that stores document embeddings in vector storage, relevant metadata
    in document storage. Also retrieves chunked texts with original document's
    metadata"""

    def __init__(
        self,
        vector_storage: BaseVectorStorage,
        document_storage: BaseDocumentStore,
        embedding_model: BaseEmbeddingModel,
        text_splitter: BaseTextSplitter
    ):
        self.vector_storage = vector_storage
        self.document_storage = document_storage
        self.embedding_model = embedding_model
        self.text_splitter = text_splitter
    
    def store_documents(self, documents: List[Document]) -> List[int]:
        """splits documents into chunks, store chunk embedding into vector storage, 
        store document into document storage. """
        # store document (e.g. original data, metadat) into document storage
        for document in documents:
            # set db id, which will be stored in vector storage
            document.db_id = self.document_storage.save_document(document)

        # split documents into chunks
        chunked_documents = self.text_splitter.split_documents(documents)
        
        # embed chunked documents
        embeddings = self.embedding_model.embed([doc.text for doc in documents])
        
        # store embedding into vector storage
        # TODO: do i want to return this/
        return self.vector_storage.insert_documents(embeddings, chunked_documents)
    
    def retrieve_documents(self, prompt: str, top_k=3):
        """given prompt, embed it and return list of chunked text with original document information"""
        embedding = self.embedding_model.embed(prompt)
        return self.retrieve_by_embedding(embedding, top_k)

    def retrieve_by_embedding(self, embedding: List[float], top_k = 3) -> List[Document]:
        """given embeddings, retrieve list of chunked text with original document information"""
        retrieved_documents = self.vector_storage.search_vectors([embedding], top_k)[0]
        
        for document in retrieved_documents:
            # retrieve corresponding data in document storage
            db_document = self.document_storage.get_document(document.db_id)

            # if document exists in document storage, set metadata
            # this should always be true
            if db_document:
                document.metadata = db_document.metadata
        
        return retrieved_documents

    def as_retriever(self, retriever_type: str = "simple") -> BaseRetriever:
        """return retriever, determined by retriever type"""
        # TODO
        if retriever_type == "simple":
            return SimpleRetriever()