import asyncio
import numpy as np
from typing import List
import logging
from pydantic import Field, PrivateAttr

from rag.vector_store.base_store import BaseVectorStore
from rag.models import Document

logger = logging.getLogger(__name__)

def cosine_similarity(a: np.ndarray, b: np.ndarray):
    """calculates cosine similarity between a and b, both 2d array of shape (n or m, embedding_dim)
    returns a 2d array of shape (n, m) where similarity[i][j] = cosine similarity between a[i] and b[j]
    """
    norm_product = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.dot(a, b.T) / norm_product

class InMemoryVectorStore(BaseVectorStore):
    documents: List[Document] = Field(default_factory=list)
    _embeddings_matrix: np.ndarray = PrivateAttr(default=None)
    
    def insert_documents(self, documents: List[Document]):
        return asyncio.run(self.async_insert_documents(documents))
    
    async def async_insert_documents(self, documents: List[Document]):
        # remove duplicate documents
        documents = [doc for doc in documents if (doc_hash := hash(doc)) not in self.texts_hashes and not self.texts_hashes.add(doc_hash)]

        embeddings = await self.embedding_model.async_embed(documents)
        if self._embeddings_matrix is None:
            self._embeddings_matrix = np.array(embeddings)
            ids = list(range(len(documents)))
        else:
            self._embeddings_matrix = np.concatenate((self._embeddings_matrix, np.array(embeddings)), axis=0)
            ids = list(range(len(self.documents), len(documents) + len(self.documents)))
        
        self.documents.extend(documents)
        return ids

    def search(self, vector: List[float], top_k: int = 3):
        top_k = min(top_k, len(self.documents))
        if (top_k == 0):
            return []
        
        similarity_scores = cosine_similarity(vector.reshape(1, -1), self._embeddings_matrix)[0]
        similarity_scores = np.nan_to_num(similarity_scores, nan=-np.inf) # treat nan values as -inf
        sorted_indices = np.argsort(similarity_scores)[::-1] # sort in descending order
        top_indices = sorted_indices[:top_k] # get top k indices

        documents = [self.documents[index] for index in top_indices]
        return documents
    
    async def async_search(self, vector: List[float], top_k: int = 3) -> List[Document]:
        return self.search(vector, top_k)
        
        
    def remove_documents(self, ids: List[int]):
        # TODO: update text hashes
        
        mask = np.ones(len(self.documents), dtype=bool)
        mask[ids] = False
        original_size = len(self.documents)
        self.documents = [doc for doc, keep in zip(self.documents, mask) if keep]
        self._embeddings_matrix = self._embeddings_matrix[mask]
        return original_size - len(self.documents)
    
    # def clear(self):
    #     super().clear()
    #     self.documents = []
    #     self._embeddings_matrix = None
