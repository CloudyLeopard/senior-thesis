from .embeddings import OpenAIEmbeddingModel
from .vector_storages import MilvusVectorStorage

class RAGCoordinator:
    def __init__(self):
        embedding_model = OpenAIEmbeddingModel()
        vector_storage = MilvusVectorStorage(...)