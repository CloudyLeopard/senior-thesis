import logging
from pydantic import model_validator

from rag.index.vectorstore_index import VectorStoreIndex
from rag.llm import  BaseLLM
from rag.text_splitters import ContextualTextSplitter, RecursiveTextSplitter

logger = logging.getLogger(__name__)

class ContextualVectorStoreIndex(VectorStoreIndex):
    llm: BaseLLM

    @model_validator(mode='after')
    def check_contextual_text_splitter(self):
        text_splitter = self.text_splitter
        if isinstance(text_splitter, ContextualTextSplitter):
            pass
        elif isinstance(text_splitter, RecursiveTextSplitter):
            # note: contextual text splitter uses recursive text splitter
            self.text_splitter = ContextualTextSplitter(
                chunk_size=text_splitter.chunk_size,
                chunk_overlap=text_splitter.chunk_overlap,
                llm=self.llm
            )
        else:
            logger.warning("Using a splitter that is not contextual or recursive. Defaulting to ContextualTextSplitter")
            self.text_splitter = ContextualTextSplitter(
                chunk_size=text_splitter.chunk_size,
                chunk_overlap=text_splitter.chunk_overlap,
                llm=self.llm
            )
        
        return self