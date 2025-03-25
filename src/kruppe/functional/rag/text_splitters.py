import asyncio
from pydantic import BaseModel, PrivateAttr
from abc import ABC, abstractmethod
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from kruppe.llm import BaseLLM
from kruppe.models import Document, Chunk
from kruppe.prompt_formatter import CustomPromptFormatter
from kruppe.prompts.rag import SPLITTER_CONTEXTUALIZE_USER


class BaseTextSplitter(ABC, BaseModel):
    """Custom Text Splitter interface"""

    chunk_size: int = 1024
    chunk_overlap: int = 64

    @abstractmethod
    def split_documents(documents: List[Document]) -> List[Chunk]:
        pass

    @abstractmethod
    async def async_split_documents(documents: List[Document]) -> List[Chunk]:
        pass


class RecursiveTextSplitter(BaseTextSplitter):
    """langchain recursive character text splitter"""
    _text_splitter = PrivateAttr()

    def model_post_init(self, __context):
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Split the given documents into chunks of text according to the recursive character text splitter

        The output chunks will have the same metadata, uuid, and db_id as the original document
        The chunks are linked together in a doubly linked list, where each chunk has a reference to the previous and next chunk

        Args:
            documents (List[Document]): the documents to split

        Returns:
            List[Chunk]: the list of chunked texts
        """
        new_chunks = []
        for document in documents:
            chunked_texts = self._text_splitter.split_text(document.text)
            prev_chunk = None

            # create a Chunk object for each chunked text
            # and link them together
            for i in range(0, len(chunked_texts)):
                curr_chunk = Chunk(
                    text=chunked_texts[i],
                    metadata=document.metadata,
                    document_id=document.id,
                    prev_chunk_id=prev_chunk.id if prev_chunk is not None else None,
                )

                # link the chunks together
                if prev_chunk is not None:
                    prev_chunk.next_chunk_id = curr_chunk.id
                prev_chunk = curr_chunk
                new_chunks.append(curr_chunk)

        return new_chunks

    async def async_split_documents(self, documents: List[Document]) -> List[Chunk]:
        return self.split_documents(documents)


class ContextualTextSplitter(BaseTextSplitter):
    """defunct - add context using ContextualVectorStoreIndex instead"""
    llm: BaseLLM
    _text_splitter = PrivateAttr()

    def model_post_init(self, __context):
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    async def async_split_documents(
        self, documents: List[Document]
    ) -> List[Chunk]:
        """
        Split the given documents into chunks of text according to the recursive character text splitter
        and generate a context for each chunk using the llm.

        The output chunks will have the same metadata, uuid, and db_id as the original document
        The chunks are linked together in a doubly linked list, where each chunk has a reference to the previous and next chunk

        Args:
            documents (List[Document]): the documents to split

        Returns:
            List[ContextualizedChunk]: the list of chunked texts
        """
        # create a prompt formatter to format the messages for the llm
        prompt_formatter = CustomPromptFormatter(
            prompt_template=SPLITTER_CONTEXTUALIZE_USER
        )
         
        # for document in documents:
        
        async def split_document_helper(document: Document) -> List[Chunk]:
            document_chunks = []  # list of Chunk objects, each representing a chunk of the document

            chunked_texts = self._text_splitter.split_text(document.text)

            # generate the message to contextualize the chunks using llm
            messages_list = []
            for chunked_text in chunked_texts:
                messages = prompt_formatter.format_messages(
                    WHOLE_DOCUMENT=document.text,
                    CHUNK_CONTENT=chunked_text,
                )
                messages_list.append(messages)

            # generate the context for each chunk using the llm
            contexts_list = await self.llm.batch_async_generate(messages_list)

            # create a Chunk object for each chunked text and link them together
            prev_chunk = None
            for i in range(0, len(chunked_texts)):
                context_text = contexts_list[i].text
                curr_chunk = Chunk(
                    text= "-CONTEXT-\n"+context_text+"\n-TEXT-\n"+chunked_texts[i], # NOTE: THIS STEP COMBINES CONTEXT WITH CHUNK
                    metadata=document.metadata,
                    document_id=document.id,
                    prev_chunk_id=prev_chunk.id if prev_chunk is not None else None,
                )

                # link the chunks together
                if prev_chunk is not None:
                    prev_chunk.next_chunk_id = curr_chunk.id
                prev_chunk = curr_chunk
                document_chunks.append(curr_chunk)
            return document_chunks

        # NOTE: i'm not sure doing it this way is actually faster, but eh it's worth a shot
        new_chunks = await asyncio.gather(*[split_document_helper(document) for document in documents])
        new_chunks = [chunk for sublist in new_chunks for chunk in sublist] # flatten the list of lists
        return new_chunks
    
    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        return asyncio.run(self.async_split_documents(documents))