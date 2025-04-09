import heapq
from typing import Callable, List, Literal, Dict, Any
import asyncio
import re

from kruppe.llm import BaseLLM
from kruppe.functional.rag.retriever.base_retriever import BaseRetriever
from kruppe.models import Chunk, Query, Document
from kruppe.prompts.rag import FUSION_GENERATE_QUERIES_USER, FUSION_GENERATE_QUERIES_SYSTEM


class QueryFusionRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    mode: Literal["simple", "rrf"] = "simple"
    llm: BaseLLM
    num_queries: int = 5

    @property
    def fusion_method(self) -> Callable:
        if self.mode == "simple":
            return self._simple_fusion
        elif self.mode == "rrf":
            return self._reciprocal_rank_fusion
        else:
            raise ValueError(f"Unknown fusion method: {self.mode}")

    def retrieve(self, query: Query, filter: Dict[str, Any] = None) -> List[Document]:
        # generate queries
        if isinstance(query, Query):
            query = query.text
        
        messages = [
            {"role": "user", "content": FUSION_GENERATE_QUERIES_USER.format(query=query, n=self.num_queries)},
            {"role": "system", "content": FUSION_GENERATE_QUERIES_SYSTEM}
        ]
        llm_response = self.llm.generate(messages)
        queries = re.split(r'\n+', llm_response.text.strip())

        assert len(queries) == self.num_queries

        # retrieve chunks (list of list of chunks)
        retrieved_chunks = [
            retriever.retrieve(query=gen_query, filter=filter)
            for gen_query in queries
            for retriever in self.retrievers
        ]
        return self.fusion_method(retrieved_chunks)[:self.top_k]
    
    async def async_retrieve(self, query: Query | str, filter: Dict[str, Any] = None) -> List[Document]:
        # generate queries
        if isinstance(query, Query):
            query = query.text
        
        messages = [
            {"role": "user", "content": FUSION_GENERATE_QUERIES_USER.format(query=query, n=self.num_queries)},
            {"role": "system", "content": FUSION_GENERATE_QUERIES_SYSTEM}
        ]
        llm_response = await self.llm.async_generate(messages)
        queries = re.split(r'\n+', llm_response.text.strip())

        assert len(queries) == self.num_queries

        # retrieve documents
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(retriever.async_retrieve(query=gen_query, filter=filter))
                for gen_query in queries
                for retriever in self.retrievers
            ]
        retrieved_chunks = [t.result() for t in tasks]
        
        # rerank documents
        return self.fusion_method(retrieved_chunks)[:self.top_k]

    
    def _simple_fusion(self, retrieved_chunks: List[List[Chunk]]) -> List[Chunk]:
        """
        Simple fusion method that combines multiple ranked lists into a single list.
        This method assumes that the input lists are already sorted in descending order of relevance.

        Args:
            retrieved_chunks (List[List[Chunk]]): A list of lists, where each inner list contains chunks
                retrieved from a different retriever.

        Returns:
            List[Chunk]: A single list of chunks, sorted by their relevance.
        """
        # Create an iterator for each non-empty list of chunks
        iterators = [iter(chunk_list) for chunk_list in retrieved_chunks if chunk_list]
        
        # Build a heap that will always give the chunk with the highest score.
        # Since each list is sorted in descending order, the first element is the best.
        # We negate the score to use Python’s min-heap as a max-heap.
        heap = []
        for idx, it in enumerate(iterators):
            try:
                chunk = next(it)
                heapq.heappush(heap, (-chunk.score, idx, chunk))
            except StopIteration:
                pass  # In case an iterator is empty

        seen = set() # to track seen chunk ids
        merged_chunks = [] # final result list

        while heap:
            neg_score, idx, chunk = heapq.heappop(heap)
            
            # Only add the chunk if we haven't added one with the same id already.
            if chunk.id not in seen:
                seen.add(chunk.id)
                merged_chunks.append(chunk)
            
            # Push the next chunk from the same sublist (if available)
            try:
                next_chunk = next(iterators[idx])
                heapq.heappush(heap, (-next_chunk.score, idx, next_chunk))
            except StopIteration:
                continue

        return merged_chunks


    def _reciprocal_rank_fusion(self, retrieved_chunks: List[List[Chunk]], k=60):
        """
        Fuse multiple ranked lists into a single ranking using Reciprocal Rank Fusion (RRF).
        
        Parameters:
            retrieved_chunks (list of list): Each inner list represents a ranked list of documents.
                It is assumed that the ranking is in order of decreasing relevance (i.e., the first document
                is the highest ranked).
            retrieved_distances(list of list): ignored
                
            k (int, optional): A constant to dampen the influence of the rank. Default is 60.
            
        Returns:
            list: A list of documents sorted by their aggregated RRF score in descending order.
                Documents with higher scores appear earlier.
        """
        # https://medium.com/@devalshah1619/mathematical-intuition-behind-reciprocal-rank-fusion-rrf-explained-in-2-mins-002df0cc5e2a

        scores = {}  # key: doc.id, value: [representative doc, aggregated score]

        # Process each ranking list.
        for ranking in retrieved_chunks:
            # Enumerate with 1-indexing.
            for rank, doc in enumerate(ranking, 1):
                # Compute the score component.
                score_component = 1 / (k + rank)
                # Use setdefault to retrieve or initialize the aggregate entry.
                entry = scores.setdefault(doc.id, [doc, 0.0])
                entry[1] += score_component

        # Sorting the aggregated entries by their score in descending order.
        # Since sorting is unavoidable for returning an ordered list,
        # this step is O(n log n) where n is the number of unique docs.
        fused_ranking = sorted(scores.values(), key=lambda entry: entry[1], reverse=True)

        # Return the document objects in sorted order.
        return [doc for doc, score in fused_ranking]