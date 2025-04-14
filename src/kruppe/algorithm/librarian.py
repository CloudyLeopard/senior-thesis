from tqdm import tqdm
from pydantic import Field, computed_field, PrivateAttr
from typing import AsyncGenerator, Callable, List, Dict, Literal, Set, Tuple
import json
import asyncio
import re
from datetime import datetime
import logging

from kruppe.algorithm.agents import Researcher
from kruppe.utils import log_io
from kruppe.data_source.news.base_news import NewsSource
from kruppe.data_source.utils import is_method_ready, combine_async_generators
from kruppe.algorithm.utils import process_request
from kruppe.models import Document, Chunk, Query
from kruppe.prompts.librarian import (
    LIBRARIAN_STANDARD_SYSTEM,
    CHOOSE_RESOURCE_USER,
    REQUEST_TO_QUERY_USER,
    LIBRARIAN_TIME_USER,
    LIBRARIAN_CONTEXT_RELEVANCE_USER
)
from kruppe.functional.rag.index.base_index import BaseIndex
from kruppe.functional.rag.retriever.base_retriever import BaseRetriever
from kruppe.functional.docstore.base_docstore import BaseDocumentStore

logger = logging.getLogger(__name__)

class Librarian(Researcher):
    """
    Used to help users or agents find information. Given a description of the information that the user wants,
    it will either consult its index or its library of online sources to retrieve the information.
    
    Note: the goal of the Librarian is *not* to make huge creative leaps - that is the job of the other, 
    more dedicated "researcher". The goal of the librarian is that, given a specific piece of information
    that we want to find, the librarian finds it. The librarian is not meant to be creative, but rather
    to be efficient and accurate.
    """

    system_message: str = LIBRARIAN_STANDARD_SYSTEM
    news_source: NewsSource
    # fin_source:
    # forum_source:
    # llm_expert_source: 
    index: BaseIndex # for retrieve_from_index
    retriever: BaseRetriever = Field(default_factory = lambda data: data['index'].as_retriever())
    docstore: BaseDocumentStore # NOTE: THIS DOCUMENT STORE NEED TO HAVE A UNIQUE INDEX TO DEAL WITH DUPLICATES
    num_retries: int = 2
    relevance_score_threshold: Literal[1, 2, 3] | None = 2
    # retrieve_from_library related
    resource_rank_threshold: Literal[1, 2, 3] = 2
    num_rsc_per_retrieve: int = 2
    _executed_funcs: Set[Tuple[str]] = PrivateAttr(default_factory=set)

    @computed_field
    @property
    def library(self) -> Dict:
        """Returns a dictionary of the Librarian's library of online sources and functions.
        Used in `retrieve_from_library`.

        Returns:
            Dict: dictionary of the Librarian's functions and their schemas
        """
        registry = {}

        # register news_source's methods
        news_source_methods = set(
            x for x, y in NewsSource.__dict__.items() if isinstance(y, Callable)
        )
        for method in news_source_methods:
            func = getattr(self.news_source, method)

            if not is_method_ready(self.news_source, method):
                # method may not yet be implemented. we remove those from the librarian's tool kit
                continue

            registry[method] = {
                "func": func,
                "schema": self.news_source.get_schema(method),
            }

        # register fin_source's methods

        # register form_source's methods

        return registry
    
    @computed_field
    @property
    def document_count(self) -> int:
        """Returns the number of documents in the Documentstore.

        Returns:
            int: number of documents in the Documentstore
        """
        return self.docstore.document_count

    @log_io
    async def execute(
        self,
        info_request: str,
        **kwargs
    ) -> List[Document]:
        """Given a description of the information that the user wants, the Librarian will
        retrieve relevant contexts to the information description and return them to the user.
        
        It first tries to look for the information in the index. If none is found (i.e. index is empty),
        or if the retrieved contexts' confidence score is low, the Librarian will call on its 'library'
        of online sources and functions to retrieve the information. The Librarian will then save the
        retrieved documents to the Documentstore and Index. If the Librarian has already executed a function,
        it will not execute it again.

        If, after a few retries, the confidence score is still low, the Librarian will return an empty list

        Args:
            info_request (str): description of the information that the user wants to know
            retries (int): number of retries to get relevant contexts from index
            kwargs: additional arguments, defined by `retrieve_from_library` and `retrieve_from_index`

        Returns:
            List[Document]: list of documents, empty if confidence score is low
        """
        logger.info("Executing librarian with info request: %s", info_request)

        # NOTE: this step is unnecessary if I optimize the prompts that generate info_requests
        # in hypothesis and background.py, such that the info_requests is concsie and clear
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": REQUEST_TO_QUERY_USER.format(info_request=info_request)},
        ]
        llm_response = await self.llm.async_generate(messages)
        llm_string = llm_response.text # info request, except more concise
        logger.debug("LLM Query\nSYSTEM=%s\nUSER=%s", messages[0]["content"], messages[1]["content"])
        logger.debug("LLM Response\nASSISTANT=%s", llm_string)

        info_query = llm_string
        logger.info("Transformed info request to query: %s", info_query)

        current_try = 0
        while current_try <= self.num_retries:
            
            logger.info("Attempt %d/%d to retrieve from index", current_try + 1, self.num_retries + 1)

            ret_chunks = await self.retrieve_from_index(
                info_request=info_query,
                **kwargs
            )

            need_new_documents = False

            if len(ret_chunks) == 0:
                logger.warning("Index is empty")
                need_new_documents = True
            elif self.relevance_score_threshold is not None:
                # use LLM to determine relevance
                
                user_message = LIBRARIAN_CONTEXT_RELEVANCE_USER.format(
                    info_request=info_query,
                    contexts = "\n".join(chunk.text for chunk in ret_chunks)
                )
                messages = [
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": user_message},
                ]
                llm_response = await self.llm.async_generate(messages)
                llm_string = llm_response.text
                logger.debug("LLM Query\nSYSTEM=%s\nUSER=%s", messages[0]["content"], messages[1]["content"])
                logger.debug("LLM Response\nASSISTANT=%s", llm_string)

                # high relevance: 1, somewhat relevant: 2, not relevant: 3
                relevance = llm_string.lower().split("relevance: ")[-1].strip()
                relevance_score_map = {"highly relevant": 1, "somewhat relevant": 2, "not relevant": 3}
                relevance_score = relevance_score_map.get(relevance, 4)

                if relevance_score == 4:
                    logger.error("Could not determine relevance score from LLM.")

                if relevance_score > self.relevance_score_threshold:
                    logger.info("Relevance score %d is too low, need to collect new documents", relevance_score)
                    need_new_documents = True
            
            if need_new_documents:
                # low relevance, try again
                logger.info("Collecting from library for info request: %s", info_query)
                await self.retrieve_from_library(
                    info_request=info_query,
                    **kwargs
                )
                current_try += 1 # decrement retries

                # await asyncio.sleep(5) # sleep for 5 seconds before trying again
            else:
                # high relevance, return
                logger.info("Retrieved %d relevant contexts", len(ret_chunks))
                return ret_chunks
        
        logger.warning("Could not find relevant contexts after %d tries", self.num_retries)
        return []
        
    async def retrieve_from_index(
            self,
            info_request: str,
            top_k: int = 10,
            llm_restrict_time: bool = False,
            start_time: str | float | datetime = None,
            end_time: str | datetime = None,
            **kwargs
        ) -> List[Chunk]:
        """Generates a list of queries based on the information description, then retrieves the chunks
        from the index.

        Args:
            info_request (str): description of the information that the user wants to know

        Returns:
            List[str]: list of queries
        """
    
        # TODO: maybe more functionalities with filters... we'll see
        filter = None

        # --- FILTER WITH TIME ---
        if start_time or end_time:
            # manual filter
            
            start_filter = None
            if start_time:
                # turn into datetime object
                if isinstance(start_time, str):
                    start_time = datetime.strptime(start_time, "%Y-%m-%d")
                elif isinstance(start_time, datetime):
                    start_date_unix = int(start_time.timestamp())
                elif isinstance(start_time, int) or isinstance(start_time, float):
                    start_date_unix = int(start_time)
                else:
                    raise ValueError("start_time must be a string, datetime, or unix timestamp")
                start_filter = {"publication_time": {"$gte": start_date_unix}}
            
            end_filter = None
            if end_time:
                if isinstance(end_time, str):
                    end_time = datetime.strptime(end_time, "%Y-%m-%d")
                elif isinstance(end_time, datetime):
                    end_date_unix = int(end_time.timestamp())
                elif isinstance(end_time, int) or isinstance(end_time, float):
                    end_date_unix = int(end_time)
                else:
                    raise ValueError("end_time must be a string, datetime, or unix timestamp")
                end_filter = {"publication_time": {"$lte": end_date_unix}}
            
            if start_time and end_time:
                # combine filters if both are present
                filter = {"$and": [start_filter, end_filter]}
            else:
                # use either start_filter or end_filter
                filter = start_filter or end_filter

        elif llm_restrict_time:
            # "smart" filter with LLM
            
            user_message = LIBRARIAN_TIME_USER.format(info_request=info_request)
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": user_message},
            ]

            llm_response = await self.llm.async_generate(messages)
            llm_string = llm_response.text
            logger.debug("LLM Query\nSYSTEM=%s\nUSER=%s", messages[0]["content"], messages[1]["content"])
            logger.debug("LLM Response\nASSISTANT=%s", llm_string)

            pattern = r'start_date:\s*(?P<start_date>\S+).*?end_date:\s*(?P<end_date>\S+)'
            match = re.search(pattern, llm_string)
            if match:
                start_date_str = match.group("start_date")
                end_date_str = match.group("end_date")
                start_date_unix = int(datetime.strptime(start_date_str, "%Y-%m-%d").timestamp())
                end_date_unix = int(datetime.strptime(end_date_str, "%Y-%m-%d").timestamp())

                filter = {
                    "$and": [
                        {"publication_time": {"$gte": start_date_unix}},
                        {"publication_time": {"$lte": end_date_unix}},
                    ]
                }
            else:
                logger.warning("Could not find start_date or end_date in LLM response")

        # query index and retrieve documents
        query = Query(text=info_request)
        relevant_documents = self.index.query(query, top_k=top_k, filter=filter)
        return relevant_documents

    async def retrieve_from_library(
        self,
        info_request: str,
        **kwargs
    ) -> None:

        # get resource requests, and limit to num_resources
        resource_requests = await self._choose_resource(info_request)
        resource_requests = resource_requests[:self.num_rsc_per_retrieve]

        combined_generator = combine_async_generators([self._retrieve_helper(request) for request in resource_requests])

        documents_added = 0
        async for doc in combined_generator:
            saved_doc = await self.docstore.asave_document(doc)
            if saved_doc: # if document was a repeat, saved_doc will be None
                await self.index.async_add_documents([doc])
                documents_added += 1

                # logs
                logger.debug("Added document: title=%s, uuid=%s", saved_doc.metadata.get('title'), str(saved_doc.id))
                if documents_added % 10 == 0:
                    logger.info("Added %d documents to index and docstore", documents_added)
                
        logger.info("Added total of %d documents to index and docstore", documents_added)



    async def _choose_resource(self, info_request: str) -> List[Dict]:
        """Given an information description (which could be a query or a description of the information
        that the user is seeking), determine which tools/functions should be used to find the information
        and the parameters needed to execute the function.

        Args:
            info_request (str): description of the information that the user wants to know

        Returns:
            List[Dict]: list of resource requests
        """  
        # get resource descriptions
        desc_lists = []
        for method in self.library.keys():
            method_schema = self.library[method]["schema"]
            method_schema_json = json.dumps(method_schema)

            desc_lists.append(f"{method}: {method_schema_json}")

        resource_desc = "\n".join(desc_lists)

        # get past function calls
        funcs_past = "\n".join(f"func_name: {tup[0]}, parameters: {tup[1]}" for tup in self._executed_funcs)

        # format librarian user message
        user_message = CHOOSE_RESOURCE_USER.format(
            info_request=info_request,
            resource_desc=resource_desc, # resource descriptions in JSON format
            funcs_past=funcs_past, # past function calls
            n=self.num_rsc_per_retrieve
        )

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]

        llm_response = await self.llm.async_generate(messages)
        llm_string = llm_response.text
        logger.debug("LLM Query\nSYSTEM=%s\nUSER=%s", messages[0]["content"], messages[1]["content"])
        logger.debug("LLM Response\nASSISTANT=%s", llm_string)

        # parse llm response into list of dict
        resource_requests = process_request(llm_string)
        if not resource_requests:
            logger.warning("No resource requests found in LLM response")
            return []

        # sort by rank
        resource_requests = sorted(resource_requests, key=lambda x: x["rank"])

        return resource_requests

    async def _retrieve_helper(self, resource_request: Dict) -> AsyncGenerator[Document, None]:
        """Executes the function specified in the resource request and yields the documents.
        If the function has already been executed, it will not be executed again.

        Args:
            resource_request (Dict): resource request, returned from `_choose_resource`

        Returns:
            AsyncGenerator[Document, None]: _description_

        Yields:
            Iterator[AsyncGenerator[Document, None]]: _description_
        """
        func_name = resource_request["func_name"]
        func = self.library[func_name]["func"]  # NOTE: warning, may not be safe
        parameters = resource_request["parameters"]

        # TODO: remove later - right now, i'm putting a cap on number of documents that can be retrieved
        parameters["max_results"] = 10

        # check if rank is higher than lowest rank
        if "rank" in resource_request and resource_request["rank"] > self.resource_rank_threshold:
            return

        # check for duplicates.
        # we do not want to execute the same function twice
        new_func = (func_name, json.dumps(parameters))
        if new_func in self._executed_funcs:
            logger.warning("Skipping duplicate function execution: func=%s, params=%s", func_name, str(parameters))
            return
        
        # if we do not find a duplicate, add to _executed_funcs
        self._executed_funcs.add(new_func)

        logger.info("Executing func=%s, params=%s", func_name, parameters)

        result = func(**parameters)
        if hasattr(result, "__aiter__"):  # some functions return async iterators
            async for document in result: # if it is an async gen
                yield document
        else:
            logger.warning("The function returned a non-async generator. Awaiting it.")
            # Await the result if it's not an async generator
            result = await result
            for document in result:
                yield document
    

        
    