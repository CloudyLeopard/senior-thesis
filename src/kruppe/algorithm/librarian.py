from pydantic import computed_field
from typing import AsyncGenerator, Callable, List, Dict, Any, Set
import json
import asyncio
import re
from datetime import datetime


from kruppe.algorithm.agents import Researcher
from kruppe.utils import log_io
from kruppe.data_source.news.base_news import NewsSource
from kruppe.data_source.utils import is_method_ready, combine_async_generators
from kruppe.algorithm.utils import process_request
from kruppe.models import Document, Chunk, Query
from kruppe.prompts.librarian import (
    LIBRARIAN_STANDARD_SYSTEM,
    LIBRARIAN_STANDARD_USER,
    LIBRARIAN_TIME_USER,
    LIBRARIAN_CONTEXT_CONFIDENCE_USER
)
from kruppe.functional.rag.index.base_index import BaseIndex
from kruppe.functional.docstore.base_docstore import BaseDocumentStore


CONFIDENCE_THRESHOLD = 5

class Librarian(Researcher):
    system_message: str = LIBRARIAN_STANDARD_SYSTEM
    news_source: NewsSource
    # fin_source:
    # forum_source:
    index: BaseIndex # for retrieve_from_index
    docstore: BaseDocumentStore # to avoid duplicates
    executed_funcs: Set[Dict[str, Any]] = set()

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
    
    @log_io
    async def execute(self, information_desc: str, retries=2) -> List[Document]:
        """Given a description of the information that the user wants, the Librarian will
        retrieve relevant contexts to the information description and return them to the user.
        
        It first tries to look for the information in the index. If none is found (i.e. index is empty),
        or if the retrieved contexts' confidence score is low, the Librarian will call on its 'library'
        of online sources and functions to retrieve the information. The Librarian will then save the
        retrieved documents to the Documentstore and Index. If the Librarian has already executed a function,
        it will not execute it again.

        If, after a few retries, the confidence score is still low, the Librarian will return an empty list

        Args:
            information_desc (str): description of the information that the user wants to know
            retries (int): number of retries to get relevant contexts from index

        Returns:
            List[Document]: list of documents, empty if confidence score is low
        """

        while retries > 0:
            ret_chunks = await self.retrieve_from_index(
                information_desc=information_desc,
                top_k=10,
            )

            # calculate confidence
            confidence_score = 0

            user_message = LIBRARIAN_CONTEXT_CONFIDENCE_USER.format(
                information_desc=information_desc,
                contexts = "\n".join(chunk.text for chunk in ret_chunks)
            )
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": user_message},
            ]

            llm_response = await self.llm.async_generate(messages)
            llm_string = llm_response.text
            confidence_score = int(llm_string.split("\n")[-1])

            if confidence_score < CONFIDENCE_THRESHOLD:
                scraped_doc = await self.retrieve_from_library(
                    information_desc=information_desc,
                    num_resources=1 # only retrieve from one resource
                )
                retries -= 1 # decrement retries
            else:
                return ret_chunks
            
        return []
        
    async def retrieve_from_index(
            self,
            information_desc: str,
            top_k: int = 10,
            llm_restrict_time: bool = False,
            start_time: str = None,
            end_time: str = None
        ) -> List[Chunk]:
        """Generates a list of queries based on the information description, then retrieves the chunks
        from the index.

        Args:
            information_desc (str): description of the information that the user wants to know

        Returns:
            List[str]: list of queries
        """

        # TODO: add function for LLM to generate sub-queries given an information description
        # maybe do this in index?
    
        # TODO: maybe more functionalities with filters... we'll see
        filter = None

        # --- FILTER WITH TIME ---
        start_date_str = None
        end_date_str = None
        if start_time or end_time:
            start_date_str = start_time if start_time else "1970-01-01"
            end_date_str = end_time if end_time else "2038-01-19"
            start_date_unix = int(datetime.strptime(start_date_str, "%Y-%m-%d").timestamp())
            end_date_unix = int(datetime.strptime(end_date_str, "%Y-%m-%d").timestamp())

            filter = {
                "$and": [
                    {"publication_time": {"$gte": start_date_unix}},
                    {"publication_time": {"$lte": end_date_unix}},
                ]
            }

        elif llm_restrict_time:
            user_message = LIBRARIAN_TIME_USER.format(information_desc=information_desc)
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": user_message},
            ]

            llm_response = await self.llm.async_generate(messages)
            llm_string = llm_response.text

            if llm_string.startswith("yes"):
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

        # query index and retrieve documents
        query = Query(text=information_desc)
        relevant_documents = self.index.query(query, top_k=top_k, filter=filter)
        return relevant_documents

    async def retrieve_from_library(
        self, information_desc: str, num_resources: int = 1
    ) -> List[Document]:
        # get resource requests, and limit to num_resources
        resource_requests = await self._choose_resource(information_desc)
        resource_requests = resource_requests[:num_resources]

        combined_generator = combine_async_generators([self._retrieve_helper(request) for request in resource_requests])

        running_save_tasks = set()
        async for doc in combined_generator:
            # create a task to save the document to docstore and index
            # https://stackoverflow.com/questions/71938799/python-asyncio-create-task-really-need-to-keep-a-reference
            save_task = asyncio.create_task(self._save_to_docstore_and_index(doc))
            running_save_tasks.add(save_task)
        
        return await asyncio.gather(*running_save_tasks) # final catch all to make sure all save tasks are done

    async def _choose_resource(self, information_desc: str) -> List[Dict]:
        """Given an information description (which could be a query or a description of the information
        that the user is seeking), determine which tools/functions should be used to find the information
        and the parameters needed to execute the function.

        Args:
            information_desc (str): description of the information that the user wants to know

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
        funcs_past = "\n".join(self.executed_funcs)

        # format librarian user message
        user_message = LIBRARIAN_STANDARD_USER.format(
            information_desc=information_desc,
            resource_desc=resource_desc, # resource descriptions in JSON format
            funcs_past=funcs_past, # past function calls
        )

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]

        llm_response = await self.llm.async_generate(messages)
        llm_string = llm_response.text

        # parse llm response into list of dict
        resource_requests = process_request(llm_string)

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

        # TODO: add function filter by rank later
        # ...

        new_func = {"func_name": func_name, "parameters": parameters}
        
        # do not execute the same function twice
        if new_func in self.executed_funcs:
            return
        
        # add to executed_funcs
        self.executed_funcs.add(new_func)

        result = func(**parameters)
        if hasattr(result, "__aiter__"):  # some functions return async iterators
            async for document in result:
                yield document
        else:
            result = await result
            for document in result:
                yield document
    
    async def _save_to_docstore_and_index(self, document: Document):
        """Saves Document to Documentstore and Index. 
        Will not save if document is a repeat.

        Args:
            document (Document): Document to save
        """
        saved_doc = await self.docstore.asave_document(document)
        if saved_doc: # if document was a repeat, saved_doc will be None
            await self.index.async_add_documents([document])
        
        return saved_doc
        
    