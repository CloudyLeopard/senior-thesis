from abc import ABC, abstractmethod
from pydantic import BaseModel, computed_field
from typing import AsyncGenerator, Callable, List, Dict
import json
import asyncio

from kruppe.utils import log_io
from kruppe.llm import BaseLLM
from kruppe.data_source.news.base_news import NewsSource
from kruppe.data_source.utils import is_method_ready, combine_async_generators
from kruppe.algorithm.utils import process_request
from kruppe.models import Document, Response
from kruppe.prompts.librarian import LIBRARIAN_STANDARD_SYSTEM, LIBRARIAN_STANDARD_USER


class Researcher(BaseModel, ABC):
    llm: BaseLLM
    system_message: str

    @abstractmethod
    async def execute(self):
        raise NotImplementedError


class Librarian(Researcher):
    system_message: str = LIBRARIAN_STANDARD_SYSTEM
    news_source: NewsSource
    # fin_source:
    # forum_source:

    @computed_field
    @property
    def library(self) -> Dict:
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
    def resource_desc(self) -> str:
        desc_lists = []
        for method in self.library.keys():
            method_schema = self.library[method]["schema"]
            method_schema_json = json.dumps(method_schema)

            desc_lists.append(f"{method}: {method_schema_json}")

        return "\n".join(desc_lists)
    
    @log_io
    async def execute(self, information_desc: str) -> AsyncGenerator[Document, None]:
        resource_requests = await self.choose_resource(information_desc)
        
        async for doc in self.retrieve(resource_requests):
            yield doc

    async def choose_resource(self, information_desc: str) -> List[Dict]:
        """Given an information description (which could be a query or a description of the information
        that the user is seeking), determine which tools/functions should be used to find the information
        and the parameters needed to execute the function.

        Args:
            information_desc (str): description of the information that the user wants to know

        Returns:
            List[Dict]: list of resource requests
        """  
        resource_desc = self.resource_desc
        user_message = LIBRARIAN_STANDARD_USER.format(
            information_desc=information_desc, resource_desc=resource_desc
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
        func_name = resource_request["func_name"]
        func = self.library[func_name]["func"]  # NOTE: warning, may not be safe
        parameters = resource_request["parameters"]

        # TODO: remove later - right now, i'm putting a cap on number of documents that can be retrieved
        parameters["max_results"] = 10

        result = func(**parameters)
        if hasattr(result, "__aiter__"):  # some functions return async iterators
            async for document in result:
                yield document
        else:
            result = await result
            for document in result:
                yield document

    async def retrieve(
        self, resource_requests: List[Dict], num_resources: int = 1
    ) -> AsyncGenerator[Document, None]:
        # limit to num_resources
        resource_requests = resource_requests[:num_resources]

        combined_generator = combine_async_generators([self._retrieve_helper(request) for request in resource_requests])

        async for doc in combined_generator:
            yield doc