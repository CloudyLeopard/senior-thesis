from abc import ABC, abstractmethod
from textwrap import dedent
from arrow import get
from pydantic import BaseModel, computed_field
from typing import Callable, List, Dict, Optional
import re
import json
import asyncio

from tqdm import tqdm

from kruppe.utils import log_io
from kruppe.data_source.utils import combine_async_generators
from kruppe.llm import BaseLLM
from kruppe.models import Document, Response
from kruppe.prompts.background import (
    RESEARCH_STANDARD_SYSTEM,
    ANALYZE_QUERY_USER_CHAIN,
    DETERMINE_INFO_QUERY_USER,
    ANALYZE_DOCUMENT_SIMPLE_USER,
    COMPILE_REPORT_USER
)
from kruppe.algorithm.utils import process_request
from kruppe.algorithm.agents import Researcher
from kruppe.algorithm.librarian import Librarian


class BackgroundResearcher(Researcher):
    query: str
    system_message: str = RESEARCH_STANDARD_SYSTEM
    history: List[Dict] = []  # research history
    librarian: Librarian


    def new_query(self, query: str):
        self.query = query
        self.history = []

    async def execute(self):
        print("Creating info requests...")
        info_requests = await self.create_info_requests()
        print(f"Created {len(info_requests)} info requests")

        # NOTE: if this breaks, its ok because partial results will be saved in self.history
        try:
            # Set a timeout (in seconds) for retrieve_and_analyze
            await asyncio.wait_for(self.retrieve_and_analyze(info_requests), timeout=360)
        except asyncio.TimeoutError:
            print("retrieve_and_analyze took too long; proceeding with partial results...")

        print("Compiling report...")
        report = await self.compile_report()
        return report


    @log_io
    async def create_info_requests(self) -> List[str]:
        user_message = DETERMINE_INFO_QUERY_USER.format(query=self.query)
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]

        llm_response = await self.llm.async_generate(messages)
        llm_string = llm_response.text

        info_requests = re.split(r'\n+', llm_string)

        return info_requests
    
    async def _analyze_helper(self, document: Document):
        user_message = ANALYZE_DOCUMENT_SIMPLE_USER.format(
            query=self.query,
            document=document.text
        )
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]

        llm_response = await self.llm.async_generate(messages)
        llm_string = llm_response.text

        research_result = {
            # "info_request": info_request, # NOTE: the hypothesis exp version should be different here
            "document": document.text, # TODO: make this document class later im too lazy rn
            "analysis": llm_string
        }

        self.history.append(research_result)


    @log_io
    async def retrieve_and_analyze(self, info_requests: List[str]) -> List[Dict]:
        """For each information request, retrieve relevant documents from the libriarian.
        Then analyze the document to extract relevant background for the query.

        Args:
            info_requests (List[str]): _description_

        Returns:
            List[Dict]: _description_
        """

        async_generators = []
        for info_request in info_requests:
            async_generators.append(self.librarian.execute(info_request))

        combined_generator = combine_async_generators(async_generators)

        tasks = []
        async for doc in combined_generator:
            tasks.append(asyncio.create_task(self._analyze_helper(doc)))

        await asyncio.gather(*tasks)
        
    @log_io
    async def compile_report(self) -> str:
        """Compile a research report on the relevant background pieces using the
        research history.

        Returns:
            str: research history
        """

        user_message = COMPILE_REPORT_USER.format(
            query=self.query,
            analyses="\n".join([item["analysis"] for item in self.history])
        )

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]

        llm_response = await self.llm.async_generate(messages)
        llm_string = llm_response.text

        return llm_string

    async def analyze_query(self) -> List[Dict]:
        """This function breaks down the query into areas to conduct background research.
        Then, it determines which tool to call on for each area. Finally, returns the
        list of tools to call, and to do what with the tool.

        Args:
            query (str): user query to research

        Returns:
            List[Dict]: list of tool request
        """

        ## ===== INFO REQUEST =====

        # extract entities from query
        entity_categories = "firm, industry, country, event, personnel, terminology"
        information_categories = "finance, economy, industry"

        user_message = ANALYZE_QUERY_USER_CHAIN[0].format(
            query=self.query,
            entity_categories=entity_categories,
            information_categories=information_categories
        )
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]

        response1 = await self.llm.async_generate(messages)
        messages.append({"role": "assistant", "content": response1.text})

        # extract additional information from query
        followup_message = ANALYZE_QUERY_USER_CHAIN[1].format(
            query=self.query, information_categories=information_categories
        )
        messages.append({"role": "user", "content": followup_message})

        response2 = await self.llm.async_generate(messages)
        messages.append({"role": "assistant", "content": response2.text})
        
        response_text = response1.text + response2.text
        info_requests = process_request(response_text)
        
        info_requests_sorted = sorted(info_requests, key=lambda x: (x["entity_name"], x["type"]) )
        info_requests_string = "\n".join([json.dumps(info_request) for info_request in info_requests_sorted])
                    
        ...

class DocumentResearcher(BaseModel):
    objective: str