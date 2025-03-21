from abc import ABC, abstractmethod
from textwrap import dedent
from arrow import get
from pydantic import BaseModel, computed_field
from typing import Callable, List, Dict, Optional
import re
import json
import asyncio

from tqdm import tqdm

from kruppe.llm import BaseLLM
from kruppe.models import Document, Response
from kruppe.prompts.background import (
    RESEARCH_STANDARD_SYSTEM,
    ANALYZE_QUERY_USER_CHAIN,
    DETERMINE_INFO_QUERY_USER,
    ANALYZE_DOCUMENT_SIMPLE_USER,
    COMPILE_REPORT_USER
)
from kruppe.data_source.news.base_news import NewsSource
from kruppe.algorithm.utils import process_request
from kruppe.algorithm.agents import Researcher, Librarian

class BackgroundResearchTeam(BaseModel):
    ...


class BackgroundResearcher(Researcher):
    query: str
    system_message: str = RESEARCH_STANDARD_SYSTEM
    history: List[Dict] = []  # research history
    librarian: Librarian


    def new_query(self, query: str):
        self.query = query
        self.history = []

    async def execute(self):
        info_requests = await self.create_info_requests()
        await self.retrieve_and_analyze(info_requests)
        report = await self.compile_report()
        return report


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
    
    async def _analyze_helper(self, info_request: str):
        doc_generator = self.librarian.execute(info_request)

        async for document in doc_generator:
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
                "info_request": info_request,
                "document": document.text, # TODO: make this document class later im too lazy rn
                "analysis": llm_string
            }

            self.history.append(research_result)        
    
    async def retrieve_and_analyze(self, info_requests: List[str]) -> List[Dict]:
        """For each information request, retrieve relevant documents from the libriarian.
        Then analyze the document to extract relevant background for the query.

        Args:
            info_requests (List[str]): _description_

        Returns:
            List[Dict]: _description_
        """

        async for info_request in tqdm(info_requests):
            await self._analyze_helper(info_request)
        
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