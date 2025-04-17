
from typing import List, Dict, Tuple
import re
import json
import asyncio
import logging
from pydantic import computed_field
from tqdm import tqdm

from kruppe.common.log import log_io
from kruppe.common.utils import combine_async_generators
from kruppe.llm import BaseLLM
from kruppe.models import Document, Response
from kruppe.prompts.background import (
    RESEARCH_STANDARD_SYSTEM,
    CREATE_INFO_REQUEST_USER,
    ANSWER_INFO_REQUEST_USER,
    COMPILE_REPORT_USER,
    ANALYZE_QUERY_USER_CHAIN,
    ANALYZE_DOCUMENT_SIMPLE_USER,
)
from kruppe.algorithm.utils import process_request
from kruppe.algorithm.agents import Researcher
from kruppe.algorithm.librarian import Librarian

logger = logging.getLogger(__name__)


class BackgroundResearcher(Researcher):
    librarian: Librarian
    system_message: str = RESEARCH_STANDARD_SYSTEM
    research_question: str
    num_info_requests: int = 3 # number of info requests to generate
    verbatim_answer: bool = False # whether to return the raw documents or the processed response
    strict_answer: bool = True # TODO: change from boolean to magnitude so i have varying degree of "strictness". This determines if the system will continue even if no documents are found
    # properties users can access
    info_history: List[Tuple[str, Response]] = []  # TODO: change this to `info_history`
    reports: List[Response] = []

    @computed_field
    @property
    def latest_report(self) -> Response:
        return self.reports[-1] if self.reports else None

    def clear_history(self, research_question: str):
        self.research_question = research_question
        self.info_history = []

    async def execute(self):
        print("Creating info requests...")
        info_requests = await self.create_info_requests()
        print(f"Created {len(info_requests)} info requests")

        # TODO: add timeout or a way to continue with partial results if something breaks

        # answer each info request
        # doing this sequentially for now cuz it breaking
        for info_request in tqdm(info_requests, desc="Answering info requests\n"):
            await self.complete_info_request(info_request)

        print("Compiling report...")
        report = await self.compile_report()
        self.reports.append(report)
        return report


    @log_io
    async def create_info_requests(self) -> List[str]:
        user_message = CREATE_INFO_REQUEST_USER.format(
            query=self.research_question,
            n=self.num_info_requests
        )
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]

        llm_response = await self.llm.async_generate(messages)
        llm_string = llm_response.text

        info_requests = re.split(r'\n+', llm_string)

        return info_requests[:self.num_info_requests]

    @log_io
    async def complete_info_request(self, info_request: str) -> Response:
        """Prompt the librarian for relevant contexts to answer the info request,
        and then answer the info request. Pretty much RAG with an extra step lol.

        Also adds the (info_request, Response) pair to history.

        Args:
            info_request (str): description of the information we want answered

        Returns:
            Response: response to the information request, with the relevant contexts
        """

        # TODO: change this to "complete_info_request" 

        ret_docs = await self.librarian.execute(info_request)
        if self.strict_answer and not ret_docs:
            logger.warning(f"Info request '{info_request}' returned no documents.")
            return Response(text="I do not know, no relevant documents were found.")
        
        contexts = "\n\n".join([doc.text for doc in ret_docs])

        if self.verbatim_answer:
            return Response(text=contexts, sources=ret_docs)
        
        # generate response
        user_message = ANSWER_INFO_REQUEST_USER.format(
            query=self.research_question,
            info_request=info_request,
            contexts=contexts
        )
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]

        llm_response = await self.llm.async_generate(messages)
        llm_response.sources = ret_docs

        self.info_history.append((info_request, llm_response))

        return llm_response

    @log_io
    async def compile_report(self) -> Response:
        """Compile a research report on the relevant background pieces using the
        research history.

        Returns:
            str: research history
        """
        info_responses_list = [f"Question: {info_request}\nAnswer: {response.text}\n\n" for info_request, response in self.info_history]
        info_responses = "\n".join(info_responses_list)

        user_message = COMPILE_REPORT_USER.format(
            query=self.research_question,
            info_responses=info_responses
        )

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]

        llm_response = await self.llm.async_generate(messages)

        return llm_response


# =========== OLD CODE ===========
# NOTE: may consider turning this into the "naive" analyst
# that processes every document

    @log_io
    async def request_and_analyze(self, info_requests: List[str]) -> List[Dict]:
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

    async def _analyze_helper(self, document: Document):
        user_message = ANALYZE_DOCUMENT_SIMPLE_USER.format(
            query=self.research_question,
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

        self.info_history.append(research_result)
 
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
            query=self.research_question,
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
            query=self.research_question, information_categories=information_categories
        )
        messages.append({"role": "user", "content": followup_message})

        response2 = await self.llm.async_generate(messages)
        messages.append({"role": "assistant", "content": response2.text})
        
        response_text = response1.text + response2.text
        info_requests = process_request(response_text)
        
        info_requests_sorted = sorted(info_requests, key=lambda x: (x["entity_name"], x["type"]) )
        info_requests_string = "\n".join([json.dumps(info_request) for info_request in info_requests_sorted])
                    
        ...