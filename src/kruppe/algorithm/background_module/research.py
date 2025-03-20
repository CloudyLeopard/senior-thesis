from json import tool
from pydantic import BaseModel
from typing import List, Dict, Optional
import re

from kruppe.llm import BaseLLM
from kruppe.prompts.background import (
    RESEARCH_STANDARD_SYSTEM,
    ANALYZE_QUERY_USER_CHAIN
)


class BackgroundResearcher(BaseModel):
    llm: BaseLLM
    system_message: str = RESEARCH_STANDARD_SYSTEM
    query: Optional[str] = None
    history: List[Dict] = []  # research history
    toolkits: List = []

    def new_query(self, query: str):
        self.query = query
        self.history = []


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
        entity_categories = [
            "firm",
            "industry",
            "country",
            "event",
            "personnel",
            "terminology",
        ]
        user_message = ANALYZE_QUERY_USER_CHAIN[0].format(
            query=self.query, entity_categories=", ".join(entity_categories)
        )
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]

        response1 = await self.llm.async_generate(messages)
        messages.append({"role": "assistant", "content": response1.text})

        # extract additional information from query
        information_categories = ["finance", "economy", "industry"]
        followup_message = ANALYZE_QUERY_USER_CHAIN[1].format(
            query=self.query, information_categories=", ".join(information_categories)
        )
        messages.append({"role": "user", "content": followup_message})

        response2 = await self.llm.async_generate(messages)
        messages.append({"role": "assistant", "content": response2.text})

        def _process_info_request(self, response_text: str) -> List[Dict]:
            pattern = re.compile(
                r'\("(?P<type>[^"]+)"\|(?P<name>[^|]*)\|(?P<category>[^|]*)\|(?P<reasoning>[^|]*)(?:\|(?P<entity_name>[^)]*))?\)'
            )
            
            results = response_text.split("\n")
            requests = [] # i call them "requests", like "info_requests" or "tool_requests"
            for result in results:
                match = pattern.match(response_text)
                if match:
                    requests.append(match.groupdict())
            return requests

        info_requests = _process_info_request(response1.text) + _process_info_request(response2.text)

        # ===== TOOL REQUEST =====
        
        return

    async def _research_helper(self, tool_request: Dict) -> Dict:
        """Helper function to "research query". Makes the individual tool request.
        After each request finish, also determines if there is anything particularly
        note worthy in this request's result.

        Args:
            tool_request (Dict): single tool request

        Returns:
            Dict: research history
        """
        return ...

    async def research_query(self, tool_requests: List[Dict]) -> List[Dict]:
        """This function takes a list of tool requests, and sends the requests to perform
         research with each tool. After each request is done, the result is saved to the
         "history" attribute.

        Args:
            tool_requests (List[Dict]): list of tool request

        Returns:
            List[Dict]: list of research history. This is also saved to "history" attribute
        """

        return ...

    async def compile_report(self) -> str:
        """Compile a research report on the relevant background pieces using the
        research history.

        Returns:
            str: research history
        """
