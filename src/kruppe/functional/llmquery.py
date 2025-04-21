from typing import Tuple, List, Dict

from pydantic import computed_field

from kruppe.functional.base_tool import BaseTool
from kruppe.llm import BaseLLM
from kruppe.prompts.functional import LLM_QUERY_USER, LLM_QUERY_SYSTEM, LLM_QUERY_TOOL_DESCRIPTION

class LLMQuery(BaseTool):
    llm: BaseLLM

    async def llm_query(self, query: str) -> Tuple[str, List]:
        messages = [
            {"role": "system", "content": LLM_QUERY_SYSTEM},
            {"role": "user", "content": LLM_QUERY_USER.format(query=query)},
        ]

        response = await self.llm.async_generate(messages)
        llm_knowledge = response.text.strip()
        obs = f"Knowledge: {llm_knowledge}"

        return obs, []
    
    @computed_field
    @property
    def llm_query_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": "llm_query",
                "description": LLM_QUERY_TOOL_DESCRIPTION,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "required": [
                        "query"
                    ],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user's query for which general knowledge is to be provided"
                        }
                    },
                    "additionalProperties": False
                }
            }
        }
