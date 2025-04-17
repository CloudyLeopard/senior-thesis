from datetime import datetime
from typing import List, Tuple, Dict
import logging

from pydantic import computed_field

from kruppe.llm import BaseLLM
from kruppe.common.utils import convert_to_datetime
from kruppe.functional.base_tool import BaseTool
from kruppe.functional.rag.retriever.base_retriever import BaseRetriever
from kruppe.models import Chunk
from kruppe.prompts.functional import RAG_QUERY_USER, RAG_QUERY_SYSTEM, RAG_QUERY_TOOL_DESCRIPTION

logger = logging.getLogger(__name__)

class RagQuery(BaseTool):
    retriever: BaseRetriever
    llm: BaseLLM

    async def rag_query(
        self, 
        query: str,
        start_date: str | float | datetime = None,
        end_date: str | float | datetime = None,
    ) -> Tuple[str, List[Chunk]]:
        """OpenAI Tool friendly version of the rag query."""
        
        # Adding filters
        filter = None

        # validating and converting start_date and end_date
        try:
            start_date = convert_to_datetime(start_date)
        except ValueError:
            start_date = None

        try:
            end_date = convert_to_datetime(end_date)
        except ValueError:
            end_date = None

        start_filter = None
        if start_date:
            start_date_unix = int(start_date.timestamp())
            start_filter = {"publication_time": {"$gte": start_date_unix}}
        
        end_filter = None
        if end_date:
            end_date_unix = int(end_date.timestamp())
            end_filter = {"publication_time": {"$lte": end_date_unix}}

        if start_filter and end_filter:
            filter = {"$and": [start_filter, end_filter]}
        else:
            filter = start_filter or end_filter
        

        # retrieval and query

        ret_chunks = await self.retriever.async_retrieve(
            query=query, 
            filter=filter
        )

        contexts = [f"{i}. {ret_chunks[i].text}" for i in range(len(ret_chunks))]

        # formatting the message
        messages = [
            {
                "role": "system",
                "content": RAG_QUERY_SYSTEM.format(contexts="\n".join(contexts))
            },
            {
                "role": "user",
                "content": RAG_QUERY_USER.format(query=query)
            }
        ]

        # querying the LLM
        response = await self.llm.async_generate(messages=messages)
        rag_response_str = response.text

        try:
            thoughts, answer = rag_response_str.split("Answer:", 1)
            thoughts = thoughts.replace("Thoughts:", "").strip()
            answer = answer.strip()
        except ValueError:
            logger.warning("RAG response format error. Expected 'Thoughts:' and 'Answer:' sections.")
            thoughts = ""
            answer = rag_response_str.strip() # fallback to the whole response if format is incorrect
        
        return answer, ret_chunks
    
    @computed_field
    @property
    def rag_query_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": "rag_query",
                "description": RAG_QUERY_TOOL_DESCRIPTION,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "required": [
                        "query",
                        "end_date",
                        "start_date",
                    ],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query string to search for."
                        },
                        "end_date": {
                            "type": ["string", "null"],
                            "description": f"End date filter (YYYY-MM-DD), or null for no filter. For reference, today's date is {datetime.now().strftime('%Y-%m-%d')}.",
                        },
                        "start_date": {
                            "type": ["string", "null"],
                            "description": f"Start date filter (YYYY-MM-DD), or null for no filter. For reference, today's date is {datetime.now().strftime('%Y-%m-%d')}.",
                        },
                    },
                    "additionalProperties": False
                }
            }
        }






