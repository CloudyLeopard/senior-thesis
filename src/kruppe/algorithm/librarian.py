from pydantic import computed_field
from typing import List, override

import logging

from kruppe.algorithm.agents import ReActResearcher
from kruppe.common.log import log_io
from kruppe.models import Document, Chunk, Response
from kruppe.prompts.librarian import (
    REACT_RESEARCH_SYSTEM,
    REACT_RESEARCH_USER
)
from kruppe.functional.rag.index.base_index import BaseIndex
from kruppe.functional.docstore.base_docstore import BaseDocumentStore

logger = logging.getLogger(__name__)

class Librarian(ReActResearcher):
    """
    Used to help users or agents find information. Given a description of the information that the user wants,
    it will either consult its index or its library of online sources to retrieve the information.
    
    Note: the goal of the Librarian is *not* to make huge creative leaps - that is the job of the other, 
    more dedicated "researcher". The goal of the librarian is that, given a specific piece of information
    that we want to find, the librarian finds it. The librarian is not meant to be creative, but rather
    to be efficient and accurate.
    """

    docstore: BaseDocumentStore # NOTE: THIS DOCUMENT STORE NEED TO HAVE A UNIQUE INDEX TO DEAL WITH DUPLICATES
    index: BaseIndex # for retrieve_from_index
    
    @computed_field
    @property
    def document_count(self) -> int:
        """Returns the number of documents in the Documentstore.

        Returns:
            int: number of documents in the Documentstore
        """
        return self.docstore.document_count

    def _react_system_prompt(self) -> str:
        return REACT_RESEARCH_SYSTEM
    
    def _react_user_prompt(self, query: str) -> str:
        return REACT_RESEARCH_USER.format(query=query)

    
    @override
    async def execute(self, query: str, to_print=True) -> Response:
        """Executes the research task using the ReAct framework."""
        all_sources: List[Chunk | Document] = []
        done = False
        ans = None

        # initialize the messages with the system prompt
        self._messages = [
            {"role": "system", "content": self._react_system_prompt()},
            {"role": "user", "content": self._react_user_prompt(query)}
        ]

        for i in range(1, self.max_steps + 1):
            # reason step
            reason_messages, action = await self.reason(i)
            self._messages.extend(reason_messages)

            if to_print:
                print(reason_messages[0]['content'], end='')

            # check if the action is a termination command
            action = action.lower()
            if action.startswith("finish[") and action.endswith("]"):
                done = True
                ans = action[len("finish["):-1].strip()
                break

            if i == self.max_steps - 1:
                break

            # act step
            tool_call_messages, obs, sources = await self.act(i)
            self._messages.extend(tool_call_messages)
            all_sources.extend(sources)

            # librarian exclusive code
            # if the returned sources are documents, not chunks -> assume they are not in index and must be added
            if len(sources) > 0 and type(sources[0]) is Document:
                # `asave_documents` will return documents successfully saved, and omit duplicates
                saved_docs = await self.docstore.asave_documents(sources)

                # add the saved documents to the index
                await self.index.async_add_documents(saved_docs)

                logger.debug("Added %d documents to index and docstore (out of %d total documents)", len(saved_docs), len(sources))
                if to_print:
                    print(f"Added {len(saved_docs)} documents to index and docstore (out of {len(sources)} total documents)")
            
            if to_print:
                tool_call_func = tool_call_messages[-2]['tool_calls'][0]['function']
                print(f"{tool_call_func['name']}({tool_call_func['arguments']})")
                print(tool_call_messages[-1]['content'], end='')
        
        
        if not done:
            logger.warning("Reached max steps without finishing the research task.")
            ans = ("Could not complete research task within the maximum number of steps. " +
                "Please try something else or discontinue this research.")
        
        return Response(text=ans, sources=all_sources)

        
    