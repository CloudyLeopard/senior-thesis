from pydantic import computed_field
from typing import Any, Dict, List, Tuple, override

import logging

from kruppe.algorithm.agents import ReActResearcher
from kruppe.common.log import log_io
from kruppe.models import Document, Chunk, Response
from kruppe.prompts.librarian import (
    REACT_RESEARCH_SYSTEM,
    REACT_RESEARCH_USER,
    REACT_RESEARCH_END_USER
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

    def _react_system_prompt(self) -> str:
        return REACT_RESEARCH_SYSTEM
    
    def _react_user_prompt(self, query: str) -> str:
        return REACT_RESEARCH_USER.format(query=query)
    
    @override
    async def _parse_reason(
        self, messages: List[Dict[str, str,]], reason_response: str, step: int
    ) -> Tuple[List[Dict], Dict[str, Any], bool]:
        reason_messages, results, done = await super()._parse_reason(messages, reason_response, step)

        if done:
            final_report_response = await self.llm.async_generate(
                messages=messages[1:]  # remove the system message
                + [{"role": "user", "content": REACT_RESEARCH_END_USER}],
            )
            final_report = final_report_response.text.strip()

            if final_report.lower().startswith("thought"):
                # remove the thoughts from final report
                # NOTE: i am assuming its always just one (and the first) paragraph
               # could be dangerous
               final_report = final_report.split("\n\n", 1)[-1].strip()
            
            results["answer"] = final_report
        
        return reason_messages, results, done

    @override
    async def _post_act(self, act_results: Dict[str, Any]) -> Dict[str, Any]:
        # if the returned sources are documents, not chunks
        # -> assume they are not in index and must be added
        # also, not indexing FinancialDocuments
        sources = act_results['sources']
        if len(sources) > 0 and type(sources[0]) is Document:
            # `asave_documents` will return documents successfully saved, and omit duplicates
            saved_docs = await self.docstore.asave_documents(sources)

            # add the saved documents to the index
            await self.index.async_add_documents(saved_docs)

            logger.debug("Added %d documents to index and docstore (out of %d total documents)", len(saved_docs), len(sources))
            
            if self.verbose:
                print(f"Added {len(saved_docs)} documents to index and docstore (out of {len(sources)} total documents)")
        
        return act_results

