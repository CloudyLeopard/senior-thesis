import re
from pydantic import computed_field, PrivateAttr
from typing import List, Dict, Literal, Tuple
import logging
from tqdm import tqdm

from kruppe.algorithm.agents import Researcher, Lead
from kruppe.algorithm.librarian import Librarian
from kruppe.prompts.hypothesis import (
    HYPOTHESIS_RESEARCHER_SYSTEM,
    CREATE_INFO_REQUEST_USER,
    ANSWER_INFO_REQUEST_USER,
    COMPILE_REPORT_USER,
    EVALUATE_LEAD_USER,
    UPDATE_LEAD_USER
)
from kruppe.utils import log_io
from kruppe.models import Response

logger = logging.getLogger(__name__)

class HypothesisResearcher(Researcher):
    librarian: Librarian
    system_message: str = HYPOTHESIS_RESEARCHER_SYSTEM
    research_question: str
    new_lead: Lead = None
    iterations: int = 3 # TODO: currently hardcoded method to stop evaluation, need to change this
    iterations_used: int = 1 # NOTE: number of past chat history to use in next iteration
    num_info_requests: int = 3 # number of info requests to make per iteration
    verbatim_answer: bool = False # whether to return the raw documents or the processed response
    strict_answer: bool = True # TODO: change from boolean to magnitude so i have varying degree of "strictness". This determines if the system will continue even if no documents are found
    # properties users can access
    past_leads: List[Lead] = []
    reports: List[Response] = [] # NOTE: not necessary same len as past_leads (due to `start_new_lead`)
    info_history: List[Tuple[str, Response]] = []
    # private attributes for internal use
    _lead_status: Literal[0, 1, 2] = PrivateAttr(default=2)
    _messages_history_list: List[List[Dict[str, str]]] = PrivateAttr(default=[]) # past messages

    @computed_field
    @property
    def lead_status(self) -> str:
        return ["Rejected", "Accepted", "Investigating"][self._lead_status]

    @computed_field
    @property
    def report_ready(self) -> bool:
        return self._lead_status == 1
    
    @computed_field
    @property
    def latest_report(self) -> Response | None:
        return self.reports[-1] if self.reports else None
    
    @computed_field
    @property
    def latest_hypothesis(self) -> str:
        if self._lead_status == 0:
            # lead is rejected, return the hypothesis of the last lead
            logger.warning("Lead is rejected. Returning the hypothesis of the last lead.")
            return self.past_leads[-1].hypothesis
        elif self._lead_status == 1:
            # lead is accepted, return the hypothesis of the current lead
            return self.new_lead.hypothesis
        elif self._lead_status == 2:
            # still investigating, return the hypothesis of the current lead
            return self.new_lead.hypothesis
        else:
            logger.error("Lead status is invalid. Returning None.")
            return None
    
    @computed_field
    @property
    def _messages_history(self) -> List[Dict[str, str]]:
        msgs_group = self._messages_history_list[-self.iterations_used:]
        if not msgs_group:
            return []
        # this is a list of list... need to flatten it
        return [msg for msgs in msgs_group for msg in msgs]
        
    def start_new_lead(self, lead: Lead, iterations: int = 3):
        """
        Start a new lead. Usually don't use this because it's probably better to just
        create a new HypothesisResearcher object.
        
        But, maybe for some reason you want to use a researcher that's already "researched",
        and built on top of previous generatedreports/leads, then use this.

        Args:
            lead (Lead): new lead
            iterations (int, optional): Number of iterations. Defaults to 3.
        """
        # reset leads
        self.past_leads.append(self.new_lead)
        self.new_lead = lead
        # reset investigation status
        self._lead_status = 2
        self.iterations = iterations
        # reset messages
        self._messages_history_list = []
    

    async def execute(self) -> Response:
        """
        Execute the Hypothesis Researcher algorithm. This will investigate the lead
        and compile a report based on the research question, lead, and hypothesis.
        """

        # TWO OPTIONS ON LIBRARIAN REQUEST (dependent on `verbatim` parameter):
        # 1. use the info answers as context, and llm answer info_request (<-- currently using this one)
        # 2. directly use the retrieved contexts from the librarian as contexts (RAG Mode really)
        verbatim_mode = False
        
        with tqdm(total=self.iterations, desc="Lead investigation iteration") as pbar:
            # repeat investigation until we finish the iterations
            continue_research = True

            while (continue_research):
                # --- MAKE INFO REQUEST AND RETRIEVE ---
                # make info requests and answer them using the library as a source
                info_requests = await self.create_info_requests()
                info_retrieved = []
                for i in range(len(info_requests)):
                    info_request = info_requests[i]

                    if i >= self.num_info_requests:
                        # adding a hard cap to how many info requests can be made
                        # to avoid excessive calls
                        # NOTE: also used in prompt (see `create_info_requests`)
                        break

                    response = await self.complete_info_request(info_request, verbatim=verbatim_mode)
                    info_retrieved.append(response)


                # --- COMPILE REPORT AND UPDATE LEAD ---
                continue_research = await self.compile_report_and_update_lead(
                    info_requests=info_requests, 
                    info_retrieved=info_retrieved
                )

                pbar.update(1)
        
        return self.latest_report
    
    @log_io
    async def create_info_requests(self) -> List[str]:
        user_message = CREATE_INFO_REQUEST_USER.format(
            query=self.research_question,
            lead=self.new_lead.lead,
            hypothesis=self.new_lead.hypothesis,
            n=self.num_info_requests # number of info requests
        )
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]

        llm_response = await self.llm.async_generate(messages)
        llm_string = llm_response.text

        info_requests = re.split(r'\n+', llm_string)

        return info_requests

    @log_io
    async def complete_info_request(self, info_request: str) -> Response:
        ret_docs = await self.librarian.execute(info_request)
        if self.strict_answer and not ret_docs:
            logger.warning(f"Strict mode is on; could not complete info request '{info_request}'")
            return Response(text="I do not know, because no documents found.", sources=[])
        
        contexts = "\n\n".join([doc.text for doc in ret_docs])

        if self.verbatim_answer:
            # if verbatim is True, return the contexts verbatim without any LLM processing
            return Response(text=contexts, sources=ret_docs)
        else:
            user_message = ANSWER_INFO_REQUEST_USER.format(
                lead=self.new_lead.lead,
                info_request=info_request,
                contexts=contexts
            )
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": user_message},
            ]

            llm_response = await self.llm.async_generate(messages)
            llm_response.sources = ret_docs

            return llm_response
    
    @log_io
    async def compile_report_and_update_lead(
            self,
            info_requests: List[str],
            info_retrieved: List[Response]
        ) -> bool:
        """
        Using the retrieved info_requests and info_retrieved, compile a report built on top of
        the research question, lead, and hypothesis.

        Args:
            info_requests (List[str]): list of information requests, from `create_info_requests`
            info_retrieved (List[Response]): list of responses to the information requests, from `complete_info_request`

        Returns:
            bool: True if there is more to investigate, False if there is nothing more to investigate
        """

        # if lead status is 0 or 1, we don't need to do anything
        if self._lead_status in [0, 1]:
            return False

        # Check if the number of info requests and info retrieved match
        # If not, truncate the longer list
        if len(info_requests) != len(info_retrieved):
            logger.warning("Number of info requests and info retrieved do not match. " \
                "Truncating the longer list.")
            min_len = min(len(info_requests), len(info_retrieved))
            info_requests = info_requests[:min_len]
            info_retrieved = info_retrieved[:min_len]

        # chat history - note that all the helper method will append to this list in the functions
        # also i am not adding a system message - see helper function __send_messages_with_history
        messages = []

        # --- COMPILE REPORT ---
        report_response = await self._compile_report(
            messages=messages,
            info_requests=info_requests,
            info_retrieved=info_retrieved
        )

        # add report to all compiled reports
        self.reports.append(report_response) 

        # --- EVALUATE LEAD ---
        continue_lead = await self._evaluate_lead(messages=messages)

        # append current lead to past leads
        self.past_leads.append(self.new_lead)
        self.new_lead = None # reset current lead

        if not continue_lead:
            self._lead_status = 0
            self.iterations = 0
            self._messages_history_list.append(messages) # since we are stopping here, store messages
            return False
        
        # TODO: add a llm thing to say "oh, we can stop now but we are good"

        # --- UPDATE LEAD ---
        new_lead = await self._update_lead(messages=messages)
        self._messages_history_list.append(messages) # last call in this chain, so store the messages

        # regex failed to match llm response --> assume lead is rejected
        if new_lead is None:
            logger.warning("LLM did not return a valid response for updating the lead. " \
                "Assuming the lead is rejected.")
            self._lead_status = 0
            self.iterations = 0
            return False
        
        # update the newest lead
        self.new_lead = new_lead
        
        # update iteration count and check if we reached final iteration
        self.iterations -= 1
        if self.iterations == 0:
            # we finished all iterations! we are done!
            self._lead_status = 1
            return False
        else:
            # we want to keep exploring
            self._lead_status = 2
            return True

    async def _compile_report(
        self,
        messages: List[Dict[str, str]],
        info_requests: List[str],
        info_retrieved: List[Response]
    ) -> Response:
        """Note: the file also modifies `messages` in place."""
        
        # Compile the info requests/retrieval into a single string
        new_info_responses = "\n\n".join(f"Q:{q}\nA:{r.text}" for q, r in zip(info_requests, info_retrieved))

        user_message = COMPILE_REPORT_USER.format(
            research_question=self.research_question,
            lead=self.new_lead.lead,
            hypothesis=self.new_lead.hypothesis,
            observation=self.new_lead.observation,
            info_responses=new_info_responses,
        )

        messages.append({"role": "user", "content": user_message})

        # llm compile report, as Response object
        report_response = await self.__send_messages_with_history(messages)

        # append llm response to chat history
        messages.append({"role": "assistant", "content": report_response.text})

        return report_response

    async def _evaluate_lead(self, messages: List[Dict[str, str]]) -> bool:
        """Evaluate the lead and return True if we should continue investigating, False otherwise."""
        user_message = EVALUATE_LEAD_USER.format(
            research_question=self.research_question,
            lead=self.new_lead.lead,
            hypothesis=self.new_lead.hypothesis,
        )

        # append user message to chat history
        messages.append({"role": "user", "content": user_message})  

        eval_response = await self.__send_messages_with_history(messages)

        # append llm response to chat history
        messages.append({"role": "assistant", "content": eval_response.text})

        # evaluate the lead using if statements
        eval_str_lower = eval_response.text.lower()
        if eval_str_lower.startswith("accept"):
            return True
        elif eval_str_lower.startswith("reject"):
            return False
        else:
            logger.warning("LLM did not return a valid response for evaluating the lead. " \
                "Assuming the lead is accepted.")
            return True
    
    async def _update_lead(self, messages: List[Dict[str, str]]) -> Lead | None:
        user_message = UPDATE_LEAD_USER
        messages.append({"role": "user", "content": user_message})

        update_response = await self.__send_messages_with_history(messages)
        messages.append({"role": "assistant", "content": update_response.text})

        # regex parse the response to extract Lead
        pattern = (
            r"\*{0,2}New Observation:\*{0,2}\s*(.*?)\s*[\r\n]+"
            r"\*{0,2}New Working Hypothesis:\*{0,2}\s*(.*?)\s*[\r\n]+"
            r"\*{0,2}New Research Lead:\*{0,2}\s*(.*)"
        )
        
        match = re.search(pattern, update_response.text, re.DOTALL)
        if match:
            # newest lead
            return Lead(
                observation=match.group(1).strip(),
                hypothesis=match.group(2).strip(),
                lead=match.group(3).strip(),
            )
        
        else:
            logger.warning("LLM did not return a valid response for updating the lead. " \
                "Assuming the lead is rejected.")
            return None
    
    async def __send_messages_with_history(self, curr_messages: List[Dict[str, str]]):
        messages = ([{"role": "system", "content": self.system_message}] # system message
                    + self._messages_history # past messages, for a chosen number of iterations
                    + curr_messages # current messages
                    )
        
        return await self.llm.async_generate(messages)
