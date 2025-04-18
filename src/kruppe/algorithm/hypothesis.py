import re
from pydantic import BaseModel, Field, computed_field, PrivateAttr
from typing import List, Dict, Literal, Tuple, Any, Callable, override
import logging
from tqdm import tqdm
from enum import Enum
import asyncio
from functools import partial

from kruppe.algorithm.agents import Researcher, ReActResearcher, Lead
from kruppe.algorithm.librarian import Librarian
from kruppe.prompts.hypothesis import (
    HYPOTHESIS_RESEARCHER_SYSTEM,
    CREATE_INFO_REQUEST_USER,
    ANSWER_INFO_REQUEST_USER,
    COMPILE_REPORT_USER,
    EVALUATE_LEAD_USER,
    UPDATE_LEAD_USER,
    CREATE_HYPOTHESIS_SYSTEM,
    CREATE_HYPOTHESIS_USER,
    REACT_HYPOTHESIS_SYSTEM,
    REACT_HYPOTHESIS_USER
)
from kruppe.common.log import log_io
from kruppe.models import Response

logger = logging.getLogger(__name__)

class Status(Enum):
    UNDISCOVERED = 0
    DISCOVERED = 1
    FINISHED = 2

class Node(BaseModel):
    step: int
    messages: List[Dict[str, str]]
    act_queued: bool
    is_leaf: bool = False
    d_time: int = None
    f_time: int = None
    parent: "Node" = None
    children: List["Node"] = []
    act_results: Dict[str, Any] = {}
    reason_results: Dict[str, Any] = {}

    @computed_field
    @property
    def status(self) -> Status:
        if self.f_time is not None:
            return Status.FINISHED
        elif self.d_time is not None:
            return Status.DISCOVERED
        else:
            return Status.UNDISCOVERED
    
class HypothesisResearcher(ReActResearcher):
    """
    Does depth-first search to implement tree of thought with ReAct, and find the best
    hypothesis.
    """
    max_degree: int = 3 # maximum number of children per node. linear if degree = 1
    max_depth: int = 10 # maximum depth of the search tree
    root_nodes: List[Node] = []

    def _react_system_prompt(self):
        ...
    
    def _react_user_prompt(self):
        ...

    @override
    async def _parse_reason(self, messages: List[Dict[str, str]], reason_response: str, step: int) -> Tuple[List[Dict], Dict[str, Any]]:
        
        # NOTE: remember to add 'done' with max_depth (compare to step) as a safety measure, in case it keeps going forever
        ...
    
    async def generate_starting_node(self, query: str) -> Node:
        """Generate the initial claim node for the given query."""
        messages = [
            {"role": "system", "content": CREATE_HYPOTHESIS_SYSTEM},
            {"role": "user", "content": CREATE_HYPOTHESIS_USER.format(query=query)}
        ]
        initial_claim_response = await self.llm.async_generate(messages)
        initial_claim_text = initial_claim_response.text.strip()
        
        # extract the hypothesis from the response
        try:
            thought, final_response = initial_claim_text.split('# Final Output', 1)
            thought = thought.strip()
            final_response = final_response.strip()
        except ValueError:
            # TODO: make this a retry
            logger.error("LLM did not return a valid response for the initial claim. " \
                "Make sure the LLM is set up correctly and the prompt is correct.")
            raise ValueError("LLM did not return a valid response for the initial claim.")

        # parse thoughts
        if thought.startswith('# Thoughts'):
            thought = thought[len('# Thoughts'):].strip()
        
        # parse hypothesis, reasoning, and research direction
        hypothesis = ""
        reasoning = ""
        research_direction = ""

        heading_map = {
            "## Working Hypothesis": "hypothesis",
            "## Reasoning": "reasoning",
            "## Research Direction": "research_direction"
        }

        lines = final_response.splitlines()
        current_key = ""
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if line_stripped in heading_map:
                current_key = heading_map[line_stripped]
            elif current_key:
                if current_key == "hypothesis":
                    hypothesis += f"{line_stripped}\n"
                elif current_key == "reasoning":
                    reasoning += f"{line_stripped}\n"
                elif current_key == "research_direction":
                    research_direction += f"{line_stripped}\n"
            else:
                logger.warning(f"Unexpected line in initial claim thoughts: {line_stripped}")

        # NOTE: i call this "reason_results", but its not really
        # because it does not have an "action" field.
        reason_results = {
            "thought": thought,
            "hypothesis": hypothesis,
            "reasoning": reasoning,
            "research_direction": research_direction
        }
        
        # create the initial node
        messages = [
            {"role": "system", "content": REACT_HYPOTHESIS_SYSTEM},
            {"role": "user", "content": REACT_HYPOTHESIS_USER.format(query=query)}
        ]

        initial_node = Node(
            step=0,
            messages=messages,
            act_queued=False,  # initial node does not have an action queued
            reason_results=reason_results, # initial node's reason_result is special
        )

        return initial_node
        
    
    @override
    async def execute(self, query: str, to_print=True) -> Response:
        # initialize root nodes: each root node should have a starting claim
        with asyncio.TaskGroup() as tg:
            initial_node_tasks = [
                tg.create_task(self.generate_starting_node(query))
                for i in range(self.max_degree)
            ]
        
        initial_nodes = [task.result() for task in initial_node_tasks]
        
        # set the root nodes
        self.root_nodes = initial_nodes

        # conduct dfs search over each node
        # TODO: parallelize this
        for root_node in self.root_nodes:
            await self.dfs_visit(root_node)



    async def dfs_visit(self, start: Node):
        """Iterative depth-first search visit function.

        Args:
            s (Node): starting node
            messages (List[Dict[str, str]]): messages to send to the LLM

        Returns:
            _type_: _description_
        """

        stack = [start]
        time = 0
        while stack:
            node = stack[-1] # stack.peek()

            if node.status == Status.UNDISCOVERED:

                # TODO: ending conditions
                if node.is_leaf:
                    # case 1: FINISHED - we found our hypothesis, return
                    # case 2: DEADEND - we need to backtrack
                    ...
                
                # if node needs to call on a tool first before the reasoning
                # should always be true except for the initial (root) node
                if node.act_queued:
                    tool_call_messages, act_results = await self.act(messages=node.messages, step=node.step)
                    node.messages.extend(tool_call_messages)
                    node.act_results = act_results

                
                # reasoning (for children)
                child_nodes = []
                for i in range(self.max_degree):
                    # step = node.step+1 because its technically the child node's reason.
                    reason_messages, reason_results, done = await self.reason(messages=node.messages, step=node.step+1)

                    child_node = Node(
                        step=node.step+1,
                        messages=node.messages + reason_messages, # shallow copy
                        act_queued=True,
                        reason_results=reason_results,
                        parent=node,
                        is_leaf=done
                    )
                
                # TODO: rank child nodes' order. Change the below to a better ranking function
                # rank it in descending order first - i.e. best child at position 0
                child_reason_results = [child.messages[-1]["content"] for child in child_nodes]
                child_nodes = sorted(child_nodes, key=lambda x: x.reason_results["research_direction"]) 
                node.children = child_nodes # update the children of the current node

                # TODO: maybe merge similar children
                ...

                # stack is last-in-first-out, so need to reverse the order (i.e. push the best child last)
                # push the children to the stack
                for child in reversed(child_nodes):
                    stack.append(child)
                
                # update discovery time
                time += 1
                node.d_time = time

                # push children to stack
            elif node.status == Status.DISCOVERED:
                # node is discovered, but not finished yet; remove from stack
                time += 1
                node.f_time = time
                stack.pop()
            else:
                # node is already finished, pop it from the stack
                # shouldn't happen here cuz i don't have backward edges, but maybe in the future
                stack.pop()


        




class HypothesisResearcherOld(Researcher):
    librarian: Librarian
    system_message: str = HYPOTHESIS_RESEARCHER_SYSTEM
    research_question: str
    init_lead: Lead
    chat_iterations: int = 3 # TODO: currently hardcoded method to stop evaluation, need to change this
    chat_depth: int = 1 # NOTE: number of past chat history to use in next iteration
    num_info_requests: int = 3 # number of info requests to make per iteration
    verbatim_answer: bool = False # whether to return the raw documents or the processed response
    strict_answer: bool = True # TODO: change from boolean to magnitude so i have varying degree of "strictness". This determines if the system will continue even if no documents are found
    # properties users can access
    leads: List[Lead] = Field(default_factory = lambda data: [data["init_lead"]])
    reports: List[Response] = [] 
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
    def latest_lead(self) -> Lead | None:
        return self.leads[-1] if self.leads else None
    
    @computed_field
    @property
    def latest_hypothesis(self) -> str:
        if self._lead_status == 0:
            # lead is rejected, return the hypothesis of the last lead
            logger.warning("Lead is rejected. Returning the hypothesis of the last lead.")
            return self.latest_lead.hypothesis
        elif self._lead_status == 1 or self._lead_status == 2:
            # 1 = lead is accepted, return the hypothesis of the current lead
            # 2 = still investigating, return the hypothesis of the current lead
            return self.latest_lead.hypothesis
        else:
            logger.error("Lead status is invalid. Returning None.")
            return None
    
    @computed_field
    @property
    def _messages_history(self) -> List[Dict[str, str]]:
        msgs_group = self._messages_history_list[-self.chat_depth:]
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
        self.init_lead = lead
        self.leads.append(lead)
        # reset investigation status
        self._lead_status = 2
        self.chat_iterations = iterations
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
        
        with tqdm(total=self.chat_iterations, desc="Lead investigation iteration") as pbar:
            # repeat investigation until we finish the iterations
            continue_research = True

            while (continue_research):
                continue_research = await self.research()

                pbar.update(1)
        
        return self.latest_report

    async def research(self) -> bool:
        """
        Create info requests, and retrieve information from the librarian to address the reqeusts.
        Using the retrieved info_requests and info_retrieved, compile a report built on top of
        the research question, lead, and hypothesis. Finally, update the lead and decide
        whether to continue investigating or not.

        Returns:
            bool: True if there is more to investigate, False if there is nothing more to investigate
        """
        # if lead status is 0 or 1, send a warning
        if self._lead_status in [0, 1]:
            logger.warning("You are researching a lead whose status is not in investigating state.")

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

            response = await self.complete_info_request(info_request)
            info_retrieved.append(response)

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

        if not continue_lead:
            self._lead_status = 0
            self.chat_iterations = 0
            self._messages_history_list.append(messages) # since we are stopping here, store messages
            return False
        
        # TODO: add a llm thing to say "oh, we can stop now but we are good"

        # --- UPDATE LEAD ---
        latest_lead = await self._create_lead(messages=messages)
        self._messages_history_list.append(messages) # last call in this chain, so store the messages

        # regex failed to match llm response --> assume lead is rejected
        if latest_lead is None:
            logger.warning("LLM did not return a valid response for updating the lead. " \
                "Assuming the lead is rejected.")
            self._lead_status = 0
            self.chat_iterations = 0
            return False
        
        # update the newest lead
        self.leads.append(latest_lead)
        
        # update iteration count and check if we reached final iteration
        self.chat_iterations -= 1
        if self.chat_iterations == 0:
            # we finished all iterations! we are done!
            self._lead_status = 1
            return False
        else:
            # we want to keep exploring
            self._lead_status = 2
            return True

    
    @log_io
    async def create_info_requests(self) -> List[str]:
        user_message = CREATE_INFO_REQUEST_USER.format(
            query=self.research_question,
            lead=self.latest_lead.lead,
            hypothesis=self.latest_lead.hypothesis,
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
                lead=self.latest_lead.lead,
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
            lead=self.lateset_lead.lead,
            hypothesis=self.lateset_lead.hypothesis,
            observation=self.lateset_lead.observation,
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
            lead=self.latest_lead.lead,
            hypothesis=self.latest_lead.hypothesis,
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
    
    async def _create_lead(self, messages: List[Dict[str, str]]) -> Lead | None:
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
        """Send messages to the LLM with the history of messages."""
        messages = ([{"role": "system", "content": self.system_message}] # system message
                    + self._messages_history # past messages, for a chosen number of iterations
                    + curr_messages # current messages
                    )
        
        return await self.llm.async_generate(messages)
