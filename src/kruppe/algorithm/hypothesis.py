import re
from pydantic import BaseModel, Field, computed_field
from typing import List, Dict, Tuple, Any, final, override
import logging
from enum import Enum
import asyncio

from kruppe.algorithm.agents import ReActResearcher

from kruppe.functional.docstore.base_docstore import BaseDocumentStore
from kruppe.functional.rag.index.base_index import BaseIndex
from kruppe.prompts.hypothesis import (
    CREATE_HYPOTHESIS_SYSTEM,
    CREATE_HYPOTHESIS_USER,
    REACT_HYPOTHESIS_SYSTEM,
    REACT_HYPOTHESIS_USER,
    REACT_HYPOTHESIS_ACCEPT_END_USER,
    REACT_HYPOTHESIS_REJECT_END_USER,
    REACT_HYPOTHESIS_REJECT_MAX_STEPS_USER,
    RANK_REASONS_SYSTEM,
    RANK_REASONS_USER,
    REACT_HYPOTHESIS_REJECT_COMBINE_USER
)
from kruppe.models import Document, Response

logger = logging.getLogger(__name__)


class Status(Enum):
    UNDISCOVERED = 0
    DISCOVERED = 1
    FINISHED = 2


class Node(BaseModel):
    tree_id: int = None
    step: int
    messages: List[Dict[str, Any]] = Field(exclude=True)
    act_queued: bool
    is_leaf: bool = False
    d_time: int = None
    f_time: int = None
    parent: "Node" = Field(default=None, exclude=True)
    children: List["Node"] = Field(default=[], exclude=True)
    act_results: Dict = Field(default={}, exclude=True)
    reason_results: Dict = Field(default={}, exclude=True)

    @computed_field
    @property
    def status(self) -> Status:
        if self.f_time is not None:
            return Status.FINISHED
        elif self.d_time is not None:
            return Status.DISCOVERED
        else:
            return Status.UNDISCOVERED

    def __str__(self):
        return (
            f"Node(tree_id={self.tree_id}, step={self.step}, is_leaf={self.is_leaf},"
            f" d_time={self.d_time}, f_time={self.f_time})"
        )


class HypothesisResearcher(ReActResearcher):
    """
    Does depth-first search to implement tree of thought with ReAct, and find the best
    hypothesis.
    """

    role: str = "Financial Analyst"
    role_description: str = "You are a financial analyst who is has great insight into financial markets, standard business practices, and financial statement analysis."
    max_degree: int = 3  # maximum number of children per node. linear if degree = 1
    background_report: Response = None  # optional background report
    docstore: BaseDocumentStore = None # optional for if i want to save documents
    index: BaseIndex = None # optional for if i want to save documents
    root_nodes: List[Node] = []
    leaf_nodes: Dict[int, List[Node]] = {}
    research_reports: List[Response] = []

    def _react_system_prompt(self):
        return REACT_HYPOTHESIS_SYSTEM.format(
            role=self.role, role_description=self.role_description
        )

    def _react_user_prompt(self, query: str, hypothesis: str, direction: str) -> str:
        return REACT_HYPOTHESIS_USER.format(
            query=query, hypothesis=hypothesis, direction=direction
        )

    @override
    async def _parse_reason(
        self, messages: List[Dict[str, str]], reason_response: str, step: int
    ) -> Tuple[List[Dict], Dict[str, Any], bool]:
        pattern = re.compile(
            r"Thought.*:\s*(?P<thought>.*?)\s*"
            r"Working Hypothesis.*:\s*(?P<hypothesis>.*?)\s*"
            r"Research Direction.*:\s*(?P<research_direction>.*?)\s*"
            r"Action.*:\s*(?P<action>.*)$",
            re.S,
        )
        match = pattern.search(reason_response.strip())

        if match:
            results = {
                "thought": match.group("thought").strip(),
                "hypothesis": match.group("hypothesis").strip(),
                "research_direction": match.group("research_direction").strip(),
                "action": match.group("action").strip(),
            }
        else:
            reason_response_lines = [
                line for line in reason_response.strip().splitlines() if line
            ]

            # assuming each line is a different field, except for thought
            # checking one line at a time, from the bottom
            if reason_response_lines[-1].lower().startswith("action"):
                action = reason_response_lines.pop(-1)
                action = action.split(":", 1)[-1].strip()
            else:
                action = ""
            
            if reason_response_lines[-1].lower().startswith("research direction"):
                research_direction = reason_response_lines.pop(-1)
                research_direction = research_direction.split(":", 1)[-1].strip()
            else:
                research_direction = ""

            if reason_response_lines[-1].lower().startswith("hypothesis"):
                hypothesis = reason_response_lines.pop(-1)
                hypothesis = hypothesis.split(":", 1)[-1].strip()
            else:
                hypothesis = ""
            
            thought = "\n".join(reason_response_lines)
            thought = thought.split(":", 1)[-1].strip()

            results = {
                "thought": thought,
                "hypothesis": hypothesis,
                "research_direction": research_direction,
                "action": action,
            }

            logger.error(
                "LLM did not return a valid response for the reasoning step. " \
                f"Missing fields: {', '.join([k for k, v in results.items() if not v])}"
            )

            
            

        # evaluate action
        done = False

        action = results["action"]
        if action.lower().startswith("finish["):
            done = True

            # final report is in action
            action, answer = action.split("]", 1)
            action = f"{action}]"

            results["action"] = action

            if "accept" in action.lower():
                results["accept_hypothesis"] = True

                # generate final report - accept hypothesis
                final_report_response = await self.llm.async_generate(
                    messages=messages
                    + [{"role": "user", "content": REACT_HYPOTHESIS_ACCEPT_END_USER}]
                )
                final_report = final_report_response.text.strip()

                if final_report.lower().startswith("thought"):
                    # remove the thoughts from the final report
                    # NOTE: im assuming its always just one (and the first) paragraph...
                    # could be dangerous
                    final_report = final_report.split("\n\n", 1)[-1].strip()

                results["answer"] = final_report
            else:
                # llm SHOULD respond with a FINISH[reject], but anything else will also get rejected
                if "reject" not in action.lower():
                    logger.warning(
                        f"LLM's final action is not technically valid: FINISH[{action}]"
                    )

                results["accept_hypothesis"] = False  # False means reject hypothesis

                # generate final report - reject hypothesis
                final_report_response = await self.llm.async_generate(
                    messages=messages
                    + [{"role": "user", "content": REACT_HYPOTHESIS_REJECT_END_USER}]
                )
                final_report = final_report_response.text.strip()
                results["answer"] = final_report

        # max step as a safeguard
        if step > self.max_steps:
            done = True
            results["accept_hypothesis"] = False

            # general final report - reject hypothesis due to max depth reached
            final_report_response = await self.llm.async_generate(
                messages=messages
                + [{"role": "user", "content": REACT_HYPOTHESIS_REJECT_MAX_STEPS_USER}]
            )
            final_report = final_report_response.text.strip()
            results["answer"] = final_report

        reason_message = {
            "role": "assistant",
            "content": f"Thought {step}: {results['thought']}\n\n"
            f"Working Hypothesis {step}: {results['hypothesis']}\n\n"
            f"Research Direction {step}: {results['research_direction']}\n\n"
            f"Action {step}: {results['action']}",
        }

        return [reason_message], results, done

    @override
    async def _post_act(self, act_results: Dict[str, Any]) -> Dict[str, Any]:
        if not self.docstore and not self.index:
            # no need to save documents if no docstore or index
            return act_results

        if not self.docstore or not self.index:
            logger.warning("Only one of docstore or index is set. Not saving documents.")
            return act_results

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


    @override
    async def execute(self, query: str) -> List[Response]:
        # reset variables
        self.root_nodes = []
        self.leaf_nodes = {}
        self.research_reports = []

        # initialize root nodes: each root node should have a starting claim
        async with asyncio.TaskGroup() as tg:
            initial_node_tasks = [
                tg.create_task(self.init_starting_node(
                    query, background=self.background_report.text))
                for i in range(self.max_degree)
            ]

        initial_nodes = [task.result() for task in initial_node_tasks]

        # set the root nodes
        self.root_nodes = initial_nodes

        if self.verbose:
            print(f"Initialized {len(self.root_nodes)} root nodes.")

        # conduct dfs search over each node
        # TODO: parallelize this

        async with asyncio.TaskGroup() as tg:
            tasks = []
            for i in range(len(self.root_nodes)):
                # set the tree id for the root node
                self.root_nodes[i].tree_id = i
                
                # conduct dfs search
                task = tg.create_task(self.dfs_research(self.root_nodes[i], query))
                tasks.append(task)

        # collect the results
        for task in tasks:
            # collect the final report from each leaf node
            self.research_reports.append(task.result())


        return self.research_reports

    async def init_starting_node(self, query: str, background: str = "") -> Node:
        """Generate the initial claim node for the given query."""
        messages = [
            {
                "role": "system",
                "content": CREATE_HYPOTHESIS_SYSTEM.format(
                    role=self.role,
                    role_description=self.role_description,
                    background_report=background,
                ),
            },
            {"role": "user", "content": CREATE_HYPOTHESIS_USER.format(query=query)},
        ]
        initial_claim_response = await self.llm.async_generate(messages)
        initial_claim_text = initial_claim_response.text.strip()

        # extract the hypothesis from the response
        try:
            thought, final_response = initial_claim_text.split("# Final Output", 1)
            thought = thought.strip()
            final_response = final_response.strip()
        except ValueError:
            # TODO: make this a retry
            logger.error(
                "LLM did not return a valid response when initializing the claim. "
                "Make sure the LLM is set up correctly and the prompt is correct."
            )
            raise ValueError(
                "LLM did not return a valid response for the initial claim."
            )

        # parse thoughts
        if thought.startswith("# Thoughts"):
            thought = thought[len("# Thoughts") :].strip()

        # parse hypothesis, reasoning, and research direction
        hypothesis = ""
        reasoning = ""
        research_direction = ""

        heading_map = {
            "## Working Hypothesis": "hypothesis",
            "## Reasoning": "reasoning",
            "## Research Direction": "research_direction",
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
                logger.warning(
                    f"Unexpected line in initial claim thoughts: {line_stripped}"
                )

        # NOTE: i call this "reason_results", but its not really
        # because it does not have an "action" field.
        reason_results = {
            "thought": thought,
            "hypothesis": hypothesis,
            "reasoning": reasoning,
            "research_direction": research_direction,
        }

        # create the initial node
        messages = [
            {"role": "system", "content": self._react_system_prompt()},
            {
                "role": "user",
                "content": self._react_user_prompt(
                    query=query, hypothesis=hypothesis, direction=research_direction
                ),
            },
        ]

        initial_node = Node(
            step=0,
            messages=messages,
            act_queued=False,  # initial node does not have an action queued
            reason_results=reason_results,  # initial node's reason_result is special
        )

        return initial_node

    async def rank_child_nodes(self, children: List[Node], query: str) -> List[Node]:
        """Rank the children nodes based on their reason results. If they are similar, merge them"""

        if len(children) == 0:
            return []
        if len(children) == 1:
            return children

        # reason output should be the last message in `node.messages`
        reason_contents = [
            f'[{i + 1}] "{children[i].messages[-1]["content"]}"'
            for i in range(len(children))
        ]

        messages = [
            {"role": "system", "content": RANK_REASONS_SYSTEM},
            {
                "role": "user",
                "content": RANK_REASONS_USER.format(
                    query=query,
                    actions="\n".join(reason_contents),
                ),
            },
        ]

        rank_response = await self.llm.async_generate(messages)
        rank_str = rank_response.text.strip()

        # parse the ranking response
        try:
            thoughts, rank_str = rank_str.split("# Final Output", 1)
            thoughts = thoughts.strip()
            rank_str = rank_str.strip()
        except ValueError:
            # fall back to assume the ranking is just the response text
            logger.error(
                "LLM did not return a valid response for ranking the children. "
                "Make sure the LLM is set up correctly and the prompt is correct."
            )
            pass

        # parse the ranking
        ranked_indices_strs = re.findall(r"\d\.\s*\[(\d+)\]", rank_str)
        ranked_indices = [int(idx) - 1 for idx in ranked_indices_strs]

        seen_indices = set()
        ranked_children = [
            children[i]
            for i in ranked_indices
            if i < len(children) and i not in seen_indices and not seen_indices.add(i)
        ]

        assert len(ranked_children) <= len(children), (
            "Ranked children should not exceed the original children count."
        )

        return ranked_children

    async def dfs_research(self, start: Node, query: str) -> Response:
        """Iterative depth-first search visit function.

        Args:
            s (Node): starting node
            messages (List[Dict[str, str]]): messages to send to the LLM

        Returns:
            _type_: _description_
        """

        if self.verbose:
            print(f"Starting research on tree {start.tree_id}")

        stack = [start]
        time = 0
        while stack:
            node = stack[-1]  # stack.peek()

            if node.status == Status.UNDISCOVERED:
                # update discovery time
                time += 1
                node.d_time = time

                if self.verbose:
                    print(f"Discovering node: {node}")

                if node.is_leaf:
                    if node.tree_id in self.leaf_nodes:
                        self.leaf_nodes[node.tree_id].append(node)
                    else:
                        self.leaf_nodes[node.tree_id] = [node]
                    
                    # node is a leaf node, so we need to check if we can accept the hypothesis

                    # case 1: ACCEPT - we found our hypothesis, return
                    if node.reason_results["accept_hypothesis"]:
                        # collect all sources from the node and its parents
                        all_sources = []
                        curr = node
                        while curr is not None:
                            all_sources.extend(curr.act_results.get("sources", []))
                            curr = curr.parent

                        if self.verbose:
                            print(f"Finished research on tree {node.tree_id}")

                        return Response(
                            text=node.reason_results["answer"],
                            sources=all_sources,
                            metadata = {
                                "expert": self.role,
                                "expert_description": self.role_description,
                                "query": query
                            }
                        )
                    else:  # case 2: REJECT hypothesis or MAX_STEP reached - we need to backtrack
                        # done with this node, so just not do anything.

                        feedback = node.reason_results["answer"]
                        
                        # NOTE: this is not used here. its used at the end

                        continue

                # if node needs to call on a tool first before the reasoning
                # should always be true except for the initial (root) node
                if node.act_queued:
                    tool_call_messages, act_results = await self.act(
                        messages=node.messages, step=node.step
                    )
                    node.messages.extend(tool_call_messages)
                    node.act_results = act_results  # stores tool call results

                # reasoning (for children)
                child_nodes = []
                for i in range(self.max_degree):
                    # step = node.step+1 because its technically the child node's reason.
                    reason_messages, reason_results, done = await self.reason(
                        messages=node.messages, step=node.step + 1
                    )

                    child_node = Node(
                        tree_id=node.tree_id,
                        step=node.step + 1,
                        messages=node.messages + reason_messages,  # shallow copy
                        act_queued=True,
                        reason_results=reason_results,
                        parent=node,
                        is_leaf=done,
                    )

                    child_nodes.append(child_node)

                # rank nodes and remove duplicates
                ranked_child_nodes = await self.rank_child_nodes(child_nodes, query)
                node.children = ranked_child_nodes  # update children

                # stack is last-in-first-out, so need to reverse the order (i.e. push the best child last)
                # push the children to the stack
                for child in reversed(ranked_child_nodes):
                    stack.append(child)

            elif node.status == Status.DISCOVERED:
                # node is discovered, but not finished yet; remove from stack
                time += 1
                node.f_time = time
                stack.pop()

                if self.verbose:
                    print(f"Finishing node: {node}")
            else:
                # node is already finished, pop it from the stack
                # shouldn't happen here cuz i don't have backward edges, but maybe in the future
                stack.pop()

                if self.verbose:
                    print(f"Visiting a finished node: {node}")
        
        # dfs failed to find a hypothesis
        if self.verbose:
            print(f"Finished research on tree {start.tree_id} without accepting a hypothesis")
        
        # combine all the final reports from the leaf nodes
        final_reports = [node.reason_results["answer"] for node in self.leaf_nodes[start.tree_id]]

        messages = [
            {"role": "system", "content": REACT_HYPOTHESIS_SYSTEM},
            {
                "role": "user",
                "content": REACT_HYPOTHESIS_REJECT_COMBINE_USER.format(
                    query=query,
                    final_reports="\n----------\n".join(final_reports),
                ),
            },
        ]

        final_report = await self.llm.async_generate(messages)
        final_report = final_report.text.strip()
        final_report.metadata = {
            "expert": self.role,
            "expert_description": self.role_description,
            "query": query
        }
        
        return final_report


