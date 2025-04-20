from abc import ABC, abstractmethod
import json
from typing import Any, Dict, List, Tuple, Callable
from pydantic import BaseModel, PrivateAttr, computed_field, field_validator
import logging
import inspect

from kruppe.functional.rag.index.base_index import BaseIndex
from kruppe.llm import BaseLLM
from kruppe.models import Document, Response
from kruppe.functional.base_tool import BaseTool

logger = logging.getLogger(__name__)

class Researcher(BaseModel, ABC):
    llm: BaseLLM
    # system_message: str

    @abstractmethod
    async def execute(self):
        raise NotImplementedError

class ReActResearcher(Researcher):
    llm: BaseLLM
    toolkit: List[Callable]
    max_steps: int = 20
    _messages: List[Dict[str, str]] = PrivateAttr(default_factory=list)

    @field_validator('toolkit', mode='after')
    @classmethod
    def is_tool_method(cls, toolkit: List[Callable]) -> List[Callable]:
        """Validates that all functions in the toolkit are methods of a BaseTool class. """
        # Ensure that all functions are methods of BaseTool
        for tool in toolkit:
            if not hasattr(tool, '__self__') or not isinstance(tool.__self__, BaseTool):
                raise ValueError(f"Function {tool} is not a method of a BaseTool class.")
        
        return toolkit

    @abstractmethod
    def _react_system_prompt(self) -> str:
        ...
    
    @abstractmethod
    def _react_user_prompt(self, query: str) -> str:
        ...
    
    @computed_field
    @property
    def _tools(self) -> Dict[str, Dict[str, Any]]:
        """Returns a dictionary of tools available to the researcher.
        The keys are tool names, and the values are dictionaries with the function and its schema.

        Example: {"func_name": { "func": Callable, "schema": Dict[str, Any] } }
        """
        tools = {}
        for tool in self.toolkit:
            hub = tool.__self__
            assert isinstance(hub, BaseTool), f"Instance {hub} is not an instance of BaseTool."

            # get tool name and schema
            tool_schema = hub.get_schema(tool.__name__)

            if 'function' in tool_schema:
                # openai chat completion tools schema
                tool_name = tool_schema['function'].get('name')
            else:
                # openai response tools schema
                tool_name = tool_schema.get('name')
            
            if not tool_name:
                raise ValueError(f"Tool {tool.__name__} does not have a name in its schema.")
            
            tools[tool_name] = {
                'func': tool,
                'schema': tool_schema
            }

        return tools

    @computed_field
    @property
    def _tools_schemas(self) -> List[Dict[str, Any]]:
        return [tool['schema'] for tool in self._tools.values()]
            

    async def call_tool(self, tool_name: str, tool_args: Dict) -> Tuple[str, List[Document]]:
        """Calls a tool with the given name and arguments.
        This method checks if the tool exists in the available tools, and if so, calls it with the provided arguments.
        If the tool does not exist, it raises a ValueError.

        Args:
            tool_name (str): name of the tool to call
            tool_args (Dict): arguments to pass to the tool

        Raises:
            ValueError: if the tool does not exist in the available tools.

        Returns:
            Tuple[str, List[Document]]: string result of the tool call, and a list of documents (sources) returned by the tool.
        """
        if tool_name not in self._tools:
            raise ValueError(f"Tool {tool_name} not found in available tools.")

        logger.debug("Calling tool %s with arguments: %s", tool_name, tool_args)

        # call the tool function with the provided arguments
        tool_func = self._tools[tool_name]['func']

        if inspect.iscoroutinefunction(tool_func):
            # if the tool function is a coroutine, we await it
            result, docs = await tool_func(**tool_args)
        else:
            # if the tool function is a regular function, we call it directly
            # (this is for compatibility with synchronous tools)
            result, docs = tool_func(**tool_args)


        return str(result), docs


    async def _parse_reason (
            self,
            messages: List[Dict[str, str]],
            reason_response: str,
            step: int
        ) -> Tuple[List[Dict], Dict[str, Any], bool]:
        # if the thought starts with "Thought {step}:", we assume it's in the correct format
        if reason_response.startswith(f"Thought {step}:"):
            thought_action = reason_response[len(f"Thought {step}:"):].strip()

        try:
            thought, action = thought_action.split(f"\nAction {step}:")
            thought = thought.strip()
            action = action.strip()
        except ValueError:
            logger.warning("Failed to split thought and action: %s", thought_action)
            thought = thought_action.strip().split('\n')[0]

            action_response = await self.llm.async_generate(
                messages + [{"role": "assistant", "content": f"Thought {step}: {thought}\nAction {step}: "}],
                tools=self._tools_schemas,
                tool_choice="none", # does not use tool, but gives model the tools' schema
                stop=["\n"]
            )
            action = action_response.text.strip()
        
        reason_message = {"role": "assistant", "content": f"Thought {step}: {thought}\nAction {step}: {action}\n"}
        
        results = {
            "thought": thought,
            "action": action,
        }

        # evaluate action
        done = False

        # check if the action is a termination command
        if action.lower().startswith("finish["):
            done = True

            # final report is in action
            action, answer = action.split("]", 1)
            action = f"{action}]"
            answer = answer.strip()
            
            results['action'] = action
            results['answer'] = answer

        return [reason_message], results, done
    
    async def reason(
            self,
            messages: List[Dict[str, str]],
            step: int
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], bool]:
        """Generates the reasoning and action for the given step.
        This method uses the LLM to generate the reasoning and action based on the current messages.
        It initializes the messages for the first step, and uses the existing messages for subsequent steps.
        It expects the LLM to return a thought and action in the format:

        Thought {step}: [your thought process]\\
        Action {step}: [action to take]

        If it fails to extract the thought and action, it will generate a new action based on the thought.

        Args:
            step (int): the current step in the reasoning process, starting from 1.

        Returns:
            Tuple[List[Dict], str]: messages containing the reasoning and action, 
            and the action string.
        """

        reason_response = await self.llm.async_generate(
            messages,
            tools=self._tools_schemas,
            tool_choice="none", # does not use tool, but gives model the tools' schema
            stop=[f'\nObservation {step}'])
        reason_response = reason_response.text

        # parse "reason" response
        reason_messages, reason_results, done = await self._parse_reason(
            messages=messages,
            reason_response=reason_response,
            step=step
        )

        # assert to check parsed output is in right format
        # this is because _parse_reason may be overridden
        assert isinstance(reason_messages, list), "Reasoning messages should be a list."
        assert all(isinstance(msg, dict) for msg in reason_messages), "Each message should be a dictionary."
        assert isinstance(reason_results, dict), "Reasoning results should be a dictionary."
        assert isinstance(done, bool), "Done should be a boolean."

        return reason_messages, reason_results, done
        
    
    async def act(self, messages: List[Dict[str, str]], step: int) -> Tuple[List[Dict], Dict[str, Any]]:
        """Executes the action for a given step. This function should be called after the
        reasoning step (i.e. after `reason` method).
        It uses the LLM to generate the tool call, and then calls the tool with the generated arguments.

        Args:
            step (int): the current step in the reasoning process, starting from 1.

        Returns:
            Tuple[List[Dict], str, List[Document]]: messages containing the tool call and observation,
            the observation string, and a list of documents (sources) returned by the tool.
        """

        text, tool_id, tool_name, tool_args_str = await self.llm.async_generate_with_tools(
            messages,
            tools=self._tools_schemas,
            tool_choice="required" # text will be empty because of this
        )

        tool_args = json.loads(tool_args_str)

        obs, sources = await self.call_tool(tool_name, tool_args)

        # message that describes what tool was called
        tool_call_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": tool_args_str
                }
            }]
        }

        # message that describes the tool's observation (i.e. result)
        tool_obs_message = {
            "role": "tool",
            "tool_call_id": tool_id,
            "content": f"Observation {step}: {obs}\n\n"
        }

        act_results = {
            "tool_id": tool_id,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_args_str": tool_args_str,
            "obs": obs,
            "sources": sources
        }

        return [tool_call_message, tool_obs_message], act_results

    async def execute(self, query: str, to_print=True) -> Response:
        """Executes the research task using the ReAct framework."""
        all_sources = []
        ans = None
        done = False

        # initialize the messages with the system prompt
        self._messages = [
            {"role": "system", "content": self._react_system_prompt()},
            {"role": "user", "content": self._react_user_prompt(query)}
        ]

        for i in range(1, self.max_steps + 1):
            # reason step
            reason_messages, reason_results, done = await self.reason(self._messages, i)
            self._messages.extend(reason_messages)

            if to_print:
                print(reason_messages[0]['content'], end='')

            if done:
                ans = reason_results['answer']
                break
            
            if i == self.max_steps - 1:
                break

            # act step
            tool_call_messages, act_results = await self.act(self._messages, i)
            self._messages.extend(tool_call_messages)
            all_sources.extend(act_results['sources'])

            if to_print:
                print(f"{act_results['tool_name']}({act_results['tool_args_str']})")
                print(tool_call_messages[-1]['content'], end='')
        
        if not done:
            logger.warning("Reached max steps without finishing the research task.")
            ans = ("Could not complete research task within the maximum number of steps. " +
                "Please try something else or discontinue this research.")
        
        return Response(text=ans, sources=all_sources)



class Lead(BaseModel):
    observation: str # observation that led to the lead/hypothesis, though i don't really use this
    lead: str # working lead
    hypothesis: str # working hypothesis
    
