from abc import ABC, abstractmethod
import time
from typing import Any, List, Dict, Literal, Tuple
import os
import asyncio
import openai
from openai import OpenAI, AsyncOpenAI
import logging
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict, computed_field
import httpx
import stamina

from kruppe.models import Embeddable, Response
from kruppe.common.log import log_io

HTTPX_CONNECTION_LIMITS = httpx.Limits(
    max_keepalive_connections=50, max_connections=400
)
HTTPX_TIMEOUT = httpx.Timeout(
    5.0, read=60.0
)  # high read timeout cuz nyu api is slow (i keep getting read timeout error)


def init_httpx_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        limits=HTTPX_CONNECTION_LIMITS,
        timeout=HTTPX_TIMEOUT,
    )
    # event_hooks={"response": lambda r: r.release()},


logger = logging.getLogger(__name__)


def retry_on_openai_error(exc: Exception) -> bool:
    """Retry on OpenAI API errors"""
    if isinstance (exc, openai.APITimeoutError):
        logger.warning("OpenAI API timeout error. Waiting for 3 seconds")
        time.sleep(3)
        return True
    elif isinstance(exc, openai.AuthenticationError):
        logger.error("OpenAI API authentication error. Please check your API key")
        return False        
    elif isinstance(exc, openai.BadRequestError):
        logger.warning("OpenAI API bad request error. Please check your request")
        return False
    elif isinstance(exc, openai.InternalServerError):
        logger.warning("OpenAI API internal server error. Waiting for 10 seconds")
        time.sleep(10)
        return True
    elif isinstance(exc, openai.RateLimitError):
        logger.warning("OpenAI API rate limit error. Waiting for 60 seconds")
        time.sleep(60)
        return True
    elif isinstance(exc, openai.APIConnectionError):
        logger.warning("OpenAI API connection error. Trying again")
        return False
    
    if isinstance(exc, httpx.TimeoutException):
        logger.warning("OpenAI HTTPX timeout error. Waiting for 10 seconds")
        time.sleep(10)
        return True
    elif isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code in [429, 500, 502, 503, 504]:
            return True
        return False
    logger.error(f"OpenAI API error: {exc}")
    return False


class BaseNYUModel(BaseModel):
    # TODO: deal with (read) timeout error
    api_key: str = Field(default_factory=lambda: os.getenv("NYU_API_KEY"), exclude=True)
    project_id: str = Field(default_factory=lambda: os.getenv("NYU_PROJECT_ID"))
    net_id: str = Field(default_factory=lambda: os.getenv("NYU_NET_ID"))
    _httpx_client: httpx.AsyncClient = PrivateAttr(default=None)

    @computed_field
    @property
    def headers(self) -> Dict[str, str]:
        return {
            "rit_access": f"{self.project_id}|{self.net_id}|{self.model}",
            "rit_timeout": "60",
            "Content-Type": "application/json",
            "AUTHORIZATION_KEY": self.api_key,
        }


class BaseLLM(ABC, BaseModel):
    """Custom generator interface"""
    model: str
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _session_token_usage: int = PrivateAttr(default=0)
    _input_token_usage: int = PrivateAttr(default=0)
    _output_token_usage: int = PrivateAttr(default=0)

    # @abstractmethod
    # def generate(self) -> str:
    #     pass

    @abstractmethod
    async def async_generate(self, messages: List[Dict], **kwargs) -> Response:
        raise NotImplementedError

    @abstractmethod
    def generate(self, messages: List[Dict], **kwargs) -> Response:
        raise NotImplementedError
    
    @abstractmethod
    async def async_generate_with_tools(
        self, messages: List[Dict], tools: List[Dict], tool_choice="auto", **kwargs
    ) -> Tuple[str, str, str, str]: 
        """_summary_

        Args:
            messages (List[Dict]): _description_
            tools (List[Dict]): _description_
            tool_choice (str, optional): _description_. Defaults to "auto".

        Returns:
            Tuple[str, str, str, str]: A 4-tuple containing
                - text
                - tool_id
                - tool_name
                - tool_args_str
        """
        raise NotImplementedError

    async def batch_async_generate(
        self, messages_list: List[List[Dict]], **kwargs
    ) -> List[Response]:
        return await asyncio.gather(
            *(
                self.async_generate(messages=messages, **kwargs)
                for messages in messages_list
            )
        )

    def price(self):
        if self.model == "gpt-4o":
            return (
                self._input_token_usage * 2.5
                  + self._output_token_usage * 10
            ) / 1_000_000
        elif self.model == "gpt-4o-mini":
            return (
                self._input_token_usage * 0.15
                 + self._output_token_usage * 0.6
            ) / 1_000_000
        elif self.model == "gpt-4.1":
            return (
                self._input_token_usage * 2
                + self._output_token_usage * 8
            ) / 1_000_000
        elif self.model == "gpt-4.1-mini":
            return (
                self._input_token_usage * 0.40
                + self._output_token_usage * 1.60
            ) / 1_000_000
        else:
            return 

class FakeLLM(BaseLLM):
    def generate(self, messages: List[Dict], **kwargs) -> Response:
        """returns openai response based on given messages"""
        content = "This is a fake response."

        if logger.isEnabledFor(logging.DEBUG):
            # only print out last input message if debug is on
            logger.debug("%s\n[%s] %s", 'completion_abc123', messages[-1]['role'], messages[-1]['content'])

        logger.info("%s\n[assistant] %s", 'completion_abc123', content)
        return Response(text=content)

    async def async_generate(self, messages: List[Dict], **kwargs) -> Response:
        return self.generate(messages, **kwargs)
    
    async def async_generate_with_tools(
        self, messages: List[Dict], tools: List[Dict], tool_choice="auto", **kwargs
    ) -> Tuple[str, str, str, str]: 
        
        import random
        import json
        
        chosen_tool = random.choice(tools)
        name = chosen_tool["function"]["name"]
        arguments = chosen_tool["function"]["parameters"]["properties"].keys()
        arguments = {arg: str(random.randint(1, 100)) for arg in arguments}

        content = None if tool_choice == "required" else "Fake response from tool call"
        
        if tool_choice == "none":
            tool_id = None
            tool_name = None
            tool_args_str = None
        else:
            tool_id = "call_123fake"
            tool_name = name
            tool_args_str = json.dumps(arguments)

        if logger.isEnabledFor(logging.DEBUG):
            # only print out last input message if debug is on
            logger.debug("%s\n[%s] %s", "completion_abc123", messages[-1]['role'], messages[-1]['content'])

        logger.info("\n[assistant] %s\n[tool %s] %s (%s)",
                    content, tool_id, tool_name, tool_args_str)

        return content, tool_id, tool_name, tool_args_str


class OpenAILLM(BaseLLM):
    model: Literal[
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4.5-preview",
        # "o1",
        # "o3",
        # "o4-mini"
    ] = "gpt-4.1-mini"
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"), exclude=True)
    params: Dict[str, Any] = Field(default={}, description="OpenAI parameters") # can be overridden by method call
    sync_client: OpenAI = Field(
        default_factory=lambda: OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    )
    async_client: AsyncOpenAI = Field(
        default_factory=lambda: AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    )

    @stamina.retry(on=retry_on_openai_error, attempts=10, wait_max=20)
    def generate(self, messages: List[Dict], **kwargs) -> Response:
        """returns openai response based on given messages"""

        # override params with kwargs
        params = self.params.copy()
        params.update(kwargs)

        completion = self.sync_client.chat.completions.create(
            model=self.model, messages=messages, **params
        )

        # https://platform.openai.com/docs/api-reference/introduction
        total_tokens = completion.usage.total_tokens
        self._session_token_usage += total_tokens
        self._input_token_usage += completion.usage.prompt_tokens
        self._output_token_usage += completion.usage.completion_tokens

        content = completion.choices[0].message.content

        # log token usages
        logger.info(
            "%s Total tokens used: %d (%d input tokens, %d output tokens)",
            completion.id,
            total_tokens,
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
        )

        # log llm output
        if logger.isEnabledFor(logging.DEBUG):
            # only print out last input message if debug is on
            logger.debug("%s\n[%s] %s", completion.id, messages[-1]['role'], messages[-1]['content'])

        logger.info("%s\n[assistant] %s", completion.id, content)

        return Response(text=content)

    @stamina.retry(on=retry_on_openai_error, attempts=10, wait_max=20)
    async def async_generate(
        self, messages: List[Dict], **kwargs
    ) -> Response:
        """returns openai response based on given messages"""

        # override params with kwargs
        params = self.params.copy()
        params.update(kwargs)

        completion = await self.async_client.chat.completions.create(
            model=self.model, messages=messages, **params
        )

        # https://platform.openai.com/docs/api-reference/introduction
        total_tokens = completion.usage.total_tokens
        self._session_token_usage += total_tokens
        self._input_token_usage += completion.usage.prompt_tokens
        self._output_token_usage += completion.usage.completion_tokens

        content = completion.choices[0].message.content

        # log token usages
        logger.info(
            "%s Total tokens used: %d (%d input tokens, %d output tokens)",
            completion.id,
            total_tokens,
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
        )

        # log llm output
        if logger.isEnabledFor(logging.DEBUG):
            # only print out last input message if debug is on
            logger.debug("%s\n[%s] %s", completion.id, messages[-1]['role'], messages[-1]['content'])

        logger.info("%s\n[assistant] %s", completion.id, content)

        return Response(text=content)

    @stamina.retry(on=retry_on_openai_error, attempts=10, wait_max=20)
    async def async_generate_with_tools(
        self, messages: List[Dict], tools: List[Dict], tool_choice="auto", **kwargs
    ) -> Tuple[str, str, str, str]: 
        
        if messages[0].get("role") != "system":
            logger.warning("Make sure the first message is a system message.")
        
        for tool in tools:
            if 'function' not in tool:
                logger.warning("Tool should follows chat completion schema.")
                raise ValueError("Tool should follows chat completion schema.")

        # override params with kwargs
        params = self.params.copy()
        params.update(kwargs)

        completion = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **params
        )
        
        total_tokens = completion.usage.total_tokens
        self._session_token_usage += total_tokens
        self._input_token_usage += completion.usage.prompt_tokens
        self._output_token_usage += completion.usage.completion_tokens

        # NOTE: if `tool_choice` is set to 'required', i do not think gpt returns a text output.
        # NOTE: or, if instruction does not require GPT to think out loud, it may also not return a text output.

        content = completion.choices[0].message.content # so this could be null

        # log token usages
        logger.info(
            "%s Total tokens used: %d (%d input tokens, %d output tokens)",
            completion.id,
            total_tokens,
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
        )

        # log llm output
        if logger.isEnabledFor(logging.DEBUG):
            # only print out last input message if debug is on
            logger.debug("%s\n[%s] %s", completion.id, messages[-1]['role'], messages[-1]['content'])
        
        # get tools; if tool_choice = "none", the following var should all remain None
        tool_id = None
        tool_name = None
        tool_args_str = None
        
        # only use one tool at a time
        if completion.choices[0].message.tool_calls:
            tool_call = completion.choices[0].message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments
            tool_id = tool_call.id
        
        # log tool output
        logger.info("%s\n[assistant] %s\n[tool %s] %s (%s)",
                    completion.id, content, tool_id, tool_name, tool_args_str)
        
        return content, tool_id, tool_name, tool_args_str


class NYUOpenAILLM(BaseLLM, BaseNYUModel):
    model: Literal["gpt-4o-mini"] = "gpt-4o-mini"
    endpoint_url: str = Field(
        default_factory=lambda: os.getenv("NYU_ENDPOINT_URL_CHAT")
    )
    params: Dict[str, Any] = Field(default={}, description="OpenAI parameters") # can be overridden by method call


    # TODO: rewrite this with stamina retry
    @stamina.retry(on=retry_on_openai_error, attempts=10, wait_max=20)
    async def async_generate(
        self,
        messages: List[Dict],
        retries=3,
        backoff_factor=0.3,
        **kwargs,
    ) -> Response:

        body = {
            "messages": messages,
            "openai_parameters": kwargs,
        }

        # init httpx client if not initialized in the model
        client = self._httpx_client or init_httpx_client()

        try:
            # override params with kwargs
            params = self.params.copy()
            params.update(kwargs)

            # TODO: add params
            response = await client.post(
                self.endpoint_url,
                headers=self.headers,
                json=body,
            )
            response.raise_for_status()

            completion = response.json()
            content = completion["choices"][0]["message"]["content"]

            total_tokens = completion["usage"]["total_tokens"]
            self._session_token_usage += total_tokens
            self._input_token_usage += completion["usage"]["prompt_tokens"]
            self._output_token_usage += completion["usage"]["completion_tokens"]

            # log token usages                        
            logger.info(
                "%s Total tokens used: %d (%d input tokens, %d output tokens)",
                completion["id"],
                total_tokens,
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
            )

            # log llm output
            if logger.isEnabledFor(logging.DEBUG):
                # only print out last input message if debug is on
                logger.debug("%s\n[%s] %s", completion["id"], messages[-1]['role'], messages[-1]['content'])

            logger.info("%s\n[assistant] %s", completion["id"], content)
            
            return Response(text=content)
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {response}")

            if e.response.status_code == 401:
                logger.warning(f"Double check your auth key: {self.api_key[:10]}...")

            raise e
        except httpx.ReadTimeout as e:
            if retries > 0:
                await asyncio.sleep(backoff_factor * 2 ** (3 - retries))
                return await self.async_generate(
                    messages, retries - 1, backoff_factor
                )
            else:
                raise e
        except (httpx.ConnectTimeout, httpx.ConnectError) as e:
            logger.error("Connection error to NYU API: %s", e)
            # logger.error(f"Error: {e}")
            logger.warning("Did you connect to NYU's VPN?")
            raise e
        finally:
            if self._httpx_client is None:
                await client.aclose()

    @stamina.retry(on=retry_on_openai_error, attempts=10, wait_max=20)
    def generate(self, messages, **kwargs):
        return asyncio.run(self.async_generate(messages, **kwargs))
    
    @stamina.retry(on=retry_on_openai_error, attempts=10, wait_max=20)
    async def async_generate_with_tools(
        self, messages: List[Dict], tools: List[Dict], tool_choice="required", **kwargs
    ) -> Tuple[str, str, Dict[str, Any]]:
        raise NotImplementedError()


class BaseEmbeddingModel(ABC, BaseModel):
    "Custom embedding model interface"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # @abstractmethod
    # def embed(self, text: List[str]) -> List[List[float]]:
    #     """embeds a list of strings, and returns a list of embeddings"""
    #     pass

    @abstractmethod
    def async_embed(self, text: List[str]) -> List[List[float]]:
        """embeds a list of strings asynchronously, and returns a list of embeddings"""
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: List[str]) -> List[List[float]]:
        """embeds a list of strings, and returns a list of embeddings"""
        raise NotImplementedError


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    model: Literal["text-embedding-3-small", "text-embedding-3-large"] = (
        "text-embedding-3-small"
    )
    api_key: str = Field(default_factory=lambda x: os.getenv("OPENAI_API_KEY"), exclude=True)
    sync_client: OpenAI = None
    async_client: AsyncOpenAI = None

    def model_post_init(self, __context):
        if self.sync_client is None:
            self.sync_client = OpenAI(api_key=self.api_key)
        if self.async_client is None:
            self.async_client = AsyncOpenAI(api_key=self.api_key)

    # TODO: work on "retry" when encountered error
    @stamina.retry(on=retry_on_openai_error, attempts=10, wait_max=20)
    def embed(self, text: List[str] | List[Embeddable]) -> List[List[float]]:
        
        if len(text) == 0:
            return []

        if isinstance(text[0], Embeddable):
            text = [x.text for x in text]

        embeddings = []
        for i in range(0, len(text), 100):
            result = self.sync_client.embeddings.create(
                input=text[i : i + 100], model=self.model
            )
            embeddings.extend([x.embedding for x in result.data])

        return embeddings

    @stamina.retry(on=retry_on_openai_error, attempts=10, wait_max=20)
    async def async_embed(
        self, text: List[str] | List[Embeddable]
    ) -> List[List[float]]:
        
        if len(text) == 0:
            return []
        
        if isinstance(text[0], Embeddable):
            text = [x.text for x in text]

        # batch requests so that we don't have a list that is too large
        tasks = []
        for i in range(0, len(text), 100):
            tasks.append(
                self.async_client.embeddings.create(
                    input=text[i : i + 100], model=self.model
                )
            )
        results = await asyncio.gather(*tasks)
        return [x.embedding for result in results for x in result.data]


class NYUOpenAIEmbeddingModel(BaseEmbeddingModel, BaseNYUModel):
    model: Literal["api-embedding-openai-text-embed-3-small"] = (
        "api-embedding-openai-text-embed-3-small"
    )
    endpoint_url: str = Field(
        default_factory=lambda: os.getenv("NYU_ENDPOINT_URL_EMBEDDING")
    )

    @stamina.retry(on=retry_on_openai_error, attempts=10, wait_max=20)
    async def async_embed(
        self, text: List[str] | List[Embeddable]
    ) -> List[List[float]]:
        
        if len(text) == 0:
            return []

        if isinstance(text[0], Embeddable):
            text = [x.text for x in text]

        async def send_embedding_request(
            line: str, client: httpx.AsyncClient, retries=3, backoff_factor=0.3
        ):
            try:
                response = await client.post(
                    self.endpoint_url,
                    headers=self.headers,
                    json={"text": line},
                )
                response.raise_for_status()
                data = response.json()
                return data["embedding"]
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error: {e}")
                if e.response.status_code == 401:
                    logger.warning(
                        f"Double check your auth key: {self.api_key[:10]}..."
                    )

                raise e
            except httpx.ReadTimeout as e:
                if retries > 0:
                    await asyncio.sleep(backoff_factor * 2 ** (3 - retries))
                    return await send_embedding_request(
                        line, client, retries - 1, backoff_factor * 2
                    )
                else:
                    raise e
            except (httpx.ConnectTimeout, httpx.ConnectError) as e:
                logger.error("Connection error to NYU API: %s", e)
                # logger.error(f"Error: {e}")
                logger.warning("Did you connect to NYU's VPN?")
                raise e

        # init httpx client if not initialized in the model
        client = self._httpx_client or init_httpx_client()

        try:
            # batch requests so that we don't have a list that is too large
            embeddings = []
            for i in range(0, len(text), 100):
                results = await asyncio.gather(
                    *[
                        send_embedding_request(line, client)
                        for line in text[i : i + 100]
                    ]
                )
                embeddings.extend(results)

            return embeddings
        finally:
            if self._httpx_client is None:
                await client.aclose()

    @stamina.retry(on=retry_on_openai_error, attempts=10, wait_max=20)
    def embed(self, text: List[str] | List[Embeddable]) -> List[List[float]]:
        return asyncio.run(self.async_embed(text))
