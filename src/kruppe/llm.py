from abc import ABC, abstractmethod
from typing import List, Dict, Literal
import os
import asyncio
from openai import OpenAI, AsyncOpenAI
import logging
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict, computed_field
import httpx

from kruppe.models import Embeddable, Response

HTTPX_CONNECTION_LIMITS = httpx.Limits(max_keepalive_connections=50, max_connections=400)
HTTPX_TIMEOUT = httpx.Timeout(5.0, read=60.0) # high read timeout cuz nyu api is slow (i keep getting read timeout error)

def init_httpx_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
            limits=HTTPX_CONNECTION_LIMITS,
            timeout=HTTPX_TIMEOUT,
        )
        # event_hooks={"response": lambda r: r.release()},

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class BaseNYUModel(BaseModel):
    # TODO: deal with (read) timeout error
    api_key: str = Field(default_factory=lambda: os.getenv("NYU_API_KEY"))
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
    model_config = ConfigDict(arbitrary_types_allowed=True)
    keep_history: bool = False
    messages: List[Dict] = []
    _session_token_usage: int = PrivateAttr(default=0)
    _input_token_usage: int = PrivateAttr(default=0)
    _output_token_usage: int = PrivateAttr(default=0)

    # @abstractmethod
    # def generate(self) -> str:
    #     pass

    @abstractmethod
    async def async_generate(self, messages: List[Dict], max_tokens=2000) -> Response:
        raise NotImplementedError
    
    @abstractmethod
    def generate(self, messages: List[Dict], max_tokens=2000) -> Response:
        raise NotImplementedError

    async def batch_async_generate(self, messages_list: List[List[Dict]], max_tokens=2000) -> List[Response]:
        return await asyncio.gather(*(self.async_generate(messages=messages, max_tokens=max_tokens) for messages in messages_list))
    
    def price(self):
        if self.model == "gpt-4o":
            return (self._input_token_usage * 2.5 + self._output_token_usage * 10) / 1_000_000
        elif self.model == "gpt-4o-mini":
            return (self._input_token_usage * 0.15 + self._output_token_usage * 0.075) / 1_000_000
        else:
            return 0

class OpenAILLM(BaseLLM):
    model: Literal["gpt-4o", "gpt-4o-mini"] = "gpt-4o-mini"
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    sync_client: OpenAI = Field(default_factory=lambda: OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    async_client: AsyncOpenAI = Field(default_factory=lambda: AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    
    def generate(
        self, messages: List[Dict], max_tokens=2000
    ) -> Response:
        """returns openai response based on given messages"""

        # if we want to keep history, add messages to history
        if self.keep_history:
            self.messages.extend(messages)
            messages = self.messages

        completion = self.sync_client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens
        )

        if self.keep_history:
            self.messages.append(completion.choices[0].message)

        # TODO: add logger to track openai response, token usage here
        # https://platform.openai.com/docs/api-reference/introduction
        total_tokens = completion.usage.total_tokens
        self._session_token_usage += total_tokens
        self._input_token_usage += completion.usage.prompt_tokens
        self._output_token_usage += completion.usage.completion_tokens

        logger.info("Total tokens used: %d", total_tokens)
        logger.info("Completion: %s", completion.choices[0].message.content)

        return Response(text=completion.choices[0].message.content)

    async def async_generate(
        self, messages: List[Dict], max_tokens=2000
    ) -> Response:
        """returns openai response based on given messages"""

        if self.keep_history:
            self.messages.extend(messages)
            messages = self.messages

        # TODO: add try/except for openai api key errors
        completion = await self.async_client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens
        )

        if self.keep_history:
            self.messages.append(completion.choices[0].message)

        total_tokens = completion.usage.total_tokens
        self._session_token_usage += total_tokens
        self._input_token_usage += completion.usage.prompt_tokens
        self._output_token_usage += completion.usage.completion_tokens

        logger.info("Total tokens used: %d", total_tokens)
        logger.info("Completion: %s", completion.choices[0].message.content)

        return Response(text=completion.choices[0].message.content)

class NYUOpenAILLM(BaseLLM, BaseNYUModel):
    model: Literal["gpt-4o-mini"] = "gpt-4o-mini"
    endpoint_url: str = Field(default_factory=lambda: os.getenv("NYU_ENDPOINT_URL_CHAT"))

    async def async_generate(self, messages: List[Dict], max_tokens=2000, retries=3, backoff_factor=0.3) -> Response:
        if self.keep_history:
            self.messages.extend(messages)
            messages = self.messages
        
        body = {
            "messages": messages,
            "openai_parameters": {"max_tokens": max_tokens},
        }
        # init httpx client if not initialized in the model
        client = self._httpx_client or init_httpx_client()
        
        try:
            response = await client.post(
                self.endpoint_url,
                headers=self.headers,
                json=body,
            )
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            if self.keep_history:
                self.messages.append({"role": "assistant", "content": content})
            
            total_tokens = data["usage"]["total_tokens"]
            self._session_token_usage += total_tokens
            self._input_token_usage += data["usage"]["prompt_tokens"]
            self._output_token_usage += data["usage"]["completion_tokens"]

            logger.info("Total tokens used: %d", total_tokens)
            logger.info("Completion: %s", content)

            return Response(text=content)
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {response}")

            if e.response.status_code == 401:
                logger.warning(f"Double check your auth key: {self.api_key[:10]}...")

            raise e
        except httpx.ReadTimeout as e:
            if retries > 0:
                await asyncio.sleep(backoff_factor * 2 ** (3 - retries))
                return await self.async_generate(messages, max_tokens, retries - 1, backoff_factor)
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
    
    def generate(self, messages, max_tokens=2000):
        return asyncio.run(self.async_generate(messages, max_tokens))
            

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
    model: Literal["text-embedding-3-small"] = "text-embedding-3-small"
    api_key: str = Field(default_factory=lambda x: os.getenv("OPENAI_API_KEY"))
    sync_client: OpenAI = None
    async_client: AsyncOpenAI = None

    def model_post_init(self, __context):
        if self.sync_client is None:
            self.sync_client = OpenAI(api_key=self.api_key)
        if self.async_client is None:
            self.async_client = AsyncOpenAI(api_key=self.api_key)
        

    # TODO: work on "retry" when encountered error
    def embed(self, text: List[str] | List[Embeddable]) -> List[List[float]]:
        if isinstance(text[0], Embeddable):
            text = [x.text for x in text]

        embeddings = []
        for i in range(0, len(text), 100):
            result = self.sync_client.embeddings.create(input=text[i : i + 100], model=self.model)
            embeddings.extend([x.embedding for x in result.data])

        return embeddings
    
    async def async_embed(self, text: List[str] | List[Embeddable]) -> List[List[float]]:
        if isinstance(text[0], Embeddable):
            text = [x.text for x in text]
            
        # batch requests so that we don't have a list that is too large
        tasks = []
        for i in range(0, len(text), 100):
            tasks.append(self.async_client.embeddings.create(input=text[i : i + 100], model=self.model))
        results = await asyncio.gather(*tasks)
        return [x.embedding for result in results for x in result.data]

class NYUOpenAIEmbeddingModel(BaseEmbeddingModel, BaseNYUModel):
    model: Literal['api-embedding-openai-text-embed-3-small'] = 'api-embedding-openai-text-embed-3-small'
    endpoint_url: str = Field(default_factory=lambda: os.getenv("NYU_ENDPOINT_URL_EMBEDDING"))
    
    async def async_embed(self, text: List[str] | List[Embeddable]) -> List[List[float]]:
        if isinstance(text[0], Embeddable):
            text = [x.text for x in text]
            
        async def send_embedding_request(line: str, client: httpx.AsyncClient, retries=3, backoff_factor=0.3):
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
                    logger.warning(f"Double check your auth key: {self.api_key[:10]}...")

                raise e
            except httpx.ReadTimeout as e:
                if retries > 0:
                    await asyncio.sleep(backoff_factor * 2 ** (3 - retries))
                    return await send_embedding_request(line, client, retries - 1, backoff_factor * 2)
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
                results = await asyncio.gather(*[send_embedding_request(line, client) for line in text[i : i + 100]])
                embeddings.extend(results)

            return embeddings
        finally:
            if self._httpx_client is None:
                await client.aclose()

    def embed(self, text: List[str] | List[Embeddable]) -> List[List[float]]:
        return asyncio.run(self.async_embed(text))
            