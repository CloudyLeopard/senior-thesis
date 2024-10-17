import os
from openai.types.chat import ChatCompletion
from openai import OpenAI, AsyncOpenAI
from typing import List, Dict
import sys
import logging

import pathlib
here = pathlib.Path(__file__).parent

def get_token_usage(completion):
    return 

def estimate_token_cost(tokens: int):
    return

def get_openai_client(async_on: bool = False):
    api_key = os.getenv("OPENAI_API_KEY")

    if async_on:
        client = AsyncOpenAI(api_key=api_key)
        pass
    else:
        client = OpenAI(api_key=api_key)
    return client


def embed(text_chunks: List[str]):
    client = get_openai_client()
    embeddings = client.embeddings.create(
        input=text_chunks, model="text-embedding-3-small"
    )

    return [x.embedding for x in embeddings.data]


async def async_embed(text_chunks: List[str]):
    client = get_openai_client(async_on=True)
    embeddings = await client.embeddings.create(
        input=text_chunks, model="text-embedding-3-small"
    )

    return [x.embedding for x in embeddings.data]


def send_llm_request(messages: List[Dict[str, str]], max_tokens: int = 500) -> str:

    # send request to llm
    client = get_openai_client()
    completion = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, max_tokens=max_tokens
    )
 
    return completion.choices[0].message.content

async def async_request_openai(
    messages: List[Dict[str, str]], max_tokens: int = 500
) -> ChatCompletion:
    # send request to llm
    client = get_openai_client(async_on=True)
    completion = await client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, max_tokens=max_tokens
    )

    return completion

async def async_request_openai_text(
    messages: List[Dict[str, str]], max_tokens: int = 500
) -> str:
    # parse openai message
    completion = await async_request_openai(messages=messages, max_tokens=max_tokens)

    return completion.choices[0].message.content


def get_logger():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
        level=logging.INFO,  # change to logging.DEBUG for detailed info (e.g. connection established)
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        # filename=here.joinpath("info.log")
    )
    logger = logging.getLogger(__name__)
    logging.getLogger("chardet.charsetprober").disabled = True

    return logger


if __name__ == "__main__":
    print("hello")
