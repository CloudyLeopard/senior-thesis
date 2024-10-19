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
