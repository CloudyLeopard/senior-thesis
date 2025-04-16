import logging
import pytest

from kruppe.llm import (
    OpenAIEmbeddingModel,
    # NYUOpenAIEmbeddingModel,
    OpenAILLM,
    # NYUOpenAILLM,
)
from kruppe.models import Response

logger = logging.getLogger(__name__)

@pytest.fixture(params=[OpenAIEmbeddingModel, ])
def embedding_model(request):
    return request.param()

@pytest.fixture(params=[OpenAILLM,])
def llm(request):
    return request.param()


@pytest.mark.asyncio
async def test_async_openai_embedding(embedding_model, documents):
    texts = [doc.text for doc in documents]
    embeddings = await embedding_model.async_embed(texts)

    assert len(embeddings) == len(texts)
    assert len(embeddings[0]) == 1536

@pytest.mark.asyncio
async def test_llm(llm):
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    response = await llm.async_generate(message)

    assert isinstance(response, Response)
    assert response is not None
    assert len(response.text) > 0
    
@pytest.mark.asyncio
async def test_generate_with_tool(llm, caplog):
    # caplog.set_level(logging.DEBUG, logger='kruppe.llm')
    
    messages = [
        {"role": "system", "content": "You answer questions about countries by calling on tools. You MUST plan extensively before each function call, no matter how simple it is, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully. Repeat your thoughts out loud verbatim."},
        {"role": "user", "content": "What is the capital of France? "},
    ]

    def get_capital(country: str) -> str:
        capitals = {
            "France": "Paris",
            "Germany": "Berlin",
            "Spain": "Madrid",
            "China": "Beijing",
        }
        return capitals.get(country, "Unknown")

    def get_continent(country: str) -> str:
        continents = {
            "France": "Europe",
            "Germany": "Europe",
            "Spain": "Europe",
            "China": "Asia",
        }
        return continents.get(country, "Unknown")
    
    tools = [
        {
            "type": "function",
            "name": "get_capital",
            "description": "Get the capital of a country",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string", "description": "The name of the country"},
                },
                "required": ["country"],
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "get_continent",
            "description": "Get the continent of a country",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string", "description": "The name of the country"},
                },
                "required": ["country"],
                "additionalProperties": False
            },
            "strict": True
        }
    ]
    
    text, func_name, func_args = await llm.async_generate_with_tools(messages, tools=tools, tool_choice='auto')

    assert isinstance(func_args, dict)
    if not text:
        logger.warning("Generated text is empty, which may indicate that the model did not provide a thought process or reasoning before the function call.")
    assert func_name == "get_capital"
    assert func_args == {"country": "France"}
    try:
        capital = get_capital(**func_args)
        assert capital == "Paris"
    except TypeError:
        pytest.fail("Function arguments do not match the expected format")
    