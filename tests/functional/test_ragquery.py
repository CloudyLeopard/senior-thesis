import pytest
from uuid import uuid4
from datetime import datetime
from kruppe.functional.ragquery import RagQuery
from kruppe.models import Chunk
from kruppe.common.utils import convert_to_datetime
from kruppe.functional.rag.retriever.base_retriever import BaseRetriever

# Fake Retriever that returns dummy chunks.
class FakeRetriever(BaseRetriever):
    top_k: int = 3  # Default to return 3 dummy chunks.
    async def async_retrieve(self, query, filter=None):
        return [Chunk(text=f"Dummy text {i}", document_id=uuid4(), metadata={}) for i in range(self.top_k)]
    def retrieve(self, query, filter=None):
        return [Chunk(text=f"Dummy text {i}", document_id=uuid4(), metadata={}) for i in range(self.top_k)]


@pytest.fixture
def rag_query_tool(llm):
    tool = RagQuery(
        retriever = FakeRetriever(),
        llm = llm,
    )

    return tool

@pytest.mark.asyncio
async def test_rag_query_success(rag_query_tool):
    query = "Test query"
    start_time = "2023-01-01"
    end_time = "2023-12-31"
    
    answer, chunks = await rag_query_tool.rag_query(query, start_time, end_time)
    # The actual LLM is expected to return a response with the format:
    # "Thoughts: ... Answer: The answer text." If not, adjust the expected answer.
    assert answer and len(answer) > 0
    # Ensure we received the expected number of dummy chunks.
    assert len(chunks) == 3
    for i, chunk in enumerate(chunks):
        assert chunk.text == f"Dummy text {i}"
    
    dt_start = convert_to_datetime(start_time)
    dt_end = convert_to_datetime(end_time)
    assert isinstance(dt_start, datetime)
    assert isinstance(dt_end, datetime)



def test_rag_query_schema(rag_query_tool):
    schema = rag_query_tool.rag_query_schema()
    assert isinstance(schema, dict)
    assert "function" in schema
    func_schema = schema["function"]
    assert func_schema["name"] == "rag_query"
    # Check required parameters exist.
    params = func_schema["parameters"]
    for req in ["query", "start_time", "end_time"]:
        assert req in params["required"]