# tests/test_prompts.py
import pytest
from kruppe.models import Document
from kruppe.prompt_formatter import PROMPTS, SimplePromptFormatter, RAGPromptFormatter


class TestSimplePromptFormatter:
    def test_init(self):
        # Test with system prompt
        system_prompt = "Hello, I am a system prompt."
        formatter = SimplePromptFormatter(system_prompt=system_prompt)
        assert formatter.system_prompt == system_prompt

        # Test without system prompt
        formatter = SimplePromptFormatter()
        assert formatter.system_prompt == "You are a helpful assistant."

    def test_format_messages(self):
        user_prompt = "What is the meaning of life?"
        system_prompt = "Hello, I am a system prompt."
        formatter = SimplePromptFormatter(system_prompt=system_prompt)
        messages = formatter.format_messages(user_prompt)
        assert messages == [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]


class TestRAGPromptFormatter:
    @pytest.fixture
    def sample_documents(self):
        return [
            Document(text="Sample text 1", metadata={"author": "John Doe", "date": "2023-01-01"}),
            Document(text="Sample text 2", metadata={"author": "Jane Doe", "date": "2023-01-02"}),
            Document(text="Sample text 3", metadata={"author": "Bob Doe", "date": "2023-01-03"}),
        ]
    
    def test_init(self, documents):
        # Test with system prompt
        system_prompt = "Hello, I am a system prompt."
        formatter = RAGPromptFormatter(system_prompt=system_prompt, documents=documents)
        assert formatter.system_prompt == system_prompt
        assert formatter.documents == documents

        # Test without system prompt
        formatter = RAGPromptFormatter(documents=documents)
        assert formatter.system_prompt == PROMPTS["rag_system_standard"]
        assert formatter.documents == documents
    
    def test_format_messages(self, sample_documents):
        # Test without metadata
        formatter = RAGPromptFormatter(documents=sample_documents)
        user_prompt = "What is the meaning of life?"
        messages = formatter.format_messages(user_prompt, method="concatenate")
        
        for message in messages:
            if message["role"] == "user":
                assert user_prompt in message["content"]
                assert sample_documents[0].text in message["content"]
                assert sample_documents[1].text in message["content"]
                assert sample_documents[2].text in message["content"]
            else:
                assert message["role"] == "system"
                assert message["content"] == PROMPTS["rag_system_standard"]
        
        # Test with metadata
        formatter = RAGPromptFormatter(documents=sample_documents)
        user_prompt = "What is the meaning of life?"
        metadata_fields = ["author", "date"]
        messages = formatter.format_messages(user_prompt, method="concatenate", use_metadata=True, metadata_fields=metadata_fields)
        
        for message in messages:
            if message["role"] == "user":
                assert user_prompt in message["content"]
                assert sample_documents[0].text in message["content"]
                assert sample_documents[0].metadata["author"] in message["content"]
                assert sample_documents[0].metadata["date"] in message["content"]
                assert sample_documents[1].text in message["content"]
                assert sample_documents[1].metadata["author"] in message["content"]
                assert sample_documents[1].metadata["date"] in message["content"]
                assert sample_documents[2].text in message["content"]
                assert sample_documents[2].metadata["author"] in message["content"]
                assert sample_documents[2].metadata["date"] in message["content"]
            else:
                assert message["role"] == "system"
                assert message["content"] == PROMPTS["rag_system_standard"]