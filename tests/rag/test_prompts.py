# tests/test_prompts.py
import pytest
from rag.models import Document
from src.rag.prompts import Prompts  # assuming this is the class name

class TestPrompts:
    def test_format_bulleted_list(self):
        # Create a list of Document objects
        docs = [
            Document(text="This is a sample text"),
            Document(text="Another sample text", metadata={"author": "John Doe", "date": "2022-01-01"}),
        ]

        # Create a Prompts object
        prompts = Prompts()

        # Test without metadata
        result = prompts.format_bulleted_list(docs, use_metadata=False)
        expected = "- This is a sample text\n- Another sample text"
        assert result == expected

        # Test with metadata
        result = prompts.format_bulleted_list(docs, use_metadata=True, metadata_fields=["author", "date"])
        expected = "- author: John Doe, date: 2022-01-01: Another sample text\n- This is a sample text"
        assert result == expected

    def test_extract_metadata(self):
        # Create a Document object with metadata
        doc = Document(text="Sample text", metadata={"author": "Jane Doe", "date": "2022-01-02"})

        # Create a Prompts object
        prompts = Prompts()

        # Test with fields
        result = prompts._extract_metadata(doc, ["author", "date"])
        expected = "author: Jane Doe, date: 2022-01-02"
        assert result == expected

        # Test without fields
        result = prompts._extract_metadata(doc, [])
        expected = ""
        assert result == expected

    def test_summarize(self):
        # Create a Prompts object
        prompts = Prompts()

        # Test with long content
        content = "This is a very long content that needs to be summarized"
        result = prompts._summarize(content)
        expected = "This is a very long content that needs to be summarized..."
        assert result == expected

        # Test with short content
        content = "Short content"
        result = prompts._summarize(content)
        expected = "Short content"
        assert result == expected