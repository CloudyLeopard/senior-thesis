from typing import Dict, Any

OPENAI_TEXT_EMBEDDING_SMALL_DIM = 1536

class Document:
    text = ""
    metadata = {}

    def __init__(self, text: str, metadata: Dict[Any, Any]):
        self.text = text
        self.metadata = metadata
    