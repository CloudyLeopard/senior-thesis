from typing import Dict, Any
import json

OPENAI_TEXT_EMBEDDING_SMALL_DIM = 1536

class Document:
    text = ""
    metadata = {}
    db_id = ""

    def __init__(self, text: str, metadata: Dict[Any, Any], db_id=""):
        self.text = text
        self.metadata = metadata
        self.db_id = db_id
    
    def __str__(self):
        return f"Text: {self.text}\nMetadata: {json.dumps(self.metadata, indent=2)}"

if __name__ == "__main__":
    document = Document("Hello", {"bye": "no"})
    print(document)