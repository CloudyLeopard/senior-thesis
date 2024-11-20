from typing import Dict, Any
import json
from uuid import uuid4, UUID

OPENAI_TEXT_EMBEDDING_SMALL_DIM = 1536

class Document:

    # TODO: move away from db_id
    def __init__(self, text: str, metadata: Dict[Any, Any], uuid: UUID=None, db_id=""):
        self.uuid: UUID = uuid if uuid else uuid4() # generate new for new documents
        self.text: str = text
        self.metadata: Dict[Any, Any] = metadata
        self.db_id: str = db_id # init db_id as none

    def __hash__(self) -> int:
        return hash(self.text)
    def __str__(self):
        return f"Text: {self.text}\nMetadata: {json.dumps(self.metadata, indent=2)}"

    def set_db_id(self, id: str):
        self.db_id = id

if __name__ == "__main__":
    document = Document("Hello", {"bye": "no"})
    print(document)