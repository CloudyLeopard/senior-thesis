from pydantic import BaseModel, Field
from typing import Dict, Any
import json
from uuid import uuid4, UUID

OPENAI_TEXT_EMBEDDING_SMALL_DIM = 1536

class Embeddable(BaseModel):
    text: str

class Document(Embeddable):
    metadata: Dict[Any, Any]
    uuid: UUID = Field(default_factory=uuid4)
    db_id: str = Field(default="")

    # TODO: move away from db_id
    # TODO: figure out how to handle uuid (or if i should use it at all)
    # TODO: figure out if i should add other "Document" classes (e.g. Node)
    def __hash__(self) -> int:
        return hash(self.text)
    def __str__(self):
        return f"Text: {self.text}\nMetadata: {json.dumps(self.metadata, indent=2)}"

    def set_db_id(self, id: str):
        self.db_id = id

class Query(Embeddable):
    metadata: Dict[Any, Any] = {}

class Result(Embeddable):
    metadata: Dict[Any, Any] = {}

if __name__ == "__main__":
    document = Document("Hello", {"bye": "no"})
    print(document)