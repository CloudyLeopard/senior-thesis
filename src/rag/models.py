from pydantic import BaseModel, Field, computed_field
from typing import Dict, Any, Optional
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

class Chunk(Document):
    previous_chunk: Optional["Chunk"] = None
    next_chunk: Optional["Chunk"] = None

class ContextualizedChunk(Chunk):
    context: str

    @computed_field
    @property
    def contextual_text(self) -> str:
        return self.context + self.text


class Query(Embeddable):
    metadata: Dict[Any, Any] = Field(default_factory=dict)

class Result(Embeddable):
    metadata: Dict[Any, Any] = Field(default_factory=dict)

if __name__ == "__main__":
    document = Document(text="Hello", metadata={"bye": "no"})
    print(document)