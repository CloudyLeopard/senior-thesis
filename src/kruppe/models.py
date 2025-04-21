from pydantic import BaseModel, Field, AfterValidator, computed_field
from typing import Dict, Any, Optional, List, Annotated
import json
from uuid import uuid4, UUID
from dateutil.parser import parse, ParserError
from datetime import datetime

OPENAI_TEXT_EMBEDDING_SMALL_DIM = 1536

def validate_metadata(v: Dict[str, Any]):

        # required_keys = [
        #     'query', # the query used to fetch the document from datasource
        #     'datasource', # document origin source (e.g. financial times, new york times)
        #     'url', # url of the document
        #     'title', # title of the document
        #     'description', # description of the document
        #     'publication_time', # publication time of the document
        # ]

        # if any(key not in v for key in required_keys):
        #     raise ValueError(f"metadata must contain keys: {required_keys}")
        
        # convert publication_time to unix
        dt = v.get('publication_time')
        try:
            if not dt:
                v['publication_time'] = None
            elif isinstance(dt, str):
                v['publication_time'] = int(parse(dt).timestamp())
            elif isinstance(dt, int) or isinstance(dt, float):
                v['publication_time'] = int(dt)
        except ParserError:
            v['publication_time'] = None

        # convert all None values to empty string
        for key in v:
            if v[key] is None:
                v[key] = ""
        
        return v

class Embeddable(BaseModel):
    text: str

class Document(Embeddable):
    id: UUID = Field(default_factory=uuid4)
    metadata: Annotated[Dict[str, Any], AfterValidator(validate_metadata)]    

    def __hash__(self) -> int:
        return hash(self.text)
    
    def __str__(self):
        doc_dict = self.model_dump(exclude={'text'})
        doc_dict['text'] = self.text[:50] + "..."

        # making it json dumpable
        for k in doc_dict:
            if isinstance(doc_dict[k], UUID):
                doc_dict[k] = str(doc_dict[k])
        return f"<{self.__class__.__name__} {json.dumps(doc_dict, indent=None)}>"

    def set_db_id(self, id: str):
        self.db_id = id

class FinancialDocument(Document):
    ...

class Chunk(Document):
    document_id: UUID
    metadata: Annotated[Dict[str, Any], AfterValidator(validate_metadata)]
    score: Optional[float] = None

    prev_chunk_id: Optional[UUID] = None
    next_chunk_id: Optional[UUID] = None

class Query(Embeddable):
    metadata: Dict[Any, Any] = Field(default_factory=dict)

class Response(Embeddable):
    sources: List[Document] = Field(default_factory=list)

    def __str__(self):
        return self.text

if __name__ == "__main__":
    document = Document(text="Hello", metadata={"bye": "no"})
    print(document)