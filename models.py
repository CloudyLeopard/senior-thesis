from typing import Dict, Any

OPENAI_TEXT_EMBEDDING_SMALL_DIM = 1536

class Document:
    text = ""
    metadata = {}

    def __init__(self, text: str, metadata: Dict[Any, Any], db_id=""):
        self.text = text
        self.metadata = metadata
        self.db_id = db_id
    
    def set_db_id(self, db_id: str):
        self.db_id = db_id
    
    def get_db_id(self):
        return self.db_id
    

if __name__ == "__main__":
    document = Document("Hello", {"bye": "no"})
    print(document.__dict__)