from pydantic import BaseModel
from abc import ABC
from typing import Dict

class BaseTool(BaseModel, ABC):
    """Base class for all tools in the system.
    This class provides a common interface for all tools, allowing them to be used interchangeably.
    It also provides a method to retrieve the schema for a specific method of the tool.
    It is expected that subclasses will implement the specific functionality of the tool.

    All functions that are intended as tools should return a tuple, 
    where the first element is a string convertable result,
    and the second element is a list of sources used (`Chunk` or `Document`).

    These functions must also implement a schema method that returns a dictionary
    of the function's schema for OpenAI. It should be named as `<method_name>_schema`.

    Args:
        BaseModel (_type_): _description_
        ABC (_type_): _description_
    """

    def get_schema(self, method_name: str) -> Dict:
        return getattr(self, f"{method_name}_schema")
    
    