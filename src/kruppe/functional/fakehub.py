from typing import Tuple, List, Dict

from pydantic import computed_field

from kruppe.functional.base_tool import BaseTool

class FakeHub(BaseTool):

    async def stupid_tool(self, input: str) -> Tuple[str, List]:
        """
        This is a stupid tool that just returns the input string.
        """
        return input, []
    
    async def dumb_tool(self, input: str) -> Tuple[str, List]:
        """
        This is a dumb tool that just returns the input string
        """
        return input, []

    @computed_field
    @property
    def stupid_tool_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": "stupid_tool",
                "description": "This is a stupid tool that just returns the input string.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "required": [
                        "input"
                    ],
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "The input string that will be returned"
                        }
                    },
                    "additionalProperties": False
                }
            }
        }
    
    @computed_field
    @property
    def dumb_tool_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": "dumb_tool",
                "description": "This is a dumb tool that just returns the input string.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "required": [
                        "input"
                    ],
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "The input string that will be returned"
                        }
                    },
                    "additionalProperties": False
                }
            }
        }