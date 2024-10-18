import os
from openai import OpenAI, AsyncOpenAI

class Generator:
    """Custom generator interface"""
    def generate(self, prompt: str):
        pass

class Generator:
    """Custom async generator interface"""
    async def async_generate(self, prompt: str):
        pass

class OpenAIGenerator:
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI(os.getenv("OPENAI_API_KEY"))
        self.model = model

class AsyncOpenAIGenerator:
    def __init__(self, model="gpt-4o-mini", max_tokens=2000):
        self.client = OpenAI(os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    async def async_generate(self, prompt: str, max_tokens=2000):
        
        await self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens
        )