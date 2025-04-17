import asyncio
from datetime import datetime
from dateutil.parser import parse, ParserError
import logging
from functools import wraps
from typing import Any


logger = logging.getLogger(__name__)

def not_ready(func):
    func._not_ready = True
    @wraps(func)
    def wrapper(*args, **kwargs):
        raise NotImplementedError(f"{func.__name__} is not implemented")
    return wrapper

def is_method_ready(obj, method_name: str):
    method = getattr(obj, method_name)
    return not getattr(method, "_not_ready", False)

def convert_to_datetime(obj: Any) -> datetime:
    """Convert various types to a datetime object."""
    
    if obj is None:
        return None

    if isinstance(obj, datetime):
        return obj
    elif isinstance(obj, str):
        try:
            return datetime.fromisoformat(obj)
        except ValueError:
            pass
        
        try:
            return parse(obj)
        except ParserError:
            pass
    elif isinstance(obj, (int, float)):
        return datetime.fromtimestamp(obj)
    
    raise ValueError(f"Cannot convert {obj} to datetime.")

async def combine_async_generators(async_gens):
    # Start by scheduling the first item from each generator.
    pending = {
        asyncio.create_task(gen.__anext__()): gen
        for gen in async_gens
    }
    
    while pending:
        # Wait until at least one task completes.
        done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            gen = pending.pop(task)
            try:
                result = task.result()
            except StopAsyncIteration:
                # This generator is exhausted.
                continue
            except Exception:
                # Propagate any other exceptions.
                raise
            # Yield the result that finished first.
            yield result
            # Schedule the next item from the same generator.
            pending[asyncio.create_task(gen.__anext__())] = gen

