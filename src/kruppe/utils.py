import functools
import logging
import asyncio
import inspect

# global flag to control logging
ENABLE_IO_LOGGING = True

# Create a dedicated logger for logging function inputs and outputs.
io_logger = logging.getLogger("io_logger")
io_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("/Users/danielliu/Workspace/fin-rag/logs/io.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
io_logger.addHandler(file_handler)
io_logger.propagate = False


# Maximum length for logging each argument, keyword, or output.
MAX_LOG_LENGTH = 150

def log_io(func):
    """
    Decorator that logs inputs and outputs for synchronous functions,
    async functions, and async generators. It applies a cap on each logged
    argument, keyword, and output.
    """
    if not ENABLE_IO_LOGGING:
        return func
    
    def cap_value(value, max_length=MAX_LOG_LENGTH):
        """
        Return a string representation of `value` capped at `max_length` characters.
        """
        try:
            s = repr(value)
        except Exception:
            s = str(value)
        return s if len(s) <= max_length else s[:max_length] + "..."

    def filter_args(args):
        """
        If the first argument is an instance (i.e. not a type), replace it with a short placeholder.
        """
        if args and not isinstance(args[0], type):
            # Replace the instance with a short placeholder.
            return (f"<{args[0].__class__.__name__} instance>",) + args[1:]
        return args

    def get_caller_name(func, args):
        """
        Returns a string representing the caller, including class name if available.
        """
        if args:
            first_arg = args[0]
            if isinstance(first_arg, type):
                # Likely a class method.
                return f"{first_arg.__name__}.{func.__name__}"
            else:
                # Likely an instance method.
                return f"{first_arg.__class__.__name__}.{func.__name__}"
        return func.__name__

    if inspect.isasyncgenfunction(func):
        @functools.wraps(func)
        async def async_gen_wrapper(*args, **kwargs):
            caller = get_caller_name(func, args)
            filtered_args = filter_args(args)
            capped_args = tuple(cap_value(arg) for arg in filtered_args)
            capped_kwargs = {k: cap_value(v) for k, v in kwargs.items()}
            io_logger.info(f"Calling async generator {caller} with args: {capped_args} and kwargs: {capped_kwargs}")
            try:
                async for item in func(*args, **kwargs):
                    io_logger.info(f"{caller} yielded {cap_value(item)}")
                    yield item
            except Exception as e:
                io_logger.exception(f"{caller} raised an exception: {cap_value(e)}")
                raise
            io_logger.info(f"{caller} finished")
        return async_gen_wrapper

    elif asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            caller = get_caller_name(func, args)
            filtered_args = filter_args(args)
            capped_args = tuple(cap_value(arg) for arg in filtered_args)
            capped_kwargs = {k: cap_value(v) for k, v in kwargs.items()}
            io_logger.info(f"Calling async function {caller} with args: {capped_args} and kwargs: {capped_kwargs}")
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                io_logger.exception(f"{caller} raised an exception: {cap_value(e)}")
                raise
            io_logger.info(f"{caller} returned {cap_value(result)}")
            return result
        return async_wrapper

    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            caller = get_caller_name(func, args)
            filtered_args = filter_args(args)
            capped_args = tuple(cap_value(arg) for arg in filtered_args)
            capped_kwargs = {k: cap_value(v) for k, v in kwargs.items()}
            io_logger.info(f"Calling {caller} with args: {capped_args} and kwargs: {capped_kwargs}")
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                io_logger.exception(f"{caller} raised an exception: {cap_value(e)}")
                raise
            io_logger.info(f"{caller} returned {cap_value(result)}")
            return result
        return sync_wrapper

# Example usage in a class:
class Calculator:
    @log_io
    def add(self, a, b):
        return a + b

    @classmethod
    @log_io
    def multiply(cls, a, b):
        return a * b

if __name__ == "__main__":
    calc = Calculator()
    print("add:", calc.add(3, 4))
    print("multiply:", Calculator.multiply(3, 4))