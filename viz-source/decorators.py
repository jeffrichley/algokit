"""Decorators for adding functionality to algorithms.

This module provides decorators that can add logging, timing, or other
functionality to pure algorithm implementations without polluting the
core algorithm code.
"""

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


def with_logging[T](func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to add logging to algorithm functions.

    This decorator adds logging functionality to track algorithm execution
    without modifying the core algorithm logic.

    Args:
        func: The algorithm function to wrap

    Returns:
        Wrapped function with logging
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        # Add logging here if needed
        # For now, just call the original function
        return func(*args, **kwargs)

    return wrapper


def with_timing[T](func: Callable[..., T]) -> Callable[..., tuple[T, float]]:
    """Decorator to add timing information to algorithm functions.

    This decorator measures execution time of algorithm functions
    without modifying the core algorithm logic.

    Args:
        func: The algorithm function to wrap

    Returns:
        Wrapped function that returns (result, execution_time) tuple
    """
    import time

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[T, float]:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time

    return wrapper
