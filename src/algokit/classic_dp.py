"""Classic dynamic programming algorithms."""

from __future__ import annotations


def fibonacci(n: int) -> int:
    """Compute the nth Fibonacci number using dynamic programming.

    Args:
        n: Index of the Fibonacci sequence (0-indexed).

    Returns:
        The nth Fibonacci number.

    Raises:
        ValueError: If ``n`` is negative.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


__all__ = ["fibonacci"]
