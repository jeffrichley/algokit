"""Tests for Fibonacci algorithm."""

import pytest

from algokit.classic_dp import fibonacci


@pytest.mark.unit
def test_fibonacci_basic() -> None:
    """Test basic Fibonacci number calculations."""
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(10) == 55


@pytest.mark.unit
def test_fibonacci_negative() -> None:
    """Test that fibonacci raises ValueError for negative inputs."""
    with pytest.raises(ValueError):
        fibonacci(-1)
