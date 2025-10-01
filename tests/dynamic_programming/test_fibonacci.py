"""Tests for Fibonacci algorithm."""

import pytest

from algokit.algorithms.dynamic_programming.fibonacci import fibonacci


@pytest.mark.unit
def test_fibonacci_basic() -> None:
    """Test basic Fibonacci number calculations."""
    # Arrange - Set up test inputs for Fibonacci calculations
    test_cases = [0, 1, 10]
    expected_results = [0, 1, 55]

    # Act - Calculate Fibonacci numbers for test inputs
    results = [fibonacci(n) for n in test_cases]

    # Assert - Verify Fibonacci calculations are correct
    assert results == expected_results


@pytest.mark.unit
def test_fibonacci_negative() -> None:
    """Test that fibonacci raises ValueError for negative inputs."""
    # Arrange - Set up negative input that should raise ValueError
    negative_input = -1

    # Act - Attempt to calculate Fibonacci for negative input
    # Assert - Verify that ValueError is raised for negative input
    with pytest.raises(ValueError):
        fibonacci(negative_input)
