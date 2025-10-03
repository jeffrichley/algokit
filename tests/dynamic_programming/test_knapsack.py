"""Tests for 0/1 knapsack algorithm."""

import pytest

from algokit.algorithms.dynamic_programming.knapsack import (
    knapsack_01,
    knapsack_01_memoized,
    knapsack_01_value_only,
    knapsack_fractional_greedy,
)


@pytest.mark.unit
def test_knapsack_01_basic() -> None:
    """Test basic 0/1 knapsack problem."""
    # Arrange - Set up test inputs for knapsack problem
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    expected_value = 220  # Items 1 and 2: 100 + 120 = 220
    expected_items = [1, 2]  # Indices of selected items

    # Act - Solve knapsack problem
    max_value, selected_items = knapsack_01(weights, values, capacity)

    # Assert - Verify maximum value and selected items are correct
    assert max_value == expected_value
    assert sorted(selected_items) == expected_items


@pytest.mark.unit
def test_knapsack_01_zero_capacity() -> None:
    """Test 0/1 knapsack with zero capacity."""
    # Arrange - Set up zero capacity case
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 0

    # Act - Solve knapsack with zero capacity
    max_value, selected_items = knapsack_01(weights, values, capacity)

    # Assert - Verify zero capacity results in zero value and no items
    assert max_value == 0
    assert selected_items == []


@pytest.mark.unit
def test_knapsack_01_all_items_fit() -> None:
    """Test 0/1 knapsack when all items fit."""
    # Arrange - Set up case where all items fit
    weights = [5, 10, 15]
    values = [10, 20, 30]
    capacity = 50

    # Act - Solve knapsack with large capacity
    max_value, selected_items = knapsack_01(weights, values, capacity)

    # Assert - Verify all items are selected and total value is correct
    assert max_value == 60  # 10 + 20 + 30
    assert sorted(selected_items) == [0, 1, 2]


@pytest.mark.unit
def test_knapsack_01_no_items_fit() -> None:
    """Test 0/1 knapsack when no items fit."""
    # Arrange - Set up case where no items fit
    weights = [50, 60, 70]
    values = [100, 120, 140]
    capacity = 40

    # Act - Solve knapsack with small capacity
    max_value, selected_items = knapsack_01(weights, values, capacity)

    # Assert - Verify no items are selected and value is zero
    assert max_value == 0
    assert selected_items == []


@pytest.mark.unit
def test_knapsack_01_empty_weights() -> None:
    """Test that knapsack_01 raises ValueError for empty weights list."""
    # Arrange - Set up empty weights list
    weights = []
    values = [60, 100, 120]
    capacity = 50

    # Act & Assert - Verify that ValueError is raised for empty weights
    with pytest.raises(ValueError, match="Weights and values lists cannot be empty"):
        knapsack_01(weights, values, capacity)


@pytest.mark.unit
def test_knapsack_01_empty_values() -> None:
    """Test that knapsack_01 raises ValueError for empty values list."""
    # Arrange - Set up empty values list
    weights = [10, 20, 30]
    values = []
    capacity = 50

    # Act & Assert - Verify that ValueError is raised for empty values
    with pytest.raises(ValueError, match="Weights and values lists cannot be empty"):
        knapsack_01(weights, values, capacity)


@pytest.mark.unit
def test_knapsack_01_length_mismatch() -> None:
    """Test that knapsack_01 raises ValueError for mismatched list lengths."""
    # Arrange - Set up mismatched list lengths
    weights = [10, 20]
    values = [60, 100, 120]
    capacity = 50

    # Act & Assert - Verify that ValueError is raised for length mismatch
    with pytest.raises(
        ValueError, match="Weights and values lists must have the same length"
    ):
        knapsack_01(weights, values, capacity)


@pytest.mark.unit
def test_knapsack_01_negative_weights() -> None:
    """Test that knapsack_01 raises ValueError for negative weights."""
    # Arrange - Set up negative weights
    weights = [10, -20, 30]
    values = [60, 100, 120]
    capacity = 50

    # Act & Assert - Verify that ValueError is raised for negative weights
    with pytest.raises(ValueError, match="All weights must be positive"):
        knapsack_01(weights, values, capacity)


@pytest.mark.unit
def test_knapsack_01_negative_values() -> None:
    """Test that knapsack_01 raises ValueError for negative values."""
    # Arrange - Set up negative values
    weights = [10, 20, 30]
    values = [60, -100, 120]
    capacity = 50

    # Act & Assert - Verify that ValueError is raised for negative values
    with pytest.raises(ValueError, match="All values must be positive"):
        knapsack_01(weights, values, capacity)


@pytest.mark.unit
def test_knapsack_01_negative_capacity() -> None:
    """Test that knapsack_01 raises ValueError for negative capacity."""
    # Arrange - Set up negative capacity
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = -10

    # Act & Assert - Verify that ValueError is raised for negative capacity
    with pytest.raises(ValueError, match="Capacity must be non-negative"):
        knapsack_01(weights, values, capacity)


@pytest.mark.unit
def test_knapsack_01_value_only_basic() -> None:
    """Test knapsack value-only function for basic case."""
    # Arrange - Set up test inputs for value-only knapsack
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    expected_value = 220

    # Act - Calculate maximum value only
    result = knapsack_01_value_only(weights, values, capacity)

    # Assert - Verify maximum value is correct
    assert result == expected_value


@pytest.mark.unit
def test_knapsack_01_value_only_zero_capacity() -> None:
    """Test knapsack value-only function with zero capacity."""
    # Arrange - Set up zero capacity case
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 0

    # Act - Calculate maximum value with zero capacity
    result = knapsack_01_value_only(weights, values, capacity)

    # Assert - Verify zero capacity results in zero value
    assert result == 0


@pytest.mark.unit
def test_knapsack_01_value_only_empty_weights() -> None:
    """Test that knapsack_01_value_only raises ValueError for empty weights."""
    # Arrange - Set up empty weights list
    weights = []
    values = [60, 100, 120]
    capacity = 50

    # Act & Assert - Verify that ValueError is raised for empty weights
    with pytest.raises(ValueError, match="Weights and values lists cannot be empty"):
        knapsack_01_value_only(weights, values, capacity)


@pytest.mark.unit
def test_knapsack_01_value_only_empty_values() -> None:
    """Test that knapsack_01_value_only raises ValueError for empty values."""
    # Arrange - Set up empty values list
    weights = [10, 20, 30]
    values = []
    capacity = 50

    # Act & Assert - Verify that ValueError is raised for empty values
    with pytest.raises(ValueError, match="Weights and values lists cannot be empty"):
        knapsack_01_value_only(weights, values, capacity)


@pytest.mark.unit
def test_knapsack_01_value_only_length_mismatch() -> None:
    """Test that knapsack_01_value_only raises ValueError for mismatched lengths."""
    # Arrange - Set up weights and values with different lengths
    weights = [10, 20, 30]
    values = [60, 100]  # Different length
    capacity = 50

    # Act & Assert - Verify that ValueError is raised for length mismatch
    with pytest.raises(
        ValueError, match="Weights and values lists must have the same length"
    ):
        knapsack_01_value_only(weights, values, capacity)


@pytest.mark.unit
def test_knapsack_01_value_only_negative_weights() -> None:
    """Test that knapsack_01_value_only raises ValueError for negative weights."""
    # Arrange - Set up weights with negative values
    weights = [10, -20, 30]
    values = [60, 100, 120]
    capacity = 50

    # Act & Assert - Verify that ValueError is raised for negative weights
    with pytest.raises(ValueError, match="All weights must be positive"):
        knapsack_01_value_only(weights, values, capacity)


@pytest.mark.unit
def test_knapsack_01_value_only_negative_values() -> None:
    """Test that knapsack_01_value_only raises ValueError for negative values."""
    # Arrange - Set up values with negative numbers
    weights = [10, 20, 30]
    values = [60, -100, 120]
    capacity = 50

    # Act & Assert - Verify that ValueError is raised for negative values
    with pytest.raises(ValueError, match="All values must be positive"):
        knapsack_01_value_only(weights, values, capacity)


@pytest.mark.unit
def test_knapsack_01_value_only_negative_capacity() -> None:
    """Test that knapsack_01_value_only raises ValueError for negative capacity."""
    # Arrange - Set up negative capacity
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = -50

    # Act & Assert - Verify that ValueError is raised for negative capacity
    with pytest.raises(ValueError, match="Capacity must be non-negative"):
        knapsack_01_value_only(weights, values, capacity)


@pytest.mark.unit
def test_knapsack_01_memoized_basic() -> None:
    """Test memoized knapsack function for basic case."""
    # Arrange - Set up test inputs for memoized knapsack
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    expected_value = 220

    # Act - Calculate maximum value using memoization
    result = knapsack_01_memoized(weights, values, capacity)

    # Assert - Verify memoized result matches expected value
    assert result == expected_value


@pytest.mark.unit
def test_knapsack_01_memoized_zero_capacity() -> None:
    """Test memoized knapsack function with zero capacity."""
    # Arrange - Set up zero capacity case
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 0

    # Act - Calculate maximum value with zero capacity using memoization
    result = knapsack_01_memoized(weights, values, capacity)

    # Assert - Verify zero capacity results in zero value
    assert result == 0


@pytest.mark.unit
def test_knapsack_fractional_greedy_basic() -> None:
    """Test fractional knapsack using greedy approach."""
    # Arrange - Set up test inputs for fractional knapsack
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    # Expected: Take all of item 0 (10), all of item 1 (20), and 2/3 of item 2 (20)
    # Value: 60 + 100 + (2/3) * 120 = 60 + 100 + 80 = 240

    # Act - Calculate maximum value using fractional knapsack
    result = knapsack_fractional_greedy(weights, values, capacity)

    # Assert - Verify fractional knapsack result is correct
    assert result == 240.0


@pytest.mark.unit
def test_knapsack_fractional_greedy_zero_capacity() -> None:
    """Test fractional knapsack with zero capacity."""
    # Arrange - Set up zero capacity case
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 0

    # Act - Calculate maximum value with zero capacity using fractional knapsack
    result = knapsack_fractional_greedy(weights, values, capacity)

    # Assert - Verify zero capacity results in zero value
    assert result == 0.0


@pytest.mark.unit
def test_knapsack_fractional_greedy_empty_weights() -> None:
    """Test that fractional knapsack raises ValueError for empty weights list."""
    # Arrange - Set up empty weights list
    weights = []
    values = [60, 100, 120]
    capacity = 50

    # Act & Assert - Verify that ValueError is raised for empty weights
    with pytest.raises(ValueError, match="Weights and values lists cannot be empty"):
        knapsack_fractional_greedy(weights, values, capacity)
