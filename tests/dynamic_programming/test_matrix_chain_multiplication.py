"""Tests for matrix chain multiplication algorithm."""

import pytest

from algokit.algorithms.dynamic_programming.matrix_chain_multiplication import (
    matrix_chain_multiplication,
    matrix_chain_multiplication_brute_force,
    matrix_chain_multiplication_memoized,
    matrix_chain_multiplication_value_only,
    matrix_chain_multiplication_with_expression,
    matrix_multiply_cost,
    print_optimal_parentheses,
)


@pytest.mark.unit
def test_matrix_chain_multiplication_basic() -> None:
    """Test basic matrix chain multiplication calculation."""
    # Arrange - Set up test inputs for matrix chain multiplication
    dimensions = [1, 2, 3, 4]  # 3 matrices: 1x2, 2x3, 3x4
    expected_min_cost = 18  # Optimal: ((A1*A2)*A3) = 6 + 12 = 18

    # Act - Calculate minimum scalar multiplications
    min_cost, parens = matrix_chain_multiplication(dimensions)

    # Assert - Verify minimum cost is correct
    assert min_cost == expected_min_cost


@pytest.mark.unit
def test_matrix_chain_multiplication_single_matrix() -> None:
    """Test matrix chain multiplication with single matrix."""
    # Arrange - Set up single matrix case
    dimensions = [5, 10]  # 1 matrix: 5x10
    expected_min_cost = 0  # No multiplication needed for single matrix

    # Act - Calculate minimum cost for single matrix
    min_cost, parens = matrix_chain_multiplication(dimensions)

    # Assert - Verify minimum cost is zero
    assert min_cost == expected_min_cost


@pytest.mark.unit
def test_matrix_chain_multiplication_two_matrices() -> None:
    """Test matrix chain multiplication with two matrices."""
    # Arrange - Set up two matrices case
    dimensions = [2, 3, 4]  # 2 matrices: 2x3, 3x4
    expected_min_cost = 24  # 2 * 3 * 4 = 24

    # Act - Calculate minimum cost for two matrices
    min_cost, parens = matrix_chain_multiplication(dimensions)

    # Assert - Verify minimum cost is correct
    assert min_cost == expected_min_cost


@pytest.mark.unit
def test_matrix_chain_multiplication_larger_example() -> None:
    """Test matrix chain multiplication with larger example."""
    # Arrange - Set up larger example
    dimensions = [40, 20, 30, 10, 30]  # 4 matrices
    expected_min_cost = 26000  # Known optimal solution

    # Act - Calculate minimum cost for larger example
    min_cost, parens = matrix_chain_multiplication(dimensions)

    # Assert - Verify minimum cost is correct
    assert min_cost == expected_min_cost


@pytest.mark.unit
def test_matrix_chain_multiplication_empty_dimensions() -> None:
    """Test that matrix_chain_multiplication raises ValueError for empty dimensions."""
    # Arrange - Set up empty dimensions list
    dimensions = []

    # Act & Assert - Verify that ValueError is raised for empty dimensions
    with pytest.raises(
        ValueError, match="Dimensions list must have at least 2 elements"
    ):
        matrix_chain_multiplication(dimensions)


@pytest.mark.unit
def test_matrix_chain_multiplication_single_element() -> None:
    """Test that matrix_chain_multiplication raises ValueError for single element."""
    # Arrange - Set up single element dimensions
    dimensions = [5]

    # Act & Assert - Verify that ValueError is raised for single element
    with pytest.raises(
        ValueError, match="Dimensions list must have at least 2 elements"
    ):
        matrix_chain_multiplication(dimensions)


@pytest.mark.unit
def test_matrix_chain_multiplication_negative_dimensions() -> None:
    """Test that matrix_chain_multiplication raises ValueError for negative dimensions."""
    # Arrange - Set up negative dimensions
    dimensions = [1, -2, 3, 4]

    # Act & Assert - Verify that ValueError is raised for negative dimensions
    with pytest.raises(ValueError, match="All dimensions must be positive"):
        matrix_chain_multiplication(dimensions)


@pytest.mark.unit
def test_matrix_chain_multiplication_zero_dimensions() -> None:
    """Test that matrix_chain_multiplication raises ValueError for zero dimensions."""
    # Arrange - Set up zero dimensions
    dimensions = [1, 0, 3, 4]

    # Act & Assert - Verify that ValueError is raised for zero dimensions
    with pytest.raises(ValueError, match="All dimensions must be positive"):
        matrix_chain_multiplication(dimensions)


@pytest.mark.unit
def test_matrix_chain_multiplication_value_only_basic() -> None:
    """Test value-only matrix chain multiplication calculation."""
    # Arrange - Set up test inputs for value-only calculation
    dimensions = [1, 2, 3, 4]
    expected_min_cost = 18

    # Act - Calculate minimum cost without parentheses tracking
    result = matrix_chain_multiplication_value_only(dimensions)

    # Assert - Verify minimum cost is correct
    assert result == expected_min_cost


@pytest.mark.unit
def test_matrix_chain_multiplication_value_only_single_matrix() -> None:
    """Test value-only calculation with single matrix."""
    # Arrange - Set up single matrix case
    dimensions = [5, 10]
    expected_min_cost = 0

    # Act - Calculate minimum cost for single matrix
    result = matrix_chain_multiplication_value_only(dimensions)

    # Assert - Verify minimum cost is zero
    assert result == expected_min_cost


@pytest.mark.unit
def test_matrix_chain_multiplication_memoized_basic() -> None:
    """Test memoized matrix chain multiplication calculation."""
    # Arrange - Set up test inputs for memoized calculation
    dimensions = [1, 2, 3, 4]
    expected_min_cost = 18

    # Act - Calculate minimum cost using memoization
    result = matrix_chain_multiplication_memoized(dimensions)

    # Assert - Verify memoized result matches expected
    assert result == expected_min_cost


@pytest.mark.unit
def test_matrix_chain_multiplication_memoized_single_matrix() -> None:
    """Test memoized calculation with single matrix."""
    # Arrange - Set up single matrix case
    dimensions = [5, 10]
    expected_min_cost = 0

    # Act - Calculate minimum cost for single matrix using memoization
    result = matrix_chain_multiplication_memoized(dimensions)

    # Assert - Verify memoized result for single matrix
    assert result == expected_min_cost


@pytest.mark.unit
def test_matrix_chain_multiplication_with_expression_basic() -> None:
    """Test matrix chain multiplication with expression generation."""
    # Arrange - Set up test inputs for expression generation
    dimensions = [1, 2, 3, 4]
    expected_min_cost = 18

    # Act - Calculate minimum cost with expression
    cost, expression = matrix_chain_multiplication_with_expression(dimensions)

    # Assert - Verify cost and expression are correct
    assert cost == expected_min_cost
    assert "(" in expression and ")" in expression  # Should have parentheses


@pytest.mark.unit
def test_matrix_chain_multiplication_with_expression_single_matrix() -> None:
    """Test expression generation with single matrix."""
    # Arrange - Set up single matrix case
    dimensions = [5, 10]
    expected_min_cost = 0
    expected_expression = "A1"

    # Act - Calculate minimum cost with expression for single matrix
    cost, expression = matrix_chain_multiplication_with_expression(dimensions)

    # Assert - Verify cost and expression are correct
    assert cost == expected_min_cost
    assert expression == expected_expression


@pytest.mark.unit
def test_print_optimal_parentheses_basic() -> None:
    """Test optimal parentheses printing."""
    # Arrange - Set up parentheses table and indices
    dimensions = [1, 2, 3, 4]
    _, parens = matrix_chain_multiplication(dimensions)

    # Act - Print optimal parentheses
    result = print_optimal_parentheses(parens, 0, 2)

    # Assert - Verify parentheses string contains matrix indices
    assert "A1" in result and "A2" in result and "A3" in result


@pytest.mark.unit
def test_print_optimal_parentheses_single_matrix() -> None:
    """Test optimal parentheses printing for single matrix."""
    # Arrange - Set up single matrix case
    parens = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    # Act - Print optimal parentheses for single matrix
    result = print_optimal_parentheses(parens, 0, 0)

    # Assert - Verify parentheses string is just the matrix index
    assert result == "A1"


@pytest.mark.unit
def test_matrix_multiply_cost_basic() -> None:
    """Test matrix multiplication cost calculation."""
    # Arrange - Set up test inputs for matrix multiplication cost
    A_rows, A_cols = 2, 3
    B_rows, B_cols = 3, 4
    expected_cost = 24  # 2 * 3 * 4 = 24

    # Act - Calculate matrix multiplication cost
    result = matrix_multiply_cost(A_rows, A_cols, B_rows, B_cols)

    # Assert - Verify multiplication cost is correct
    assert result == expected_cost


@pytest.mark.unit
def test_matrix_multiply_cost_incompatible_dimensions() -> None:
    """Test that matrix_multiply_cost raises ValueError for incompatible dimensions."""
    # Arrange - Set up incompatible matrix dimensions
    A_rows, A_cols = 2, 3
    B_rows, B_cols = 4, 5  # 3 != 4, so incompatible

    # Act & Assert - Verify that ValueError is raised for incompatible dimensions
    with pytest.raises(
        ValueError, match="Number of columns in A must equal number of rows in B"
    ):
        matrix_multiply_cost(A_rows, A_cols, B_rows, B_cols)


@pytest.mark.unit
def test_matrix_chain_multiplication_brute_force_basic() -> None:
    """Test brute force matrix chain multiplication calculation."""
    # Arrange - Set up test inputs for brute force calculation
    dimensions = [1, 2, 3, 4]
    expected_min_cost = 18

    # Act - Calculate minimum cost using brute force
    result = matrix_chain_multiplication_brute_force(dimensions)

    # Assert - Verify brute force result matches expected
    assert result == expected_min_cost


@pytest.mark.unit
def test_matrix_chain_multiplication_brute_force_single_matrix() -> None:
    """Test brute force calculation with single matrix."""
    # Arrange - Set up single matrix case
    dimensions = [5, 10]
    expected_min_cost = 0

    # Act - Calculate minimum cost for single matrix using brute force
    result = matrix_chain_multiplication_brute_force(dimensions)

    # Assert - Verify brute force result for single matrix
    assert result == expected_min_cost


@pytest.mark.unit
def test_matrix_chain_multiplication_brute_force_empty_dimensions() -> None:
    """Test that brute force raises ValueError for empty dimensions."""
    # Arrange - Set up empty dimensions list
    dimensions = []

    # Act & Assert - Verify that ValueError is raised for empty dimensions
    with pytest.raises(
        ValueError, match="Dimensions list must have at least 2 elements"
    ):
        matrix_chain_multiplication_brute_force(dimensions)
