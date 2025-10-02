"""Matrix Chain Multiplication implementation using dynamic programming."""


def matrix_chain_multiplication(dimensions: list[int]) -> tuple[int, list[list[int]]]:
    """Find the minimum number of scalar multiplications needed for matrix chain multiplication.

    Given a sequence of matrices with their dimensions, find the optimal way to
    parenthesize the multiplication to minimize the total number of scalar multiplications.

    Args:
        dimensions: List of dimensions where dimensions[i-1] x dimensions[i]
                   represents the dimensions of matrix i (1-indexed).
                   For n matrices, this list should have n+1 elements.

    Returns:
        Tuple of (minimum_scalar_multiplications, optimal_parentheses_table).

    Raises:
        ValueError: If dimensions list has less than 2 elements or contains non-positive values.

    Examples:
        >>> dimensions = [1, 2, 3, 4]  # 3 matrices: 1x2, 2x3, 3x4
        >>> min_cost, parens = matrix_chain_multiplication(dimensions)
        >>> min_cost
        18
    """
    if len(dimensions) < 2:
        raise ValueError("Dimensions list must have at least 2 elements")
    if any(d <= 0 for d in dimensions):
        raise ValueError("All dimensions must be positive")

    n = len(dimensions) - 1  # Number of matrices

    if n == 1:
        return 0, [[0] * 2 for _ in range(2)]

    # Initialize DP tables
    # dp[i][j] = minimum scalar multiplications for matrices i through j (0-indexed)
    dp: list[list[int]] = [[0 for _ in range(n)] for _ in range(n)]

    # parens[i][j] = optimal split point for matrices i through j
    parens = [[0 for _ in range(n)] for _ in range(n)]

    # Fill DP table using bottom-up approach
    for length in range(2, n + 1):  # length of chain
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = 999999999  # Large number instead of inf

            for k in range(i, j):
                # Cost of multiplying matrices i to k and k+1 to j
                cost = (
                    dp[i][k]
                    + dp[k + 1][j]
                    + dimensions[i] * dimensions[k + 1] * dimensions[j + 1]
                )

                if cost < dp[i][j]:
                    dp[i][j] = cost
                    parens[i][j] = k

    return dp[0][n - 1], parens


def matrix_chain_multiplication_value_only(dimensions: list[int]) -> int:
    """Find minimum scalar multiplications without tracking parentheses.

    This is a space-optimized version that only returns the minimum cost
    without tracking the optimal parenthesization.

    Args:
        dimensions: List of dimensions where dimensions[i-1] x dimensions[i]
                   represents the dimensions of matrix i (1-indexed).

    Returns:
        Minimum number of scalar multiplications.

    Raises:
        ValueError: If dimensions list has less than 2 elements or contains non-positive values.
    """
    if len(dimensions) < 2:
        raise ValueError("Dimensions list must have at least 2 elements")
    if any(d <= 0 for d in dimensions):
        raise ValueError("All dimensions must be positive")

    n = len(dimensions) - 1  # Number of matrices

    if n == 1:
        return 0

    # Initialize DP table
    dp: list[list[int]] = [[0 for _ in range(n)] for _ in range(n)]

    # Fill DP table
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = 999999999  # Large number instead of inf

            for k in range(i, j):
                cost = (
                    dp[i][k]
                    + dp[k + 1][j]
                    + dimensions[i] * dimensions[k + 1] * dimensions[j + 1]
                )
                dp[i][j] = min(dp[i][j], cost)

    return dp[0][n - 1]


def matrix_chain_multiplication_memoized(dimensions: list[int]) -> int:
    """Find minimum scalar multiplications using memoized recursion.

    This implementation uses top-down dynamic programming with memoization
    as an alternative to the bottom-up approach.

    Args:
        dimensions: List of dimensions where dimensions[i-1] x dimensions[i]
                   represents the dimensions of matrix i (1-indexed).

    Returns:
        Minimum number of scalar multiplications.

    Raises:
        ValueError: If dimensions list has less than 2 elements or contains non-positive values.
    """
    if len(dimensions) < 2:
        raise ValueError("Dimensions list must have at least 2 elements")
    if any(d <= 0 for d in dimensions):
        raise ValueError("All dimensions must be positive")

    n = len(dimensions) - 1  # Number of matrices

    if n == 1:
        return 0

    # Memoization table
    memo: dict[tuple[int, int], int] = {}

    def matrix_chain_rec(i: int, j: int) -> int:
        if i == j:
            return 0

        if (i, j) in memo:
            return memo[(i, j)]

        result = 999999999
        for k in range(i, j):
            cost = (
                matrix_chain_rec(i, k)
                + matrix_chain_rec(k + 1, j)
                + dimensions[i] * dimensions[k + 1] * dimensions[j + 1]
            )
            result = min(result, cost)

        memo[(i, j)] = result
        return result

    return matrix_chain_rec(0, n - 1)


def print_optimal_parentheses(parens: list[list[int]], i: int, j: int) -> str:
    """Print the optimal parenthesization of matrix chain multiplication.

    Args:
        parens: Parentheses table from matrix_chain_multiplication.
        i: Start index of matrix chain.
        j: End index of matrix chain.

    Returns:
        String representation of optimal parenthesization.

    Examples:
        >>> dimensions = [1, 2, 3, 4]
        >>> _, parens = matrix_chain_multiplication(dimensions)
        >>> print_optimal_parentheses(parens, 0, 2)
        "((A1A2)A3)"
    """
    if i == j:
        return f"A{i + 1}"

    k = parens[i][j]
    left = print_optimal_parentheses(parens, i, k)
    right = print_optimal_parentheses(parens, k + 1, j)

    return f"({left}{right})"


def matrix_chain_multiplication_with_expression(
    dimensions: list[int],
) -> tuple[int, str]:
    """Find minimum cost and optimal parenthesization expression.

    Args:
        dimensions: List of dimensions where dimensions[i-1] x dimensions[i]
                   represents the dimensions of matrix i (1-indexed).

    Returns:
        Tuple of (minimum_cost, optimal_expression).

    Raises:
        ValueError: If dimensions list has less than 2 elements or contains non-positive values.

    Examples:
        >>> dimensions = [1, 2, 3, 4]
        >>> cost, expr = matrix_chain_multiplication_with_expression(dimensions)
        >>> cost
        18
        >>> expr
        "((A1A2)A3)"
    """
    if len(dimensions) < 2:
        raise ValueError("Dimensions list must have at least 2 elements")
    if any(d <= 0 for d in dimensions):
        raise ValueError("All dimensions must be positive")

    n = len(dimensions) - 1  # Number of matrices

    if n == 1:
        return 0, "A1"

    min_cost, parens = matrix_chain_multiplication(dimensions)
    expression = print_optimal_parentheses(parens, 0, n - 1)

    return min_cost, expression


def matrix_multiply_cost(A_rows: int, A_cols: int, B_rows: int, B_cols: int) -> int:
    """Calculate the cost of multiplying two matrices.

    Args:
        A_rows: Number of rows in matrix A.
        A_cols: Number of columns in matrix A.
        B_rows: Number of rows in matrix B.
        B_cols: Number of columns in matrix B.

    Returns:
        Number of scalar multiplications needed.

    Raises:
        ValueError: If matrices cannot be multiplied (A_cols != B_rows).
    """
    if A_cols != B_rows:
        raise ValueError("Number of columns in A must equal number of rows in B")

    return A_rows * A_cols * B_cols


def matrix_chain_multiplication_brute_force(dimensions: list[int]) -> int:
    """Find minimum cost using brute force approach (for comparison).

    This implementation tries all possible parenthesizations and is
    primarily used for testing and educational purposes.

    Args:
        dimensions: List of dimensions where dimensions[i-1] x dimensions[i]
                   represents the dimensions of matrix i (1-indexed).

    Returns:
        Minimum number of scalar multiplications.

    Raises:
        ValueError: If dimensions list has less than 2 elements or contains non-positive values.
    """
    if len(dimensions) < 2:
        raise ValueError("Dimensions list must have at least 2 elements")
    if any(d <= 0 for d in dimensions):
        raise ValueError("All dimensions must be positive")

    n = len(dimensions) - 1  # Number of matrices

    if n == 1:
        return 0

    def brute_force_rec(i: int, j: int) -> int:
        if i == j:
            return 0

        min_cost: int = 999999999
        for k in range(i, j):
            cost = (
                brute_force_rec(i, k)
                + brute_force_rec(k + 1, j)
                + dimensions[i] * dimensions[k + 1] * dimensions[j + 1]
            )
            min_cost = min(min_cost, cost)

        return min_cost

    return brute_force_rec(0, n - 1)
