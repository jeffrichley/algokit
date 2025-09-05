---
algorithm_key: "matrix-chain-multiplication"
tags: [dynamic-programming, algorithms, optimization, matrix-chain-multiplication, parenthesization, linear-algebra]
title: "Matrix Chain Multiplication"
family: "dynamic-programming"
---

{{ algorithm_card(config.extra.algorithm_key) }}

!!! abstract "Overview"
    The Matrix Chain Multiplication problem finds the most efficient way to multiply a sequence of matrices by determining the optimal parenthesization. While matrix multiplication is associative (meaning the result is the same regardless of grouping), the order of operations dramatically affects the total number of scalar multiplications required.

    This classic dynamic programming problem demonstrates how the choice of computation order can lead to exponential differences in performance. It's essential in scientific computing, graphics programming, and any application where large matrix operations are performed frequently. The algorithm provides an elegant solution that guarantees the optimal parenthesization while avoiding the exponential complexity of brute force approaches.

## Mathematical Formulation

!!! math "Problem Definition"
    Given:
    - A sequence of $n$ matrices: $A_1, A_2, ..., A_n$
    - Matrix $A_i$ has dimensions $p_{i-1} \times p_i$

    Find the optimal parenthesization that minimizes the total number of scalar multiplications:

    $$\min \sum_{k=1}^{n-1} \text{scalar\_multiplications}(A_k, A_{k+1})$$

    The recurrence relation for the optimal solution:

    $$m[i,j] = \begin{cases}
    0 & \text{if } i = j \\
    \min_{i \leq k < j} \{m[i,k] + m[k+1,j] + p_{i-1} \times p_k \times p_j\} & \text{if } i < j
    \end{cases}$$

    Where $m[i,j]$ represents the minimum cost of multiplying matrices $A_i$ through $A_j$.

!!! success "Key Properties"
    - **Optimal Substructure**: The optimal parenthesization contains optimal parenthesizations of subproblems
    - **Overlapping Subproblems**: The same subproblems are solved multiple times
    - **State Transition**: Builds solutions by combining optimal solutions of smaller matrix chains

## Implementation Approaches

=== "Dynamic Programming (Recommended)"
    ```python
    def matrix_chain_multiplication(dimensions: list[int]) -> tuple[int, list[list[int]]]:
        """
        Find optimal parenthesization for matrix chain multiplication.

        Args:
            dimensions: List of dimensions where matrix i has size dimensions[i] × dimensions[i+1]

        Returns:
            Tuple of (minimum cost, parenthesization table)

        Example:
            >>> dimensions = [10, 30, 5, 60]
            >>> cost, parens = matrix_chain_multiplication(dimensions)
            >>> print(f"Minimum cost: {cost}")
            Minimum cost: 4500
        """
        n = len(dimensions) - 1  # Number of matrices

        # Initialize DP tables
        # m[i][j] = minimum cost of multiplying matrices i through j
        m = [[0] * n for _ in range(n)]
        # s[i][j] = optimal split point for matrices i through j
        s = [[0] * n for _ in range(n)]

        # Fill the DP table diagonally
        for length in range(2, n + 1):  # length of matrix chain
            for i in range(n - length + 1):
                j = i + length - 1
                m[i][j] = float('inf')

                # Try all possible split points
                for k in range(i, j):
                    cost = m[i][k] + m[k+1][j] + dimensions[i] * dimensions[k+1] * dimensions[j+1]
                    if cost < m[i][j]:
                        m[i][j] = cost
                        s[i][j] = k

        return m[0][n-1], s
    ```

=== "Space-Optimized DP (Alternative)"
    ```python
    def matrix_chain_optimized(dimensions: list[int]) -> int:
        """
        Space-optimized version that only returns the minimum cost.
        """
        n = len(dimensions) - 1

        # Use only one row of the DP table
        dp = [0] * n

        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                min_cost = float('inf')

                for k in range(i, j):
                    cost = dp[k] + dp[j] + dimensions[i] * dimensions[k+1] * dimensions[j+1]
                    min_cost = min(min_cost, cost)

                dp[j] = min_cost

        return dp[n-1]
    ```

=== "Recursive with Memoization"
    ```python
    def matrix_chain_memoized(dimensions: list[int], i: int, j: int,
                             memo: dict[tuple[int, int], int] = None) -> int:
        """
        Recursive solution with memoization for matrix chain multiplication.
        """
        if memo is None:
            memo = {}

        if (i, j) in memo:
            return memo[(i, j)]

        if i == j:
            return 0  # Single matrix, no multiplication needed

        min_cost = float('inf')
        for k in range(i, j):
            cost = (matrix_chain_memoized(dimensions, i, k, memo) +
                   matrix_chain_memoized(dimensions, k+1, j, memo) +
                   dimensions[i] * dimensions[k+1] * dimensions[j+1])
            min_cost = min(min_cost, cost)

        memo[(i, j)] = min_cost
        return min_cost
    ```

!!! tip "Complete Implementation"
    The full implementation with parenthesization reconstruction, comprehensive testing, and additional variants is available in the source code:

    - **Main Implementation**: [`src/algokit/dynamic_programming/matrix_chain.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/dynamic_programming/matrix_chain.py)
    - **Tests**: [`tests/unit/dynamic_programming/test_matrix_chain.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/dynamic_programming/test_matrix_chain.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    | **Dynamic Programming** | O(n³) | O(n²) | Optimal solution, full table |
    | **Space-Optimized DP** | O(n³) | O(n) | Same time, reduced space |
    | **Memoized Recursion** | O(n³) | O(n²) | Same complexity, recursive |

!!! warning "Performance Considerations"
    - **Cubic time complexity** makes it expensive for very long matrix chains
    - **Space optimization** is crucial for large-scale problems
    - **Memory usage** scales quadratically with the number of matrices
    - **Parenthesization tracking** requires additional O(n²) space

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Scientific Computing"
        - **Linear Algebra**: Optimizing matrix operations in numerical libraries
        - **Machine Learning**: Efficient neural network computations
        - **Physics Simulations**: Optimizing computational kernels
        - **Signal Processing**: Filter chain optimization

    !!! grid-item "Computer Graphics"
        - **3D Transformations**: Optimizing matrix multiplication chains
        - **Animation Systems**: Efficient skeletal animation calculations
        - **Rendering Pipelines**: Optimizing transformation matrices
        - **Game Development**: Efficient matrix operations for physics

    !!! grid-item "Real-World Scenarios"
        - **Compiler Optimization**: Code generation for matrix operations
        - **Database Systems**: Query optimization with matrix operations
        - **Financial Modeling**: Portfolio optimization calculations
        - **Network Analysis**: Graph algorithm optimizations

    !!! grid-item "Educational Value"
        - **Dynamic Programming**: Understanding optimal substructure
        - **Algorithm Design**: Learning to handle complex constraints
        - **State Transitions**: Building solutions incrementally
        - **Problem Decomposition**: Breaking complex problems into subproblems

!!! success "Educational Value"
    - **Dynamic Programming**: Perfect example of optimal substructure and overlapping subproblems
    - **Algorithm Design**: Shows how to handle complex optimization problems
    - **State Management**: Demonstrates building solutions from smaller subproblems
    - **Performance Analysis**: Illustrates the impact of computation order on efficiency

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Textbooks"
        1. **Cormen, T. H., et al.** (2009). *Introduction to Algorithms*. MIT Press. ISBN 978-0-262-03384-8.
        2. **Kleinberg, J., & Tardos, É.** (2006). *Algorithm Design*. Pearson. ISBN 978-0-321-29535-4.

    !!! grid-item "Dynamic Programming"
        3. **Bellman, R.** (1957). *Dynamic Programming*. Princeton University Press.
        4. **Dreyfus, S. E., & Law, A. M.** (1977). *The Art and Theory of Dynamic Programming*. Academic Press.

    !!! grid-item "Online Resources"
        5. [Matrix Chain Multiplication - GeeksforGeeks](https://www.geeksforgeeks.org/matrix-chain-multiplication-dp-8/)
        6. [Dynamic Programming - GeeksforGeeks](https://www.geeksforgeeks.org/dynamic-programming/)
        7. [Matrix Chain Multiplication - Wikipedia](https://en.wikipedia.org/wiki/Matrix_chain_multiplication)

    !!! grid-item "Implementation & Practice"
        8. [Python Official Documentation](https://docs.python.org/3/)
        9. [Dynamic Programming Patterns](https://leetcode.com/discuss/general-discussion/458695/dynamic-programming-patterns)
        10. [Algorithm Visualization](https://visualgo.net/en/dp)

!!! tip "Interactive Learning"
    Try implementing the matrix chain multiplication algorithm yourself! Start with simple examples like 3 matrices, then trace through the DP table to see how the solution is built. Implement the solution reconstruction to understand how you can determine the optimal parenthesization. Test with different matrix dimensions to see how the algorithm handles various size combinations. This will give you deep insight into dynamic programming's power in optimization problems.

## Navigation

{{ nav_grid(current_algorithm=config.extra.algorithm_key, current_family=config.extra.family, max_related=5) }}
