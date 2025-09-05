---
algorithm_key: "knapsack"
tags: [dynamic-programming, algorithms, optimization, knapsack, 0-1-knapsack, resource-allocation]
title: "0/1 Knapsack Problem"
family: "dynamic-programming"
---

{{ algorithm_card(config.extra.algorithm_key) }}

!!! abstract "Overview"
    The 0/1 Knapsack problem is a fundamental optimization challenge: given a set of items with weights and values, and a knapsack with a maximum weight capacity, determine which items to include to maximize the total value while staying within the weight limit. Each item can only be chosen once (hence "0/1"), making this a classic example of dynamic programming's power in solving constrained optimization problems.

    This problem appears in numerous real-world scenarios, from resource allocation and portfolio optimization to cutting stock problems and even game theory. It demonstrates how dynamic programming can handle complex constraints while finding globally optimal solutions.

## Mathematical Formulation

!!! math "Problem Definition"
    Given:
    - A set of $n$ items, each with weight $w_i$ and value $v_i$
    - A knapsack with maximum weight capacity $W$

    Maximize the total value:

    $$\max \sum_{i=1}^{n} v_i x_i$$

    Subject to the weight constraint:

    $$\sum_{i=1}^{n} w_i x_i \leq W$$

    Where $x_i \in \{0, 1\}$ represents whether item $i$ is included.

!!! success "Key Properties"
    - **Optimal Substructure**: The optimal solution for capacity $W$ contains optimal solutions for smaller capacities
    - **Overlapping Subproblems**: The same subproblems are solved multiple times
    - **State Transition**: $dp[i][w] = \max(dp[i-1][w], dp[i-1][w-w_i] + v_i)$

## Implementation Approaches

=== "Dynamic Programming (Recommended)"
    ```python
    def knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:
        """
        Solve 0/1 knapsack problem using dynamic programming.

        Args:
            weights: List of item weights
            values: List of item values
            capacity: Maximum weight capacity of knapsack

        Returns:
            Maximum value that can be achieved

        Example:
            >>> weights = [2, 1, 3, 2]
            >>> values = [12, 10, 20, 15]
            >>> capacity = 5
            >>> knapsack_01(weights, values, capacity)
            37  # Items 0, 1, and 3: 12 + 10 + 15 = 37
        """
        n = len(weights)

        # Initialize DP table: dp[i][w] = max value using first i items with capacity w
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]

        # Fill the DP table
        for i in range(1, n + 1):
            for w in range(capacity + 1):
                if weights[i-1] <= w:
                    # Can include item i
                    dp[i][w] = max(
                        dp[i-1][w],  # Don't include item i
                        dp[i-1][w - weights[i-1]] + values[i-1]  # Include item i
                    )
                else:
                    # Cannot include item i due to weight constraint
                    dp[i][w] = dp[i-1][w]

        return dp[n][capacity]
    ```

=== "Space-Optimized DP (Alternative)"
    ```python
    def knapsack_01_optimized(weights: list[int], values: list[int], capacity: int) -> int:
        """
        Space-optimized version using only one row of DP table.
        """
        n = len(weights)
        dp = [0] * (capacity + 1)

        for i in range(n):
            # Iterate backwards to avoid overwriting values we need
            for w in range(capacity, weights[i] - 1, -1):
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

        return dp[capacity]
    ```

=== "Recursive with Memoization"
    ```python
    def knapsack_01_memoized(weights: list[int], values: list[int], capacity: int,
                             memo: dict[tuple[int, int], int] = None) -> int:
        """
        Recursive solution with memoization for 0/1 knapsack.
        """
        if memo is None:
            memo = {}

        if (len(weights), capacity) in memo:
            return memo[(len(weights), capacity)]

        if len(weights) == 0 or capacity == 0:
            return 0

        # Don't include current item
        value_without = knapsack_01_memoized(weights[1:], values[1:], capacity, memo)

        # Include current item if possible
        value_with = 0
        if weights[0] <= capacity:
            value_with = values[0] + knapsack_01_memoized(
                weights[1:], values[1:], capacity - weights[0], memo
            )

        result = max(value_without, value_with)
        memo[(len(weights), capacity)] = result
        return result
    ```

!!! tip "Complete Implementation"
    The full implementation with item tracking, comprehensive testing, and additional variants is available in the source code:

    - **Main Implementation**: [`src/algokit/dynamic_programming/knapsack.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/dynamic_programming/knapsack.py)
    - **Tests**: [`tests/unit/dynamic_programming/test_knapsack.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/dynamic_programming/test_knapsack.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    | **Dynamic Programming** | O(n × W) | O(n × W) | Optimal solution, full table |
    | **Space-Optimized DP** | O(n × W) | O(W) | Same time, reduced space |
    | **Memoized Recursion** | O(n × W) | O(n × W) | Same complexity, recursive |

!!! warning "Performance Considerations"
    - **Time complexity** depends on both number of items and capacity
    - **Large capacities** can make the problem computationally expensive
    - **Space optimization** is crucial for large-scale problems
    - **Pseudo-polynomial** complexity: O(n × W) is not truly polynomial

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Resource Allocation"
        - **Portfolio Optimization**: Investment selection with budget constraints
        - **Project Management**: Task selection with time/resource limits
        - **Inventory Management**: Product selection with storage constraints
        - **Budget Planning**: Expense allocation within budget limits

    !!! grid-item "Computer Science"
        - **Algorithm Design**: Dynamic programming principles
        - **Optimization Problems**: Constraint satisfaction
        - **Game Development**: Item selection and character building
        - **Data Structures**: Understanding state management

    !!! grid-item "Real-World Scenarios"
        - **Cutting Stock**: Material optimization in manufacturing
        - **Container Loading**: Cargo optimization for shipping
        - **Network Design**: Resource allocation in telecommunications
        - **Machine Learning**: Feature selection with constraints

    !!! grid-item "Educational Value"
        - **Dynamic Programming**: Understanding optimal substructure
        - **Constraint Optimization**: Learning to handle limitations
        - **State Transitions**: Building solutions incrementally
        - **Problem Decomposition**: Breaking complex problems into subproblems

!!! success "Educational Value"
    - **Dynamic Programming**: Perfect example of optimal substructure and overlapping subproblems
    - **Constraint Handling**: Shows how to work within given limitations
    - **State Management**: Demonstrates building solutions from smaller subproblems
    - **Optimization**: Illustrates the trade-off between correctness and efficiency

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Textbooks"
        1. **Cormen, T. H., et al.** (2009). *Introduction to Algorithms*. MIT Press. ISBN 978-0-262-03384-8.
        2. **Kleinberg, J., & Tardos, É.** (2006). *Algorithm Design*. Pearson. ISBN 978-0-321-29535-4.

    !!! grid-item "Dynamic Programming"
        3. **Bellman, R.** (1957). *Dynamic Programming*. Princeton University Press.
        4. **Dreyfus, S. E., & Law, A. M.** (1977). *The Art and Theory of Dynamic Programming*. Academic Press.

    !!! grid-item "Online Resources"
        5. [0/1 Knapsack Problem - LeetCode](https://leetcode.com/problems/ones-and-zeroes/)
        6. [Dynamic Programming - GeeksforGeeks](https://www.geeksforgeeks.org/dynamic-programming/)
        7. [Knapsack Problem - Wikipedia](https://en.wikipedia.org/wiki/Knapsack_problem)

    !!! grid-item "Implementation & Practice"
        8. [Python Official Documentation](https://docs.python.org/3/)
        9. [Dynamic Programming Patterns](https://leetcode.com/discuss/general-discussion/458695/dynamic-programming-patterns)
        10. [Algorithm Visualization](https://visualgo.net/en/dp)

!!! tip "Interactive Learning"
    Try implementing the 0/1 knapsack problem yourself! Start with a simple recursive solution, then add memoization, and finally implement the dynamic programming approach. Test with different weight and value combinations to understand how the algorithm handles various constraints. This will give you deep insight into dynamic programming's power in handling constraints.

## Navigation

{{ nav_grid(current_algorithm=config.extra.algorithm_key, current_family=config.extra.family, max_related=5) }}
