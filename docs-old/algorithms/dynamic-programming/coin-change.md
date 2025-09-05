---
algorithm_key: "coin-change"
tags: [dynamic-programming, algorithms, optimization, coin-change, greedy, dp]
title: "Coin Change Problem"
family: "dynamic-programming"
---

{{ algorithm_card(config.extra.algorithm_key) }}

!!! abstract "Overview"
    The Coin Change problem is a classic dynamic programming challenge that asks: given a set of coin denominations and a target amount, what is the minimum number of coins needed to make up that amount? This problem demonstrates the power of dynamic programming in solving optimization problems with overlapping subproblems.

    While a greedy approach might work for some coin sets (like US coins), it fails for arbitrary denominations. Dynamic programming provides an elegant solution that guarantees the optimal result by building solutions from smaller subproblems.

## Mathematical Formulation

!!! math "Problem Definition"
    Given:
    - A set of coin denominations: $C = \{c_1, c_2, ..., c_n\}$
    - A target amount: $A$

    Find the minimum number of coins needed to make amount $A$:

    $$\min \sum_{i=1}^{n} x_i \text{ subject to } \sum_{i=1}^{n} c_i x_i = A$$

    Where $x_i$ represents the number of coins of denomination $c_i$.

!!! success "Key Properties"
    - **Optimal Substructure**: The optimal solution for amount $A$ contains optimal solutions for amounts $A - c_i$
    - **Overlapping Subproblems**: The same subproblems are solved multiple times
    - **State Transition**: $dp[A] = \min(dp[A - c_i] + 1)$ for all valid coins $c_i$

## Implementation Approaches

=== "Dynamic Programming (Recommended)"
    ```python
    def coin_change(coins: list[int], amount: int) -> int:
        """
        Find minimum number of coins needed to make the given amount.

        Args:
            coins: List of available coin denominations
            amount: Target amount to make

        Returns:
            Minimum number of coins needed, or -1 if impossible

        Example:
            >>> coin_change([1, 2, 5], 11)
            3  # 5 + 5 + 1 = 11
        """
        if amount == 0:
            return 0

        # Initialize DP array with infinity
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0

        # Build solutions for all amounts from 1 to target
        for current_amount in range(1, amount + 1):
            for coin in coins:
                if coin <= current_amount:
                    dp[current_amount] = min(
                        dp[current_amount],
                        dp[current_amount - coin] + 1
                    )

        return dp[amount] if dp[amount] != float('inf') else -1
    ```

=== "Memoized Recursion (Alternative)"
    ```python
    def coin_change_memoized(coins: list[int], amount: int, memo: dict[int, int] = None) -> int:
        """
        Recursive solution with memoization for coin change problem.
        """
        if memo is None:
            memo = {}

        if amount in memo:
            return memo[amount]

        if amount == 0:
            return 0

        if amount < 0:
            return -1

        min_coins = float('inf')
        for coin in coins:
            result = coin_change_memoized(coins, amount - coin, memo)
            if result != -1:
                min_coins = min(min_coins, result + 1)

        memo[amount] = min_coins if min_coins != float('inf') else -1
        return memo[amount]
    ```

=== "Greedy Approach (Limited Use)"
    ```python
    def coin_change_greedy(coins: list[int], amount: int) -> int:
        """
        Greedy approach - only works for certain coin sets.
        WARNING: This may not give optimal results for arbitrary denominations!
        """
        coins.sort(reverse=True)  # Use largest coins first
        total_coins = 0
        remaining = amount

        for coin in coins:
            if remaining >= coin:
                count = remaining // coin
                total_coins += count
                remaining -= count * coin

        return total_coins if remaining == 0 else -1
    ```

!!! tip "Complete Implementation"
    The full implementation with error handling, comprehensive testing, and additional variants is available in the source code:

    - **Main Implementation**: [`src/algokit/dynamic_programming/coin_change.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/dynamic_programming/coin_change.py)
    - **Tests**: [`tests/unit/dynamic_programming/test_coin_change.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/dynamic_programming/test_coin_change.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    | **Dynamic Programming** | O(amount × coins) | O(amount) | Optimal solution, guaranteed correct |
    | **Memoized Recursion** | O(amount × coins) | O(amount) | Same complexity, recursive structure |
    | **Greedy** | O(coins log coins) | O(1) | Fast but may not be optimal |

!!! warning "Performance Considerations"
    - **DP approach** is optimal but requires building solutions for all amounts up to target
    - **Memoization** useful when you need solutions for multiple amounts
    - **Greedy approach** only works for coin sets with specific properties (like US coins)
    - **Large amounts** can make DP memory-intensive

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Financial Systems"
        - **Vending Machines**: Optimal coin dispensing
        - **Cash Registers**: Minimum coin change calculation
        - **Banking**: ATM cash withdrawal optimization
        - **Payment Processing**: Efficient change distribution

    !!! grid-item "Computer Science"
        - **Algorithm Design**: Dynamic programming principles
        - **Optimization Problems**: Resource allocation
        - **Game Development**: Score systems and rewards
        - **Data Structures**: Understanding state management

    !!! grid-item "Real-World Scenarios"
        - **Retail**: Cashier change optimization
        - **Transportation**: Fare collection systems
        - **Gaming**: Point systems and achievements
        - **Manufacturing**: Part quantity optimization

    !!! grid-item "Educational Value"
        - **Dynamic Programming**: Understanding optimal substructure
        - **State Transitions**: Learning recurrence relations
        - **Optimization**: Comparing greedy vs optimal approaches
        - **Problem Solving**: Breaking complex problems into subproblems

!!! success "Educational Value"
    - **Dynamic Programming**: Perfect example of optimal substructure and overlapping subproblems
    - **Algorithm Design**: Shows when greedy approaches fail and DP succeeds
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
        5. [Coin Change Problem - LeetCode](https://leetcode.com/problems/coin-change/)
        6. [Dynamic Programming - GeeksforGeeks](https://www.geeksforgeeks.org/dynamic-programming/)
        7. [Coin Change - Wikipedia](https://en.wikipedia.org/wiki/Change-making_problem)

    !!! grid-item "Implementation & Practice"
        8. [Python Official Documentation](https://docs.python.org/3/)
        9. [Dynamic Programming Patterns](https://leetcode.com/discuss/general-discussion/458695/dynamic-programming-patterns)
        10. [Algorithm Visualization](https://visualgo.net/en/dp)

!!! tip "Interactive Learning"
    Try implementing the coin change problem yourself! Start with the greedy approach to see when it fails, then implement the dynamic programming solution. Test with different coin sets like [1, 3, 4] and amount 6 to see why greedy fails. This will give you deep insight into when to use greedy vs dynamic programming approaches.

## Navigation

{{ nav_grid(current_algorithm=config.extra.algorithm_key, current_family=config.extra.family, max_related=5) }}
