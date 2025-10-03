"""Coin change problem implementation using dynamic programming."""


def coin_change(coins: list[int], amount: int) -> int:
    """Find the minimum number of coins needed to make up the given amount.

    This function solves the coin change problem using dynamic programming.
    It finds the minimum number of coins needed to make up a given amount
    using coins of specified denominations.

    Args:
        coins: List of coin denominations (positive integers).
        amount: Target amount to make up.

    Returns:
        Minimum number of coins needed. Returns -1 if impossible.

    Raises:
        ValueError: If coins list is empty or contains non-positive values.

    Examples:
        >>> coin_change([1, 3, 4], 6)
        2
        >>> coin_change([2], 3)
        -1
    """
    if not coins:
        raise ValueError("Coins list cannot be empty")
    if any(coin <= 0 for coin in coins):
        raise ValueError("All coins must be positive integers")
    if amount < 0:
        raise ValueError("Amount must be non-negative")

    if amount == 0:
        return 0

    # Initialize DP table: dp[i] = minimum coins needed for amount i
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0  # Base case: 0 coins needed for amount 0

    # Fill DP table
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return int(dp[amount]) if dp[amount] != float("inf") else -1


def coin_change_with_coins(coins: list[int], amount: int) -> list[int]:
    """Find the actual coins used to make up the given amount.

    This function extends the basic coin change problem to return
    the actual coins used in the optimal solution.

    Args:
        coins: List of coin denominations (positive integers).
        amount: Target amount to make up.

    Returns:
        List of coins used in the optimal solution. Returns empty list if impossible.

    Raises:
        ValueError: If coins list is empty or contains non-positive values.
    """
    if not coins:
        raise ValueError("Coins list cannot be empty")
    if any(coin <= 0 for coin in coins):
        raise ValueError("All coins must be positive integers")
    if amount < 0:
        raise ValueError("Amount must be non-negative")

    if amount == 0:
        return []

    # Initialize DP table and parent tracking
    dp = [float("inf")] * (amount + 1)
    parent = [-1] * (amount + 1)
    dp[0] = 0

    # Fill DP table with parent tracking
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                parent[i] = coin

    # Reconstruct solution
    if dp[amount] == float("inf"):
        return []

    result = []
    current = amount
    while current > 0:
        coin = parent[current]
        result.append(coin)
        current -= coin

    return result


def coin_change_count_ways(coins: list[int], amount: int) -> int:
    """Count the number of ways to make up the given amount.

    This function counts all possible combinations of coins that
    make up the given amount (order doesn't matter).

    Args:
        coins: List of coin denominations (positive integers).
        amount: Target amount to make up.

    Returns:
        Number of ways to make up the amount.

    Raises:
        ValueError: If coins list is empty or contains non-positive values.
    """
    if not coins:
        raise ValueError("Coins list cannot be empty")
    if any(coin <= 0 for coin in coins):
        raise ValueError("All coins must be positive integers")
    if amount < 0:
        raise ValueError("Amount must be non-negative")

    if amount == 0:
        return 1

    # Initialize DP table: dp[i] = number of ways to make amount i
    dp = [0] * (amount + 1)
    dp[0] = 1  # Base case: 1 way to make amount 0 (no coins)

    # Fill DP table
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]

    return dp[amount]
