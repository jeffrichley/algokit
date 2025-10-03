"""0/1 Knapsack problem implementation using dynamic programming."""


def knapsack_01(
    weights: list[int], values: list[int], capacity: int
) -> tuple[int, list[int]]:
    """Solve the 0/1 knapsack problem using dynamic programming.

    The 0/1 knapsack problem involves selecting items to maximize value
    while respecting a weight constraint. Each item can be selected at most once.

    Args:
        weights: List of item weights (positive integers).
        values: List of item values (positive integers).
        capacity: Maximum weight capacity of the knapsack.

    Returns:
        Tuple of (maximum_value, selected_items) where selected_items
        is a list of indices of selected items.

    Raises:
        ValueError: If inputs are invalid (empty lists, negative values, etc.).

    Examples:
        >>> weights = [10, 20, 30]
        >>> values = [60, 100, 120]
        >>> capacity = 50
        >>> max_value, items = knapsack_01(weights, values, capacity)
        >>> max_value
        220
        >>> sorted(items)
        [1, 2]
    """
    if not weights or not values:
        raise ValueError("Weights and values lists cannot be empty")
    if len(weights) != len(values):
        raise ValueError("Weights and values lists must have the same length")
    if any(w <= 0 for w in weights):
        raise ValueError("All weights must be positive")
    if any(v <= 0 for v in values):
        raise ValueError("All values must be positive")
    if capacity < 0:
        raise ValueError("Capacity must be non-negative")

    n = len(weights)

    if capacity == 0:
        return 0, []

    # Initialize DP table: dp[i][w] = maximum value using first i items with capacity w
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    # Fill DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Option 1: Don't take item i-1
            dp[i][w] = dp[i - 1][w]

            # Option 2: Take item i-1 (if it fits)
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])

    # Reconstruct solution
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:  # Item i-1 was selected
            selected_items.append(i - 1)
            w -= weights[i - 1]

    return dp[n][capacity], selected_items


def knapsack_01_value_only(weights: list[int], values: list[int], capacity: int) -> int:
    """Solve the 0/1 knapsack problem and return only the maximum value.

    This is a space-optimized version that only returns the maximum value
    without tracking which items were selected. Uses 1D array update-in-place.

    Args:
        weights: List of item weights (positive integers).
        values: List of item values (positive integers).
        capacity: Maximum weight capacity of the knapsack.

    Returns:
        Maximum value achievable.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not weights or not values:
        raise ValueError("Weights and values lists cannot be empty")
    if len(weights) != len(values):
        raise ValueError("Weights and values lists must have the same length")
    if any(w <= 0 for w in weights):
        raise ValueError("All weights must be positive")
    if any(v <= 0 for v in values):
        raise ValueError("All values must be positive")
    if capacity < 0:
        raise ValueError("Capacity must be non-negative")

    n = len(weights)

    if capacity == 0:
        return 0

    # 1D DP array with update-in-place (iterate backwards to avoid using updated values)
    dp = [0 for _ in range(capacity + 1)]

    for i in range(n):
        # Iterate backwards to avoid using updated values in the same iteration
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]


def knapsack_fractional_greedy(
    weights: list[int], values: list[int], capacity: int
) -> float:
    """Solve the fractional knapsack problem using greedy approach.

    The fractional knapsack allows taking fractions of items.
    This problem can be solved optimally using a greedy approach.

    Args:
        weights: List of item weights (positive integers).
        values: List of item values (positive integers).
        capacity: Maximum weight capacity of the knapsack.

    Returns:
        Maximum value achievable with fractional items.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not weights or not values:
        raise ValueError("Weights and values lists cannot be empty")
    if len(weights) != len(values):
        raise ValueError("Weights and values lists must have the same length")
    if any(w <= 0 for w in weights):
        raise ValueError("All weights must be positive")
    if any(v <= 0 for v in values):
        raise ValueError("All values must be positive")
    if capacity < 0:
        raise ValueError("Capacity must be non-negative")

    if capacity == 0:
        return 0.0

    # Calculate value-to-weight ratios and sort items by ratio (descending)
    items = [
        (values[i] / weights[i], weights[i], values[i], i) for i in range(len(weights))
    ]
    items.sort(reverse=True)

    total_value = 0.0
    remaining_capacity = capacity

    for _ratio, weight, value, _ in items:
        if remaining_capacity <= 0:
            break

        # Take as much as possible (either full item or fraction)
        take_weight = min(weight, remaining_capacity)
        total_value += (take_weight / weight) * value
        remaining_capacity -= take_weight

    return total_value


def knapsack_01_memoized(weights: list[int], values: list[int], capacity: int) -> int:
    """Solve the 0/1 knapsack problem using memoized recursion.

    This implementation uses top-down dynamic programming with memoization
    as an alternative to the bottom-up approach.

    Args:
        weights: List of item weights (positive integers).
        values: List of item values (positive integers).
        capacity: Maximum weight capacity of the knapsack.

    Returns:
        Maximum value achievable.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not weights or not values:
        raise ValueError("Weights and values lists cannot be empty")
    if len(weights) != len(values):
        raise ValueError("Weights and values lists must have the same length")
    if any(w <= 0 for w in weights):
        raise ValueError("All weights must be positive")
    if any(v <= 0 for v in values):
        raise ValueError("All values must be positive")
    if capacity < 0:
        raise ValueError("Capacity must be non-negative")

    n = len(weights)

    if capacity == 0:
        return 0

    # Memoization table
    memo: dict[tuple[int, int], int] = {}

    def knapsack_rec(i: int, w: int) -> int:
        if i == 0 or w == 0:
            return 0

        if (i, w) in memo:
            return memo[(i, w)]

        # Don't take current item
        result = knapsack_rec(i - 1, w)

        # Take current item (if it fits)
        if weights[i - 1] <= w:
            result = max(
                result, knapsack_rec(i - 1, w - weights[i - 1]) + values[i - 1]
            )

        memo[(i, w)] = result
        return result

    return knapsack_rec(n, capacity)
