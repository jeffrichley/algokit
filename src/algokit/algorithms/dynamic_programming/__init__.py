"""Dynamic Programming algorithms package."""

from algokit.algorithms.dynamic_programming.coin_change import (
    coin_change,
    coin_change_count_ways,
    coin_change_with_coins,
)
from algokit.algorithms.dynamic_programming.edit_distance import (
    edit_distance,
    edit_distance_memoized,
    edit_distance_space_optimized,
    edit_distance_with_operations,
    hamming_distance,
    longest_common_substring,
    longest_common_substring_string,
)
from algokit.algorithms.dynamic_programming.fibonacci import fibonacci
from algokit.algorithms.dynamic_programming.knapsack import (
    knapsack_01,
    knapsack_01_memoized,
    knapsack_01_value_only,
    knapsack_fractional_greedy,
)
from algokit.algorithms.dynamic_programming.longest_common_subsequence import (
    longest_common_subsequence,
    longest_common_subsequence_memoized,
    longest_common_subsequence_space_optimized,
    longest_common_subsequence_string,
    longest_increasing_subsequence,
    longest_increasing_subsequence_optimized,
)
from algokit.algorithms.dynamic_programming.matrix_chain_multiplication import (
    matrix_chain_multiplication,
    matrix_chain_multiplication_brute_force,
    matrix_chain_multiplication_memoized,
    matrix_chain_multiplication_value_only,
    matrix_chain_multiplication_with_expression,
    matrix_multiply_cost,
    print_optimal_parentheses,
)

__all__ = [
    # Fibonacci
    "fibonacci",
    # Coin Change
    "coin_change",
    "coin_change_count_ways",
    "coin_change_with_coins",
    # Knapsack
    "knapsack_01",
    "knapsack_01_memoized",
    "knapsack_01_value_only",
    "knapsack_fractional_greedy",
    # Longest Common Subsequence
    "longest_common_subsequence",
    "longest_common_subsequence_memoized",
    "longest_common_subsequence_space_optimized",
    "longest_common_subsequence_string",
    "longest_increasing_subsequence",
    "longest_increasing_subsequence_optimized",
    # Edit Distance
    "edit_distance",
    "edit_distance_memoized",
    "edit_distance_space_optimized",
    "edit_distance_with_operations",
    "hamming_distance",
    "longest_common_substring",
    "longest_common_substring_string",
    # Matrix Chain Multiplication
    "matrix_chain_multiplication",
    "matrix_chain_multiplication_brute_force",
    "matrix_chain_multiplication_memoized",
    "matrix_chain_multiplication_value_only",
    "matrix_chain_multiplication_with_expression",
    "matrix_multiply_cost",
    "print_optimal_parentheses",
]
