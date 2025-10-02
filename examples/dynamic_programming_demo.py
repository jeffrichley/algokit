#!/usr/bin/env python3
"""Demo script showcasing dynamic programming algorithms."""

from algokit.algorithms.dynamic_programming import (
    coin_change,
    coin_change_with_coins,
    edit_distance,
    fibonacci,
    knapsack_01,
    longest_common_subsequence,
    matrix_chain_multiplication_with_expression,
)


def main() -> None:
    """Run dynamic programming algorithm demonstrations."""
    print("🐍 Dynamic Programming Algorithms Demo")
    print("=" * 50)
    
    # Fibonacci Sequence
    print("\n📊 Fibonacci Sequence")
    print("-" * 20)
    n = 10
    result = fibonacci(n)
    print(f"Fibonacci({n}) = {result}")
    
    # Coin Change Problem
    print("\n🪙 Coin Change Problem")
    print("-" * 20)
    coins = [1, 3, 4]
    amount = 6
    min_coins = coin_change(coins, amount)
    actual_coins = coin_change_with_coins(coins, amount)
    print(f"Coins: {coins}")
    print(f"Amount: {amount}")
    print(f"Minimum coins needed: {min_coins}")
    print(f"Actual coins used: {actual_coins}")
    
    # 0/1 Knapsack Problem
    print("\n🎒 0/1 Knapsack Problem")
    print("-" * 20)
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    max_value, selected_items = knapsack_01(weights, values, capacity)
    print(f"Weights: {weights}")
    print(f"Values: {values}")
    print(f"Capacity: {capacity}")
    print(f"Maximum value: {max_value}")
    print(f"Selected items (indices): {selected_items}")
    
    # Longest Common Subsequence
    print("\n🔤 Longest Common Subsequence")
    print("-" * 20)
    text1 = "ABCDGH"
    text2 = "AEDFHR"
    lcs_length = longest_common_subsequence(text1, text2)
    print(f"String 1: {text1}")
    print(f"String 2: {text2}")
    print(f"LCS length: {lcs_length}")
    
    # Edit Distance
    print("\n✏️ Edit Distance (Levenshtein)")
    print("-" * 20)
    str1 = "kitten"
    str2 = "sitting"
    distance = edit_distance(str1, str2)
    print(f"String 1: {str1}")
    print(f"String 2: {str2}")
    print(f"Edit distance: {distance}")
    
    # Matrix Chain Multiplication
    print("\n🔢 Matrix Chain Multiplication")
    print("-" * 20)
    dimensions = [1, 2, 3, 4]  # 3 matrices: 1x2, 2x3, 3x4
    min_cost, expression = matrix_chain_multiplication_with_expression(dimensions)
    print(f"Matrix dimensions: {dimensions}")
    print(f"Minimum scalar multiplications: {min_cost}")
    print(f"Optimal parenthesization: {expression}")
    
    print("\n✨ Demo completed successfully!")


if __name__ == "__main__":
    main()
