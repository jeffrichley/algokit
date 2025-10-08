# Dynamic Programming Overview

## 📚 Introduction

**Dynamic Programming (DP)** is a powerful algorithmic paradigm that solves complex problems by breaking them down into simpler subproblems. The key insight is that many problems have overlapping subproblems and optimal substructure, allowing us to store and reuse solutions to avoid redundant computation.

## 🎯 Core Principles

### Optimal Substructure
A problem exhibits optimal substructure if an optimal solution to the problem contains optimal solutions to its subproblems. This allows us to build up solutions from smaller pieces.

### Overlapping Subproblems
Unlike divide-and-conquer algorithms, dynamic programming problems have overlapping subproblems that are solved multiple times. DP stores these solutions to avoid recomputation.

### Memoization vs Tabulation

- **Memoization** (Top-Down): Solve problems recursively and cache results as you go
- **Tabulation** (Bottom-Up): Build solutions iteratively, filling a table from base cases

## 🔑 Key Characteristics

- **Optimal Substructure**: Problems can be broken into optimal subproblems
- **Overlapping Subproblems**: Same subproblems are solved multiple times
- **Memoization/Tabulation**: Store and reuse computed solutions
- **State Transition**: Clear transitions between problem states
- **Base Cases**: Well-defined starting conditions
- **Recurrence Relation**: Mathematical formula relating subproblems

## 🎨 Implemented Algorithms

### 1. **Fibonacci Sequence** 🔢
Classic DP example demonstrating memoization and tabulation.
- **Time Complexity**: O(n)
- **Space Complexity**: O(n) or O(1) optimized
- **Use Case**: Introduction to DP concepts

### 2. **Coin Change** 💰
Find minimum coins needed to make change for a given amount.
- **Time Complexity**: O(n × m) where n = amount, m = coins
- **Space Complexity**: O(n)
- **Use Case**: Currency systems, resource allocation

### 3. **0/1 Knapsack** 🎒
Maximize value of items in knapsack with weight constraint.
- **Time Complexity**: O(n × W) where n = items, W = capacity
- **Space Complexity**: O(n × W) or O(W) optimized
- **Use Case**: Resource allocation, portfolio optimization

### 4. **Longest Common Subsequence** 📝
Find longest subsequence common to two sequences.
- **Time Complexity**: O(m × n)
- **Space Complexity**: O(m × n) or O(min(m,n)) optimized
- **Use Case**: DNA analysis, diff tools, text comparison

### 5. **Edit Distance** ✏️
Minimum edits (insert/delete/replace) to transform one string to another.
- **Time Complexity**: O(m × n)
- **Space Complexity**: O(m × n) or O(min(m,n)) optimized
- **Use Case**: Spell checkers, plagiarism detection, DNA analysis

### 6. **Matrix Chain Multiplication** 🔗
Find optimal parenthesization for multiplying chain of matrices.
- **Time Complexity**: O(n³)
- **Space Complexity**: O(n²)
- **Use Case**: Compiler optimization, graphics, scientific computing

## 🌟 Common Applications

### Optimization Problems
- Knapsack variants
- Coin change problem
- Rod cutting
- Longest increasing subsequence

### Sequence Problems
- Fibonacci sequences
- Longest common subsequence
- Edit distance
- Palindrome problems

### Path Finding
- Shortest path in DAGs
- Floyd-Warshall algorithm
- Bellman-Ford algorithm
- Grid path counting

### Resource Allocation
- Task scheduling
- Job assignment
- Partition problems
- Subset sum

### String Processing
- String matching
- Regular expression matching
- Wildcard pattern matching
- Text justification

## 🔬 When to Use Dynamic Programming

DP is ideal when:
- ✅ Problem has optimal substructure
- ✅ Subproblems overlap significantly
- ✅ You can define recurrence relations
- ✅ Problem asks for "optimal" solution (min/max)
- ✅ Problem involves counting or enumeration

DP may not be suitable when:
- ❌ Subproblems are independent (use divide-and-conquer instead)
- ❌ No optimal substructure exists
- ❌ Memory constraints are severe
- ❌ Simple greedy approach suffices

## 💡 Problem-Solving Strategy

### 1. **Identify DP Nature**
- Look for optimal substructure
- Check for overlapping subproblems
- Identify if it's optimization/counting/enumeration

### 2. **Define State**
- What information uniquely identifies a subproblem?
- Example: `dp[i][j]` could represent LCS length up to indices i, j

### 3. **Establish Recurrence Relation**
- How does solution to state depend on previous states?
- Example: `dp[i][j] = dp[i-1][j-1] + 1` if chars match

### 4. **Determine Base Cases**
- What are the simplest subproblems?
- Example: `dp[0][j] = 0` for LCS

### 5. **Choose Approach**
- **Top-Down (Memoization)**: Easier to write, natural recursion
- **Bottom-Up (Tabulation)**: Better performance, no stack overflow risk

### 6. **Optimize Space**
- Can you reduce dimensions? (e.g., 2D → 1D)
- Do you only need previous row/column?
- Rolling array optimization

## 📊 Complexity Analysis

| Algorithm | Time | Space | Optimized Space |
|-----------|------|-------|-----------------|
| Fibonacci | O(n) | O(n) | O(1) |
| Coin Change | O(n×m) | O(n) | O(n) |
| 0/1 Knapsack | O(n×W) | O(n×W) | O(W) |
| LCS | O(m×n) | O(m×n) | O(min(m,n)) |
| Edit Distance | O(m×n) | O(m×n) | O(min(m,n)) |
| Matrix Chain | O(n³) | O(n²) | O(n²) |

## 🎓 Learning Path

### Beginner
1. Start with **Fibonacci** - understand memoization
2. Practice **Coin Change** - learn tabulation
3. Explore **Climbing Stairs** - recognize patterns

### Intermediate
4. Master **0/1 Knapsack** - classic DP
5. Study **LCS** - 2D DP tables
6. Implement **Edit Distance** - practical applications

### Advanced
7. Tackle **Matrix Chain** - 3D thinking
8. Explore state compression techniques
9. Study space optimization methods

## 🔗 Related Families

- **Greedy Algorithms**: Sometimes optimal, but doesn't always work
- **Divide and Conquer**: Independent subproblems vs overlapping
- **Backtracking**: Explores all possibilities, DP caches results
- **Graph Algorithms**: Many graph problems use DP (shortest paths)

## 📚 Further Reading

### Books
- "Introduction to Algorithms" (CLRS) - Chapter 15
- "Algorithm Design Manual" by Skiena
- "Dynamic Programming for Coding Interviews" by Meenakshi

### Online Resources
- [MIT OCW - Dynamic Programming](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/)
- [GeeksforGeeks DP Section](https://www.geeksforgeeks.org/dynamic-programming/)
- [LeetCode DP Problems](https://leetcode.com/tag/dynamic-programming/)

## 🚀 Getting Started

```python
from algokit.algorithms.dynamic_programming import (
    fibonacci,
    coin_change,
    knapsack,
    longest_common_subsequence,
    edit_distance,
    matrix_chain_multiplication
)

# Example: Fibonacci
result = fibonacci(10)
print(f"Fibonacci(10) = {result}")

# Example: Coin Change
coins = [1, 5, 10, 25]
amount = 99
min_coins = coin_change(coins, amount)
print(f"Minimum coins for {amount}: {min_coins}")
```

## 🎯 Practice Problems

### Easy
- Fibonacci variations
- Climbing stairs
- House robber
- Min cost climbing stairs

### Medium
- Coin change II (counting ways)
- Longest increasing subsequence
- Partition equal subset sum
- Unique paths in grid

### Hard
- Regular expression matching
- Burst balloons
- Palindrome partitioning
- Interleaving strings

---

**Ready to dive in?** Start with our [Fibonacci implementation](fibonacci.md) or explore other algorithms in the sidebar!
