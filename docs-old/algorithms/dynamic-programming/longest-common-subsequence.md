---
algorithm_key: "longest-common-subsequence"
tags: [dynamic-programming, algorithms, optimization, lcs, longest-common-subsequence, string-algorithms, bioinformatics]
title: "Longest Common Subsequence"
family: "dynamic-programming"
---

{{ algorithm_card(config.extra.algorithm_key) }}

!!! abstract "Overview"
    The Longest Common Subsequence (LCS) problem finds the longest sequence that appears in the same order in two given sequences, but not necessarily consecutively. This fundamental algorithm has applications ranging from DNA sequence analysis and version control systems to natural language processing and plagiarism detection.

    Unlike substring problems that require consecutive elements, LCS allows for gaps between matching elements, making it more flexible for real-world applications where exact positioning may vary. The dynamic programming approach elegantly handles the overlapping subproblems that arise when comparing sequences of different lengths.

## Mathematical Formulation

!!! math "Problem Definition"
    Given:
    - Two sequences: $X = x_1, x_2, ..., x_m$ and $Y = y_1, y_2, ..., y_n$
    
    Find the longest sequence $Z = z_1, z_2, ..., z_k$ such that:
    - $Z$ is a subsequence of both $X$ and $Y$
    - $k$ is maximized
    
    The LCS length can be computed using the recurrence relation:
    
    $$LCS(i,j) = \begin{cases} 
    0 & \text{if } i = 0 \text{ or } j = 0 \\
    LCS(i-1,j-1) + 1 & \text{if } x_i = y_j \\
    \max(LCS(i-1,j), LCS(i,j-1)) & \text{if } x_i \neq y_j
    \end{cases}$$

!!! success "Key Properties"
    - **Optimal Substructure**: The LCS of prefixes contains the LCS of smaller prefixes
    - **Overlapping Subproblems**: The same subproblems are solved multiple times
    - **State Transition**: Builds solutions incrementally from smaller subsequences

## Implementation Approaches

=== "Dynamic Programming (Recommended)"
    ```python
    def longest_common_subsequence(text1: str, text2: str) -> int:
        """
        Find the length of the longest common subsequence between two strings.
        
        Args:
            text1: First input string
            text2: Second input string
            
        Returns:
            Length of the longest common subsequence
            
        Example:
            >>> longest_common_subsequence("abcde", "ace")
            3  # "ace" is the LCS
            >>> longest_common_subsequence("abcba", "abcbcba")
            5  # "abcba" is the LCS
        """
        m, n = len(text1), len(text2)
        
        # Initialize DP table: dp[i][j] = LCS length for text1[:i] and text2[:j]
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    # Characters match, extend the LCS
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    # Characters don't match, take maximum of previous solutions
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    ```

=== "Space-Optimized DP (Alternative)"
    ```python
    def lcs_optimized(text1: str, text2: str) -> int:
        """
        Space-optimized version using only two rows of DP table.
        """
        m, n = len(text1), len(text2)
        
        # Use only two rows to save space
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            
            # Swap rows for next iteration
            prev, curr = curr, prev
        
        return prev[n]
    ```

=== "Recursive with Memoization"
    ```python
    def lcs_memoized(text1: str, text2: str, i: int, j: int, 
                     memo: dict[tuple[int, int], int] = None) -> int:
        """
        Recursive solution with memoization for LCS.
        """
        if memo is None:
            memo = {}
        
        if (i, j) in memo:
            return memo[(i, j)]
        
        if i == 0 or j == 0:
            return 0
        
        if text1[i-1] == text2[j-1]:
            result = lcs_memoized(text1, text2, i-1, j-1, memo) + 1
        else:
            result = max(
                lcs_memoized(text1, text2, i-1, j, memo),
                lcs_memoized(text1, text2, i, j-1, memo)
            )
        
        memo[(i, j)] = result
        return result
    ```

!!! tip "Complete Implementation"
    The full implementation with subsequence reconstruction, comprehensive testing, and additional variants is available in the source code:

    - **Main Implementation**: [`src/algokit/dynamic_programming/lcs.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/dynamic_programming/lcs.py)
    - **Tests**: [`tests/unit/dynamic_programming/test_lcs.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/dynamic_programming/test_lcs.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    | **Dynamic Programming** | O(m × n) | O(m × n) | Optimal solution, full table |
    | **Space-Optimized DP** | O(m × n) | O(min(m, n)) | Same time, reduced space |
    | **Memoized Recursion** | O(m × n) | O(m × n) | Same complexity, recursive |

!!! warning "Performance Considerations"
    - **Quadratic complexity** makes it expensive for very long sequences
    - **Space optimization** is crucial for large-scale problems
    - **Memory usage** scales with the product of sequence lengths
    - **Reconstruction** requires additional O(m × n) space

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Bioinformatics"
        - **DNA Sequence Analysis**: Finding common genetic patterns
        - **Protein Comparison**: Identifying similar protein structures
        - **Phylogenetic Analysis**: Evolutionary relationship studies
        - **Genome Assembly**: Sequence alignment and comparison

    !!! grid-item "Computer Science"
        - **Version Control**: Git diff algorithms and file comparison
        - **Plagiarism Detection**: Text similarity analysis
        - **Natural Language Processing**: Sentence similarity and alignment
        - **Data Mining**: Pattern recognition in sequences

    !!! grid-item "Real-World Scenarios"
        - **File Comparison**: Document similarity and change tracking
        - **Spell Checking**: Finding closest matches in dictionaries
        - **Speech Recognition**: Audio pattern matching
        - **Image Processing**: Visual pattern recognition

    !!! grid-item "Educational Value"
        - **Dynamic Programming**: Understanding optimal substructure
        - **String Algorithms**: Learning sequence comparison techniques
        - **State Transitions**: Building solutions incrementally
        - **Problem Decomposition**: Breaking complex problems into subproblems

!!! success "Educational Value"
    - **Dynamic Programming**: Perfect example of optimal substructure and overlapping subproblems
    - **String Processing**: Shows how to handle sequence comparison efficiently
    - **State Management**: Demonstrates building solutions from smaller subproblems
    - **Algorithm Design**: Illustrates the power of incremental solution building

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Textbooks"
        1. **Cormen, T. H., et al.** (2009). *Introduction to Algorithms*. MIT Press. ISBN 978-0-262-03384-8.
        2. **Gusfield, D.** (1997). *Algorithms on Strings, Trees, and Sequences*. Cambridge University Press. ISBN 978-0-521-58519-4.

    !!! grid-item "Dynamic Programming"
        3. **Bellman, R.** (1957). *Dynamic Programming*. Princeton University Press.
        4. **Dreyfus, S. E., & Law, A. M.** (1977). *The Art and Theory of Dynamic Programming*. Academic Press.

    !!! grid-item "Online Resources"
        5. [Longest Common Subsequence - LeetCode](https://leetcode.com/problems/longest-common-subsequence/)
        6. [Dynamic Programming - GeeksforGeeks](https://www.geeksforgeeks.org/dynamic-programming/)
        7. [LCS Problem - Wikipedia](https://en.wikipedia.org/wiki/Longest_common_subsequence_problem)

    !!! grid-item "Implementation & Practice"
        8. [Python Official Documentation](https://docs.python.org/3/)
        9. [Dynamic Programming Patterns](https://leetcode.com/discuss/general-discussion/458695/dynamic-programming-patterns)
        10. [Algorithm Visualization](https://visualgo.net/en/dp)

!!! tip "Interactive Learning"
    Try implementing the longest common subsequence algorithm yourself! Start with a simple recursive solution, then add memoization, and finally implement the dynamic programming approach. Test with different string pairs to understand how the algorithm handles various sequence lengths and patterns. This will give you deep insight into dynamic programming's power in sequence analysis.

## Navigation

{{ nav_grid(current_algorithm=config.extra.algorithm_key, current_family=config.extra.family, max_related=5) }}
