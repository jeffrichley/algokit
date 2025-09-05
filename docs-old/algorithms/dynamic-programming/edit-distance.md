---
algorithm_key: "edit-distance"
tags: [dynamic-programming, algorithms, optimization, edit-distance, levenshtein, string-algorithms, nlp]
title: "Edit Distance (Levenshtein Distance)"
family: "dynamic-programming"
---

{{ algorithm_card(config.extra.algorithm_key) }}

!!! abstract "Overview"
    The Edit Distance problem, also known as Levenshtein Distance, measures the minimum number of single-character operations (insertions, deletions, or substitutions) required to transform one string into another. This fundamental algorithm has become essential in natural language processing, spell checking, DNA sequence analysis, and many other applications where string similarity is crucial.

    The dynamic programming approach elegantly handles the overlapping subproblems that arise when comparing strings of different lengths, providing an optimal solution that considers all possible transformation paths. This makes it invaluable for applications ranging from autocorrect systems to bioinformatics sequence alignment.

## Mathematical Formulation

!!! math "Problem Definition"
    Given:
    - Two strings: $X = x_1, x_2, ..., x_m$ and $Y = y_1, y_2, ..., y_n$

    Find the minimum number of operations to transform $X$ into $Y$:
    - **Insert**: Add a character
    - **Delete**: Remove a character
    - **Substitute**: Replace a character with another

    The edit distance can be computed using the recurrence relation:

    $$ED(i,j) = \begin{cases}
    j & \text{if } i = 0 \\
    i & \text{if } j = 0 \\
    ED(i-1,j-1) & \text{if } x_i = y_j \\
    1 + \min(ED(i-1,j), ED(i,j-1), ED(i-1,j-1)) & \text{if } x_i \neq y_j
    \end{cases}$$

!!! success "Key Properties"
    - **Optimal Substructure**: The optimal solution for prefixes contains optimal solutions for smaller prefixes
    - **Overlapping Subproblems**: The same subproblems are solved multiple times
    - **State Transition**: Builds solutions incrementally from smaller subproblems

## Implementation Approaches

=== "Dynamic Programming (Recommended)"
    ```python
    def edit_distance(word1: str, word2: str) -> int:
        """
        Calculate the minimum edit distance between two strings.

        Args:
            word1: First input string
            word2: Second input string

        Returns:
            Minimum number of operations to transform word1 to word2

        Example:
            >>> edit_distance("horse", "ros")
            3  # horse -> rorse -> rose -> ros
            >>> edit_distance("intention", "execution")
            5  # Multiple operations needed
        """
        m, n = len(word1), len(word2)

        # Initialize DP table: dp[i][j] = edit distance for word1[:i] and word2[:j]
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Base cases: empty string to string of length j requires j insertions
        for j in range(n + 1):
            dp[0][j] = j

        # Base cases: string of length i to empty string requires i deletions
        for i in range(m + 1):
            dp[i][0] = i

        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    # Characters match, no operation needed
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # Take minimum of insert, delete, or substitute
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # Delete character from word1
                        dp[i][j-1],    # Insert character into word1
                        dp[i-1][j-1]   # Substitute character
                    )

        return dp[m][n]
    ```

=== "Space-Optimized DP (Alternative)"
    ```python
    def edit_distance_optimized(word1: str, word2: str) -> int:
        """
        Space-optimized version using only two rows of DP table.
        """
        m, n = len(word1), len(word2)

        # Use only two rows to save space
        prev = list(range(n + 1))
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            curr[0] = i  # Base case for empty string2

            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    curr[j] = prev[j-1]
                else:
                    curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])

            # Swap rows for next iteration
            prev, curr = curr, prev

        return prev[n]
    ```

=== "Recursive with Memoization"
    ```python
    def edit_distance_memoized(word1: str, word2: str, i: int, j: int,
                              memo: dict[tuple[int, int], int] = None) -> int:
        """
        Recursive solution with memoization for edit distance.
        """
        if memo is None:
            memo = {}

        if (i, j) in memo:
            return memo[(i, j)]

        if i == 0:
            return j  # Insert j characters

        if j == 0:
            return i  # Delete i characters

        if word1[i-1] == word2[j-1]:
            result = edit_distance_memoized(word1, word2, i-1, j-1, memo)
        else:
            result = 1 + min(
                edit_distance_memoized(word1, word2, i-1, j, memo),    # Delete
                edit_distance_memoized(word1, word2, i, j-1, memo),    # Insert
                edit_distance_memoized(word1, word2, i-1, j-1, memo)   # Substitute
            )

        memo[(i, j)] = result
        return result
    ```

!!! tip "Complete Implementation"
    The full implementation with operation tracking, comprehensive testing, and additional variants is available in the source code:

    - **Main Implementation**: [`src/algokit/dynamic_programming/edit_distance.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/dynamic_programming/edit_distance.py)
    - **Tests**: [`tests/unit/dynamic_programming/test_edit_distance.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/dynamic_programming/test_edit_distance.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    | **Dynamic Programming** | O(m × n) | O(m × n) | Optimal solution, full table |
    | **Space-Optimized DP** | O(m × n) | O(min(m, n)) | Same time, reduced space |
    | **Memoized Recursion** | O(m × n) | O(m × n) | Same complexity, recursive |

!!! warning "Performance Considerations"
    - **Quadratic complexity** makes it expensive for very long strings
    - **Space optimization** is crucial for large-scale problems
    - **Memory usage** scales with the product of string lengths
    - **Operation tracking** requires additional O(m × n) space

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Natural Language Processing"
        - **Spell Checking**: Finding closest word matches
        - **Autocorrect**: Suggesting corrections for typos
        - **Text Similarity**: Measuring document similarity
        - **Machine Translation**: Evaluating translation quality

    !!! grid-item "Bioinformatics"
        - **DNA Sequence Analysis**: Measuring genetic similarity
        - **Protein Comparison**: Identifying similar protein structures
        - **Sequence Alignment**: Finding optimal alignments
        - **Phylogenetic Analysis**: Evolutionary relationship studies

    !!! grid-item "Computer Science"
        - **Fuzzy Search**: Approximate string matching
        - **Data Cleaning**: Identifying similar records
        - **Version Control**: File difference analysis
        - **Plagiarism Detection**: Text similarity analysis

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
        5. [Edit Distance - LeetCode](https://leetcode.com/problems/edit-distance/)
        6. [Dynamic Programming - GeeksforGeeks](https://www.geeksforgeeks.org/dynamic-programming/)
        7. [Levenshtein Distance - Wikipedia](https://en.wikipedia.org/wiki/Levenshtein_distance)

    !!! grid-item "Implementation & Practice"
        8. [Python Official Documentation](https://docs.python.org/3/)
        9. [Dynamic Programming Patterns](https://leetcode.com/discuss/general-discussion/458695/dynamic-programming-patterns)
        10. [Algorithm Visualization](https://visualgo.net/en/dp)

!!! tip "Interactive Learning"
    Try implementing the edit distance algorithm yourself! Start with simple examples like "kitten" to "sitting", then trace through the DP table to see how the solution is built. Implement the space-optimized version to understand how you can reduce memory usage. Try reconstructing the actual sequence of operations by backtracking through the DP table. This will give you deep insight into dynamic programming's power in string analysis.

## Navigation

{{ nav_grid(current_algorithm=config.extra.algorithm_key, current_family=config.extra.family, max_related=5) }}
