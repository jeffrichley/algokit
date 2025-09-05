---
algorithm_key: "fibonacci"
tags: [dynamic-programming, algorithms, optimization, fibonacci, recursion]
title: "Fibonacci Sequence"
family: "dynamic-programming"
---

{{ algorithm_card("fibonacci") }}

!!! abstract "Overview"
    The Fibonacci sequence is a classic problem in computer science and mathematics that demonstrates the power of dynamic programming. The sequence is defined as: each number is the sum of the two preceding ones, usually starting with 0 and 1. The sequence begins: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...

    While this can be solved with simple recursion, the naive approach has exponential time complexity due to repeated calculations. Dynamic programming provides an elegant solution that reduces this to linear time complexity.

## Mathematical Formulation

!!! math "Recurrence Relation"
    The Fibonacci sequence is defined by the recurrence relation:

    $$
    F(n) = \begin{cases}
    0 & \text{if } n = 0 \\
    1 & \text{if } n = 1 \\
    F(n-1) + F(n-2) & \text{if } n > 1
    \end{cases}
    $$

!!! success "Key Properties"
    - **Golden Ratio**: As n approaches infinity, $\frac{F(n+1)}{F(n)} \approx \phi = 1.618033988749...$
    - **Closed Form**: $F(n) = \frac{\phi^n - (-\phi)^{-n}}{\sqrt{5}}$ (Binet's formula)
    - **Growth Rate**: $F(n) = O(\phi^n)$

## Implementation Approaches

=== "Iterative (Recommended)"
    ```python
    def fibonacci(n: int) -> int:
        """Calculate the nth Fibonacci number using dynamic programming."""
        if n <= 1:
            return n

        # Initialize base cases
        prev, curr = 0, 1

        # Build up the sequence iteratively
        for _ in range(2, n + 1):
            prev, curr = curr, prev + curr

        return curr
    ```

=== "Memoized (Key Pattern)"
    ```python
    def fibonacci_memoized(n: int, memo: dict[int, int] = None) -> int:
        """Calculate the nth Fibonacci number using memoization."""
        if memo is None:
            memo = {}

        if n in memo:
            return memo[n]

        if n <= 1:
            return n

        memo[n] = fibonacci_memoized(n - 1, memo) + fibonacci_memoized(n - 2, memo)
        return memo[n]
    ```

=== "Generator Pattern"
    ```python
    def fibonacci_generator():
        """Generate Fibonacci numbers indefinitely."""
        prev, curr = 0, 1
        yield prev
        yield curr

        while True:
            prev, curr = curr, prev + curr
            yield curr
    ```

!!! tip "Complete Implementation"
    The full implementation with error handling, comprehensive testing, and additional variants is available in the source code:

    - **Main Implementation**: [`src/algokit/dynamic_programming/fibonacci.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/dynamic_programming/fibonacci.py)
    - **Tests**: [`tests/unit/dynamic_programming/test_fibonacci.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/dynamic_programming/test_fibonacci.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    | **Naive Recursion** | O(2^n) | O(n) | Exponential due to repeated calculations |
    | **Dynamic Programming** | O(n) | O(1) | Linear time with constant space |
    | **Memoization** | O(n) | O(n) | Linear time with space trade-off |

!!! warning "Performance Considerations"
    - **Naive recursion** becomes impractical for n > 40
    - **Iterative approach** is optimal for most use cases
    - **Memoization** useful when you need multiple Fibonacci numbers

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Computer Science"
        - **Algorithm Analysis**: Demonstrates dynamic programming concepts
        - **Recursion Examples**: Teaching recursive vs iterative approaches
        - **Performance Testing**: Benchmarking different implementation strategies
        - **Memory Management**: Understanding space-time trade-offs

    !!! grid-item "Finance & Economics"
        - **Fibonacci Retracements**: Technical analysis in trading
        - **Growth Models**: Population and economic growth patterns
        - **Risk Assessment**: Modeling exponential growth scenarios

    !!! grid-item "Biology & Nature"
        - **Population Growth**: Rabbit breeding models
        - **Spiral Patterns**: Shells, flowers, and natural structures
        - **Genetic Sequences**: DNA pattern analysis

    !!! grid-item "Design & Architecture"
        - **Golden Ratio**: Aesthetic proportions in design
        - **Spiral Structures**: Architectural and artistic applications
        - **Harmonic Balance**: Musical scales and compositions

!!! success "Educational Value"
    - **Dynamic Programming**: Perfect introduction to memoization
    - **Algorithm Design**: Shows importance of avoiding repeated work
    - **Mathematical Induction**: Demonstrates recursive problem-solving
    - **Optimization**: Illustrates space-time complexity trade-offs

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Textbooks"
        1. **Knuth, D. E.** (1997). *The Art of Computer Programming, Volume 1: Fundamental Algorithms*. Addison-Wesley. ISBN 0-201-89683-4.
        2. **Cormen, T. H., et al.** (2009). *Introduction to Algorithms*. MIT Press. ISBN 978-0-262-03384-8.

    !!! grid-item "Historical & Cultural"
        3. **Livio, M.** (2002). *The Golden Ratio: The Story of Phi, the World's Most Astonishing Number*. Broadway Books. ISBN 978-0-7679-0816-0.
        4. **Devlin, K.** (2011). *The Man of Numbers: Fibonacci's Arithmetic Revolution*. Walker & Company. ISBN 978-0-8027-7812-3.

    !!! grid-item "Online Resources"
        5. [Fibonacci Number - Wikipedia](https://en.wikipedia.org/wiki/Fibonacci_number)
        6. [Dynamic Programming - GeeksforGeeks](https://www.geeksforgeeks.org/dynamic-programming/)
        7. [Fibonacci Sequence - MathWorld](https://mathworld.wolfram.com/FibonacciNumber.html)

    !!! grid-item "Implementation & Practice"
        8. [Python Official Documentation](https://docs.python.org/3/)
        9. [Python Performance Tips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
        10. [Dynamic Programming Patterns](https://leetcode.com/discuss/general-discussion/458695/dynamic-programming-patterns)

!!! tip "Interactive Learning"
    Try implementing the different approaches yourself! Start with the naive recursive version, then optimize it with memoization, and finally implement the iterative solution. This progression will give you deep insight into dynamic programming principles.

## Navigation

{{ nav_grid(current_algorithm="fibonacci", current_family="dynamic-programming", max_related=5) }}
