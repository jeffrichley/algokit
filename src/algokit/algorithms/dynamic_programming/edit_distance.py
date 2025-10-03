"""Edit Distance (Levenshtein Distance) implementation using dynamic programming."""


def edit_distance(str1: str, str2: str) -> int:
    """Calculate the minimum edit distance between two strings.

    Edit distance is the minimum number of single-character edits
    (insertions, deletions, or substitutions) required to change
    one string into another.

    Args:
        str1: First input string.
        str2: Second input string.

    Returns:
        Minimum edit distance between the strings.

    Examples:
        >>> edit_distance("kitten", "sitting")
        3
        >>> edit_distance("abc", "abc")
        0
        >>> edit_distance("abc", "def")
        3
    """
    m, n = len(str1), len(str2)

    # Initialize DP table: dp[i][j] = edit distance between str1[:i] and str2[:j]
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i  # Delete all characters from str1
    for j in range(n + 1):
        dp[0][j] = j  # Insert all characters from str2

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                # Characters match, no edit needed
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Take minimum of three operations
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # Delete from str1
                    dp[i][j - 1],  # Insert into str1
                    dp[i - 1][j - 1],  # Substitute
                )

    return dp[m][n]


def edit_distance_space_optimized(str1: str, str2: str) -> int:
    """Calculate edit distance using space-optimized dynamic programming.

    This implementation uses only O(min(m, n)) space instead of O(m * n)
    by using a single array and updating it efficiently.

    Args:
        str1: First input string.
        str2: Second input string.

    Returns:
        Minimum edit distance between the strings.
    """
    m, n = len(str1), len(str2)

    # Ensure str1 is the shorter string for space optimization
    if m > n:
        str1, str2 = str2, str1
        m, n = n, m

    # Use single array for space optimization
    dp = list(range(m + 1))

    for j in range(1, n + 1):
        # Store the previous value before it gets overwritten
        prev_val = dp[0]  # This will be j-1
        dp[0] = j  # Update first element

        for i in range(1, m + 1):
            temp = dp[i]  # Store current value before overwriting
            if str1[i - 1] == str2[j - 1]:
                dp[i] = prev_val
            else:
                dp[i] = 1 + min(
                    dp[i],  # Delete (current row, same column)
                    dp[i - 1],  # Insert (current row, previous column)
                    prev_val,  # Substitute (previous row, previous column)
                )
            prev_val = temp  # Update prev_val for next iteration

    return dp[m]


def edit_distance_with_operations(str1: str, str2: str) -> tuple[int, list[str]]:
    """Calculate edit distance and return the sequence of operations.

    This function extends the basic edit distance to return the actual
    sequence of operations (insert, delete, substitute) that transforms
    str1 into str2.

    Args:
        str1: First input string.
        str2: Second input string.

    Returns:
        Tuple of (edit_distance, operations_list) where operations_list
        contains the sequence of operations.

    Examples:
        >>> distance, ops = edit_distance_with_operations("kitten", "sitting")
        >>> distance
        3
        >>> ops  # Example operations: ['substitute', 'substitute', 'insert']
    """
    m, n = len(str1), len(str2)

    # Initialize DP table and operation tracking
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    operations = [["" for _ in range(n + 1)] for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
        operations[i][0] = "delete" if i > 0 else ""
    for j in range(n + 1):
        dp[0][j] = j
        operations[0][j] = "insert" if j > 0 else ""

    # Fill DP table with operation tracking
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                operations[i][j] = "match"
            else:
                delete_cost = dp[i - 1][j] + 1
                insert_cost = dp[i][j - 1] + 1
                substitute_cost = dp[i - 1][j - 1] + 1

                if delete_cost <= insert_cost and delete_cost <= substitute_cost:
                    dp[i][j] = delete_cost
                    operations[i][j] = "delete"
                elif insert_cost <= substitute_cost:
                    dp[i][j] = insert_cost
                    operations[i][j] = "insert"
                else:
                    dp[i][j] = substitute_cost
                    operations[i][j] = "substitute"

    # Reconstruct operations sequence
    ops_list = []
    i, j = m, n

    while i > 0 or j > 0:
        op = operations[i][j]
        ops_list.append(op)

        if op == "match" or op == "substitute":
            i -= 1
            j -= 1
        elif op == "delete":
            i -= 1
        elif op == "insert":
            j -= 1

    ops_list.reverse()
    return dp[m][n], ops_list


def edit_distance_memoized(str1: str, str2: str) -> int:
    """Calculate edit distance using memoized recursion.

    This implementation uses top-down dynamic programming with memoization
    as an alternative to the bottom-up approach.

    Args:
        str1: First input string.
        str2: Second input string.

    Returns:
        Minimum edit distance between the strings.
    """
    # Memoization table
    memo: dict[tuple[int, int], int] = {}

    def edit_dist_rec(i: int, j: int) -> int:
        if i == 0:
            return j
        if j == 0:
            return i

        if (i, j) in memo:
            return memo[(i, j)]

        if str1[i - 1] == str2[j - 1]:
            result = edit_dist_rec(i - 1, j - 1)
        else:
            result = 1 + min(
                edit_dist_rec(i - 1, j),  # Delete
                edit_dist_rec(i, j - 1),  # Insert
                edit_dist_rec(i - 1, j - 1),  # Substitute
            )

        memo[(i, j)] = result
        return result

    return edit_dist_rec(len(str1), len(str2))


def hamming_distance(str1: str, str2: str) -> int:
    """Calculate the Hamming distance between two strings of equal length.

    Hamming distance is the number of positions at which the corresponding
    characters are different. This is a simpler version of edit distance
    that only allows substitutions.

    Args:
        str1: First input string.
        str2: Second input string.

    Returns:
        Hamming distance between the strings.

    Raises:
        ValueError: If strings have different lengths.

    Examples:
        >>> hamming_distance("abc", "abx")
        1
        >>> hamming_distance("abc", "abc")
        0
    """
    if len(str1) != len(str2):
        raise ValueError("Strings must have the same length for Hamming distance")

    return sum(c1 != c2 for c1, c2 in zip(str1, str2))


def longest_common_substring(str1: str, str2: str) -> int:
    """Find the length of the longest common substring between two strings.

    A substring is a contiguous sequence of characters within a string.
    This is different from subsequence as it requires contiguity.

    Args:
        str1: First input string.
        str2: Second input string.

    Returns:
        Length of the longest common substring.

    Examples:
        >>> longest_common_substring("abcde", "cdefg")
        2
        >>> longest_common_substring("abc", "def")
        0
    """
    m, n = len(str1), len(str2)

    if m == 0 or n == 0:
        return 0

    # Initialize DP table
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    max_length = 0

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_length = max(max_length, dp[i][j])
            else:
                dp[i][j] = 0  # Reset count for non-matching characters

    return max_length


def longest_common_substring_string(str1: str, str2: str) -> str:
    """Find the actual longest common substring between two strings.

    Args:
        str1: First input string.
        str2: Second input string.

    Returns:
        The longest common substring.

    Examples:
        >>> longest_common_substring_string("abcde", "cdefg")
        "cd"
    """
    m, n = len(str1), len(str2)

    if m == 0 or n == 0:
        return ""

    # Initialize DP table and tracking variables
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    max_length = 0
    end_pos = 0

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_pos = i
            else:
                dp[i][j] = 0

    if max_length == 0:
        return ""

    return str1[end_pos - max_length : end_pos]
