"""Longest Common Subsequence (LCS) implementation using dynamic programming."""


def longest_common_subsequence(text1: str, text2: str) -> int:
    """Find the length of the longest common subsequence between two strings.

    A subsequence is a sequence that appears in the same relative order,
    but not necessarily contiguous. For example, "ace" is a subsequence of "abcde".

    Args:
        text1: First input string.
        text2: Second input string.

    Returns:
        Length of the longest common subsequence.

    Examples:
        >>> longest_common_subsequence("abcde", "ace")
        3
        >>> longest_common_subsequence("abc", "abc")
        3
        >>> longest_common_subsequence("abc", "def")
        0
    """
    if not text1 or not text2:
        return 0

    m, n = len(text1), len(text2)

    # Initialize DP table: dp[i][j] = LCS length of text1[:i] and text2[:j]
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                # Characters match, extend LCS
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # Characters don't match, take maximum from previous states
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def longest_common_subsequence_string(text1: str, text2: str) -> str:
    """Find the actual longest common subsequence string between two strings.

    This function extends the basic LCS problem to return the actual
    subsequence string, not just its length.

    Args:
        text1: First input string.
        text2: Second input string.

    Returns:
        The longest common subsequence string.

    Examples:
        >>> longest_common_subsequence_string("abcde", "ace")
        "ace"
        >>> longest_common_subsequence_string("abc", "def")
        ""
    """
    if not text1 or not text2:
        return ""

    m, n = len(text1), len(text2)

    # Initialize DP table
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Reconstruct the LCS string
    lcs = []
    i, j = m, n

    while i > 0 and j > 0:
        if text1[i - 1] == text2[j - 1]:
            lcs.append(text1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return "".join(reversed(lcs))


def longest_common_subsequence_space_optimized(text1: str, text2: str) -> int:
    """Find LCS length using space-optimized dynamic programming.

    This implementation uses only O(min(m, n)) space instead of O(m * n)
    by using only the previous row in the DP table.

    Args:
        text1: First input string.
        text2: Second input string.

    Returns:
        Length of the longest common subsequence.
    """
    if not text1 or not text2:
        return 0

    # Ensure text1 is the shorter string for space optimization
    if len(text1) > len(text2):
        text1, text2 = text2, text1

    m, n = len(text1), len(text2)

    # Use only previous row
    prev_dp = [0 for _ in range(m + 1)]

    for j in range(1, n + 1):
        curr_dp = [0 for _ in range(m + 1)]
        for i in range(1, m + 1):
            if text1[i - 1] == text2[j - 1]:
                curr_dp[i] = prev_dp[i - 1] + 1
            else:
                curr_dp[i] = max(prev_dp[i], curr_dp[i - 1])
        prev_dp = curr_dp

    return prev_dp[m]


def longest_common_subsequence_memoized(text1: str, text2: str) -> int:
    """Find LCS length using memoized recursion.

    This implementation uses top-down dynamic programming with memoization
    as an alternative to the bottom-up approach.

    Args:
        text1: First input string.
        text2: Second input string.

    Returns:
        Length of the longest common subsequence.
    """
    if not text1 or not text2:
        return 0

    # Memoization table
    memo: dict[tuple[int, int], int] = {}

    def lcs_rec(i: int, j: int) -> int:
        if i == 0 or j == 0:
            return 0

        if (i, j) in memo:
            return memo[(i, j)]

        if text1[i - 1] == text2[j - 1]:
            result = lcs_rec(i - 1, j - 1) + 1
        else:
            result = max(lcs_rec(i - 1, j), lcs_rec(i, j - 1))

        memo[(i, j)] = result
        return result

    return lcs_rec(len(text1), len(text2))


def longest_increasing_subsequence(nums: list[int]) -> int:
    """Find the length of the longest increasing subsequence.

    This is a related problem that can be solved using similar DP techniques.
    A subsequence is increasing if for every i < j, nums[i] < nums[j].

    Args:
        nums: List of integers.

    Returns:
        Length of the longest increasing subsequence.

    Examples:
        >>> longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18])
        4
        >>> longest_increasing_subsequence([0, 1, 0, 3, 2, 3])
        4
    """
    if not nums:
        return 0

    n = len(nums)

    # dp[i] = length of LIS ending at index i
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


def longest_increasing_subsequence_optimized(nums: list[int]) -> int:
    """Find LIS length using binary search optimization.

    This implementation uses binary search to achieve O(n log n) time complexity
    instead of the O(n²) naive approach.

    Args:
        nums: List of integers.

    Returns:
        Length of the longest increasing subsequence.
    """
    if not nums:
        return 0

    # tails[i] = smallest tail of all increasing subsequences of length i+1
    tails: list[int] = []

    for num in nums:
        # Binary search for the position to insert/replace
        left, right = 0, len(tails)
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid

        # If num is larger than all tails, append it
        if left == len(tails):
            tails.append(num)
        else:
            # Replace the first element that is >= num
            tails[left] = num

    return len(tails)
