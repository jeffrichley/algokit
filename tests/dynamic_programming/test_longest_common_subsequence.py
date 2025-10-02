"""Tests for longest common subsequence algorithm."""

import pytest

from algokit.algorithms.dynamic_programming.longest_common_subsequence import (
    longest_common_subsequence,
    longest_common_subsequence_memoized,
    longest_common_subsequence_space_optimized,
    longest_common_subsequence_string,
    longest_increasing_subsequence,
    longest_increasing_subsequence_optimized,
)


@pytest.mark.unit
def test_lcs_basic() -> None:
    """Test basic longest common subsequence calculation."""
    # Arrange - Set up test inputs for LCS calculation
    text1 = "abcde"
    text2 = "ace"
    expected_length = 3  # "ace" is the LCS

    # Act - Calculate LCS length
    result = longest_common_subsequence(text1, text2)

    # Assert - Verify LCS length is correct
    assert result == expected_length


@pytest.mark.unit
def test_lcs_identical_strings() -> None:
    """Test LCS with identical strings."""
    # Arrange - Set up identical strings
    text1 = "abc"
    text2 = "abc"
    expected_length = 3  # Entire string is LCS

    # Act - Calculate LCS length for identical strings
    result = longest_common_subsequence(text1, text2)

    # Assert - Verify LCS length equals string length
    assert result == expected_length


@pytest.mark.unit
def test_lcs_no_common_subsequence() -> None:
    """Test LCS with no common subsequence."""
    # Arrange - Set up strings with no common characters
    text1 = "abc"
    text2 = "def"
    expected_length = 0  # No common subsequence

    # Act - Calculate LCS length for strings with no common characters
    result = longest_common_subsequence(text1, text2)

    # Assert - Verify LCS length is zero
    assert result == expected_length


@pytest.mark.unit
def test_lcs_empty_strings() -> None:
    """Test LCS with empty strings."""
    # Arrange - Set up empty strings
    text1 = ""
    text2 = ""
    expected_length = 0  # No LCS for empty strings

    # Act - Calculate LCS length for empty strings
    result = longest_common_subsequence(text1, text2)

    # Assert - Verify LCS length is zero
    assert result == expected_length


@pytest.mark.unit
def test_lcs_one_empty_string() -> None:
    """Test LCS with one empty string."""
    # Arrange - Set up one empty string
    text1 = "abc"
    text2 = ""
    expected_length = 0  # No LCS when one string is empty

    # Act - Calculate LCS length with one empty string
    result = longest_common_subsequence(text1, text2)

    # Assert - Verify LCS length is zero
    assert result == expected_length


@pytest.mark.unit
def test_lcs_single_character() -> None:
    """Test LCS with single character strings."""
    # Arrange - Set up single character strings
    text1 = "a"
    text2 = "a"
    expected_length = 1  # Single character LCS

    # Act - Calculate LCS length for single characters
    result = longest_common_subsequence(text1, text2)

    # Assert - Verify LCS length is one
    assert result == expected_length


@pytest.mark.unit
def test_lcs_string_basic() -> None:
    """Test LCS string reconstruction for basic case."""
    # Arrange - Set up test inputs for LCS string reconstruction
    text1 = "abcde"
    text2 = "ace"
    expected_lcs = "ace"

    # Act - Calculate actual LCS string
    result = longest_common_subsequence_string(text1, text2)

    # Assert - Verify LCS string is correct
    assert result == expected_lcs


@pytest.mark.unit
def test_lcs_string_no_common() -> None:
    """Test LCS string reconstruction with no common subsequence."""
    # Arrange - Set up strings with no common characters
    text1 = "abc"
    text2 = "def"
    expected_lcs = ""

    # Act - Calculate LCS string for strings with no common characters
    result = longest_common_subsequence_string(text1, text2)

    # Assert - Verify LCS string is empty
    assert result == expected_lcs


@pytest.mark.unit
def test_lcs_string_empty_strings() -> None:
    """Test LCS string reconstruction with empty strings."""
    # Arrange - Set up empty strings
    text1 = ""
    text2 = ""
    expected_lcs = ""

    # Act - Calculate LCS string for empty strings
    result = longest_common_subsequence_string(text1, text2)

    # Assert - Verify LCS string is empty
    assert result == expected_lcs


@pytest.mark.unit
def test_lcs_space_optimized_basic() -> None:
    """Test space-optimized LCS calculation."""
    # Arrange - Set up test inputs for space-optimized LCS
    text1 = "abcde"
    text2 = "ace"
    expected_length = 3

    # Act - Calculate LCS length using space-optimized approach
    result = longest_common_subsequence_space_optimized(text1, text2)

    # Assert - Verify space-optimized result matches expected
    assert result == expected_length


@pytest.mark.unit
def test_lcs_space_optimized_empty_strings() -> None:
    """Test space-optimized LCS with empty strings."""
    # Arrange - Set up empty strings
    text1 = ""
    text2 = ""
    expected_length = 0

    # Act - Calculate LCS length using space-optimized approach
    result = longest_common_subsequence_space_optimized(text1, text2)

    # Assert - Verify space-optimized result for empty strings
    assert result == expected_length


@pytest.mark.unit
def test_lcs_memoized_basic() -> None:
    """Test memoized LCS calculation."""
    # Arrange - Set up test inputs for memoized LCS
    text1 = "abcde"
    text2 = "ace"
    expected_length = 3

    # Act - Calculate LCS length using memoized approach
    result = longest_common_subsequence_memoized(text1, text2)

    # Assert - Verify memoized result matches expected
    assert result == expected_length


@pytest.mark.unit
def test_lcs_memoized_empty_strings() -> None:
    """Test memoized LCS with empty strings."""
    # Arrange - Set up empty strings
    text1 = ""
    text2 = ""
    expected_length = 0

    # Act - Calculate LCS length using memoized approach
    result = longest_common_subsequence_memoized(text1, text2)

    # Assert - Verify memoized result for empty strings
    assert result == expected_length


@pytest.mark.unit
def test_longest_increasing_subsequence_basic() -> None:
    """Test basic longest increasing subsequence calculation."""
    # Arrange - Set up test inputs for LIS calculation
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    expected_length = 4  # [2, 3, 7, 18] or [2, 3, 7, 101]

    # Act - Calculate LIS length
    result = longest_increasing_subsequence(nums)

    # Assert - Verify LIS length is correct
    assert result == expected_length


@pytest.mark.unit
def test_longest_increasing_subsequence_empty() -> None:
    """Test LIS with empty list."""
    # Arrange - Set up empty list
    nums = []
    expected_length = 0

    # Act - Calculate LIS length for empty list
    result = longest_increasing_subsequence(nums)

    # Assert - Verify LIS length is zero
    assert result == expected_length


@pytest.mark.unit
def test_longest_increasing_subsequence_single_element() -> None:
    """Test LIS with single element."""
    # Arrange - Set up single element list
    nums = [5]
    expected_length = 1

    # Act - Calculate LIS length for single element
    result = longest_increasing_subsequence(nums)

    # Assert - Verify LIS length is one
    assert result == expected_length


@pytest.mark.unit
def test_longest_increasing_subsequence_decreasing() -> None:
    """Test LIS with decreasing sequence."""
    # Arrange - Set up decreasing sequence
    nums = [5, 4, 3, 2, 1]
    expected_length = 1  # Each element forms LIS of length 1

    # Act - Calculate LIS length for decreasing sequence
    result = longest_increasing_subsequence(nums)

    # Assert - Verify LIS length is one
    assert result == expected_length


@pytest.mark.unit
def test_longest_increasing_subsequence_optimized_basic() -> None:
    """Test optimized LIS calculation."""
    # Arrange - Set up test inputs for optimized LIS
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    expected_length = 4

    # Act - Calculate LIS length using optimized approach
    result = longest_increasing_subsequence_optimized(nums)

    # Assert - Verify optimized result matches expected
    assert result == expected_length


@pytest.mark.unit
def test_longest_increasing_subsequence_optimized_empty() -> None:
    """Test optimized LIS with empty list."""
    # Arrange - Set up empty list
    nums = []
    expected_length = 0

    # Act - Calculate LIS length using optimized approach
    result = longest_increasing_subsequence_optimized(nums)

    # Assert - Verify optimized result for empty list
    assert result == expected_length


@pytest.mark.unit
def test_longest_increasing_subsequence_optimized_single_element() -> None:
    """Test optimized LIS with single element."""
    # Arrange - Set up single element list
    nums = [5]
    expected_length = 1

    # Act - Calculate LIS length using optimized approach
    result = longest_increasing_subsequence_optimized(nums)

    # Assert - Verify optimized result for single element
    assert result == expected_length
