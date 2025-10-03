"""Tests for edit distance (Levenshtein distance) algorithm."""

import pytest

from algokit.algorithms.dynamic_programming.edit_distance import (
    edit_distance,
    edit_distance_memoized,
    edit_distance_space_optimized,
    edit_distance_with_operations,
    hamming_distance,
    longest_common_substring,
    longest_common_substring_string,
)


@pytest.mark.unit
def test_edit_distance_basic() -> None:
    """Test basic edit distance calculation."""
    # Arrange - Set up test inputs for edit distance calculation
    str1 = "kitten"
    str2 = "sitting"
    expected_distance = 3  # kitten -> sitten -> sitting

    # Act - Calculate edit distance
    result = edit_distance(str1, str2)

    # Assert - Verify edit distance is correct
    assert result == expected_distance


@pytest.mark.unit
def test_edit_distance_identical_strings() -> None:
    """Test edit distance with identical strings."""
    # Arrange - Set up identical strings
    str1 = "abc"
    str2 = "abc"
    expected_distance = 0  # No edits needed

    # Act - Calculate edit distance for identical strings
    result = edit_distance(str1, str2)

    # Assert - Verify edit distance is zero
    assert result == expected_distance


@pytest.mark.unit
def test_edit_distance_completely_different() -> None:
    """Test edit distance with completely different strings."""
    # Arrange - Set up completely different strings
    str1 = "abc"
    str2 = "def"
    expected_distance = 3  # All characters need substitution

    # Act - Calculate edit distance for different strings
    result = edit_distance(str1, str2)

    # Assert - Verify edit distance equals string length
    assert result == expected_distance


@pytest.mark.unit
def test_edit_distance_empty_strings() -> None:
    """Test edit distance with empty strings."""
    # Arrange - Set up empty strings
    str1 = ""
    str2 = ""
    expected_distance = 0  # No edits needed

    # Act - Calculate edit distance for empty strings
    result = edit_distance(str1, str2)

    # Assert - Verify edit distance is zero
    assert result == expected_distance


@pytest.mark.unit
def test_edit_distance_one_empty_string() -> None:
    """Test edit distance with one empty string."""
    # Arrange - Set up one empty string
    str1 = "abc"
    str2 = ""
    expected_distance = 3  # Need to delete all characters

    # Act - Calculate edit distance with one empty string
    result = edit_distance(str1, str2)

    # Assert - Verify edit distance equals length of non-empty string
    assert result == expected_distance


@pytest.mark.unit
def test_edit_distance_single_characters() -> None:
    """Test edit distance with single character strings."""
    # Arrange - Set up single character strings
    str1 = "a"
    str2 = "b"
    expected_distance = 1  # One substitution needed

    # Act - Calculate edit distance for single characters
    result = edit_distance(str1, str2)

    # Assert - Verify edit distance is one
    assert result == expected_distance


@pytest.mark.unit
def test_edit_distance_space_optimized_basic() -> None:
    """Test space-optimized edit distance calculation."""
    # Arrange - Set up test inputs for space-optimized edit distance
    str1 = "kitten"
    str2 = "sitting"
    expected_distance = 3

    # Act - Calculate edit distance using space-optimized approach
    result = edit_distance_space_optimized(str1, str2)

    # Assert - Verify space-optimized result matches expected
    assert result == expected_distance


@pytest.mark.unit
def test_edit_distance_space_optimized_empty_strings() -> None:
    """Test space-optimized edit distance with empty strings."""
    # Arrange - Set up empty strings
    str1 = ""
    str2 = ""
    expected_distance = 0

    # Act - Calculate edit distance using space-optimized approach
    result = edit_distance_space_optimized(str1, str2)

    # Assert - Verify space-optimized result for empty strings
    assert result == expected_distance


@pytest.mark.unit
def test_edit_distance_memoized_basic() -> None:
    """Test memoized edit distance calculation."""
    # Arrange - Set up test inputs for memoized edit distance
    str1 = "kitten"
    str2 = "sitting"
    expected_distance = 3

    # Act - Calculate edit distance using memoized approach
    result = edit_distance_memoized(str1, str2)

    # Assert - Verify memoized result matches expected
    assert result == expected_distance


@pytest.mark.unit
def test_edit_distance_memoized_empty_strings() -> None:
    """Test memoized edit distance with empty strings."""
    # Arrange - Set up empty strings
    str1 = ""
    str2 = ""
    expected_distance = 0

    # Act - Calculate edit distance using memoized approach
    result = edit_distance_memoized(str1, str2)

    # Assert - Verify memoized result for empty strings
    assert result == expected_distance


@pytest.mark.unit
def test_edit_distance_with_operations_basic() -> None:
    """Test edit distance with operations tracking."""
    # Arrange - Set up test inputs for edit distance with operations
    str1 = "kitten"
    str2 = "sitting"
    expected_distance = 3

    # Act - Calculate edit distance with operations
    distance, operations = edit_distance_with_operations(str1, str2)

    # Assert - Verify distance and operations are correct
    assert distance == expected_distance
    # Operations should include all characters processed, not just edit operations
    assert len(operations) >= expected_distance


@pytest.mark.unit
def test_edit_distance_with_operations_identical() -> None:
    """Test edit distance with operations for identical strings."""
    # Arrange - Set up identical strings
    str1 = "abc"
    str2 = "abc"
    expected_distance = 0

    # Act - Calculate edit distance with operations for identical strings
    distance, operations = edit_distance_with_operations(str1, str2)

    # Assert - Verify distance is zero and operations are minimal
    assert distance == expected_distance
    assert len(operations) == 3  # Should have 3 "match" operations


@pytest.mark.unit
def test_hamming_distance_basic() -> None:
    """Test basic Hamming distance calculation."""
    # Arrange - Set up test inputs for Hamming distance
    str1 = "abc"
    str2 = "abx"
    expected_distance = 1  # Only one character differs

    # Act - Calculate Hamming distance
    result = hamming_distance(str1, str2)

    # Assert - Verify Hamming distance is correct
    assert result == expected_distance


@pytest.mark.unit
def test_hamming_distance_identical() -> None:
    """Test Hamming distance with identical strings."""
    # Arrange - Set up identical strings
    str1 = "abc"
    str2 = "abc"
    expected_distance = 0  # No differences

    # Act - Calculate Hamming distance for identical strings
    result = hamming_distance(str1, str2)

    # Assert - Verify Hamming distance is zero
    assert result == expected_distance


@pytest.mark.unit
def test_hamming_distance_different_lengths() -> None:
    """Test that Hamming distance raises ValueError for different lengths."""
    # Arrange - Set up strings of different lengths
    str1 = "abc"
    str2 = "abcd"

    # Act & Assert - Verify that ValueError is raised for different lengths
    with pytest.raises(ValueError, match="Strings must have the same length"):
        hamming_distance(str1, str2)


@pytest.mark.unit
def test_longest_common_substring_basic() -> None:
    """Test basic longest common substring calculation."""
    # Arrange - Set up test inputs for longest common substring
    str1 = "abcde"
    str2 = "cdefg"
    expected_length = 3  # "cde" is the longest common substring

    # Act - Calculate longest common substring length
    result = longest_common_substring(str1, str2)

    # Assert - Verify longest common substring length is correct
    assert result == expected_length


@pytest.mark.unit
def test_longest_common_substring_no_common() -> None:
    """Test longest common substring with no common characters."""
    # Arrange - Set up strings with no common characters
    str1 = "abc"
    str2 = "def"
    expected_length = 0  # No common substring

    # Act - Calculate longest common substring for strings with no common characters
    result = longest_common_substring(str1, str2)

    # Assert - Verify longest common substring length is zero
    assert result == expected_length


@pytest.mark.unit
def test_longest_common_substring_empty_strings() -> None:
    """Test longest common substring with empty strings."""
    # Arrange - Set up empty strings
    str1 = ""
    str2 = ""
    expected_length = 0  # No common substring for empty strings

    # Act - Calculate longest common substring for empty strings
    result = longest_common_substring(str1, str2)

    # Assert - Verify longest common substring length is zero
    assert result == expected_length


@pytest.mark.unit
def test_longest_common_substring_string_basic() -> None:
    """Test longest common substring string reconstruction."""
    # Arrange - Set up test inputs for longest common substring string
    str1 = "abcde"
    str2 = "cdefg"
    expected_substring = "cde"  # Longest common substring

    # Act - Calculate actual longest common substring
    result = longest_common_substring_string(str1, str2)

    # Assert - Verify longest common substring is correct
    assert result == expected_substring


@pytest.mark.unit
def test_longest_common_substring_string_no_common() -> None:
    """Test longest common substring string with no common characters."""
    # Arrange - Set up strings with no common characters
    str1 = "abc"
    str2 = "def"
    expected_substring = ""  # No common substring

    # Act - Calculate longest common substring for strings with no common characters
    result = longest_common_substring_string(str1, str2)

    # Assert - Verify longest common substring is empty
    assert result == expected_substring


@pytest.mark.unit
def test_longest_common_substring_string_empty_strings() -> None:
    """Test longest common substring string with empty strings."""
    # Arrange - Set up empty strings
    str1 = ""
    str2 = ""
    expected_substring = ""  # No common substring for empty strings

    # Act - Calculate longest common substring for empty strings
    result = longest_common_substring_string(str1, str2)

    # Assert - Verify longest common substring is empty
    assert result == expected_substring
