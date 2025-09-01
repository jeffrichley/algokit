"""Example unit tests for Algorithm Kit."""

import pytest

from algokit import main_function


@pytest.mark.unit
def test_main_function_returns_processed_string() -> None:
    """Test that main_function returns correctly formatted string."""
    # Arrange
    input_data = "test input"

    # Act
    result = main_function(input_data)

    # Assert
    assert result == "Processed: test input"


@pytest.mark.unit
def test_main_function_raises_value_error_on_empty_input() -> None:
    """Test that main_function raises ValueError when input is empty."""
    # Arrange
    input_data = ""

    # Act & Assert
    with pytest.raises(ValueError, match="input_data cannot be empty"):
        main_function(input_data)


@pytest.mark.unit
def test_main_function_raises_value_error_on_none_input() -> None:
    """Test that main_function raises ValueError when input is None."""
    # Arrange
    input_data: str | None = None

    # Act & Assert
    with pytest.raises(ValueError, match="input_data cannot be empty"):
        main_function(input_data)


@pytest.mark.unit
def test_main_function_handles_whitespace_input() -> None:
    """Test that main_function processes whitespace-only input correctly."""
    # Arrange
    input_data = "   "

    # Act
    result = main_function(input_data)

    # Assert
    assert result == "Processed:    "


@pytest.mark.unit
@pytest.mark.parametrize(
    "input_data,expected",
    [
        ("hello", "Processed: hello"),
        ("world", "Processed: world"),
        ("123", "Processed: 123"),
        ("special chars!@#", "Processed: special chars!@#"),
    ],
)
def test_main_function_with_various_inputs(input_data: str, expected: str) -> None:
    """Test main_function with various input types using parametrization."""
    # Act
    result = main_function(input_data)

    # Assert
    assert result == expected
