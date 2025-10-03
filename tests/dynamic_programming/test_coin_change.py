"""Tests for coin change algorithm."""

import pytest

from algokit.algorithms.dynamic_programming.coin_change import (
    coin_change,
    coin_change_count_ways,
    coin_change_with_coins,
)


@pytest.mark.unit
def test_coin_change_basic() -> None:
    """Test basic coin change calculations."""
    # Arrange - Set up test inputs for coin change calculations
    coins = [1, 3, 4]
    amount = 6
    expected_min_coins = 2  # 3 + 3 = 6 (2 coins)

    # Act - Calculate minimum coins needed
    result = coin_change(coins, amount)

    # Assert - Verify minimum coins calculation is correct
    assert result == expected_min_coins


@pytest.mark.unit
def test_coin_change_impossible() -> None:
    """Test coin change when amount cannot be made."""
    # Arrange - Set up coins that cannot make the target amount
    coins = [2]
    amount = 3

    # Act - Attempt to make impossible amount
    result = coin_change(coins, amount)

    # Assert - Verify that impossible case returns -1
    assert result == -1


@pytest.mark.unit
def test_coin_change_zero_amount() -> None:
    """Test coin change with zero amount."""
    # Arrange - Set up zero amount case
    coins = [1, 2, 5]
    amount = 0

    # Act - Calculate coins for zero amount
    result = coin_change(coins, amount)

    # Assert - Verify zero amount requires 0 coins
    assert result == 0


@pytest.mark.unit
def test_coin_change_single_coin() -> None:
    """Test coin change with single coin denomination."""
    # Arrange - Set up single coin case
    coins = [5]
    amount = 15
    expected_min_coins = 3  # 5 + 5 + 5 = 15 (3 coins)

    # Act - Calculate minimum coins with single denomination
    result = coin_change(coins, amount)

    # Assert - Verify single coin calculation is correct
    assert result == expected_min_coins


@pytest.mark.unit
def test_coin_change_empty_coins_list() -> None:
    """Test that coin_change raises ValueError for empty coins list."""
    # Arrange - Set up empty coins list
    coins = []
    amount = 5

    # Act & Assert - Verify that ValueError is raised for empty coins list
    with pytest.raises(ValueError, match="Coins list cannot be empty"):
        coin_change(coins, amount)


@pytest.mark.unit
def test_coin_change_negative_coins() -> None:
    """Test that coin_change raises ValueError for negative coin values."""
    # Arrange - Set up coins list with negative values
    coins = [1, -2, 5]
    amount = 5

    # Act & Assert - Verify that ValueError is raised for negative coins
    with pytest.raises(ValueError, match="All coins must be positive integers"):
        coin_change(coins, amount)


@pytest.mark.unit
def test_coin_change_zero_coins() -> None:
    """Test that coin_change raises ValueError for zero coin values."""
    # Arrange - Set up coins list with zero values
    coins = [1, 0, 5]
    amount = 5

    # Act & Assert - Verify that ValueError is raised for zero coins
    with pytest.raises(ValueError, match="All coins must be positive integers"):
        coin_change(coins, amount)


@pytest.mark.unit
def test_coin_change_negative_amount() -> None:
    """Test that coin_change raises ValueError for negative amount."""
    # Arrange - Set up negative amount
    coins = [1, 2, 5]
    amount = -5

    # Act & Assert - Verify that ValueError is raised for negative amount
    with pytest.raises(ValueError, match="Amount must be non-negative"):
        coin_change(coins, amount)


@pytest.mark.unit
def test_coin_change_with_coins_basic() -> None:
    """Test coin change with coins tracking for basic case."""
    # Arrange - Set up test inputs for coin tracking
    coins = [1, 3, 4]
    amount = 6
    expected_coins = [3, 3]  # 3 + 3 = 6

    # Act - Calculate actual coins used
    result = coin_change_with_coins(coins, amount)

    # Assert - Verify coins used are correct
    assert result == expected_coins


@pytest.mark.unit
def test_coin_change_with_coins_impossible() -> None:
    """Test coin change with coins tracking when impossible."""
    # Arrange - Set up impossible case
    coins = [2]
    amount = 3

    # Act - Attempt to get coins for impossible case
    result = coin_change_with_coins(coins, amount)

    # Assert - Verify impossible case returns empty list
    assert result == []


@pytest.mark.unit
def test_coin_change_with_coins_zero_amount() -> None:
    """Test coin change with coins tracking for zero amount."""
    # Arrange - Set up zero amount case
    coins = [1, 2, 5]
    amount = 0

    # Act - Calculate coins for zero amount
    result = coin_change_with_coins(coins, amount)

    # Assert - Verify zero amount uses no coins
    assert result == []


@pytest.mark.unit
def test_coin_change_count_ways_basic() -> None:
    """Test counting ways to make change."""
    # Arrange - Set up test inputs for counting ways
    coins = [1, 2, 3]
    amount = 4
    expected_ways = 4  # (1,1,1,1), (1,1,2), (1,3), (2,2)

    # Act - Count number of ways to make change
    result = coin_change_count_ways(coins, amount)

    # Assert - Verify number of ways is correct
    assert result == expected_ways


@pytest.mark.unit
def test_coin_change_count_ways_zero_amount() -> None:
    """Test counting ways to make zero amount."""
    # Arrange - Set up zero amount case
    coins = [1, 2, 5]
    amount = 0

    # Act - Count ways for zero amount
    result = coin_change_count_ways(coins, amount)

    # Assert - Verify zero amount has 1 way (no coins)
    assert result == 1


@pytest.mark.unit
def test_coin_change_count_ways_impossible() -> None:
    """Test counting ways when impossible."""
    # Arrange - Set up impossible case
    coins = [2]
    amount = 3

    # Act - Count ways for impossible case
    result = coin_change_count_ways(coins, amount)

    # Assert - Verify impossible case has 0 ways
    assert result == 0


@pytest.mark.unit
def test_coin_change_count_ways_empty_coins_list() -> None:
    """Test that coin_change_count_ways raises ValueError for empty coins list."""
    # Arrange - Set up empty coins list
    coins = []
    amount = 5

    # Act & Assert - Verify that ValueError is raised for empty coins list
    with pytest.raises(ValueError, match="Coins list cannot be empty"):
        coin_change_count_ways(coins, amount)


@pytest.mark.unit
def test_coin_change_with_coins_empty_coins_list() -> None:
    """Test that coin_change_with_coins raises ValueError for empty coins list."""
    # Arrange - Set up empty coins list
    coins = []
    amount = 5

    # Act & Assert - Verify that ValueError is raised for empty coins list
    with pytest.raises(ValueError, match="Coins list cannot be empty"):
        coin_change_with_coins(coins, amount)


@pytest.mark.unit
def test_coin_change_with_coins_negative_coins() -> None:
    """Test that coin_change_with_coins raises ValueError for negative coins."""
    # Arrange - Set up coins with negative values
    coins = [1, -5, 10]
    amount = 11

    # Act & Assert - Verify that ValueError is raised for negative coins
    with pytest.raises(ValueError, match="All coins must be positive integers"):
        coin_change_with_coins(coins, amount)


@pytest.mark.unit
def test_coin_change_with_coins_negative_amount() -> None:
    """Test that coin_change_with_coins raises ValueError for negative amount."""
    # Arrange - Set up negative amount
    coins = [1, 5, 10]
    amount = -11

    # Act & Assert - Verify that ValueError is raised for negative amount
    with pytest.raises(ValueError, match="Amount must be non-negative"):
        coin_change_with_coins(coins, amount)
