import pytest

from algokit.classic_dp import fibonacci


def test_fibonacci_basic() -> None:
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(10) == 55


def test_fibonacci_negative() -> None:
    with pytest.raises(ValueError):
        fibonacci(-1)
