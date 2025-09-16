"""Base algorithm fixtures for RL testing."""

from dataclasses import dataclass
from typing import Any, Protocol

import pytest


@dataclass
class TestAlgorithmConfig:
    """Configuration for testing algorithms."""

    episodes: int = 1
    max_steps_per_episode: int = 5
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    epsilon: float = 0.1
    verbose: bool = False
    render: bool = False


class RLAlgorithm(Protocol):
    """Protocol for RL algorithms."""

    def train(self, **kwargs) -> dict[str, Any]: ...
    def test(self, **kwargs) -> dict[str, Any]: ...
    def demo(self, **kwargs) -> dict[str, Any]: ...


@pytest.fixture
def fast_test_config():
    """Fast test configuration."""
    return TestAlgorithmConfig(
        episodes=1,
        max_steps_per_episode=5,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.1,
        verbose=False,
        render=False,
    )


@pytest.fixture
def integration_test_config():
    """Integration test configuration."""
    return TestAlgorithmConfig(
        episodes=2,
        max_steps_per_episode=10,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.1,
        verbose=False,
        render=False,
    )


@pytest.fixture
def performance_test_config():
    """Performance test configuration."""
    return TestAlgorithmConfig(
        episodes=100,
        max_steps_per_episode=200,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.1,
        verbose=True,
        render=False,
    )


@pytest.fixture
def benchmark_test_config():
    """Benchmark test configuration."""
    return TestAlgorithmConfig(
        episodes=50,
        max_steps_per_episode=100,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.1,
        verbose=False,
        render=False,
    )
