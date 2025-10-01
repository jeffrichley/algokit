"""Shared test configurations."""

import pytest


@pytest.fixture
def test_timeouts():
    """Test timeout configurations."""
    return {
        "unit": 5,  # 5 seconds for unit tests
        "integration": 30,  # 30 seconds for integration tests
        "benchmark": 600,  # 10 minutes for benchmark tests
    }


@pytest.fixture
def test_environments_config():
    """Test environments configuration."""
    return {
        "fast": {
            "CartPole-v1": {"max_episodes": 2, "max_steps": 10},
            "FrozenLake-v1": {"max_episodes": 2, "max_steps": 10},
        },
        "medium": {
            "MountainCar-v0": {"max_episodes": 5, "max_steps": 50},
            "Acrobot-v1": {"max_episodes": 5, "max_steps": 50},
        },
        "slow": {
            "LunarLander-v2": {"max_episodes": 10, "max_steps": 100},
            "BipedalWalker-v3": {"max_episodes": 10, "max_steps": 100},
        },
    }


@pytest.fixture
def algorithm_test_configs():
    """Algorithm-specific test configurations."""
    return {
        "q_learning": {
            "unit": {"episodes": 1, "max_steps": 5},
            "integration": {"episodes": 2, "max_steps": 10},
        },
    }


@pytest.fixture
def test_output_config():
    """Test output configuration."""
    return {
        "save_models": False,  # Don't save models in tests
        "save_plots": False,  # Don't save plots in tests
        "verbose": False,  # Minimal output in tests
        "render": False,  # No rendering in tests
    }
