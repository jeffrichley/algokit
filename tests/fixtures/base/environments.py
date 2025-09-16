"""Base environment fixtures for RL testing."""

from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest


def mock_environment_factory(
    state_space_size: int = 4,
    action_space_size: int = 2,
    episode_length: int = 5,
    deterministic: bool = True,
):
    """Factory for creating mock environments.

    Args:
        state_space_size: Number of discrete states
        action_space_size: Number of discrete actions
        episode_length: Length of episodes
        deterministic: Whether environment is deterministic

    Returns:
        Mock environment
    """
    mock_env = Mock(spec=gym.Env)

    # Observation space
    mock_obs_space = Mock()
    mock_obs_space.n = state_space_size
    mock_env.observation_space = mock_obs_space

    # Action space
    mock_action_space = Mock()
    mock_action_space.n = action_space_size
    mock_action_space.sample.return_value = 0
    mock_env.action_space = mock_action_space

    # Make sure the n attributes return actual integers, not Mock objects
    type(mock_obs_space).n = property(lambda self: state_space_size)
    type(mock_action_space).n = property(lambda self: action_space_size)

    # Environment behavior
    if deterministic:
        mock_env.reset.return_value = (0, {})
        mock_env.step.side_effect = [
            (i % state_space_size, 1.0, i == episode_length - 1, False, {})
            for i in range(1, episode_length + 1)
        ]
    else:
        # Random behavior for stochastic testing
        mock_env.reset.return_value = (np.random.randint(0, state_space_size), {})
        mock_env.step.side_effect = lambda action: (
            np.random.randint(0, state_space_size),
            np.random.uniform(-1, 1),
            np.random.random() < 0.1,  # 10% chance of termination
            False,
            {},
        )

    return mock_env


@pytest.fixture
def fast_cartpole():
    """Fast CartPole environment for integration tests."""
    return gym.make("CartPole-v1", render_mode=None)


@pytest.fixture
def fast_frozenlake():
    """Fast FrozenLake environment for integration tests."""
    return gym.make("FrozenLake-v1", render_mode=None, is_slippery=False)


@pytest.fixture
def test_environments():
    """Registry of test environments."""
    return {
        "fast": ["CartPole-v1", "FrozenLake-v1"],
        "medium": ["MountainCar-v0", "Acrobot-v1"],
        "slow": ["LunarLander-v2", "BipedalWalker-v3"],
    }


@pytest.fixture
def deterministic_mock_env(mock_environment_factory):
    """Deterministic mock environment for unit tests."""
    return mock_environment_factory(
        state_space_size=4, action_space_size=2, episode_length=5, deterministic=True
    )


@pytest.fixture
def stochastic_mock_env(mock_environment_factory):
    """Stochastic mock environment for testing exploration."""
    return mock_environment_factory(
        state_space_size=4, action_space_size=2, episode_length=5, deterministic=False
    )
