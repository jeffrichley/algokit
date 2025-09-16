"""Test data generators for RL algorithms."""

import numpy as np
import pytest


@pytest.fixture
def sample_states():
    """Sample states for testing."""
    return [0, 1, 2, 3, 4]


@pytest.fixture
def sample_actions():
    """Sample actions for testing."""
    return [0, 1]


@pytest.fixture
def sample_rewards():
    """Sample rewards for testing."""
    return [1.0, -1.0, 0.5, 0.0, -0.5]


@pytest.fixture
def sample_q_table():
    """Sample Q-table for testing."""
    return np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])


@pytest.fixture
def sample_eligibility_traces():
    """Sample eligibility traces for testing."""
    return np.array([[0.5, 0.3], [0.2, 0.8], [0.1, 0.9], [0.4, 0.6]])


@pytest.fixture
def sample_episode_data():
    """Sample episode data for testing."""
    return {
        "states": [0, 1, 2, 3, 4],
        "actions": [0, 1, 0, 1, 0],
        "rewards": [1.0, 0.5, 1.0, 0.5, 1.0],
        "next_states": [1, 2, 3, 4, 0],
        "dones": [False, False, False, False, True],
    }


@pytest.fixture
def sample_training_results():
    """Sample training results for testing."""
    return {
        "episodes_trained": 10,
        "final_avg_reward": 150.0,
        "convergence_episode": 8,
        "total_rewards": [100, 120, 140, 150, 150, 150, 150, 150, 150, 150],
        "avg_rewards": [100, 110, 120, 130, 140, 145, 147, 149, 149, 150],
        "epsilon_values": [
            0.1,
            0.0995,
            0.099,
            0.0985,
            0.098,
            0.0975,
            0.097,
            0.0965,
            0.096,
            0.0955,
        ],
    }


@pytest.fixture
def sample_model_data():
    """Sample model data for saving/loading tests."""
    return {
        "q_table": {
            (0, 0, 0, 0): np.array([0.1, 0.2]),
            (1, 1, 1, 1): np.array([0.3, 0.4]),
        },
        "parameters": {
            "learning_rate": 0.1,
            "discount_factor": 0.9,
            "epsilon": 0.1,
            "epsilon_decay": 0.995,
            "mode": "tabular",
            "lambda_param": 0.9,
        },
        "metadata": {
            "episodes_trained": 10,
            "final_avg_reward": 150.0,
            "environment": "CartPole-v1",
        },
    }
