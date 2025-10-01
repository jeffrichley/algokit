"""Test metrics and assertions for RL algorithms."""

from typing import Any

import numpy as np
import pytest


class RLTestAssertions:
    """Assertion helpers for RL algorithm testing."""

    @staticmethod
    def assert_training_results(results: dict[str, Any]) -> None:
        """Assert basic training results structure."""
        required_keys = ["episodes_trained", "final_avg_reward"]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"

        assert results["episodes_trained"] > 0
        assert isinstance(results["final_avg_reward"], int | float)

    @staticmethod
    def assert_convergence(results: dict[str, Any], min_episodes: int = 1) -> None:
        """Assert algorithm converged properly."""
        assert results["episodes_trained"] >= min_episodes
        if "convergence_episode" in results:
            assert results["convergence_episode"] <= results["episodes_trained"]

    @staticmethod
    def assert_q_table_valid(q_table: np.ndarray) -> None:
        """Assert Q-table is valid."""
        assert q_table is not None
        assert q_table.shape[0] > 0  # At least one state
        assert q_table.shape[1] > 0  # At least one action
        assert not np.isnan(q_table).any()
        assert not np.isinf(q_table).any()

    @staticmethod
    def assert_episode_results(results: dict[str, Any]) -> None:
        """Assert episode results are valid."""
        assert "episode_rewards" in results
        assert "episode_lengths" in results
        assert isinstance(results["episode_rewards"], list)
        assert isinstance(results["episode_lengths"], list)
        assert len(results["episode_rewards"]) > 0
        assert len(results["episode_lengths"]) > 0

    @staticmethod
    def assert_learning_progress(results: dict[str, Any]) -> None:
        """Assert learning progress is reasonable."""
        if "episode_rewards" in results and len(results["episode_rewards"]) > 1:
            rewards = results["episode_rewards"]
            # Check that rewards are not all the same (some learning occurred)
            assert not all(r == rewards[0] for r in rewards), (
                "No learning progress detected"
            )


@pytest.fixture
def rl_assertions():
    """RL test assertions fixture."""
    return RLTestAssertions()




@pytest.fixture
def test_metrics():
    """Standard test metrics for validation."""
    return {
        "min_episode_reward": -500,
        "max_episode_reward": 500,
        "min_episode_length": 1,
        "max_episode_length": 1000,
        "min_learning_rate": 0.001,
        "max_learning_rate": 1.0,
        "min_discount_factor": 0.0,
        "max_discount_factor": 1.0,
    }
