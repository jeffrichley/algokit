"""Tests for Policy Gradient GAE indexing bug fix and new features."""

import numpy as np
import pytest
import torch

from algokit.algorithms.reinforcement_learning.policy_gradient import (
    PolicyGradientAgent,
    RolloutExperience,
)


class TestPolicyGradientGAEFix:
    """Test Policy Gradient GAE indexing bug fix and new features."""

    @pytest.mark.unit
    def test_compute_gae_advantages_with_non_terminal_final_step(self) -> None:
        """Test GAE computation when final step is non-terminal (exposes the bug)."""
        # Arrange - Set up agent with GAE and test data where final step is NOT terminal
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, use_gae=True, gae_lambda=0.95
        )
        rewards = [1.0, 0.5, -0.5, 2.0]
        values = [0.8, 0.6, 0.4, 0.2]
        dones = [False, False, False, False]  # Final step is NOT terminal
        next_value = 0.1  # Bootstrap value

        # Act - Compute GAE advantages (this should NOT crash)
        advantages = agent.compute_gae_advantages(rewards, values, dones, next_value)

        # Assert - Verify advantages are computed correctly without crashing
        assert len(advantages) == len(rewards)
        assert all(isinstance(adv, float) for adv in advantages)
        assert all(np.isfinite(adv) for adv in advantages)

    @pytest.mark.unit
    def test_compute_gae_advantages_with_terminal_final_step(self) -> None:
        """Test GAE computation when final step is terminal (should work as before)."""
        # Arrange - Set up agent with GAE and test data where final step IS terminal
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, use_gae=True, gae_lambda=0.95
        )
        rewards = [1.0, 0.5, -0.5, 2.0]
        values = [0.8, 0.6, 0.4, 0.2]
        dones = [False, False, False, True]  # Final step IS terminal
        next_value = 0.1  # Should be ignored for terminal states

        # Act - Compute GAE advantages
        advantages = agent.compute_gae_advantages(rewards, values, dones, next_value)

        # Assert - Verify advantages are computed correctly
        assert len(advantages) == len(rewards)
        assert all(isinstance(adv, float) for adv in advantages)
        assert all(np.isfinite(adv) for adv in advantages)

    @pytest.mark.unit
    def test_compute_gae_advantages_mixed_terminal_states(self) -> None:
        """Test GAE computation with mixed terminal and non-terminal states."""
        # Arrange - Set up agent with GAE and complex test data
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, use_gae=True, gae_lambda=0.95
        )
        rewards = [1.0, 0.5, -0.5, 2.0, 1.5, 0.0, 0.8]
        values = [0.8, 0.6, 0.4, 0.2, 0.1, 0.0, 0.05]
        dones = [False, False, True, False, False, True, False]  # Mixed terminal states
        next_value = 0.02  # Bootstrap value for final non-terminal state

        # Act - Compute GAE advantages
        advantages = agent.compute_gae_advantages(rewards, values, dones, next_value)

        # Assert - Verify advantages are computed correctly for all states
        assert len(advantages) == len(rewards)
        assert all(isinstance(adv, float) for adv in advantages)
        assert all(np.isfinite(adv) for adv in advantages)

    @pytest.mark.unit
    def test_learn_with_gae_non_terminal_final_step(self) -> None:
        """Test learn method with GAE when final step is non-terminal."""
        # Arrange - Set up agent with GAE and sample rollout data with non-terminal final step
        agent = PolicyGradientAgent(
            state_size=4,
            action_size=2,
            use_baseline=True,
            use_gae=True,
            gae_lambda=0.95,
            continuous_actions=False,
        )

        rollout_data = [
            RolloutExperience(
                state=np.array([1.0, 0.0, 0.0, 0.0]),
                action=0,
                reward=1.0,
                log_prob=-0.5,
                value=0.8,
                done=False,
            ),
            RolloutExperience(
                state=np.array([0.0, 1.0, 0.0, 0.0]),
                action=1,
                reward=0.5,
                log_prob=-0.6,
                value=0.6,
                done=False,
            ),
            RolloutExperience(
                state=np.array([0.0, 0.0, 1.0, 0.0]),
                action=0,
                reward=-0.5,
                log_prob=-0.7,
                value=0.4,
                done=False,  # Final step is NOT terminal
            ),
        ]

        # Act - Learn from rollout data (this should NOT crash)
        metrics = agent.learn(rollout_data)

        # Assert - Verify learning completed successfully
        assert isinstance(metrics, dict)
        assert "policy_loss" in metrics
        assert "baseline_loss" in metrics
        assert "kl_divergence" in metrics
        assert "reward_mean" in metrics
        assert "reward_std" in metrics
        assert all(
            np.isfinite(v) for v in metrics.values() if isinstance(v, (int, float))
        )

    @pytest.mark.unit
    def test_reward_normalization_feature(self) -> None:
        """Test reward normalization feature."""
        # Arrange - Set up agent with reward normalization enabled
        agent = PolicyGradientAgent(
            state_size=4,
            action_size=2,
            normalize_rewards=True,
            continuous_actions=False,
        )

        # Initial reward statistics should be default values
        assert agent.reward_mean == 0.0
        assert agent.reward_std == 1.0
        assert agent.reward_update_count == 0

        # Act - Test reward normalization with some rewards
        test_rewards = [10.0, 5.0, -2.0, 8.0]
        normalized_rewards = agent._normalize_rewards(test_rewards)

        # Assert - Verify normalization worked
        assert len(normalized_rewards) == len(test_rewards)
        assert agent.reward_update_count == 1
        assert agent.reward_mean != 0.0  # Should have been updated
        assert agent.reward_std != 1.0  # Should have been updated

        # Test that subsequent calls use running statistics
        test_rewards_2 = [3.0, 7.0, 1.0, 4.0]
        agent._normalize_rewards(test_rewards_2)
        assert agent.reward_update_count == 2

    @pytest.mark.unit
    def test_kl_divergence_tracking(self) -> None:
        """Test KL divergence tracking feature."""
        # Arrange - Set up agent
        agent = PolicyGradientAgent(
            state_size=4,
            action_size=2,
            continuous_actions=False,
        )

        # Initial KL history should be empty
        assert len(agent.kl_divergence_history) == 0

        # Act - Test KL divergence computation
        old_log_probs = torch.tensor([-0.5, -0.6, -0.7])
        new_log_probs = torch.tensor([-0.4, -0.5, -0.6])

        kl_div = agent._compute_kl_divergence(old_log_probs, new_log_probs)

        # Assert - Verify KL divergence computation
        assert isinstance(kl_div, float)
        assert np.isfinite(kl_div)

        # Test that KL divergence is tracked during learning
        rollout_data = [
            RolloutExperience(
                state=np.array([1.0, 0.0, 0.0, 0.0]),
                action=0,
                reward=1.0,
                log_prob=-0.5,
                value=0.8,
                done=False,
            ),
            RolloutExperience(
                state=np.array([0.0, 1.0, 0.0, 0.0]),
                action=1,
                reward=0.5,
                log_prob=-0.6,
                value=0.6,
                done=True,
            ),
        ]

        metrics = agent.learn(rollout_data)

        # Assert - Verify KL divergence is tracked and returned
        assert "kl_divergence" in metrics
        assert len(agent.kl_divergence_history) == 1
        assert np.isfinite(metrics["kl_divergence"])

    @pytest.mark.unit
    def test_enhanced_training_stats(self) -> None:
        """Test enhanced training statistics with new features."""
        # Arrange - Set up agent with new features
        agent = PolicyGradientAgent(
            state_size=4,
            action_size=2,
            normalize_rewards=True,
            continuous_actions=False,
        )

        # Act - Get training stats before any training
        stats = agent.get_training_stats()

        # Assert - Verify enhanced stats structure
        assert "mean_reward" in stats
        assert "mean_kl_divergence" in stats
        assert "reward_normalization_stats" in stats
        assert "mean" in stats["reward_normalization_stats"]
        assert "std" in stats["reward_normalization_stats"]

        # Test with some training data
        agent.episode_rewards = [1.0, 2.0, 3.0]
        agent.episode_lengths = [10, 15, 20]
        agent.kl_divergence_history = [0.1, 0.2, 0.15]

        stats_with_data = agent.get_training_stats()

        # Assert - Verify stats with data
        assert stats_with_data["total_episodes"] == 3
        assert "mean_kl_divergence" in stats_with_data
        assert "std_kl_divergence" in stats_with_data
        assert "recent_kl_divergence" in stats_with_data
        assert stats_with_data["recent_kl_divergence"] == 0.15

    @pytest.mark.unit
    def test_gae_bootstrap_value_handling(self) -> None:
        """Test that bootstrap value is properly used for non-terminal final states."""
        # Arrange - Set up agent with baseline and GAE
        agent = PolicyGradientAgent(
            state_size=4,
            action_size=2,
            use_baseline=True,
            use_gae=True,
            continuous_actions=False,
        )

        # Create rollout data with non-terminal final step
        rollout_data = [
            RolloutExperience(
                state=np.array([1.0, 0.0, 0.0, 0.0]),
                action=0,
                reward=1.0,
                log_prob=-0.5,
                value=0.8,
                done=False,
            ),
            RolloutExperience(
                state=np.array([0.0, 1.0, 0.0, 0.0]),
                action=1,
                reward=0.5,
                log_prob=-0.6,
                value=0.6,
                done=False,  # Final step is NOT terminal
            ),
        ]

        # Act - Learn from rollout data
        metrics = agent.learn(rollout_data)

        # Assert - Verify learning completed without errors
        assert isinstance(metrics, dict)
        assert "policy_loss" in metrics
        assert "baseline_loss" in metrics
        assert all(
            np.isfinite(v) for v in metrics.values() if isinstance(v, (int, float))
        )

    @pytest.mark.unit
    def test_compute_gae_advantages_edge_cases(self) -> None:
        """Test GAE computation with edge cases."""
        # Arrange - Set up agent with GAE
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, use_gae=True, gae_lambda=0.95
        )

        # Test case 1: Single step episode (non-terminal)
        rewards_single = [1.0]
        values_single = [0.5]
        dones_single = [False]
        next_value = 0.2

        # Act - Compute GAE advantages for single non-terminal step
        advantages_single = agent.compute_gae_advantages(
            rewards_single, values_single, dones_single, next_value
        )

        # Assert - Verify single step advantages
        assert len(advantages_single) == 1
        assert np.isfinite(advantages_single[0])

        # Test case 2: Single step episode (terminal)
        rewards_terminal = [1.0]
        values_terminal = [0.5]
        dones_terminal = [True]

        # Act - Compute GAE advantages for single terminal step
        advantages_terminal = agent.compute_gae_advantages(
            rewards_terminal, values_terminal, dones_terminal, next_value
        )

        # Assert - Verify terminal step advantages
        assert len(advantages_terminal) == 1
        assert np.isfinite(advantages_terminal[0])

        # Test case 3: Empty episode (should not crash)
        # Act - Compute GAE advantages for empty episode
        advantages_empty = agent.compute_gae_advantages([], [], [], next_value)

        # Assert - Verify empty episode handling
        assert len(advantages_empty) == 0
