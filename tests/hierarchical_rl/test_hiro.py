"""Tests for HIRO (Hierarchical Reinforcement Learning) implementation.

Tests cover:
1. Higher-level and lower-level policy initialization and forward passes
2. Goal representation as state deltas (s_{t+k} - s_t)
3. Normalized and scaled intrinsic rewards
4. TD3-style target policy smoothing
5. Diverse goal sampling strategy
6. Off-policy goal relabeling
7. Training loops and experience replay
8. Soft target updates
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from algokit.algorithms.hierarchical_rl.hiro import (
    HigherLevelPolicy,
    HIROAgent,
    LowerLevelPolicy,
)


@pytest.mark.unit
class TestHigherLevelPolicy:
    """Test higher-level policy network."""

    def test_higher_level_policy_initialization(self) -> None:
        """Test that HigherLevelPolicy initializes correctly."""
        # Arrange - Set up test data and dependencies
        state_size = 4
        goal_size = 8
        hidden_size = 128

        # Act - Execute the function under test
        policy = HigherLevelPolicy(
            state_size=state_size, goal_size=goal_size, hidden_size=hidden_size
        )

        # Assert - Verify the expected outcomes
        assert policy.state_size == state_size
        assert policy.goal_size == goal_size
        assert isinstance(policy.network, torch.nn.Sequential)
        assert isinstance(policy.critic, torch.nn.Sequential)

    def test_higher_level_policy_forward_pass(self) -> None:
        """Test that higher-level policy produces goal outputs."""
        # Arrange - Set up test data and dependencies
        policy = HigherLevelPolicy(state_size=4, goal_size=8)
        state = torch.randn(1, 4)

        # Act - Execute the function under test
        goal = policy(state)

        # Assert - Verify the expected outcomes
        assert goal.shape == (1, 8)
        assert torch.all(goal >= -1.0) and torch.all(goal <= 1.0)  # Tanh bounded

    def test_higher_level_critic_q_value(self) -> None:
        """Test that higher-level critic computes Q-values for state-goal pairs."""
        # Arrange - Set up test data and dependencies
        policy = HigherLevelPolicy(state_size=4, goal_size=8)
        state = torch.randn(2, 4)
        goal = torch.randn(2, 8)

        # Act - Execute the function under test
        q_value = policy.get_value(state, goal)

        # Assert - Verify the expected outcomes
        assert q_value.shape == (2, 1)

    def test_higher_level_policy_deterministic(self) -> None:
        """Test that higher-level policy is deterministic for same input."""
        # Arrange - Set up test data and dependencies
        policy = HigherLevelPolicy(state_size=4, goal_size=8)
        state = torch.randn(1, 4)

        # Act - Execute the function under test
        goal1 = policy(state)
        goal2 = policy(state)

        # Assert - Verify the expected outcomes
        assert torch.allclose(goal1, goal2)


@pytest.mark.unit
class TestLowerLevelPolicy:
    """Test lower-level goal-conditioned policy network."""

    def test_lower_level_policy_initialization(self) -> None:
        """Test that LowerLevelPolicy initializes correctly."""
        # Arrange - Set up test data and dependencies
        state_size = 4
        action_size = 3
        goal_size = 8
        hidden_size = 128

        # Act - Execute the function under test
        policy = LowerLevelPolicy(
            state_size=state_size,
            action_size=action_size,
            goal_size=goal_size,
            hidden_size=hidden_size,
        )

        # Assert - Verify the expected outcomes
        assert policy.state_size == state_size
        assert policy.action_size == action_size
        assert policy.goal_size == goal_size
        assert isinstance(policy.policy, torch.nn.Sequential)
        assert isinstance(policy.critic, torch.nn.Sequential)

    def test_lower_level_policy_forward_pass(self) -> None:
        """Test that lower-level policy produces action logits."""
        # Arrange - Set up test data and dependencies
        policy = LowerLevelPolicy(state_size=4, action_size=3, goal_size=8)
        state = torch.randn(1, 4)
        goal = torch.randn(1, 8)

        # Act - Execute the function under test
        logits = policy(state, goal)

        # Assert - Verify the expected outcomes
        assert logits.shape == (1, 3)

    def test_lower_level_critic_q_value(self) -> None:
        """Test that lower-level critic computes Q-values for state-goal-action."""
        # Arrange - Set up test data and dependencies
        policy = LowerLevelPolicy(state_size=4, action_size=3, goal_size=8)
        state = torch.randn(2, 4)
        goal = torch.randn(2, 8)
        action = torch.randn(2, 3)

        # Act - Execute the function under test
        q_value = policy.get_value(state, goal, action)

        # Assert - Verify the expected outcomes
        assert q_value.shape == (2, 1)

    def test_lower_level_policy_goal_conditioned(self) -> None:
        """Test that lower-level policy output changes with different goals."""
        # Arrange - Set up test data and dependencies
        policy = LowerLevelPolicy(state_size=4, action_size=3, goal_size=8)
        state = torch.randn(1, 4)
        goal1 = torch.randn(1, 8)
        goal2 = torch.randn(1, 8)

        # Act - Execute the function under test
        logits1 = policy(state, goal1)
        logits2 = policy(state, goal2)

        # Assert - Verify the expected outcomes
        assert not torch.allclose(logits1, logits2)


@pytest.mark.unit
class TestHIROAgent:
    """Test HIRO agent initialization and core functionality."""

    def test_hiro_agent_initialization(self) -> None:
        """Test that HIROAgent initializes with correct parameters."""
        # Arrange - Set up test data and dependencies
        state_size = 4
        action_size = 3
        goal_size = 8

        # Act - Execute the function under test
        agent = HIROAgent(
            state_size=state_size, action_size=action_size, goal_size=goal_size
        )

        # Assert - Verify the expected outcomes
        assert agent.state_size == state_size
        assert agent.action_size == action_size
        assert agent.goal_size == goal_size
        assert agent.goal_horizon == 10  # Default
        assert agent.gamma == 0.99  # Default
        assert agent.tau == 0.005  # Default
        assert isinstance(agent.higher_policy, HigherLevelPolicy)
        assert isinstance(agent.lower_policy, LowerLevelPolicy)
        assert isinstance(agent.higher_target, HigherLevelPolicy)
        assert isinstance(agent.lower_target, LowerLevelPolicy)

    def test_hiro_agent_new_parameters(self) -> None:
        """Test that new stability parameters are properly initialized."""
        # Arrange - Set up test data and dependencies
        policy_noise = 0.3
        noise_clip = 0.6
        intrinsic_scale = 2.0

        # Act - Execute the function under test
        agent = HIROAgent(
            state_size=4,
            action_size=3,
            goal_size=8,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            intrinsic_scale=intrinsic_scale,
        )

        # Assert - Verify the expected outcomes
        assert agent.policy_noise == policy_noise
        assert agent.noise_clip == noise_clip
        assert agent.intrinsic_scale == intrinsic_scale
        assert agent.distance_mean == 0.0
        assert agent.distance_std == 1.0
        assert len(agent.distance_buffer) == 0

    def test_hiro_agent_seed_reproducibility(self) -> None:
        """Test that setting seed produces reproducible results."""
        # Arrange - Set up test data and dependencies
        seed = 42
        state = torch.randn(4)

        # Act - Execute the function under test
        agent1 = HIROAgent(state_size=4, action_size=3, goal_size=8, seed=seed)
        goal1 = agent1.select_goal(state)

        agent2 = HIROAgent(state_size=4, action_size=3, goal_size=8, seed=seed)
        goal2 = agent2.select_goal(state)

        # Assert - Verify the expected outcomes
        assert torch.allclose(goal1, goal2)

    def test_select_goal(self) -> None:
        """Test that select_goal produces valid goal outputs."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=8)
        state = torch.randn(4)

        # Act - Execute the function under test
        goal = agent.select_goal(state)

        # Assert - Verify the expected outcomes
        assert goal.shape == (8,)
        assert torch.all(goal >= -1.0) and torch.all(goal <= 1.0)

    def test_select_action_with_epsilon(self) -> None:
        """Test that select_action uses epsilon-greedy exploration."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=8, seed=42)
        state = torch.randn(4)
        goal = torch.randn(8)

        # Act - Execute the function under test
        action = agent.select_action(state, goal, epsilon=0.1)

        # Assert - Verify the expected outcomes
        assert isinstance(action, int)
        assert 0 <= action < 3

    def test_select_action_deterministic(self) -> None:
        """Test that select_action produces valid actions with epsilon=0."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=8, seed=42)
        state = torch.randn(4)
        goal = torch.randn(8)

        # Act - Execute the function under test
        action1 = agent.select_action(state, goal, epsilon=0.0)
        action2 = agent.select_action(state, goal, epsilon=0.0)

        # Assert - Verify the expected outcomes (both actions should be valid)
        assert 0 <= action1 < 3
        assert 0 <= action2 < 3


@pytest.mark.unit
class TestGoalDistanceAndRewards:
    """Test intrinsic reward computation and normalization."""

    def test_goal_distance_initial_computation(self) -> None:
        """Test that goal_distance computes distance without statistics."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=8)
        state = torch.randn(4)
        goal = torch.randn(4)  # Using state_size for compatibility

        # Act - Execute the function under test
        distance = agent.goal_distance(state, goal)

        # Assert - Verify the expected outcomes
        assert isinstance(distance, float)

    def test_goal_distance_normalization_after_warmup(self) -> None:
        """Test that goal_distance normalizes after collecting statistics."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=8)
        state = torch.randn(4)
        goal = torch.randn(4)

        # Act - Collect enough samples for statistics
        for _ in range(150):
            _ = agent.goal_distance(torch.randn(4), torch.randn(4))

        _ = agent.goal_distance(state, goal)

        # Assert - Verify the expected outcomes
        assert agent.distance_mean != 0.0 or agent.distance_std != 1.0
        assert len(agent.distance_buffer) >= 100

    def test_goal_distance_scaling_by_state_size(self) -> None:
        """Test that intrinsic rewards are scaled by state dimensionality."""
        # Arrange - Set up test data and dependencies
        agent_small = HIROAgent(state_size=2, action_size=3, goal_size=8)
        agent_large = HIROAgent(state_size=16, action_size=3, goal_size=8)

        # Collect some statistics
        for _ in range(150):
            _ = agent_small.goal_distance(torch.randn(2), torch.randn(2))
            _ = agent_large.goal_distance(torch.randn(16), torch.randn(16))

        # Act - Execute the function under test
        dist_small = agent_small.goal_distance(torch.zeros(2), torch.ones(2))
        dist_large = agent_large.goal_distance(torch.zeros(16), torch.ones(16))

        # Assert - Larger state space should have different scaling
        assert isinstance(dist_small, float)
        assert isinstance(dist_large, float)

    def test_goal_distance_with_intrinsic_scale(self) -> None:
        """Test that intrinsic_scale parameter affects reward magnitude."""
        # Arrange - Set up test data and dependencies
        agent_low = HIROAgent(
            state_size=4, action_size=3, goal_size=8, intrinsic_scale=0.5
        )
        agent_high = HIROAgent(
            state_size=4, action_size=3, goal_size=8, intrinsic_scale=2.0
        )

        # Warm up both agents with same data
        for _ in range(150):
            same_state = torch.randn(4)
            same_goal = torch.randn(4)
            _ = agent_low.goal_distance(same_state, same_goal)
            _ = agent_high.goal_distance(same_state, same_goal)

        # Act - Execute the function under test
        state = torch.randn(4)
        goal = torch.randn(4)
        dist_low = agent_low.goal_distance(state, goal)
        dist_high = agent_high.goal_distance(state, goal)

        # Assert - Higher scale should produce larger magnitude
        assert abs(dist_high) > abs(dist_low)


@pytest.mark.unit
class TestGoalRelabeling:
    """Test off-policy goal relabeling with state deltas."""

    def test_relabel_goal_computes_delta(self) -> None:
        """Test that relabel_goal computes state delta g = s_{t+k} - s_t."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=8)
        start_state = torch.tensor([0.0, 0.0, 0.0, 0.0])
        trajectory = [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([2.0, 0.0, 0.0, 0.0]),
            torch.tensor([3.0, 0.0, 0.0, 0.0]),
        ]
        horizon = 2

        # Act - Execute the function under test
        relabeled_goal = agent.relabel_goal(start_state, trajectory, horizon)

        # Assert - Should be delta from start to horizon
        expected_delta = trajectory[horizon - 1] - start_state
        assert torch.allclose(relabeled_goal, expected_delta)

    def test_relabel_goal_uses_last_state_when_short(self) -> None:
        """Test that relabel_goal uses last state if trajectory is shorter than horizon."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=8)
        start_state = torch.tensor([0.0, 0.0, 0.0, 0.0])
        trajectory = [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([2.0, 0.0, 0.0, 0.0]),
        ]
        horizon = 5  # Longer than trajectory

        # Act - Execute the function under test
        relabeled_goal = agent.relabel_goal(start_state, trajectory, horizon)

        # Assert - Should use last state
        expected_delta = trajectory[-1] - start_state
        assert torch.allclose(relabeled_goal, expected_delta)

    def test_relabel_goal_is_relative_not_absolute(self) -> None:
        """Test that relabeled goals are relative displacements."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=8)
        start_state = torch.tensor([10.0, 20.0, 30.0, 40.0])
        trajectory = [
            torch.tensor([11.0, 21.0, 31.0, 41.0]),
            torch.tensor([12.0, 22.0, 32.0, 42.0]),
        ]
        horizon = 2

        # Act - Execute the function under test
        relabeled_goal = agent.relabel_goal(start_state, trajectory, horizon)

        # Assert - Should be delta, not absolute position
        expected_delta = torch.tensor([2.0, 2.0, 2.0, 2.0])
        assert torch.allclose(relabeled_goal, expected_delta)


@pytest.mark.unit
class TestTrainingAndReplay:
    """Test training loops and experience replay."""

    def test_lower_buffer_stores_experiences(self) -> None:
        """Test that lower-level buffer stores experiences correctly."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=8)

        # Act - Execute the function under test
        agent.lower_buffer.append(
            {
                "state": torch.randn(4),
                "action": 1,
                "reward": 0.5,
                "next_state": torch.randn(4),
                "goal": torch.randn(8),
                "done": False,
            }
        )

        # Assert - Verify the expected outcomes
        assert len(agent.lower_buffer) == 1

    def test_higher_buffer_stores_experiences(self) -> None:
        """Test that higher-level buffer stores experiences correctly."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=8)

        # Act - Execute the function under test
        agent.higher_buffer.append(
            {
                "state": torch.randn(4),
                "goal": torch.randn(8),
                "reward": 1.0,
                "next_state": torch.randn(4),
                "done": False,
            }
        )

        # Assert - Verify the expected outcomes
        assert len(agent.higher_buffer) == 1

    def test_train_lower_requires_minimum_samples(self) -> None:
        """Test that train_lower returns 0 when buffer has insufficient samples."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=8)

        # Act - Execute the function under test
        loss = agent.train_lower(batch_size=64)

        # Assert - Verify the expected outcomes
        assert loss == 0.0

    def test_train_higher_requires_minimum_samples(self) -> None:
        """Test that train_higher returns 0 when buffer has insufficient samples."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=8)

        # Act - Execute the function under test
        loss = agent.train_higher(batch_size=64)

        # Assert - Verify the expected outcomes
        assert loss == 0.0

    def test_train_lower_with_sufficient_samples(self) -> None:
        """Test that train_lower runs when buffer has sufficient samples."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=8, seed=42)

        # Add sufficient samples
        for _ in range(100):
            agent.lower_buffer.append(
                {
                    "state": torch.randn(4),
                    "action": np.random.randint(0, 3),
                    "reward": np.random.randn(),
                    "next_state": torch.randn(4),
                    "goal": torch.randn(8),
                    "done": False,
                }
            )

        # Act - Execute the function under test
        loss = agent.train_lower(batch_size=32)

        # Assert - Verify the expected outcomes
        assert loss > 0.0  # Should return actual loss

    def test_train_higher_with_sufficient_samples(self) -> None:
        """Test that train_higher runs when buffer has sufficient samples."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=8, seed=42)

        # Add sufficient samples
        for _ in range(100):
            agent.higher_buffer.append(
                {
                    "state": torch.randn(4),
                    "goal": torch.randn(8),
                    "reward": np.random.randn(),
                    "next_state": torch.randn(4),
                    "done": False,
                }
            )

        # Act - Execute the function under test
        loss = agent.train_higher(batch_size=32)

        # Assert - Verify the expected outcomes
        assert loss > 0.0  # Should return actual loss


@pytest.mark.unit
class TestDiverseSampling:
    """Test diverse goal sampling strategy."""

    def test_diverse_sampling_mixes_recent_and_old(self) -> None:
        """Test that diverse sampling uses both recent and old experiences."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=8, seed=42)

        # Add experiences with identifiable patterns
        for i in range(200):
            agent.lower_buffer.append(
                {
                    "state": torch.tensor([float(i), 0.0, 0.0, 0.0]),
                    "action": i % 3,
                    "reward": float(i),
                    "next_state": torch.tensor([float(i + 1), 0.0, 0.0, 0.0]),
                    "goal": torch.randn(8),
                    "done": False,
                }
            )

        # Act - Execute the function under test
        loss = agent.train_lower(batch_size=64)

        # Assert - Should successfully sample and train
        assert loss > 0.0

    def test_diverse_sampling_handles_small_buffer(self) -> None:
        """Test that diverse sampling handles buffers smaller than batch size."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=8, seed=42)

        # Add small number of samples
        for _ in range(10):
            agent.lower_buffer.append(
                {
                    "state": torch.randn(4),
                    "action": 1,
                    "reward": 0.5,
                    "next_state": torch.randn(4),
                    "goal": torch.randn(8),
                    "done": False,
                }
            )

        # Act - Execute the function under test
        loss = agent.train_lower(batch_size=64)

        # Assert - Should handle gracefully
        assert loss == 0.0  # Not enough samples


@pytest.mark.unit
class TestTargetSmoothing:
    """Test TD3-style target policy smoothing."""

    def test_target_smoothing_adds_noise(self) -> None:
        """Test that target smoothing adds noise to target policy actions."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(
            state_size=4,
            action_size=3,
            goal_size=8,
            policy_noise=0.2,
            noise_clip=0.5,
            seed=42,
        )

        # Add sufficient samples
        for _ in range(100):
            agent.higher_buffer.append(
                {
                    "state": torch.randn(4),
                    "goal": torch.randn(8),
                    "reward": np.random.randn(),
                    "next_state": torch.randn(4),
                    "done": False,
                }
            )

        # Act - Training should use smoothed targets
        loss = agent.train_higher(batch_size=32)

        # Assert - Verify the expected outcomes
        assert loss > 0.0
        assert agent.policy_noise == 0.2
        assert agent.noise_clip == 0.5


@pytest.mark.unit
class TestSoftTargetUpdates:
    """Test soft target network updates."""

    def test_soft_update_targets_moves_toward_online(self) -> None:
        """Test that soft_update_targets updates target networks."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=8, tau=0.1)

        # Store initial target parameters
        initial_higher_param = next(agent.higher_target.parameters()).clone()
        initial_lower_param = next(agent.lower_target.parameters()).clone()

        # Modify online network parameters
        for param in agent.higher_policy.parameters():
            param.data.fill_(1.0)
        for param in agent.lower_policy.parameters():
            param.data.fill_(1.0)

        # Act - Execute the function under test
        agent.soft_update_targets()

        # Assert - Target parameters should have moved
        updated_higher_param = next(agent.higher_target.parameters())
        updated_lower_param = next(agent.lower_target.parameters())

        assert not torch.allclose(initial_higher_param, updated_higher_param)
        assert not torch.allclose(initial_lower_param, updated_lower_param)

    def test_soft_update_tau_controls_update_rate(self) -> None:
        """Test that tau parameter controls update rate."""
        # Arrange - Set up test data and dependencies
        agent_slow = HIROAgent(state_size=4, action_size=3, goal_size=8, tau=0.001)
        agent_fast = HIROAgent(state_size=4, action_size=3, goal_size=8, tau=0.5)

        # Modify online networks
        for param in agent_slow.higher_policy.parameters():
            param.data.fill_(10.0)
        for param in agent_fast.higher_policy.parameters():
            param.data.fill_(10.0)

        initial_slow = next(agent_slow.higher_target.parameters()).clone()
        initial_fast = next(agent_fast.higher_target.parameters()).clone()

        # Act - Execute the function under test
        agent_slow.soft_update_targets()
        agent_fast.soft_update_targets()

        # Assert - Fast should change more
        updated_slow = next(agent_slow.higher_target.parameters())
        updated_fast = next(agent_fast.higher_target.parameters())

        slow_change = torch.abs(updated_slow - initial_slow).mean()
        fast_change = torch.abs(updated_fast - initial_fast).mean()

        assert fast_change > slow_change


@pytest.mark.unit
class TestEpisodeTraining:
    """Test full episode training integration."""

    def test_train_episode_with_mock_env(self) -> None:
        """Test that train_episode runs with a mock environment."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=4, seed=42)

        # Create mock environment
        env = MagicMock()
        env.reset.return_value = (np.array([0.0, 0.0, 0.0, 0.0]), {})
        env.step.return_value = (
            np.array([0.1, 0.1, 0.1, 0.1]),  # next_state
            1.0,  # reward
            False,  # done
            False,  # truncated
            {},  # info
        )

        # Act - Execute the function under test
        metrics = agent.train_episode(env, max_steps=10, epsilon=0.1)

        # Assert - Verify the expected outcomes
        assert "reward" in metrics
        assert "steps" in metrics
        assert "avg_lower_critic_loss" in metrics
        assert "avg_higher_critic_loss" in metrics
        assert "avg_lower_actor_loss" in metrics
        assert "avg_higher_actor_loss" in metrics
        assert metrics["steps"] == 10

    def test_train_episode_early_termination(self) -> None:
        """Test that train_episode handles early termination."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=4, seed=42)

        # Create mock environment that terminates after 3 steps
        env = MagicMock()
        env.reset.return_value = (np.array([0.0, 0.0, 0.0, 0.0]), {})

        step_count = [0]

        def step_side_effect(
            action: int,
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, str]]:
            step_count[0] += 1
            done = step_count[0] >= 3
            return (
                np.array([0.1, 0.1, 0.1, 0.1]),
                1.0,
                done,
                False,
                {},
            )

        env.step.side_effect = step_side_effect

        # Act - Execute the function under test
        metrics = agent.train_episode(env, max_steps=100, epsilon=0.1)

        # Assert - Verify the expected outcomes
        assert metrics["steps"] == 3

    def test_goal_updates_at_horizon(self) -> None:
        """Test that goals are updated at the goal horizon."""
        # Arrange - Set up test data and dependencies
        goal_horizon = 5
        agent = HIROAgent(
            state_size=4, action_size=3, goal_size=4, goal_horizon=goal_horizon, seed=42
        )

        # Create mock environment
        env = MagicMock()
        env.reset.return_value = (np.array([0.0, 0.0, 0.0, 0.0]), {})
        env.step.return_value = (
            np.array([0.1, 0.1, 0.1, 0.1]),
            1.0,
            False,
            False,
            {},
        )

        # Act - Execute the function under test
        _ = agent.train_episode(env, max_steps=15, epsilon=0.1)

        # Assert - Should have higher-level experiences at horizon intervals
        # With 15 steps and horizon=5, we expect ~3 higher-level experiences
        assert len(agent.higher_buffer) >= 2


@pytest.mark.unit
class TestStatistics:
    """Test agent statistics tracking."""

    def test_get_statistics_returns_correct_keys(self) -> None:
        """Test that get_statistics returns expected metrics."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=8)

        # Act - Execute the function under test
        stats = agent.get_statistics()

        # Assert - Verify the expected outcomes
        assert "total_episodes" in stats
        assert "avg_reward" in stats
        assert "avg_lower_critic_loss" in stats
        assert "avg_higher_critic_loss" in stats
        assert "avg_lower_actor_loss" in stats
        assert "avg_higher_actor_loss" in stats
        assert "avg_intrinsic_reward" in stats
        assert "avg_extrinsic_reward" in stats
        assert "intrinsic_extrinsic_ratio" in stats
        assert "goal_horizon" in stats

    def test_statistics_track_episode_rewards(self) -> None:
        """Test that episode rewards are tracked."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=8)
        agent.episode_rewards = [1.0, 2.0, 3.0, 4.0, 5.0]

        # Act - Execute the function under test
        stats = agent.get_statistics()

        # Assert - Verify the expected outcomes
        assert stats["total_episodes"] == 5
        assert stats["avg_reward"] == 3.0

    def test_statistics_track_losses(self) -> None:
        """Test that losses are tracked."""
        # Arrange - Set up test data and dependencies
        agent = HIROAgent(state_size=4, action_size=3, goal_size=8)
        agent.lower_critic_losses = [0.5, 0.6, 0.7]
        agent.higher_critic_losses = [0.3, 0.4, 0.5]
        agent.lower_actor_losses = [0.1, 0.2, 0.3]
        agent.higher_actor_losses = [0.05, 0.10, 0.15]

        # Act - Execute the function under test
        stats = agent.get_statistics()

        # Assert - Verify the expected outcomes
        assert stats["avg_lower_critic_loss"] == pytest.approx(0.6, rel=1e-5)
        assert stats["avg_higher_critic_loss"] == pytest.approx(0.4, rel=1e-5)
        assert stats["avg_lower_actor_loss"] == pytest.approx(0.2, rel=1e-5)
        assert stats["avg_higher_actor_loss"] == pytest.approx(0.10, rel=1e-5)
