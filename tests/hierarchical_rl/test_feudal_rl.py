"""Tests for Feudal RL (FeudalNet) implementation.

This test suite covers all components of the improved Feudal RL:
- StateEncoder for latent representations
- ManagerNetwork for goal generation
- WorkerNetwork for goal-conditioned actions
- FeudalAgent with temporal coordination and improvements
"""

import gymnasium as gym
import numpy as np
import pytest
import torch
from pydantic import ValidationError

from algokit.algorithms.hierarchical_rl.feudal_rl import (
    FeudalAgent,
    FeudalConfig,
    ManagerNetwork,
    StateEncoder,
    WorkerNetwork,
)


@pytest.mark.unit
class TestStateEncoder:
    """Test the shared state encoder."""

    def test_encoder_initialization(self) -> None:
        """Test that state encoder initializes correctly."""
        # Arrange - create state encoder with specified dimensions
        # Act - initialize encoder
        encoder = StateEncoder(state_size=4, latent_size=16, hidden_size=32)

        # Assert - verify encoder properties are set correctly
        assert encoder.state_size == 4
        assert encoder.latent_size == 16
        assert isinstance(encoder.encoder, torch.nn.Sequential)

    def test_encoder_forward_pass(self) -> None:
        """Test that encoder produces correct output shape."""
        # Arrange - create encoder and input state tensor
        encoder = StateEncoder(state_size=4, latent_size=16, hidden_size=32)
        state = torch.randn(2, 4)

        # Act - pass state through encoder
        latent = encoder(state)

        # Assert - verify output shape and valid values
        assert latent.shape == (2, 16)
        assert not torch.isnan(latent).any()
        assert not torch.isinf(latent).any()

    def test_encoder_deterministic(self) -> None:
        """Test that encoder produces deterministic outputs."""
        # Arrange - create encoder in eval mode with fixed state
        encoder = StateEncoder(state_size=4, latent_size=16, hidden_size=32)
        encoder.eval()
        state = torch.randn(1, 4)

        # Act - encode same state twice
        output1 = encoder(state)
        output2 = encoder(state)

        # Assert - verify outputs are identical
        assert torch.allclose(output1, output2)


@pytest.mark.unit
class TestManagerNetwork:
    """Test the manager network."""

    def test_manager_initialization(self) -> None:
        """Test that manager network initializes correctly."""
        # Arrange - create manager network with specified dimensions
        # Act - initialize manager
        manager = ManagerNetwork(latent_size=16, goal_size=8, hidden_size=32)

        # Assert - verify manager properties are set correctly
        assert manager.latent_size == 16
        assert manager.goal_size == 8
        assert isinstance(manager.goal_generator, torch.nn.Sequential)
        assert isinstance(manager.value, torch.nn.Sequential)

    def test_manager_forward_pass(self) -> None:
        """Test that manager produces goals and values."""
        # Arrange - create manager and latent state tensor
        manager = ManagerNetwork(latent_size=16, goal_size=8, hidden_size=32)
        latent_state = torch.randn(2, 16)

        # Act - pass latent state through manager
        goal, value = manager(latent_state)

        # Assert - verify output shapes and valid values
        assert goal.shape == (2, 8)
        assert value.shape == (2, 1)
        assert not torch.isnan(goal).any()
        assert not torch.isnan(value).any()

    def test_manager_goal_normalization(self) -> None:
        """Test that manager normalizes goals to unit length."""
        # Arrange - create manager and latent state
        manager = ManagerNetwork(latent_size=16, goal_size=8, hidden_size=32)
        latent_state = torch.randn(3, 16)

        # Act - generate goals from manager
        goal, _ = manager(latent_state)

        # Assert - goals should have unit L2 norm
        norms = torch.norm(goal, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


@pytest.mark.unit
class TestWorkerNetwork:
    """Test the worker network."""

    def test_worker_initialization(self) -> None:
        """Test that worker network initializes correctly."""
        # Arrange - create worker network with specified dimensions
        # Act - initialize worker
        worker = WorkerNetwork(
            latent_size=16, action_size=4, goal_size=8, hidden_size=32
        )

        # Assert - verify worker properties are set correctly
        assert worker.latent_size == 16
        assert worker.action_size == 4
        assert worker.goal_size == 8
        assert isinstance(worker.policy, torch.nn.Sequential)
        assert isinstance(worker.value, torch.nn.Sequential)

    def test_worker_forward_pass(self) -> None:
        """Test that worker produces action logits and values."""
        # Arrange - create worker with latent state and goal tensors
        worker = WorkerNetwork(
            latent_size=16, action_size=4, goal_size=8, hidden_size=32
        )
        latent_state = torch.randn(2, 16)
        goal = torch.randn(2, 8)

        # Act - pass latent state and goal through worker
        logits, value = worker(latent_state, goal)

        # Assert - verify output shapes and valid values
        assert logits.shape == (2, 4)
        assert value.shape == (2, 1)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(value).any()

    def test_worker_goal_conditioning(self) -> None:
        """Test that worker output changes with different goals."""
        # Arrange - create worker in eval mode with two different goals
        worker = WorkerNetwork(
            latent_size=16, action_size=4, goal_size=8, hidden_size=32
        )
        worker.eval()
        latent_state = torch.randn(1, 16)
        goal1 = torch.randn(1, 8)
        goal2 = torch.randn(1, 8)

        # Act - generate outputs for same state with different goals
        logits1, _ = worker(latent_state, goal1)
        logits2, _ = worker(latent_state, goal2)

        # Assert - different goals should produce different outputs
        assert not torch.allclose(logits1, logits2)


@pytest.mark.unit
class TestFeudalAgent:
    """Test the complete Feudal RL agent."""

    def test_agent_initialization(self) -> None:
        """Test that agent initializes with correct parameters."""
        # Arrange - create feudal agent with specified parameters
        # Act - initialize agent
        agent = FeudalAgent(
            state_size=4,
            action_size=2,
            latent_size=16,
            goal_size=8,
            hidden_size=32,
            manager_horizon=10,
            seed=42,
        )

        # Assert - verify agent properties and components are set correctly
        assert agent.state_size == 4
        assert agent.action_size == 2
        assert agent.latent_size == 16
        assert agent.goal_size == 8
        assert agent.manager_horizon == 10
        assert isinstance(agent.state_encoder, StateEncoder)
        assert isinstance(agent.manager, ManagerNetwork)
        assert isinstance(agent.worker, WorkerNetwork)

    def test_agent_device_handling(self) -> None:
        """Test that agent handles device placement correctly."""
        # Arrange - create agent on CPU device
        device = "cpu"
        agent = FeudalAgent(state_size=4, action_size=2, device=device, seed=42)

        # Act - encode a state tensor
        state = torch.randn(4)
        latent = agent.encode_state(state)

        # Assert - verify latent is on correct device
        assert latent.device.type == device

    def test_encode_state(self) -> None:
        """Test state encoding functionality."""
        # Arrange - create agent and random state
        agent = FeudalAgent(state_size=4, action_size=2, latent_size=16, seed=42)
        state = torch.randn(4)

        # Act - encode the state
        latent = agent.encode_state(state)

        # Assert - verify latent shape and valid values
        assert latent.shape == (16,)
        assert not torch.isnan(latent).any()

    def test_select_action_updates_goal(self) -> None:
        """Test that select_action updates goals at horizon intervals."""
        # Arrange - create agent with horizon of 5 and latent state
        agent = FeudalAgent(
            state_size=4,
            action_size=2,
            latent_size=16,
            goal_size=8,
            manager_horizon=5,
            seed=42,
        )
        latent_state = torch.randn(16)

        # Act - select actions and track goal updates across horizon
        # First action creates initial goal
        action1 = agent.select_action(latent_state)
        goal1 = agent.current_goal.clone() if agent.current_goal is not None else None

        # Advance without reaching horizon
        for _ in range(4):
            agent.select_action(latent_state)

        goal2 = agent.current_goal.clone() if agent.current_goal is not None else None

        # Advance to horizon (5th step)
        agent.select_action(latent_state)
        goal3 = agent.current_goal.clone() if agent.current_goal is not None else None

        # Assert - verify goal behavior across horizon
        assert isinstance(action1, int)
        assert 0 <= action1 < 2
        assert goal1 is not None
        assert goal2 is not None
        assert goal3 is not None
        # Goal should not change until horizon
        assert torch.allclose(goal1, goal2)
        # Goal should change after horizon
        # Note: May be same by chance, so just check it was recomputed
        assert agent.steps_since_goal_update < agent.manager_horizon

    def test_intrinsic_reward_computation(self) -> None:
        """Test intrinsic reward calculation using cosine similarity."""
        # Arrange - create agent with state transition and goal
        agent = FeudalAgent(
            state_size=4, action_size=2, latent_size=16, goal_size=8, seed=42
        )
        latent_state = torch.randn(16)
        next_latent_state = latent_state + torch.randn(16) * 0.1  # Small change
        goal = torch.randn(8)

        # Act - compute intrinsic reward for the transition
        intrinsic = agent.intrinsic_reward(latent_state, next_latent_state, goal)

        # Assert - verify intrinsic reward is valid cosine similarity
        assert isinstance(intrinsic, float)
        assert -1.0 <= intrinsic <= 1.0  # Cosine similarity range

    def test_intrinsic_reward_aligned_goal(self) -> None:
        """Test intrinsic reward when transition aligns with goal."""
        # Arrange - create agent with transition aligned with goal
        agent = FeudalAgent(
            state_size=4, action_size=2, latent_size=8, goal_size=8, seed=42
        )
        latent_state = torch.zeros(8)
        goal = torch.ones(8)
        # Transition in direction of goal
        next_latent_state = latent_state + goal * 0.1

        # Act - compute intrinsic reward for aligned transition
        intrinsic = agent.intrinsic_reward(latent_state, next_latent_state, goal)

        # Assert - should be positive (aligned)
        assert intrinsic > 0.5

    def test_compute_n_step_return(self) -> None:
        """Test n-step return computation."""
        # Arrange - create agent with reward sequence
        agent = FeudalAgent(state_size=4, action_size=2, seed=42)
        rewards = [1.0, 2.0, 3.0]
        gamma = 0.9

        # Act - compute n-step return
        n_step_return = agent.compute_n_step_return(rewards, gamma)

        # Assert - verify correct discounted return calculation
        expected = 1.0 + 0.9 * 2.0 + 0.9**2 * 3.0
        assert abs(n_step_return - expected) < 1e-6

    def test_train_worker_buffer_too_small(self) -> None:
        """Test that worker training returns 0 when buffer is too small."""
        # Arrange - create agent with empty buffer
        agent = FeudalAgent(state_size=4, action_size=2, seed=42)

        # Act - attempt to train with insufficient data
        loss = agent.train_worker(batch_size=32)

        # Assert - verify loss is zero when buffer too small
        assert loss == 0.0

    def test_train_manager_buffer_too_small(self) -> None:
        """Test that manager training returns 0 when buffer is too small."""
        # Arrange - create agent with empty buffer
        agent = FeudalAgent(state_size=4, action_size=2, seed=42)

        # Act - attempt to train with insufficient data
        loss = agent.train_manager(batch_size=32)

        # Assert - verify loss is zero when buffer too small
        assert loss == 0.0

    def test_train_worker_with_experiences(self) -> None:
        """Test worker training with sufficient experiences."""
        # Arrange - create agent and populate buffer with experiences
        agent = FeudalAgent(state_size=4, action_size=2, latent_size=16, seed=42)

        # Add experiences to buffer
        for _ in range(40):
            agent.worker_buffer.append(
                {
                    "latent_state": torch.randn(agent.latent_size),
                    "action": np.random.randint(0, agent.action_size),
                    "reward": np.random.randn(),
                    "next_latent_state": torch.randn(agent.latent_size),
                    "goal": torch.randn(agent.goal_size),
                    "done": False,
                }
            )

        # Act - train worker with sufficient data
        loss = agent.train_worker(batch_size=32)

        # Assert - verify valid loss is returned
        assert isinstance(loss, float)
        assert loss > 0
        assert not np.isnan(loss)

    def test_train_manager_with_experiences(self) -> None:
        """Test manager training with sufficient experiences."""
        # Arrange - create agent and populate buffer with experiences
        agent = FeudalAgent(state_size=4, action_size=2, latent_size=16, seed=42)

        # Add experiences to buffer
        for _ in range(40):
            agent.manager_buffer.append(
                {
                    "latent_state": torch.randn(16),
                    "n_step_return": np.random.randn(),
                    "next_latent_state": torch.randn(16),
                    "done": False,
                }
            )

        # Act - train manager with sufficient data
        loss = agent.train_manager(batch_size=32)

        # Assert - verify valid loss is returned
        assert isinstance(loss, float)
        assert loss > 0
        assert not np.isnan(loss)

    def test_train_episode_basic(self) -> None:
        """Test basic episode training."""
        # Arrange - create environment and agent
        env = gym.make("CartPole-v1")
        agent = FeudalAgent(
            state_size=4,
            action_size=2,
            latent_size=16,
            goal_size=8,
            manager_horizon=5,
            seed=42,
        )

        # Act - train for one episode
        metrics = agent.train_episode(env, max_steps=50)

        # Assert - verify metrics are returned and valid
        assert "reward" in metrics
        assert "steps" in metrics
        assert "manager_updates" in metrics
        assert "avg_worker_loss" in metrics
        assert "avg_manager_loss" in metrics
        assert isinstance(metrics["reward"], float)
        assert isinstance(metrics["steps"], int)
        assert metrics["steps"] > 0
        assert metrics["steps"] <= 50

        env.close()

    def test_train_episode_manager_updates_at_horizons(self) -> None:
        """Test that manager updates occur at horizon intervals."""
        # Arrange - create environment and agent with horizon of 10
        env = gym.make("CartPole-v1")
        horizon = 10
        agent = FeudalAgent(
            state_size=4,
            action_size=2,
            latent_size=16,
            goal_size=8,
            manager_horizon=horizon,
            seed=42,
        )

        # Act - train for one episode
        metrics = agent.train_episode(env, max_steps=50)

        # Assert - verify manager updates match horizon intervals
        expected_updates = metrics["steps"] // horizon
        # Allow some tolerance for episode termination
        assert metrics["manager_updates"] <= expected_updates + 1
        assert metrics["manager_updates"] >= expected_updates - 1

        env.close()

    def test_episode_rewards_tracked(self) -> None:
        """Test that episode rewards are tracked."""
        # Arrange - create environment and agent
        env = gym.make("CartPole-v1")
        agent = FeudalAgent(state_size=4, action_size=2, seed=42)

        # Act - train for two episodes
        agent.train_episode(env, max_steps=50)
        agent.train_episode(env, max_steps=50)

        # Assert - verify rewards are tracked for both episodes
        assert len(agent.episode_rewards) == 2
        assert all(isinstance(r, float) for r in agent.episode_rewards)

        env.close()

    def test_get_statistics(self) -> None:
        """Test statistics retrieval."""
        # Arrange - create agent and populate tracking data
        agent = FeudalAgent(state_size=4, action_size=2, manager_horizon=10, seed=42)
        agent.episode_rewards = [10.0, 20.0, 30.0]
        agent.worker_losses = [0.5, 0.4, 0.3]
        agent.manager_losses = [0.2, 0.1]

        # Act - retrieve statistics
        stats = agent.get_statistics()

        # Assert - verify statistics are calculated correctly
        assert stats["total_episodes"] == 3
        assert stats["avg_reward"] == 20.0
        assert stats["avg_worker_loss"] > 0
        assert stats["avg_manager_loss"] > 0
        assert stats["manager_horizon"] == 10

    def test_gradient_clipping_applied(self) -> None:
        """Test that gradient clipping is applied during training."""
        # Arrange - create agent and add experiences with large rewards
        agent = FeudalAgent(state_size=4, action_size=2, latent_size=16, seed=42)

        # Add experiences with extreme values to potentially cause gradient explosion
        for _ in range(40):
            agent.worker_buffer.append(
                {
                    "latent_state": torch.randn(agent.latent_size),
                    "action": np.random.randint(0, agent.action_size),
                    "reward": np.random.randn() * 100,  # Large rewards
                    "next_latent_state": torch.randn(agent.latent_size),
                    "goal": torch.randn(agent.goal_size),
                    "done": False,
                }
            )

        # Act - train worker with extreme values
        loss = agent.train_worker(batch_size=32)

        # Assert - loss should be finite due to gradient clipping
        assert np.isfinite(loss)

    def test_advantage_normalization(self) -> None:
        """Test that advantage normalization stabilizes training."""
        # Arrange - create agent and add experiences with varying rewards
        agent = FeudalAgent(state_size=4, action_size=2, latent_size=16, seed=42)

        # Add experiences with varying rewards
        rewards = [1.0, 10.0, 0.1, 100.0] * 10
        for reward in rewards:
            agent.worker_buffer.append(
                {
                    "latent_state": torch.randn(agent.latent_size),
                    "action": np.random.randint(0, agent.action_size),
                    "reward": reward,
                    "next_latent_state": torch.randn(agent.latent_size),
                    "goal": torch.randn(agent.goal_size),
                    "done": False,
                }
            )

        # Act - train worker with varied rewards
        loss = agent.train_worker(batch_size=32)

        # Assert - should handle varied rewards without issues
        assert np.isfinite(loss)
        assert loss > 0

    def test_entropy_regularization_present(self) -> None:
        """Test that entropy coefficient affects training."""
        # Arrange - create two agents with different entropy coefficients
        agent_high_entropy = FeudalAgent(
            state_size=4, action_size=2, latent_size=16, entropy_coef=0.1, seed=42
        )
        agent_low_entropy = FeudalAgent(
            state_size=4, action_size=2, latent_size=16, entropy_coef=0.001, seed=43
        )

        # Add same experiences to both
        experiences = []
        for _ in range(40):
            exp = {
                "latent_state": torch.randn(agent_high_entropy.latent_size),
                "action": np.random.randint(0, agent_high_entropy.action_size),
                "reward": np.random.randn(),
                "next_latent_state": torch.randn(agent_high_entropy.latent_size),
                "goal": torch.randn(agent_high_entropy.goal_size),
                "done": False,
            }
            experiences.append(exp)

        for exp in experiences:
            agent_high_entropy.worker_buffer.append(exp)
            agent_low_entropy.worker_buffer.append(exp)

        # Act - train both agents on same experiences
        loss_high = agent_high_entropy.train_worker(batch_size=32)
        loss_low = agent_low_entropy.train_worker(batch_size=32)

        # Assert - different entropy coefficients should produce different losses
        assert loss_high != loss_low

    def test_deterministic_action_selection(self) -> None:
        """Test deterministic action selection mode."""
        # Arrange - create agent and latent state
        agent = FeudalAgent(state_size=4, action_size=2, latent_size=16, seed=42)
        latent_state = torch.randn(16)

        # Act - select actions deterministically twice
        action1 = agent.select_action(latent_state, deterministic=True)
        # Reset goal to same state
        agent.steps_since_goal_update = 0
        action2 = agent.select_action(latent_state, deterministic=True)

        # Assert - actions should be identical in deterministic mode
        assert action1 == action2

    def test_seed_reproducibility(self) -> None:
        """Test that setting seed produces reproducible results."""
        # Arrange - create two agents with same seed
        agent1 = FeudalAgent(state_size=4, action_size=2, latent_size=16, seed=42)
        agent2 = FeudalAgent(state_size=4, action_size=2, latent_size=16, seed=42)
        latent_state = torch.randn(16)

        # Act - select actions from both agents
        action1 = agent1.select_action(latent_state, deterministic=True)
        action2 = agent2.select_action(latent_state, deterministic=True)

        # Assert - actions should be identical with same seed
        assert action1 == action2


@pytest.mark.integration
class TestFeudalAgentIntegration:
    """Integration tests for Feudal RL agent."""

    def test_full_training_loop(self) -> None:
        """Test complete training loop over multiple episodes."""
        # Arrange - create environment and agent
        env = gym.make("CartPole-v1")
        agent = FeudalAgent(
            state_size=4,
            action_size=2,
            latent_size=16,
            goal_size=8,
            manager_horizon=10,
            learning_rate=0.001,
            seed=42,
        )

        # Act - train for multiple episodes (more episodes to ensure buffer fills)
        total_rewards = []
        for _ in range(5):
            metrics = agent.train_episode(env, max_steps=100)
            total_rewards.append(metrics["reward"])

        # Assert - verify training produces valid results
        assert len(total_rewards) == 5
        assert all(r > 0 for r in total_rewards)
        assert len(agent.episode_rewards) == 5
        # Buffer should accumulate experiences across episodes
        assert len(agent.worker_buffer) > 0

        env.close()

    def test_buffers_maintain_max_size(self) -> None:
        """Test that experience buffers don't exceed max size."""
        # Arrange - create environment and agent
        env = gym.make("CartPole-v1")
        max_buffer_size = 10000
        agent = FeudalAgent(
            state_size=4,
            action_size=2,
            manager_horizon=5,
            seed=42,
        )

        # Act - run multiple episodes to fill buffers
        for _ in range(5):
            agent.train_episode(env, max_steps=100)

        # Assert - verify buffers don't exceed max size
        assert len(agent.worker_buffer) <= max_buffer_size
        assert len(agent.manager_buffer) <= max_buffer_size

        env.close()

    def test_temporal_coordination_maintained(self) -> None:
        """Test that manager-worker temporal coordination is maintained."""
        # Arrange - create environment and agent with horizon of 8
        env = gym.make("CartPole-v1")
        horizon = 8
        agent = FeudalAgent(
            state_size=4,
            action_size=2,
            manager_horizon=horizon,
            seed=42,
        )

        # Act - train for one episode
        metrics = agent.train_episode(env, max_steps=80)

        # Assert - verify manager updates match horizon intervals
        # Manager should update approximately every horizon steps
        expected_manager_updates = metrics["steps"] // horizon
        assert abs(metrics["manager_updates"] - expected_manager_updates) <= 2

        env.close()


@pytest.mark.unit
class TestFeudalAgentInterpretability:
    """Test interpretability features of FeudalAgent."""

    def test_separate_learning_rates_default(self) -> None:
        """Test that separate learning rates are set to defaults correctly."""
        # Arrange - create agent without specifying separate learning rates
        # Act - initialize agent with base learning rate
        agent = FeudalAgent(state_size=4, action_size=2, learning_rate=0.0001)

        # Assert - verify manager and worker learning rates use recommended defaults
        assert agent.manager_lr == 1e-4  # Manager slower for stability
        assert agent.worker_lr == 3e-4  # Worker faster for adaptation

    def test_separate_learning_rates_custom(self) -> None:
        """Test that custom learning rates can be specified."""
        # Arrange - create agent with custom manager and worker learning rates
        # Act - initialize agent with custom LRs
        agent = FeudalAgent(
            state_size=4,
            action_size=2,
            manager_lr=5e-5,
            worker_lr=2e-4,
        )

        # Assert - verify custom learning rates are used
        assert agent.manager_lr == 5e-5
        assert agent.worker_lr == 2e-4

    def test_goal_kl_divergence_computation(self) -> None:
        """Test that KL divergence between goals is computed correctly."""
        # Arrange - create agent and two normalized goal vectors
        agent = FeudalAgent(state_size=4, action_size=2, latent_size=8, seed=42)
        goal1 = torch.randn(8)
        goal1 = torch.nn.functional.normalize(goal1, p=2, dim=-1)
        goal2 = torch.randn(8)
        goal2 = torch.nn.functional.normalize(goal2, p=2, dim=-1)

        # Act - compute KL divergence between goals
        kl_div = agent.compute_goal_kl_divergence(goal1, goal2)

        # Assert - verify KL divergence is a valid non-negative number
        assert isinstance(kl_div, float)
        assert kl_div >= 0.0
        assert not np.isnan(kl_div)
        assert not np.isinf(kl_div)

    def test_goal_kl_divergence_tracking(self) -> None:
        """Test that KL divergence is tracked during action selection."""
        # Arrange - create agent and set initial goal
        agent = FeudalAgent(state_size=4, action_size=2, latent_size=16, seed=42)
        state = torch.randn(4)
        latent_state = agent.encode_state(state)

        # Act - select actions to trigger goal updates
        # First action sets initial goal
        agent.select_action(latent_state)
        initial_kl_count = len(agent.goal_kl_divergences)

        # Force goal update by setting steps to horizon
        agent.steps_since_goal_update = agent.manager_horizon

        # Second action should compute KL divergence
        agent.select_action(latent_state)

        # Assert - verify KL divergence was tracked
        assert len(agent.goal_kl_divergences) == initial_kl_count + 1
        assert agent.goal_kl_divergences[-1] >= 0.0

    def test_gradient_norm_tracking_worker(self) -> None:
        """Test that worker gradient norms are tracked during training."""
        # Arrange - create agent and add experience to buffer
        agent = FeudalAgent(state_size=4, action_size=2, latent_size=16, seed=42)

        # Add sufficient experiences for training
        for _ in range(35):
            latent_state = agent.encode_state(torch.randn(4))
            next_latent_state = agent.encode_state(torch.randn(4))
            goal = torch.randn(16)
            agent.worker_buffer.append(
                {
                    "latent_state": latent_state,
                    "action": 0,
                    "reward": 0.5,
                    "next_latent_state": next_latent_state,
                    "goal": goal,
                    "done": False,
                }
            )

        # Act - train worker
        initial_grad_count = len(agent.worker_grad_norms)
        agent.train_worker(batch_size=32)

        # Assert - verify gradient norm was tracked
        assert len(agent.worker_grad_norms) == initial_grad_count + 1
        assert agent.worker_grad_norms[-1] >= 0.0

    def test_gradient_norm_tracking_manager(self) -> None:
        """Test that manager gradient norms are tracked during training."""
        # Arrange - create agent and add experience to buffer
        agent = FeudalAgent(state_size=4, action_size=2, latent_size=16, seed=42)

        # Add sufficient experiences for training
        for _ in range(35):
            latent_state = agent.encode_state(torch.randn(4))
            next_latent_state = agent.encode_state(torch.randn(4))
            agent.manager_buffer.append(
                {
                    "latent_state": latent_state,
                    "n_step_return": 1.0,
                    "next_latent_state": next_latent_state,
                    "done": False,
                }
            )

        # Act - train manager
        initial_grad_count = len(agent.manager_grad_norms)
        agent.train_manager(batch_size=32)

        # Assert - verify gradient norm was tracked
        assert len(agent.manager_grad_norms) == initial_grad_count + 1
        assert agent.manager_grad_norms[-1] >= 0.0

    def test_statistics_include_interpretability_metrics(self) -> None:
        """Test that statistics include all interpretability metrics."""
        # Arrange - create agent and populate metric lists
        agent = FeudalAgent(state_size=4, action_size=2, latent_size=16, seed=42)
        agent.goal_kl_divergences = [0.1, 0.2, 0.15]
        agent.manager_grad_norms = [0.5, 0.6, 0.55]
        agent.worker_grad_norms = [0.8, 0.9, 0.85]

        # Act - get statistics
        stats = agent.get_statistics()

        # Assert - verify all interpretability metrics are present
        assert "avg_goal_kl_divergence" in stats
        assert "avg_manager_grad_norm" in stats
        assert "avg_worker_grad_norm" in stats
        assert "manager_lr" in stats
        assert "worker_lr" in stats
        assert stats["avg_goal_kl_divergence"] > 0.0
        assert stats["avg_manager_grad_norm"] > 0.0
        assert stats["avg_worker_grad_norm"] > 0.0

    def test_train_episode_returns_interpretability_metrics(self) -> None:
        """Test that train_episode returns interpretability metrics."""
        # Arrange - create environment and agent
        env = gym.make("CartPole-v1")
        agent = FeudalAgent(
            state_size=4,
            action_size=2,
            latent_size=16,
            manager_horizon=5,
            seed=42,
        )

        # Act - train for one episode
        metrics = agent.train_episode(env, max_steps=50)

        # Assert - verify interpretability metrics are in return value
        assert "avg_goal_kl_divergence" in metrics
        assert "avg_worker_grad_norm" in metrics
        assert "avg_manager_grad_norm" in metrics
        # Metrics should be non-negative
        assert metrics["avg_goal_kl_divergence"] >= 0.0
        assert metrics["avg_worker_grad_norm"] >= 0.0
        assert metrics["avg_manager_grad_norm"] >= 0.0

        env.close()

    def test_optimizer_learning_rates_match_settings(self) -> None:
        """Test that optimizer learning rates match agent settings."""
        # Arrange - create agent with custom learning rates
        manager_lr = 8e-5
        worker_lr = 2.5e-4
        encoder_lr = 1e-4

        # Act - initialize agent with custom learning rates
        agent = FeudalAgent(
            state_size=4,
            action_size=2,
            manager_lr=manager_lr,
            worker_lr=worker_lr,
            learning_rate=encoder_lr,
        )

        # Assert - verify optimizer learning rates are set correctly
        assert agent.manager_optimizer.param_groups[0]["lr"] == manager_lr
        assert agent.worker_optimizer.param_groups[0]["lr"] == worker_lr
        assert agent.encoder_optimizer.param_groups[0]["lr"] == encoder_lr

    def test_goal_kl_divergence_identical_goals(self) -> None:
        """Test that KL divergence is zero for identical goals."""
        # Arrange - create agent and identical normalized goal vectors
        agent = FeudalAgent(state_size=4, action_size=2, latent_size=8, seed=42)
        goal = torch.randn(8)
        goal = torch.nn.functional.normalize(goal, p=2, dim=-1)

        # Act - compute KL divergence between identical goals
        kl_div = agent.compute_goal_kl_divergence(goal, goal)

        # Assert - verify KL divergence is very close to zero
        assert kl_div < 1e-5  # Should be essentially zero

    def test_previous_goal_tracking(self) -> None:
        """Test that previous goal is tracked correctly."""
        # Arrange - create agent and two different states
        agent = FeudalAgent(state_size=4, action_size=2, latent_size=16, seed=42)
        state1 = torch.randn(4)
        state2 = torch.randn(4)  # Different state for different goal
        latent_state1 = agent.encode_state(state1)
        latent_state2 = agent.encode_state(state2)

        # Act - select action to set initial goal
        agent.select_action(latent_state1)
        assert agent.current_goal is not None  # Goal should be set after first action
        first_goal = agent.current_goal.clone()

        # Force goal update with different state
        agent.steps_since_goal_update = agent.manager_horizon
        agent.select_action(latent_state2)

        # Assert - verify previous goal was stored
        assert agent.previous_goal is not None
        assert torch.allclose(agent.previous_goal, first_goal)
        assert agent.current_goal is not None
        # Different states should produce different goals (with high probability)
        goal_diff = torch.norm(agent.current_goal - first_goal).item()
        assert goal_diff > 0.01  # Goals should be meaningfully different


@pytest.mark.unit
class TestFeudalConfig:
    """Test Pydantic configuration validation for FeudalAgent."""

    def test_config_validates_negative_state_size(self) -> None:
        """Test that Config rejects negative state_size."""
        # Arrange - prepare invalid parameters with negative state_size
        # Act - attempt to create config with negative value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="state_size"):
            FeudalConfig(state_size=-1, action_size=4)

    def test_config_validates_zero_state_size(self) -> None:
        """Test that Config rejects zero state_size."""
        # Arrange - prepare invalid parameters with zero state_size
        # Act - attempt to create config with zero value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="state_size"):
            FeudalConfig(state_size=0, action_size=4)

    def test_config_validates_negative_action_size(self) -> None:
        """Test that Config rejects negative action_size."""
        # Arrange - prepare invalid parameters with negative action_size
        # Act - attempt to create config with negative value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="action_size"):
            FeudalConfig(state_size=4, action_size=-1)

    def test_config_validates_zero_action_size(self) -> None:
        """Test that Config rejects zero action_size."""
        # Arrange - prepare invalid parameters with zero action_size
        # Act - attempt to create config with zero value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="action_size"):
            FeudalConfig(state_size=4, action_size=0)

    def test_config_validates_negative_latent_size(self) -> None:
        """Test that Config rejects negative latent_size."""
        # Arrange - prepare invalid parameters with negative latent_size
        # Act - attempt to create config with negative value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="latent_size"):
            FeudalConfig(state_size=4, action_size=2, latent_size=-1)

    def test_config_validates_zero_latent_size(self) -> None:
        """Test that Config rejects zero latent_size."""
        # Arrange - prepare invalid parameters with zero latent_size
        # Act - attempt to create config with zero value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="latent_size"):
            FeudalConfig(state_size=4, action_size=2, latent_size=0)

    def test_config_validates_negative_goal_size(self) -> None:
        """Test that Config rejects negative goal_size."""
        # Arrange - prepare invalid parameters with negative goal_size
        # Act - attempt to create config with negative value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="goal_size"):
            FeudalConfig(state_size=4, action_size=2, goal_size=-1)

    def test_config_validates_zero_goal_size(self) -> None:
        """Test that Config rejects zero goal_size."""
        # Arrange - prepare invalid parameters with zero goal_size
        # Act - attempt to create config with zero value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="goal_size"):
            FeudalConfig(state_size=4, action_size=2, goal_size=0)

    def test_config_validates_negative_hidden_size(self) -> None:
        """Test that Config rejects negative hidden_size."""
        # Arrange - prepare invalid parameters with negative hidden_size
        # Act - attempt to create config with negative value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="hidden_size"):
            FeudalConfig(state_size=4, action_size=2, hidden_size=-1)

    def test_config_validates_zero_hidden_size(self) -> None:
        """Test that Config rejects zero hidden_size."""
        # Arrange - prepare invalid parameters with zero hidden_size
        # Act - attempt to create config with zero value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="hidden_size"):
            FeudalConfig(state_size=4, action_size=2, hidden_size=0)

    def test_config_validates_negative_manager_horizon(self) -> None:
        """Test that Config rejects negative manager_horizon."""
        # Arrange - prepare invalid parameters with negative manager_horizon
        # Act - attempt to create config with negative value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="manager_horizon"):
            FeudalConfig(state_size=4, action_size=2, manager_horizon=-1)

    def test_config_validates_zero_manager_horizon(self) -> None:
        """Test that Config rejects zero manager_horizon."""
        # Arrange - prepare invalid parameters with zero manager_horizon
        # Act - attempt to create config with zero value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="manager_horizon"):
            FeudalConfig(state_size=4, action_size=2, manager_horizon=0)

    def test_config_validates_negative_learning_rate(self) -> None:
        """Test that Config rejects negative learning_rate."""
        # Arrange - prepare invalid parameters with negative learning_rate
        # Act - attempt to create config with negative value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="learning_rate"):
            FeudalConfig(state_size=4, action_size=2, learning_rate=-0.001)

    def test_config_validates_zero_learning_rate(self) -> None:
        """Test that Config rejects zero learning_rate."""
        # Arrange - prepare invalid parameters with zero learning_rate
        # Act - attempt to create config with zero value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="learning_rate"):
            FeudalConfig(state_size=4, action_size=2, learning_rate=0.0)

    def test_config_validates_learning_rate_too_high(self) -> None:
        """Test that Config rejects learning_rate > 1."""
        # Arrange - prepare invalid parameters with learning_rate > 1
        # Act - attempt to create config with too high value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="learning_rate"):
            FeudalConfig(state_size=4, action_size=2, learning_rate=1.5)

    def test_config_validates_negative_manager_lr(self) -> None:
        """Test that Config rejects negative manager_lr."""
        # Arrange - prepare invalid parameters with negative manager_lr
        # Act - attempt to create config with negative value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="manager_lr"):
            FeudalConfig(state_size=4, action_size=2, manager_lr=-0.001)

    def test_config_validates_zero_manager_lr(self) -> None:
        """Test that Config rejects zero manager_lr."""
        # Arrange - prepare invalid parameters with zero manager_lr
        # Act - attempt to create config with zero value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="manager_lr"):
            FeudalConfig(state_size=4, action_size=2, manager_lr=0.0)

    def test_config_validates_manager_lr_too_high(self) -> None:
        """Test that Config rejects manager_lr > 1."""
        # Arrange - prepare invalid parameters with manager_lr > 1
        # Act - attempt to create config with too high value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="manager_lr"):
            FeudalConfig(state_size=4, action_size=2, manager_lr=1.5)

    def test_config_validates_negative_worker_lr(self) -> None:
        """Test that Config rejects negative worker_lr."""
        # Arrange - prepare invalid parameters with negative worker_lr
        # Act - attempt to create config with negative value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="worker_lr"):
            FeudalConfig(state_size=4, action_size=2, worker_lr=-0.001)

    def test_config_validates_zero_worker_lr(self) -> None:
        """Test that Config rejects zero worker_lr."""
        # Arrange - prepare invalid parameters with zero worker_lr
        # Act - attempt to create config with zero value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="worker_lr"):
            FeudalConfig(state_size=4, action_size=2, worker_lr=0.0)

    def test_config_validates_worker_lr_too_high(self) -> None:
        """Test that Config rejects worker_lr > 1."""
        # Arrange - prepare invalid parameters with worker_lr > 1
        # Act - attempt to create config with too high value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="worker_lr"):
            FeudalConfig(state_size=4, action_size=2, worker_lr=1.5)

    def test_config_validates_gamma_too_low(self) -> None:
        """Test that Config rejects gamma <= 0."""
        # Arrange - prepare invalid parameters with gamma <= 0
        # Act - attempt to create config with too low value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="gamma"):
            FeudalConfig(state_size=4, action_size=2, gamma=0.0)

    def test_config_validates_gamma_too_high(self) -> None:
        """Test that Config rejects gamma >= 1."""
        # Arrange - prepare invalid parameters with gamma >= 1
        # Act - attempt to create config with too high value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="gamma"):
            FeudalConfig(state_size=4, action_size=2, gamma=1.0)

    def test_config_validates_negative_entropy_coef(self) -> None:
        """Test that Config rejects negative entropy_coef."""
        # Arrange - prepare invalid parameters with negative entropy_coef
        # Act - attempt to create config with negative value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="entropy_coef"):
            FeudalConfig(state_size=4, action_size=2, entropy_coef=-0.1)

    def test_config_validates_invalid_device(self) -> None:
        """Test that Config rejects invalid device strings."""
        # Arrange - prepare invalid parameters with unknown device
        # Act - attempt to create config with invalid device string
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="device"):
            FeudalConfig(state_size=4, action_size=2, device="invalid_device")

    def test_config_accepts_valid_cpu_device(self) -> None:
        """Test that Config accepts 'cpu' device."""
        # Arrange - set up test configuration

        # Act - create config object
        config = FeudalConfig(state_size=4, action_size=2, device="cpu")

        # Assert - verify device is set to cpu
        assert config.device == "cpu"

    def test_config_accepts_valid_cuda_device(self) -> None:
        """Test that Config accepts 'cuda' device."""
        # Arrange - set up test configuration

        # Act - create config object
        config = FeudalConfig(state_size=4, action_size=2, device="cuda")

        # Assert - verify device is set to cuda
        assert config.device == "cuda"

    def test_config_accepts_valid_cuda_numbered_device(self) -> None:
        """Test that Config accepts 'cuda:N' device."""
        # Arrange - set up test configuration

        # Act - create config object
        config = FeudalConfig(state_size=4, action_size=2, device="cuda:0")

        # Assert - verify device is set to cuda:0
        assert config.device == "cuda:0"

    def test_config_accepts_mps_device(self) -> None:
        """Test that Config accepts 'mps' device for Apple Silicon."""
        # Arrange - set up test configuration

        # Act - create config object
        config = FeudalConfig(state_size=4, action_size=2, device="mps")

        # Assert - verify device is set to mps
        assert config.device == "mps"

    def test_config_defaults_are_set_correctly(self) -> None:
        """Test that Config sets default values correctly."""
        # Arrange - set up test configuration

        # Act - create config object
        config = FeudalConfig(state_size=4, action_size=2)

        # Assert - verify all default values are correct
        assert config.state_size == 4
        assert config.action_size == 2
        assert config.latent_size == 64
        assert config.goal_size is None
        assert config.hidden_size == 256
        assert config.manager_horizon == 10
        assert config.learning_rate == 0.0001
        assert config.manager_lr is None
        assert config.worker_lr is None
        assert config.gamma == 0.99
        assert config.entropy_coef == 0.01
        assert config.device == "cpu"
        assert config.seed is None

    def test_backwards_compatible_kwargs_initialization(self) -> None:
        """Test that agent accepts kwargs for backwards compatibility."""
        # Arrange - set up test configuration

        # Act - create agent with kwargs
        agent = FeudalAgent(state_size=4, action_size=2, latent_size=32, seed=42)

        # Assert - verify agent initialized with correct values
        assert agent.state_size == 4
        assert agent.action_size == 2
        assert agent.latent_size == 32
        assert agent.config.state_size == 4
        assert agent.config.action_size == 2
        assert agent.config.latent_size == 32

    def test_config_object_initialization(self) -> None:
        """Test that agent accepts config object."""
        # Arrange - create config object
        config = FeudalConfig(state_size=4, action_size=2, latent_size=32, seed=42)

        # Act - initialize agent with config
        agent = FeudalAgent(config=config)

        # Assert - verify agent uses config correctly
        assert agent.config == config
        assert agent.state_size == 4
        assert agent.action_size == 2
        assert agent.latent_size == 32

    def test_config_object_takes_precedence(self) -> None:
        """Test that config object takes precedence over kwargs."""
        # Arrange - create config with specific values
        config = FeudalConfig(state_size=4, action_size=2, latent_size=32)

        # Act - create agent with config and conflicting kwargs
        agent = FeudalAgent(config=config, state_size=100, action_size=100)

        # Assert - verify config values are used not kwargs
        assert agent.state_size == 4
        assert agent.action_size == 2

    def test_config_validation_happens_automatically(self) -> None:
        """Test that validation happens automatically during init."""
        # Arrange - prepare test parameters
        # Act - perform operation
        # Assert - verify result - should raise ValidationError during init
        with pytest.raises(ValidationError, match="state_size"):
            FeudalAgent(state_size=-1, action_size=2)
