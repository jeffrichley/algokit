"""Additional tests for Policy Gradient to improve coverage with proper AAA structure."""

import numpy as np
import pytest
import torch

from algokit.algorithms.reinforcement_learning.policy_gradient import (
    BaselineNetwork,
    PolicyGradientAgent,
    PolicyNetwork,
    RolloutExperience,
)


class MockEnvironment:
    """Mock environment for testing."""

    def __init__(self, state_size: int = 4, action_size: int = 2):
        self.state_size = state_size
        self.action_size = action_size
        self.state = np.random.random(state_size)
        self.step_count = 0
        self.episode_count = 0

    def reset(self):
        self.state = np.random.random(self.state_size)
        self.step_count = 0
        self.episode_count += 1
        return self.state

    def step(self, action):
        self.step_count += 1
        reward = 1.0 if action == 0 else -0.5
        self.state = np.random.random(self.state_size)
        # Ensure episodes terminate properly
        done = self.step_count >= 3  # Episode ends after 3 steps
        return self.state, reward, done, {}


class TestPolicyGradientCoverage:
    """Additional tests to improve coverage with proper AAA structure."""

    @pytest.mark.unit
    def test_collect_rollout_fixed_steps(self) -> None:
        """Test collect_rollout with fixed number of steps."""
        # Arrange - Set up agent and mock environment for rollout collection
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, continuous_actions=False
        )
        env = MockEnvironment()

        # Act - Collect rollout data with fixed number of steps
        rollout_data = agent.collect_rollout(env, n_steps=10)

        # Assert - Verify correct number of experiences and their properties
        assert len(rollout_data) == 10
        for exp in rollout_data:
            assert isinstance(exp.state, np.ndarray)
            assert exp.state.shape == (4,)
            assert isinstance(exp.action, int | np.integer)
            assert isinstance(exp.reward, float | np.floating)
            assert isinstance(exp.log_prob, float | np.floating)
            assert isinstance(exp.value, float | np.floating)
            assert isinstance(exp.done, bool)

    @pytest.mark.unit
    def test_collect_rollout_full_episodes(self) -> None:
        """Test collect_rollout with full episodes."""
        # Arrange - Set up agent and environment for full episode collection
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, continuous_actions=False
        )
        env = MockEnvironment()

        # Act - Collect rollout data for full episodes with max length limit
        rollout_data = agent.collect_rollout(env, n_steps=None, max_episode_length=3)

        # Assert - Verify episodes are collected and properly terminated
        assert len(rollout_data) >= 3  # At least one episode
        done_indices = [i for i, exp in enumerate(rollout_data) if exp.done]
        assert len(done_indices) > 0

    @pytest.mark.unit
    def test_collect_rollout_max_episode_length(self) -> None:
        """Test collect_rollout with max episode length."""
        # Arrange - Set up agent and environment with episode length constraints
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, continuous_actions=False
        )
        env = MockEnvironment()

        # Act - Collect rollout data with max episode length constraint
        rollout_data = agent.collect_rollout(env, n_steps=15, max_episode_length=3)

        # Assert - Verify correct number of steps and episode length constraints
        assert len(rollout_data) == 15
        episode_lengths = []
        current_length = 0
        for exp in rollout_data:
            current_length += 1
            if exp.done:
                episode_lengths.append(current_length)
                current_length = 0

        if episode_lengths:
            assert max(episode_lengths) <= 3

    @pytest.mark.unit
    def test_evaluate_with_baseline(self) -> None:
        """Test evaluate method with baseline enabled."""
        # Arrange - Set up agent with baseline and test state
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, continuous_actions=False, use_baseline=True
        )
        state = np.array([1.0, 2.0, 3.0, 4.0])

        # Act - Evaluate state to get action, log_prob, and value
        action, log_prob, value = agent.evaluate(state)

        # Assert - Verify action, log_prob, and value are valid
        # For discrete actions, action should be a scalar
        assert isinstance(action, (int, np.integer))
        assert isinstance(log_prob, (float, np.floating))
        assert isinstance(value, (float, np.floating))
        assert np.isfinite(value)

    @pytest.mark.unit
    def test_evaluate_without_baseline(self) -> None:
        """Test evaluate method without baseline."""
        # Arrange - Set up agent without baseline and test state
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, continuous_actions=False, use_baseline=False
        )
        state = np.array([1.0, 2.0, 3.0, 4.0])

        # Act - Evaluate state to get action, log_prob, and value
        action, log_prob, value = agent.evaluate(state)

        # Assert - Verify action, log_prob, and value (should be 0 without baseline)
        assert isinstance(action, (int, np.integer))
        assert isinstance(log_prob, (float, np.floating))
        assert isinstance(value, (float, np.floating))
        assert value == 0.0  # Should be 0 when no baseline

    @pytest.mark.unit
    def test_compute_gae_advantages(self) -> None:
        """Test compute_gae_advantages method."""
        # Arrange - Set up agent with GAE and test data
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, use_gae=True, gae_lambda=0.95
        )
        rewards = [1.0, 0.5, -0.5, 2.0]
        values = [0.8, 0.6, 0.4, 0.2]
        dones = [False, False, False, True]

        # Act - Compute GAE advantages from rewards, values, and dones
        advantages = agent.compute_gae_advantages(rewards, values, dones)

        # Assert - Verify advantages are computed correctly
        assert len(advantages) == len(rewards)
        assert all(isinstance(adv, float) for adv in advantages)
        assert all(np.isfinite(adv) for adv in advantages)

    @pytest.mark.unit
    def test_compute_gae_advantages_with_done_episodes(self) -> None:
        """Test compute_gae_advantages with multiple done episodes."""
        # Arrange - Set up agent with GAE and test data with multiple episodes
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, use_gae=True, gae_lambda=0.95
        )
        rewards = [1.0, 0.5, -0.5, 2.0, 1.5, 0.0]
        values = [0.8, 0.6, 0.4, 0.2, 0.1, 0.0]
        dones = [False, False, True, False, False, True]

        # Act - Compute GAE advantages with multiple episode boundaries
        advantages = agent.compute_gae_advantages(rewards, values, dones)

        # Assert - Verify advantages are computed correctly for multiple episodes
        assert len(advantages) == len(rewards)
        assert all(isinstance(adv, float) for adv in advantages)
        assert all(np.isfinite(adv) for adv in advantages)

    @pytest.mark.unit
    def test_learn_with_gae(self) -> None:
        """Test learn method with GAE enabled."""
        # Arrange - Set up agent with GAE and sample rollout data
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
                state=np.array([1.0, 2.0, 3.0, 4.0]),
                action=0,
                reward=1.0,
                log_prob=0.5,
                value=0.8,
                done=False,
            ),
            RolloutExperience(
                state=np.array([2.0, 3.0, 4.0, 5.0]),
                action=1,
                reward=2.0,
                log_prob=0.3,
                value=1.2,
                done=True,
            ),
        ]

        # Act - Learn from rollout data using GAE
        metrics = agent.learn(rollout_data)

        # Assert - Verify learning metrics are returned correctly
        assert "policy_loss" in metrics
        assert "baseline_loss" in metrics
        assert "entropy_loss" in metrics
        assert "mean_return" in metrics
        assert "mean_advantage" in metrics
        assert isinstance(metrics["policy_loss"], float)
        assert isinstance(metrics["baseline_loss"], float)

    @pytest.mark.unit
    def test_learn_with_gae_and_normalize_advantages(self) -> None:
        """Test learn method with GAE and advantage normalization."""
        # Arrange - Set up agent with GAE and advantage normalization
        agent = PolicyGradientAgent(
            state_size=4,
            action_size=2,
            use_baseline=True,
            use_gae=True,
            gae_lambda=0.95,
            normalize_advantages=True,
            continuous_actions=False,
        )

        rollout_data = [
            RolloutExperience(
                state=np.array([1.0, 2.0, 3.0, 4.0]),
                action=0,
                reward=1.0,
                log_prob=0.5,
                value=0.8,
                done=False,
            ),
            RolloutExperience(
                state=np.array([2.0, 3.0, 4.0, 5.0]),
                action=1,
                reward=2.0,
                log_prob=0.3,
                value=1.2,
                done=False,
            ),
            RolloutExperience(
                state=np.array([3.0, 4.0, 5.0, 6.0]),
                action=0,
                reward=1.5,
                log_prob=0.4,
                value=1.0,
                done=True,
            ),
        ]

        # Act - Learn from rollout data with GAE and normalization
        metrics = agent.learn(rollout_data)

        # Assert - Verify learning metrics are returned correctly
        assert "policy_loss" in metrics
        assert "baseline_loss" in metrics
        assert "entropy_loss" in metrics
        assert "mean_return" in metrics
        assert "mean_advantage" in metrics
        assert isinstance(metrics["policy_loss"], float)
        assert isinstance(metrics["baseline_loss"], float)

    @pytest.mark.unit
    def test_learn_without_gae(self) -> None:
        """Test learn method without GAE."""
        # Arrange - Set up agent without GAE and sample rollout data
        agent = PolicyGradientAgent(
            state_size=4,
            action_size=2,
            use_baseline=True,
            use_gae=False,
            continuous_actions=False,
        )

        rollout_data = [
            RolloutExperience(
                state=np.array([1.0, 2.0, 3.0, 4.0]),
                action=0,
                reward=1.0,
                log_prob=0.5,
                value=0.8,
                done=False,
            ),
            RolloutExperience(
                state=np.array([2.0, 3.0, 4.0, 5.0]),
                action=1,
                reward=2.0,
                log_prob=0.3,
                value=1.2,
                done=True,
            ),
        ]

        # Act - Learn from rollout data without GAE
        metrics = agent.learn(rollout_data)

        # Assert - Verify learning metrics are returned correctly
        assert "policy_loss" in metrics
        assert "baseline_loss" in metrics
        assert "entropy_loss" in metrics
        assert "mean_return" in metrics
        assert "mean_advantage" in metrics
        assert isinstance(metrics["policy_loss"], float)
        assert isinstance(metrics["baseline_loss"], float)

    @pytest.mark.unit
    def test_learn_continuous_actions(self) -> None:
        """Test learn method with continuous actions."""
        # Arrange - Set up agent with continuous actions and sample rollout data
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, use_baseline=True, continuous_actions=True
        )

        rollout_data = [
            RolloutExperience(
                state=np.array([1.0, 2.0, 3.0, 4.0]),
                action=np.array([0.5, -0.3]),
                reward=1.0,
                log_prob=0.5,
                value=0.8,
                done=False,
            ),
            RolloutExperience(
                state=np.array([2.0, 3.0, 4.0, 5.0]),
                action=np.array([-0.2, 0.7]),
                reward=2.0,
                log_prob=0.3,
                value=1.2,
                done=True,
            ),
        ]

        # Act - Learn from rollout data with continuous actions
        metrics = agent.learn(rollout_data)

        # Assert - Verify learning metrics are returned correctly
        assert "policy_loss" in metrics
        assert "baseline_loss" in metrics
        assert "entropy_loss" in metrics
        assert "mean_return" in metrics
        assert "mean_advantage" in metrics
        assert isinstance(metrics["policy_loss"], float)
        assert isinstance(metrics["baseline_loss"], float)

    @pytest.mark.unit
    def test_set_training_mode_with_baseline(self) -> None:
        """Test set_training_mode with baseline enabled."""
        # Arrange - Set up agent with baseline enabled
        agent = PolicyGradientAgent(state_size=4, action_size=2, use_baseline=True)

        # Act - Set training mode to False
        agent.set_training_mode(False)

        # Assert - Verify training mode is set correctly for all networks
        assert agent.training is False
        assert agent.policy.training is False
        assert agent.baseline is not None
        assert agent.baseline.training is False

        # Act - Set back to training mode
        agent.set_training_mode(True)

        # Assert - Verify training mode is set correctly for all networks
        assert agent.training is True
        assert agent.policy.training is True
        assert agent.baseline is not None
        assert agent.baseline.training is True

    @pytest.mark.unit
    def test_set_training_mode_without_baseline(self) -> None:
        """Test set_training_mode without baseline."""
        # Arrange - Set up agent without baseline
        agent = PolicyGradientAgent(state_size=4, action_size=2, use_baseline=False)

        # Act - Set training mode to False
        agent.set_training_mode(False)

        # Assert - Verify training mode is set correctly for policy only
        assert agent.training is False
        assert agent.policy.training is False
        assert agent.baseline is None

        # Act - Set back to training mode
        agent.set_training_mode(True)

        # Assert - Verify training mode is set correctly for policy only
        assert agent.training is True
        assert agent.policy.training is True
        assert agent.baseline is None

    @pytest.mark.unit
    def test_learn_with_single_experience(self) -> None:
        """Test learn method with single experience."""
        # Arrange - Set up agent and single experience data
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, use_baseline=True, continuous_actions=False
        )

        rollout_data = [
            RolloutExperience(
                state=np.array([1.0, 2.0, 3.0, 4.0]),
                action=0,
                reward=1.0,
                log_prob=0.5,
                value=0.8,
                done=True,
            ),
        ]

        # Act - Learn from single experience
        metrics = agent.learn(rollout_data)

        # Assert - Verify learning metrics are returned correctly
        assert "policy_loss" in metrics
        assert "baseline_loss" in metrics
        assert "entropy_loss" in metrics
        assert "mean_return" in metrics
        assert "mean_advantage" in metrics
        assert isinstance(metrics["policy_loss"], float)
        assert isinstance(metrics["baseline_loss"], float)

    @pytest.mark.unit
    def test_learn_without_advantage_normalization(self) -> None:
        """Test learn method without advantage normalization."""
        # Arrange - Set up agent without advantage normalization
        agent = PolicyGradientAgent(
            state_size=4,
            action_size=2,
            use_baseline=True,
            normalize_advantages=False,
            continuous_actions=False,
        )

        rollout_data = [
            RolloutExperience(
                state=np.array([1.0, 2.0, 3.0, 4.0]),
                action=0,
                reward=1.0,
                log_prob=0.5,
                value=0.8,
                done=False,
            ),
            RolloutExperience(
                state=np.array([2.0, 3.0, 4.0, 5.0]),
                action=1,
                reward=2.0,
                log_prob=0.3,
                value=1.2,
                done=True,
            ),
        ]

        # Act - Learn from rollout data without advantage normalization
        metrics = agent.learn(rollout_data)

        # Assert - Verify learning metrics are returned correctly
        assert "policy_loss" in metrics
        assert "baseline_loss" in metrics
        assert "entropy_loss" in metrics
        assert "mean_return" in metrics
        assert "mean_advantage" in metrics
        assert isinstance(metrics["policy_loss"], float)
        assert isinstance(metrics["baseline_loss"], float)

    @pytest.mark.unit
    def test_learn_with_single_experience_no_normalization(self) -> None:
        """Test learn method with single experience and no normalization."""
        # Arrange - Set up agent with normalization but single experience
        agent = PolicyGradientAgent(
            state_size=4,
            action_size=2,
            use_baseline=True,
            normalize_advantages=True,
            continuous_actions=False,
        )

        rollout_data = [
            RolloutExperience(
                state=np.array([1.0, 2.0, 3.0, 4.0]),
                action=0,
                reward=1.0,
                log_prob=0.5,
                value=0.8,
                done=True,
            ),
        ]

        # Act - Learn from single experience with normalization enabled
        metrics = agent.learn(rollout_data)

        # Assert - Verify learning metrics are returned correctly
        assert "policy_loss" in metrics
        assert "baseline_loss" in metrics
        assert "entropy_loss" in metrics
        assert "mean_return" in metrics
        assert "mean_advantage" in metrics
        assert isinstance(metrics["policy_loss"], float)
        assert isinstance(metrics["baseline_loss"], float)

    @pytest.mark.unit
    def test_act_method_backward_compatibility(self) -> None:
        """Test act method for backward compatibility."""
        # Arrange - Set up agent and test state
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, continuous_actions=False
        )
        state = np.array([1.0, 2.0, 3.0, 4.0])

        # Act - Get action using act method
        action = agent.act(state, training=True)

        # Assert - Verify action is valid discrete action
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 2

    @pytest.mark.unit
    def test_act_method_continuous_actions(self) -> None:
        """Test act method with continuous actions."""
        # Arrange - Set up agent with continuous actions and test state
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, continuous_actions=True
        )
        state = np.array([1.0, 2.0, 3.0, 4.0])

        # Act - Get action using act method
        action = agent.act(state, training=True)

        # Assert - Verify action is valid continuous action
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)
        assert np.isfinite(action).all()

    @pytest.mark.unit
    def test_compute_returns_method(self) -> None:
        """Test compute_returns method."""
        # Arrange - Set up agent and test data
        agent = PolicyGradientAgent(state_size=4, action_size=2)
        rewards = [1.0, 0.5, -0.5, 2.0]
        values = [0.8, 0.6, 0.4, 0.2]
        dones = [False, False, False, True]

        # Act - Compute returns from rewards, values, and dones
        returns = agent.compute_returns(rewards, values, dones)

        # Assert - Verify returns are computed correctly
        assert len(returns) == len(rewards)
        assert all(isinstance(ret, float) for ret in returns)
        assert all(np.isfinite(ret) for ret in returns)

    @pytest.mark.unit
    def test_compute_returns_with_multiple_done_episodes(self) -> None:
        """Test compute_returns with multiple done episodes."""
        # Arrange - Set up agent and test data with multiple episodes
        agent = PolicyGradientAgent(state_size=4, action_size=2)
        rewards = [1.0, 0.5, -0.5, 2.0, 1.5, 0.0]
        values = [0.8, 0.6, 0.4, 0.2, 0.1, 0.0]
        dones = [False, False, True, False, False, True]

        # Act - Compute returns with multiple episode boundaries
        returns = agent.compute_returns(rewards, values, dones)

        # Assert - Verify returns are computed correctly for multiple episodes
        assert len(returns) == len(rewards)
        assert all(isinstance(ret, float) for ret in returns)
        assert all(np.isfinite(ret) for ret in returns)

    @pytest.mark.unit
    def test_policy_network_default_hidden_sizes(self) -> None:
        """Test PolicyNetwork with default hidden sizes."""
        # Arrange & Act - Create network with default parameters
        network = PolicyNetwork(state_size=4, action_size=2)

        # Assert - Verify network structure with default hidden sizes
        assert network.state_size == 4
        assert network.action_size == 2
        assert (
            len(network.network) == 6
        )  # 2 layers * 3 components (Linear, ReLU, Dropout)

    @pytest.mark.unit
    def test_baseline_network_default_hidden_sizes(self) -> None:
        """Test BaselineNetwork with default hidden sizes."""
        # Arrange & Act - Create network with default parameters
        network = BaselineNetwork(state_size=4)

        # Assert - Verify network structure with default hidden sizes
        assert len(network.network) == 7  # 2 layers * 3 components + output layer

    @pytest.mark.unit
    def test_policy_network_continuous_actions_log_std_clamping(self) -> None:
        """Test PolicyNetwork continuous actions with log_std clamping."""
        # Arrange - Set up network with continuous actions and test state
        network = PolicyNetwork(state_size=4, action_size=2, continuous_actions=True)
        state = torch.randn(1, 4)

        # Act - Forward pass and action sampling
        mean, log_std = network.forward(state)
        action, log_prob = network.get_action(state)

        # Assert - Verify outputs are valid and finite
        assert mean.shape == (1, 2)
        assert log_std.shape == (1, 2)
        assert torch.isfinite(mean).all()
        assert torch.isfinite(log_std).all()
        assert action.shape == (1, 2)
        assert log_prob.shape == (1, 1)
        assert torch.isfinite(action).all()
        assert torch.isfinite(log_prob).all()

    @pytest.mark.unit
    def test_agent_initialization_with_all_parameters(self) -> None:
        """Test agent initialization with all parameters."""
        # Arrange & Act - Create agent with all parameters
        agent = PolicyGradientAgent(
            state_size=4,
            action_size=2,
            learning_rate=0.001,
            gamma=0.99,
            use_baseline=True,
            hidden_sizes=[64, 32],
            dropout_rate=0.1,
            continuous_actions=False,
            device="cpu",
            seed=42,
            entropy_coefficient=0.01,
            use_gae=True,
            gae_lambda=0.95,
            normalize_advantages=True,
        )

        # Assert - Verify all parameters are set correctly
        assert agent.state_size == 4
        assert agent.action_size == 2
        assert agent.learning_rate == 0.001
        assert agent.gamma == 0.99
        assert agent.use_baseline is True
        assert agent.continuous_actions is False
        assert agent.device.type == "cpu"
        assert agent.entropy_coefficient == 0.01
        assert agent.use_gae is True
        assert agent.gae_lambda == 0.95
        assert agent.normalize_advantages is True
        assert agent.policy is not None
        assert agent.baseline is not None

    @pytest.mark.unit
    def test_learn_with_empty_rollout(self) -> None:
        """Test learn method with empty rollout data."""
        # Arrange - Set up agent with empty rollout data
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, use_baseline=True, continuous_actions=False
        )
        rollout_data = []

        # Act - Learn from empty rollout data
        metrics = agent.learn(rollout_data)

        # Assert - Verify default metrics are returned
        assert metrics["policy_loss"] == 0.0
        assert metrics["baseline_loss"] == 0.0
        assert metrics["entropy_loss"] == 0.0
        assert metrics["mean_return"] == 0.0
        assert metrics["mean_advantage"] == 0.0

    @pytest.mark.unit
    def test_save_and_load_with_baseline(self) -> None:
        """Test save and load methods with baseline."""
        # Arrange - Set up agent with baseline and filepath
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, use_baseline=True, continuous_actions=False
        )
        filepath = "/tmp/test_agent_baseline.pth"

        # Act - Save and load agent with baseline
        agent.save(filepath)
        new_agent = PolicyGradientAgent(
            state_size=4, action_size=2, use_baseline=True, continuous_actions=False
        )
        new_agent.load(filepath)

        # Assert - Verify loaded agent structure matches original
        assert new_agent.state_size == agent.state_size
        assert new_agent.action_size == agent.action_size
        assert new_agent.use_baseline == agent.use_baseline
        assert new_agent.baseline is not None

    @pytest.mark.unit
    def test_save_and_load_without_baseline(self) -> None:
        """Test save and load methods without baseline."""
        # Arrange - Set up agent without baseline and filepath
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, use_baseline=False, continuous_actions=False
        )
        filepath = "/tmp/test_agent_no_baseline.pth"

        # Act - Save and load agent without baseline
        agent.save(filepath)
        new_agent = PolicyGradientAgent(
            state_size=4, action_size=2, use_baseline=False, continuous_actions=False
        )
        new_agent.load(filepath)

        # Assert - Verify loaded agent structure matches original
        assert new_agent.state_size == agent.state_size
        assert new_agent.action_size == agent.action_size
        assert new_agent.use_baseline == agent.use_baseline
        assert new_agent.baseline is None

    @pytest.mark.unit
    def test_load_with_missing_baseline_optimizer(self) -> None:
        """Test load method with missing baseline optimizer state."""
        # Arrange - Set up agent and create incomplete save state
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, use_baseline=True, continuous_actions=False
        )
        filepath = "/tmp/test_agent_incomplete.pth"

        # Create incomplete state dict (missing baseline optimizer)
        state = {
            "policy_state_dict": agent.policy.state_dict(),
            "policy_optimizer_state_dict": agent.policy_optimizer.state_dict(),
            "state_size": agent.state_size,
            "action_size": agent.action_size,
            "learning_rate": agent.learning_rate,
            "gamma": agent.gamma,
            "use_baseline": agent.use_baseline,
            "continuous_actions": agent.continuous_actions,
            "baseline_state_dict": agent.baseline.state_dict()
            if agent.baseline
            else None,
            # Don't include baseline_optimizer_state_dict at all
        }
        torch.save(state, filepath)

        # Act - Load agent with incomplete state
        new_agent = PolicyGradientAgent(
            state_size=4, action_size=2, use_baseline=True, continuous_actions=False
        )
        new_agent.load(filepath)

        # Assert - Verify agent loads without error
        assert new_agent.state_size == agent.state_size
        assert new_agent.action_size == agent.action_size
        assert new_agent.use_baseline == agent.use_baseline

    @pytest.mark.unit
    def test_learn_without_baseline_baseline_loss_zero(self) -> None:
        """Test learn method without baseline sets baseline_loss to zero."""
        # Arrange - Set up agent without baseline and sample rollout data
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, use_baseline=False, continuous_actions=False
        )

        rollout_data = [
            RolloutExperience(
                state=np.array([1.0, 2.0, 3.0, 4.0]),
                action=0,
                reward=1.0,
                log_prob=0.5,
                value=0.0,
                done=False,
            ),
            RolloutExperience(
                state=np.array([2.0, 3.0, 4.0, 5.0]),
                action=1,
                reward=2.0,
                log_prob=0.3,
                value=0.0,
                done=True,
            ),
        ]

        # Act - Learn from rollout data without baseline
        metrics = agent.learn(rollout_data)

        # Assert - Verify baseline_loss is zero when no baseline is used
        assert metrics["baseline_loss"] == 0.0
        assert isinstance(metrics["policy_loss"], float)
        assert isinstance(metrics["entropy_loss"], float)

    @pytest.mark.unit
    def test_get_training_stats_with_no_episodes(self) -> None:
        """Test get_training_stats method with no episodes."""
        # Arrange - Set up agent with no episode data
        agent = PolicyGradientAgent(state_size=4, action_size=2)
        agent.episode_rewards = []
        agent.episode_lengths = []

        # Act - Get training statistics with no episodes
        stats = agent.get_training_stats()

        # Assert - Verify default statistics are returned
        assert stats["mean_reward"] == 0.0
        assert stats["mean_length"] == 0.0
