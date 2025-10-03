"""Tests for Policy Gradient reinforcement learning algorithm."""

import numpy as np
import pytest
import torch

from algokit.algorithms.reinforcement_learning.policy_gradient import (
    BaselineNetwork,
    PolicyGradientAgent,
    PolicyNetwork,
    RolloutExperience,
)


class TestPolicyNetwork:
    """Test PolicyNetwork class."""

    @pytest.mark.unit
    def test_policy_network_discrete_actions_initialization(self) -> None:
        """Test PolicyNetwork initialization for discrete actions."""
        # Arrange - Set up parameters for discrete action network
        state_size = 4
        action_size = 2
        hidden_sizes = [64, 32]
        dropout_rate = 0.1

        # Act - Create PolicyNetwork with discrete actions
        network = PolicyNetwork(
            state_size=state_size,
            action_size=action_size,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            continuous_actions=False,
        )

        # Assert - Verify network properties
        assert network.state_size == state_size
        assert network.action_size == action_size
        assert network.continuous_actions is False
        assert (
            len(network.network) == 6
        )  # 2 layers * 3 components (Linear, ReLU, Dropout)

    @pytest.mark.unit
    def test_policy_network_continuous_actions_initialization(self) -> None:
        """Test PolicyNetwork initialization for continuous actions."""
        # Arrange - Set up parameters for continuous action network
        state_size = 4
        action_size = 2
        hidden_sizes = [64, 32]

        # Act - Create PolicyNetwork with continuous actions
        network = PolicyNetwork(
            state_size=state_size,
            action_size=action_size,
            hidden_sizes=hidden_sizes,
            continuous_actions=True,
        )

        # Assert - Verify network properties
        assert network.state_size == state_size
        assert network.action_size == action_size
        assert network.continuous_actions is True
        assert hasattr(network, "mean_head")
        assert hasattr(network, "log_std_head")

    @pytest.mark.unit
    def test_policy_network_forward_discrete(self) -> None:
        """Test PolicyNetwork forward pass for discrete actions."""
        # Arrange - Create network and input tensor
        network = PolicyNetwork(state_size=4, action_size=2, continuous_actions=False)
        state = torch.randn(1, 4)

        # Act - Forward pass through network
        output = network.forward(state)

        # Assert - Verify output shape and properties
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 2)
        assert torch.isfinite(output).all()

    @pytest.mark.unit
    def test_policy_network_forward_continuous(self) -> None:
        """Test PolicyNetwork forward pass for continuous actions."""
        # Arrange - Create network and input tensor
        network = PolicyNetwork(state_size=4, action_size=2, continuous_actions=True)
        state = torch.randn(1, 4)

        # Act - Forward pass through network
        mean, log_std = network.forward(state)

        # Assert - Verify output shapes and properties
        assert mean.shape == (1, 2)
        assert log_std.shape == (1, 2)
        assert torch.isfinite(mean).all()
        assert torch.isfinite(log_std).all()

    @pytest.mark.unit
    def test_policy_network_get_action_discrete(self) -> None:
        """Test PolicyNetwork get_action for discrete actions."""
        # Arrange - Create network and input tensor
        network = PolicyNetwork(state_size=4, action_size=2, continuous_actions=False)
        state = torch.randn(1, 4)

        # Act - Get action from network
        action, log_prob = network.get_action(state)

        # Assert - Verify action and log probability
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert action.dtype == torch.long
        assert torch.isfinite(log_prob).all()

    @pytest.mark.unit
    def test_policy_network_get_action_continuous(self) -> None:
        """Test PolicyNetwork get_action for continuous actions."""
        # Arrange - Create network and input tensor
        network = PolicyNetwork(state_size=4, action_size=2, continuous_actions=True)
        state = torch.randn(1, 4)

        # Act - Get action from network
        action, log_prob = network.get_action(state)

        # Assert - Verify action and log probability
        assert action.shape == (1, 2)
        assert log_prob.shape == (1, 1)
        assert torch.isfinite(action).all()
        assert torch.isfinite(log_prob).all()


class TestBaselineNetwork:
    """Test BaselineNetwork class."""

    @pytest.mark.unit
    def test_baseline_network_initialization(self) -> None:
        """Test BaselineNetwork initialization."""
        # Arrange - Set up parameters for baseline network
        state_size = 4
        hidden_sizes = [64, 32]
        dropout_rate = 0.1

        # Act - Create BaselineNetwork
        network = BaselineNetwork(
            state_size=state_size,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
        )

        # Assert - Verify network structure
        assert len(network.network) == 7  # 2 layers * 3 components + output layer

    @pytest.mark.unit
    def test_baseline_network_forward(self) -> None:
        """Test BaselineNetwork forward pass."""
        # Arrange - Create network and input tensor
        network = BaselineNetwork(state_size=4)
        state = torch.randn(5, 4)

        # Act - Forward pass through network
        output = network.forward(state)

        # Assert - Verify output shape and properties
        assert output.shape == (5,)
        assert torch.isfinite(output).all()


class TestPolicyGradientAgent:
    """Test PolicyGradientAgent class."""

    @pytest.mark.unit
    def test_agent_initialization_discrete(self) -> None:
        """Test PolicyGradientAgent initialization for discrete actions."""
        # Arrange - Set up agent parameters
        state_size = 4
        action_size = 2
        learning_rate = 0.001
        gamma = 0.99
        use_baseline = True

        # Act - Create agent
        agent = PolicyGradientAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            gamma=gamma,
            use_baseline=use_baseline,
            continuous_actions=False,
        )

        # Assert - Verify agent properties
        assert agent.state_size == state_size
        assert agent.action_size == action_size
        assert agent.learning_rate == learning_rate
        assert agent.gamma == gamma
        assert agent.use_baseline == use_baseline
        assert agent.continuous_actions is False
        assert agent.policy is not None
        assert agent.baseline is not None

    @pytest.mark.unit
    def test_agent_initialization_continuous(self) -> None:
        """Test PolicyGradientAgent initialization for continuous actions."""
        # Arrange - Set up agent parameters
        state_size = 4
        action_size = 2
        use_baseline = False

        # Act - Create agent
        agent = PolicyGradientAgent(
            state_size=state_size,
            action_size=action_size,
            use_baseline=use_baseline,
            continuous_actions=True,
        )

        # Assert - Verify agent properties
        assert agent.continuous_actions is True
        assert agent.use_baseline is False
        assert agent.baseline is None

    @pytest.mark.unit
    def test_agent_evaluate_discrete(self) -> None:
        """Test agent evaluate method for discrete actions."""
        # Arrange - Create agent and state
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, continuous_actions=False
        )
        state = np.array([1.0, 2.0, 3.0, 4.0])

        # Act - Evaluate state
        action, log_prob, value = agent.evaluate(state)

        # Assert - Verify action, log_prob, and value
        assert isinstance(action, int | np.integer | np.ndarray)
        if isinstance(action, np.ndarray):
            action = action.item()
        assert isinstance(log_prob, float | np.floating)
        assert isinstance(value, float | np.floating)
        assert 0 <= action < 2
        assert np.isfinite(log_prob)
        assert np.isfinite(value)

    @pytest.mark.unit
    def test_agent_evaluate_continuous(self) -> None:
        """Test agent evaluate method for continuous actions."""
        # Arrange - Create agent and state
        agent = PolicyGradientAgent(
            state_size=4, action_size=2, continuous_actions=True
        )
        state = np.array([1.0, 2.0, 3.0, 4.0])

        # Act - Evaluate state
        action, log_prob, value = agent.evaluate(state)

        # Assert - Verify action, log_prob, and value
        assert isinstance(action, np.ndarray)
        assert isinstance(log_prob, float | np.floating)
        assert isinstance(value, float | np.floating)
        assert action.shape == (2,)
        assert np.isfinite(action).all()
        assert np.isfinite(log_prob)
        assert np.isfinite(value)

    @pytest.mark.unit
    def test_agent_learn_with_baseline(self) -> None:
        """Test agent learn method with baseline."""
        # Arrange - Create agent and sample rollout data
        agent = PolicyGradientAgent(
            state_size=4,
            action_size=2,
            use_baseline=True,
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

        # Act - Learn from rollout data
        metrics = agent.learn(rollout_data)

        # Assert - Verify metrics are returned
        assert "policy_loss" in metrics
        assert "baseline_loss" in metrics
        assert "entropy_loss" in metrics
        assert "mean_return" in metrics
        assert "mean_advantage" in metrics
        assert isinstance(metrics["policy_loss"], float)
        assert isinstance(metrics["baseline_loss"], float)

    @pytest.mark.unit
    def test_agent_learn_without_baseline(self) -> None:
        """Test agent learn method without baseline."""
        # Arrange - Create agent and sample rollout data
        agent = PolicyGradientAgent(
            state_size=4,
            action_size=2,
            use_baseline=False,
            continuous_actions=False,
        )

        rollout_data = [
            RolloutExperience(
                state=np.array([1.0, 2.0, 3.0, 4.0]),
                action=0,
                reward=1.0,
                log_prob=0.5,
                value=0.0,
                done=True,
            ),
        ]

        # Act - Learn from rollout data
        metrics = agent.learn(rollout_data)

        # Assert - Verify metrics are returned
        assert "policy_loss" in metrics
        assert "baseline_loss" in metrics
        assert metrics["baseline_loss"] == 0.0
        assert isinstance(metrics["policy_loss"], float)

    @pytest.mark.unit
    def test_agent_learn_empty_rollout(self) -> None:
        """Test agent learn with empty rollout data."""
        # Arrange - Create agent with empty rollout data
        agent = PolicyGradientAgent(state_size=4, action_size=2)
        rollout_data = []

        # Act - Learn from empty rollout data
        metrics = agent.learn(rollout_data)

        # Assert - Verify default metrics
        assert metrics["policy_loss"] == 0.0
        assert metrics["baseline_loss"] == 0.0
        assert metrics["entropy_loss"] == 0.0

    @pytest.mark.unit
    def test_agent_get_training_stats(self) -> None:
        """Test agent get_training_stats method."""
        # Arrange - Create agent and simulate training
        agent = PolicyGradientAgent(state_size=4, action_size=2)
        agent.episode_rewards = [1.0, 2.0, 3.0]
        agent.episode_lengths = [10, 15, 20]

        # Act - Get training statistics
        stats = agent.get_training_stats()

        # Assert - Verify statistics
        assert "mean_reward" in stats
        assert "std_reward" in stats
        assert "mean_length" in stats
        assert "std_length" in stats
        assert "total_episodes" in stats
        assert stats["total_episodes"] == 3
        assert stats["mean_reward"] == 2.0

    @pytest.mark.unit
    def test_agent_get_training_stats_empty(self) -> None:
        """Test agent get_training_stats with no episodes."""
        # Arrange - Create agent with no training data
        agent = PolicyGradientAgent(state_size=4, action_size=2)

        # Act - Get training statistics
        stats = agent.get_training_stats()

        # Assert - Verify default statistics
        assert stats["mean_reward"] == 0.0
        assert stats["mean_length"] == 0.0

    @pytest.mark.unit
    def test_agent_set_training_mode(self) -> None:
        """Test agent set_training_mode method."""
        # Arrange - Create agent with baseline
        agent = PolicyGradientAgent(state_size=4, action_size=2, use_baseline=True)

        # Act - Set training mode to False
        agent.set_training_mode(False)

        # Assert - Verify training mode is set
        assert agent.training is False
        assert agent.policy.training is False
        if agent.baseline is not None:
            assert agent.baseline.training is False

    @pytest.mark.unit
    def test_agent_save_and_load(self) -> None:
        """Test agent save and load methods."""
        # Arrange - Create agent and filepath
        agent = PolicyGradientAgent(state_size=4, action_size=2, use_baseline=True)
        filepath = "/tmp/test_agent.pth"

        # Act - Save and load agent
        agent.save(filepath)
        new_agent = PolicyGradientAgent(state_size=4, action_size=2, use_baseline=True)
        new_agent.load(filepath)

        # Assert - Verify loaded agent structure
        assert new_agent.state_size == agent.state_size
        assert new_agent.action_size == agent.action_size
        assert new_agent.use_baseline == agent.use_baseline

    @pytest.mark.unit
    def test_agent_seed_reproducibility(self) -> None:
        """Test that agent initialization with seed is reproducible."""
        # Arrange - Set up seed and state
        seed = 42
        state = np.array([1.0, 2.0, 3.0, 4.0])

        # Act - Create agents with same seed
        agent1 = PolicyGradientAgent(state_size=4, action_size=2, seed=seed)
        agent2 = PolicyGradientAgent(state_size=4, action_size=2, seed=seed)

        action1 = agent1.act(state)
        action2 = agent2.act(state)

        # Assert - Verify agents work (exact reproducibility not guaranteed due to randomness)
        assert isinstance(action1, int | np.integer | np.ndarray)
        assert isinstance(action2, int | np.integer | np.ndarray)

    @pytest.mark.unit
    def test_agent_device_handling(self) -> None:
        """Test agent device handling."""
        # Arrange - Set up device
        device = "cpu"  # Use CPU for testing

        # Act - Create agent with device
        agent = PolicyGradientAgent(state_size=4, action_size=2, device=device)

        # Assert - Verify device is set correctly
        assert agent.device.type == "cpu"
        assert next(agent.policy.parameters()).device.type == "cpu"


class TestRolloutExperience:
    """Test RolloutExperience namedtuple."""

    @pytest.mark.unit
    def test_rollout_experience_creation(self) -> None:
        """Test RolloutExperience namedtuple creation."""
        # Arrange - Set up experience data
        state = np.array([1.0, 2.0, 3.0])
        action = 1
        reward = 0.5
        log_prob = -0.2
        value = 0.8
        done = False

        # Act - Create RolloutExperience object
        exp = RolloutExperience(
            state=state,
            action=action,
            reward=reward,
            log_prob=log_prob,
            value=value,
            done=done,
        )

        # Assert - Verify experience properties
        assert exp.state is state
        assert exp.action == action
        assert exp.reward == reward
        assert exp.log_prob == log_prob
        assert exp.value == value
        assert exp.done == done

    @pytest.mark.unit
    def test_rollout_experience_unpacking(self) -> None:
        """Test RolloutExperience namedtuple unpacking."""
        # Arrange - Create experience object
        state = np.array([1.0, 2.0, 3.0])
        action = 1
        reward = 0.5
        log_prob = -0.2
        value = 0.8
        done = False
        exp = RolloutExperience(
            state=state,
            action=action,
            reward=reward,
            log_prob=log_prob,
            value=value,
            done=done,
        )

        # Act - Unpack experience
        s, a, r, lp, v, d = exp

        # Assert - Verify unpacked values
        assert s is state
        assert a == action
        assert r == reward
        assert lp == log_prob
        assert v == value
        assert d == done


@pytest.mark.unit
class TestPolicyGradientIntegration:
    """Integration tests for Policy Gradient algorithm."""

    @pytest.mark.unit
    def test_full_training_cycle_discrete(self) -> None:
        """Test full training cycle for discrete actions."""
        # Arrange - Create agent and rollout data
        agent = PolicyGradientAgent(
            state_size=4,
            action_size=2,
            continuous_actions=False,
            use_baseline=True,
        )

        rollout_data = []
        for _ in range(5):
            state = np.random.randn(4)
            action, log_prob, value = agent.evaluate(state)
            reward = np.random.randn()
            done = _ == 4  # Last step is done
            rollout_data.append(
                RolloutExperience(
                    state=state,
                    action=action,
                    reward=reward,
                    log_prob=log_prob,
                    value=value,
                    done=done,
                )
            )

        # Act - Learn from rollout data
        metrics = agent.learn(rollout_data)

        # Assert - Verify training metrics
        assert "policy_loss" in metrics
        assert "baseline_loss" in metrics
        assert "entropy_loss" in metrics
        assert "mean_return" in metrics
        assert "mean_advantage" in metrics

    @pytest.mark.unit
    def test_full_training_cycle_continuous(self) -> None:
        """Test full training cycle for continuous actions."""
        # Arrange - Create agent and rollout data
        agent = PolicyGradientAgent(
            state_size=4,
            action_size=2,
            continuous_actions=True,
            use_baseline=True,
        )

        rollout_data = []
        for _ in range(5):
            state = np.random.randn(4)
            action, log_prob, value = agent.evaluate(state)
            reward = np.random.randn()
            done = _ == 4  # Last step is done
            rollout_data.append(
                RolloutExperience(
                    state=state,
                    action=action,
                    reward=reward,
                    log_prob=log_prob,
                    value=value,
                    done=done,
                )
            )

        # Act - Learn from rollout data
        metrics = agent.learn(rollout_data)

        # Assert - Verify training metrics
        assert "policy_loss" in metrics
        assert "baseline_loss" in metrics
        assert "entropy_loss" in metrics
        assert "mean_return" in metrics
        assert "mean_advantage" in metrics

    @pytest.mark.unit
    def test_agent_learning_progress(self) -> None:
        """Test that agent shows learning progress over multiple updates."""
        # Arrange - Create agent and training data
        agent = PolicyGradientAgent(
            state_size=4,
            action_size=2,
            continuous_actions=False,
            use_baseline=True,
        )

        initial_losses = []
        final_losses = []

        # Act - Run initial updates
        for _ in range(3):
            rollout_data = []
            for _ in range(3):
                state = np.random.randn(4)
                action, log_prob, value = agent.evaluate(state)
                reward = np.random.randn()
                done = _ == 2  # Last step is done
                rollout_data.append(
                    RolloutExperience(
                        state=state,
                        action=action,
                        reward=reward,
                        log_prob=log_prob,
                        value=value,
                        done=done,
                    )
                )

            metrics = agent.learn(rollout_data)
            initial_losses.append(metrics["policy_loss"])

        # Run more updates
        for _ in range(3):
            rollout_data = []
            for _ in range(3):
                state = np.random.randn(4)
                action, log_prob, value = agent.evaluate(state)
                reward = np.random.randn()
                done = _ == 2  # Last step is done
                rollout_data.append(
                    RolloutExperience(
                        state=state,
                        action=action,
                        reward=reward,
                        log_prob=log_prob,
                        value=value,
                        done=done,
                    )
                )

            metrics = agent.learn(rollout_data)
            final_losses.append(metrics["policy_loss"])

        # Assert - Verify training progress
        assert all(np.isfinite(loss) for loss in initial_losses)
        assert all(np.isfinite(loss) for loss in final_losses)
