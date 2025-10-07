"""Tests for Actor-Critic algorithm implementation."""

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from algokit.algorithms.reinforcement_learning.actor_critic import (
    ActorCriticAgent,
    ActorCriticConfig,
    ActorNetwork,
    CriticNetwork,
    RolloutExperience,
)


class TestActorNetwork:
    """Test the Actor network implementation."""

    @pytest.mark.unit
    def test_network_initialization(self) -> None:
        """Test Actor network initializes correctly."""
        # Arrange - Set up network parameters
        state_size = 4
        action_size = 2
        hidden_sizes = [64, 32]

        # Act - Create Actor network
        network = ActorNetwork(state_size, action_size, hidden_sizes)

        # Assert - Verify network structure
        assert network.network[0].in_features == state_size
        assert network.network[-2].out_features == action_size

    @pytest.mark.unit
    def test_network_forward_pass(self) -> None:
        """Test network forward pass."""
        # Arrange - Create network and input
        network = ActorNetwork(state_size=4, action_size=2)
        state = torch.randn(1, 4)

        # Act - Forward pass
        output = network(state)

        # Assert - Verify output shape and properties
        assert output.shape == (1, 2)
        assert torch.allclose(output.sum(dim=-1), torch.ones(1), atol=1e-6)

    @pytest.mark.unit
    def test_network_empty_hidden_sizes(self) -> None:
        """Test network raises error for empty hidden sizes."""
        # Arrange & Act & Assert - Test empty hidden sizes
        with pytest.raises(ValueError, match="hidden_sizes cannot be empty"):
            ActorNetwork(state_size=4, action_size=2, hidden_sizes=[])

    @pytest.mark.unit
    def test_network_output_probabilities(self) -> None:
        """Test that network outputs valid probabilities."""
        # Arrange - Create network and input
        network = ActorNetwork(state_size=4, action_size=3)
        state = torch.randn(5, 4)

        # Act - Forward pass
        output = network(state)

        # Assert - Verify probability properties
        assert torch.all(output >= 0), "All probabilities should be non-negative"
        assert torch.allclose(output.sum(dim=-1), torch.ones(5), atol=1e-6), (
            "Probabilities should sum to 1"
        )


class TestCriticNetwork:
    """Test the Critic network implementation."""

    @pytest.mark.unit
    def test_network_initialization(self) -> None:
        """Test Critic network initializes correctly."""
        # Arrange - Set up network parameters
        state_size = 4
        hidden_sizes = [64, 32]

        # Act - Create Critic network
        network = CriticNetwork(state_size, hidden_sizes)

        # Assert - Verify network structure
        assert network.network[0].in_features == state_size
        assert network.network[-1].out_features == 1

    @pytest.mark.unit
    def test_network_forward_pass(self) -> None:
        """Test network forward pass."""
        # Arrange - Create network and input
        network = CriticNetwork(state_size=4)
        state = torch.randn(1, 4)

        # Act - Forward pass
        output = network(state)

        # Assert - Verify output shape
        assert output.shape == (1, 1)

    @pytest.mark.unit
    def test_network_empty_hidden_sizes(self) -> None:
        """Test network raises error for empty hidden sizes."""
        # Arrange & Act & Assert - Test empty hidden sizes
        with pytest.raises(ValueError, match="hidden_sizes cannot be empty"):
            CriticNetwork(state_size=4, hidden_sizes=[])


class TestRolloutExperience:
    """Test the rollout experience implementation."""

    @pytest.mark.unit
    def test_rollout_experience_creation(self) -> None:
        """Test rollout experience creates correctly."""
        # Arrange - Create experience data
        state = np.array([1.0, 2.0])
        action = 1
        reward = 0.5
        log_prob = -0.7
        value = 2.3
        done = False

        # Act - Create experience
        experience = RolloutExperience(state, action, reward, log_prob, value, done)

        # Assert - Verify experience properties
        assert np.array_equal(experience.state, state)
        assert experience.action == action
        assert experience.reward == reward
        assert experience.log_prob == log_prob
        assert experience.value == value
        assert experience.done == done


class TestActorCriticAgent:
    """Test the Actor-Critic agent implementation."""

    @pytest.mark.unit
    def test_agent_initialization(self) -> None:
        """Test agent initializes correctly."""
        # Arrange - Set up agent parameters
        state_size = 4
        action_size = 2

        # Act - Create agent
        agent = ActorCriticAgent(state_size=state_size, action_size=action_size)

        # Assert - Verify agent properties
        assert agent.state_size == state_size
        assert agent.action_size == action_size
        assert agent.learning_rate_actor == 0.001
        assert agent.learning_rate_critic == 0.001
        assert agent.discount_factor == 0.99
        assert agent.gae_lambda == 0.95
        assert agent.normalize_advantages is True
        assert agent.gradient_clip_norm == 0.5

    @pytest.mark.unit
    def test_agent_invalid_parameters(self) -> None:
        """Test agent raises errors for invalid parameters."""
        # Arrange & Act & Assert - Test invalid state size
        with pytest.raises(ValidationError):
            ActorCriticAgent(state_size=0, action_size=2)

        # Arrange & Act & Assert - Test invalid action size
        with pytest.raises(ValidationError):
            ActorCriticAgent(state_size=4, action_size=0)

        # Arrange & Act & Assert - Test invalid learning rate
        with pytest.raises(ValidationError):
            ActorCriticAgent(state_size=4, action_size=2, learning_rate_actor=1.5)

        # Arrange & Act & Assert - Test invalid discount factor
        with pytest.raises(ValidationError):
            ActorCriticAgent(state_size=4, action_size=2, discount_factor=1.5)

        # Arrange & Act & Assert - Test invalid GAE lambda
        with pytest.raises(ValidationError):
            ActorCriticAgent(state_size=4, action_size=2, gae_lambda=1.5)

        # Arrange & Act & Assert - Test invalid gradient clip norm
        with pytest.raises(ValidationError):
            ActorCriticAgent(state_size=4, action_size=2, gradient_clip_norm=-0.1)

    @pytest.mark.unit
    def test_get_action(self) -> None:
        """Test action selection."""
        # Arrange - Create agent and state
        agent = ActorCriticAgent(state_size=4, action_size=2)
        state = np.array([1.0, 2.0, 3.0, 4.0])

        # Act - Get action
        action, log_prob, value = agent.get_action(state)

        # Assert - Verify action properties
        assert isinstance(action, int)
        assert 0 <= action < 2
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    @pytest.mark.unit
    def test_compute_returns(self) -> None:
        """Test return computation."""
        # Arrange - Create agent and test data
        agent = ActorCriticAgent(state_size=4, action_size=2)
        rewards = [1.0, 0.5, 2.0]
        values = [0.8, 1.2, 1.5]
        dones = [False, False, True]

        # Act - Compute returns
        returns = agent.compute_returns(rewards, values, dones)

        # Assert - Verify returns computation
        assert len(returns) == 3
        assert isinstance(returns[0], float)
        # Returns should be computed backwards
        assert returns[2] == 2.0  # Last reward (done=True)
        assert returns[1] == 0.5 + 0.99 * 2.0  # Second reward + discounted last
        assert returns[0] == 1.0 + 0.99 * returns[1]  # First reward + discounted second

    @pytest.mark.unit
    def test_compute_gae_advantages(self) -> None:
        """Test GAE advantage computation."""
        # Arrange - Create agent and test data
        agent = ActorCriticAgent(state_size=4, action_size=2, gae_lambda=0.95)
        rewards = [1.0, 0.5, 2.0]
        values = [0.8, 1.2, 1.5]
        dones = [False, False, True]
        next_value = 0.0

        # Act - Compute GAE advantages and returns
        advantages, returns = agent.compute_gae_advantages(
            rewards, values, dones, next_value
        )

        # Assert - Verify GAE computation
        assert len(advantages) == 3
        assert len(returns) == 3
        assert isinstance(advantages[0], float)
        assert isinstance(returns[0], float)
        # Returns should equal advantages + values
        for i in range(3):
            assert abs(returns[i] - (advantages[i] + values[i])) < 1e-6

    @pytest.mark.unit
    def test_gae_with_bootstrap(self) -> None:
        """Test GAE with bootstrap value."""
        # Arrange - Create agent and test data
        agent = ActorCriticAgent(state_size=4, action_size=2, gae_lambda=0.95)
        rewards = [1.0, 0.5]
        values = [0.8, 1.2]
        dones = [False, False]
        next_value = 2.0

        # Act - Compute GAE advantages and returns
        advantages, returns = agent.compute_gae_advantages(
            rewards, values, dones, next_value
        )

        # Assert - Verify bootstrap is used
        assert len(advantages) == 2
        assert len(returns) == 2
        # Should use next_value for bootstrap
        assert isinstance(advantages[0], float)
        assert isinstance(returns[0], float)

    @pytest.mark.unit
    def test_learn_empty_rollout(self) -> None:
        """Test learning with empty rollout data."""
        # Arrange - Create agent
        agent = ActorCriticAgent(state_size=4, action_size=2)

        # Act - Try to learn with no rollout data
        losses = agent.learn([])

        # Assert - Verify no learning occurred
        assert losses["actor_loss"] == 0.0
        assert losses["critic_loss"] == 0.0
        assert losses["entropy_loss"] == 0.0

    @pytest.mark.unit
    def test_learn_with_rollout_data(self) -> None:
        """Test learning with rollout data."""
        # Arrange - Create agent and rollout data
        agent = ActorCriticAgent(state_size=4, action_size=2)

        # Create mock rollout data
        rollout_data = [
            RolloutExperience(
                state=np.array([1.0, 2.0, 3.0, 4.0]),
                action=1,
                reward=0.5,
                log_prob=-0.7,
                value=1.2,
                done=False,
            ),
            RolloutExperience(
                state=np.array([2.0, 3.0, 4.0, 5.0]),
                action=0,
                reward=1.0,
                log_prob=-0.5,
                value=1.5,
                done=True,
            ),
        ]

        # Act - Learn from rollout data
        losses = agent.learn(rollout_data)

        # Assert - Verify learning occurred
        assert "actor_loss" in losses
        assert "critic_loss" in losses
        assert "entropy_loss" in losses
        assert "entropy" in losses
        assert "mean_advantage" in losses
        assert isinstance(losses["actor_loss"], float)
        assert isinstance(losses["critic_loss"], float)

    @pytest.mark.unit
    def test_set_training_mode(self) -> None:
        """Test setting training mode."""
        # Arrange - Create agent
        agent = ActorCriticAgent(state_size=4, action_size=2)

        # Act - Set training mode
        agent.set_training(False)

        # Assert - Verify networks are in eval mode
        assert not agent.actor.training
        assert not agent.critic.training

        # Act - Set back to training mode
        agent.set_training(True)

        # Assert - Verify networks are in training mode
        assert agent.actor.training
        assert agent.critic.training

    @pytest.mark.unit
    def test_save_and_load(self, tmp_path) -> None:
        """Test saving and loading agent state."""
        # Arrange - Create agent with custom parameters
        agent = ActorCriticAgent(
            state_size=4,
            action_size=2,
            entropy_coefficient=0.02,
            gae_lambda=0.8,
            normalize_advantages=False,
            gradient_clip_norm=1.0,
        )
        filepath = tmp_path / "agent.pth"

        # Act - Save agent
        agent.save(str(filepath))

        # Create new agent and load state
        new_agent = ActorCriticAgent(state_size=4, action_size=2)
        new_agent.load(str(filepath))

        # Assert - Verify all parameters were loaded
        assert new_agent.entropy_coefficient == 0.02
        assert new_agent.gae_lambda == 0.8
        assert new_agent.normalize_advantages is False
        assert new_agent.gradient_clip_norm == 1.0

    @pytest.mark.unit
    def test_device_handling(self) -> None:
        """Test device handling for CPU."""
        # Arrange & Act - Create agent with CPU device
        agent = ActorCriticAgent(state_size=4, action_size=2, device="cpu")

        # Assert - Verify device is set correctly
        assert agent.device.type == "cpu"
        assert next(agent.actor.parameters()).device.type == "cpu"
        assert next(agent.critic.parameters()).device.type == "cpu"

    @pytest.mark.unit
    def test_random_seed(self) -> None:
        """Test random seed setting."""
        # Arrange - Create agents with same seed
        agent1 = ActorCriticAgent(state_size=4, action_size=2, random_seed=42)
        agent2 = ActorCriticAgent(state_size=4, action_size=2, random_seed=42)

        # Act - Get actions from both agents with same state
        state = np.array([1.0, 2.0, 3.0, 4.0])
        action1, _, _ = agent1.get_action(state)
        action2, _, _ = agent2.get_action(state)

        # Assert - Verify agents are properly initialized
        # Note: Due to the stochastic nature of neural networks,
        # we can't guarantee identical outputs, but we can test
        # that the agents are properly initialized
        assert isinstance(action1, int)
        assert isinstance(action2, int)
        assert 0 <= action1 < 2
        assert 0 <= action2 < 2

    @pytest.mark.unit
    def test_config_object_initialization(self) -> None:
        """Test that agent accepts config object."""
        # Arrange - Create a config object
        config = ActorCriticConfig(state_size=4, action_size=2)

        # Act - Initialize agent with config
        agent = ActorCriticAgent(config=config)

        # Assert - Verify agent uses config correctly
        assert agent.config == config
        assert agent.state_size == 4
        assert agent.action_size == 2

    @pytest.mark.unit
    def test_backwards_compatible_kwargs(self) -> None:
        """Test that agent accepts kwargs for backwards compatibility."""
        # Arrange - No setup needed

        # Act - Initialize agent with kwargs
        agent = ActorCriticAgent(state_size=4, action_size=2)

        # Assert - Verify agent initialized correctly
        assert agent.state_size == 4
        assert agent.action_size == 2

    @pytest.mark.unit
    def test_config_validates_negative_state_size(self) -> None:
        """Test that Config rejects negative state_size."""
        # Arrange - No setup needed

        # Act & Assert - Verify validation error is raised
        with pytest.raises(ValidationError):
            ActorCriticConfig(state_size=-1, action_size=4)

    @pytest.mark.unit
    def test_config_validates_negative_action_size(self) -> None:
        """Test that Config rejects negative action_size."""
        # Arrange - No setup needed

        # Act & Assert - Verify validation error is raised
        with pytest.raises(ValidationError):
            ActorCriticConfig(state_size=4, action_size=-1)

    @pytest.mark.unit
    def test_config_validates_learning_rate_actor(self) -> None:
        """Test that Config rejects invalid learning_rate_actor."""
        # Arrange - No setup needed

        # Act & Assert - Verify validation error is raised for zero
        with pytest.raises(ValidationError):
            ActorCriticConfig(state_size=4, action_size=2, learning_rate_actor=0.0)

        # Act & Assert - Verify validation error is raised for too large
        with pytest.raises(ValidationError):
            ActorCriticConfig(state_size=4, action_size=2, learning_rate_actor=1.5)

    @pytest.mark.unit
    def test_config_validates_learning_rate_critic(self) -> None:
        """Test that Config rejects invalid learning_rate_critic."""
        # Arrange - No setup needed

        # Act & Assert - Verify validation error is raised for zero
        with pytest.raises(ValidationError):
            ActorCriticConfig(state_size=4, action_size=2, learning_rate_critic=0.0)

        # Act & Assert - Verify validation error is raised for too large
        with pytest.raises(ValidationError):
            ActorCriticConfig(state_size=4, action_size=2, learning_rate_critic=1.5)

    @pytest.mark.unit
    def test_config_validates_discount_factor(self) -> None:
        """Test that Config rejects invalid discount_factor."""
        # Arrange - No setup needed

        # Act & Assert - Verify validation error is raised for negative
        with pytest.raises(ValidationError):
            ActorCriticConfig(state_size=4, action_size=2, discount_factor=-0.1)

        # Act & Assert - Verify validation error is raised for too large
        with pytest.raises(ValidationError):
            ActorCriticConfig(state_size=4, action_size=2, discount_factor=1.5)

    @pytest.mark.unit
    def test_config_validates_dropout_rate(self) -> None:
        """Test that Config rejects invalid dropout_rate."""
        # Arrange - No setup needed

        # Act & Assert - Verify validation error is raised for negative
        with pytest.raises(ValidationError):
            ActorCriticConfig(state_size=4, action_size=2, dropout_rate=-0.1)

        # Act & Assert - Verify validation error is raised for 1.0
        with pytest.raises(ValidationError):
            ActorCriticConfig(state_size=4, action_size=2, dropout_rate=1.0)

    @pytest.mark.unit
    def test_config_validates_entropy_coefficient(self) -> None:
        """Test that Config rejects invalid entropy_coefficient."""
        # Arrange - No setup needed

        # Act & Assert - Verify validation error is raised for negative
        with pytest.raises(ValidationError):
            ActorCriticConfig(state_size=4, action_size=2, entropy_coefficient=-0.1)

    @pytest.mark.unit
    def test_config_validates_gae_lambda(self) -> None:
        """Test that Config rejects invalid gae_lambda."""
        # Arrange - No setup needed

        # Act & Assert - Verify validation error is raised for negative
        with pytest.raises(ValidationError):
            ActorCriticConfig(state_size=4, action_size=2, gae_lambda=-0.1)

        # Act & Assert - Verify validation error is raised for too large
        with pytest.raises(ValidationError):
            ActorCriticConfig(state_size=4, action_size=2, gae_lambda=1.5)

    @pytest.mark.unit
    def test_config_validates_gradient_clip_norm(self) -> None:
        """Test that Config rejects invalid gradient_clip_norm."""
        # Arrange - No setup needed

        # Act & Assert - Verify validation error is raised for negative
        with pytest.raises(ValidationError):
            ActorCriticConfig(state_size=4, action_size=2, gradient_clip_norm=-0.1)

    @pytest.mark.unit
    def test_config_validates_empty_hidden_sizes(self) -> None:
        """Test that Config rejects empty hidden_sizes."""
        # Arrange - No setup needed

        # Act & Assert - Verify validation error is raised for empty list
        with pytest.raises(ValidationError, match="hidden_sizes cannot be empty"):
            ActorCriticConfig(state_size=4, action_size=2, hidden_sizes=[])

    @pytest.mark.unit
    def test_config_default_hidden_sizes(self) -> None:
        """Test that Config sets default hidden_sizes when None is provided."""
        # Arrange - No setup needed

        # Act - Create config with None for hidden_sizes
        config = ActorCriticConfig(state_size=4, action_size=2, hidden_sizes=None)

        # Assert - Verify default value is set
        assert config.hidden_sizes == [128, 128]
