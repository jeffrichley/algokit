"""Tests for DQN algorithm implementation."""

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from algokit.algorithms.reinforcement_learning.dqn import (
    DQNAgent,
    DQNConfig,
    DQNNetwork,
    ReplayBuffer,
)


class TestDQNNetwork:
    """Test the DQN neural network implementation."""

    @pytest.mark.unit
    def test_network_initialization(self) -> None:
        """Test DQN network initializes correctly."""
        # Arrange - Set up network parameters
        state_size = 4
        action_size = 2
        hidden_sizes = [64, 32]

        # Act - Create DQN network
        network = DQNNetwork(state_size, action_size, hidden_sizes)

        # Assert - Verify network structure
        assert network.network[0].in_features == state_size
        assert network.network[-1].out_features == action_size

    @pytest.mark.unit
    def test_network_forward_pass(self) -> None:
        """Test network forward pass."""
        # Arrange - Create network and input
        network = DQNNetwork(state_size=4, action_size=2)
        state = torch.randn(1, 4)

        # Act - Forward pass
        output = network(state)

        # Assert - Verify output shape
        assert output.shape == (1, 2)

    @pytest.mark.unit
    def test_network_empty_hidden_sizes(self) -> None:
        """Test network raises error for empty hidden sizes."""
        # Arrange & Act & Assert - Test empty hidden sizes
        with pytest.raises(ValueError, match="hidden_sizes cannot be empty"):
            DQNNetwork(state_size=4, action_size=2, hidden_sizes=[])


class TestReplayBuffer:
    """Test the experience replay buffer implementation."""

    @pytest.mark.unit
    def test_buffer_initialization(self) -> None:
        """Test replay buffer initializes correctly."""
        # Arrange - Set up buffer capacity
        capacity = 1000

        # Act - Create replay buffer
        buffer = ReplayBuffer(capacity)

        # Assert - Verify buffer properties
        assert buffer.capacity == capacity
        assert len(buffer) == 0

    @pytest.mark.unit
    def test_buffer_push_and_sample(self) -> None:
        """Test adding experiences and sampling from buffer."""
        # Arrange - Create buffer and sample experience
        buffer = ReplayBuffer(capacity=10)
        state = np.array([1.0, 2.0])
        action = 0
        reward = 1.0
        next_state = np.array([2.0, 3.0])
        done = False

        # Act - Add experience and sample
        buffer.push(state, action, reward, next_state, done)
        experiences = buffer.sample(1)

        # Assert - Verify experience was stored correctly
        assert len(experiences) == 1
        assert np.array_equal(experiences[0].state, state)
        assert experiences[0].action == action
        assert experiences[0].reward == reward
        assert np.array_equal(experiences[0].next_state, next_state)
        assert experiences[0].done == done

    @pytest.mark.unit
    def test_buffer_capacity_limit(self) -> None:
        """Test buffer respects capacity limit."""
        # Arrange - Create small buffer
        buffer = ReplayBuffer(capacity=3)

        # Act - Add more experiences than capacity
        for i in range(5):
            state = np.array([float(i)])
            buffer.push(state, i % 2, 1.0, state, False)

        # Assert - Buffer should not exceed capacity
        assert len(buffer) == 3

    @pytest.mark.unit
    def test_buffer_sample_insufficient_experiences(self) -> None:
        """Test buffer raises error when sampling more than available."""
        # Arrange - Create buffer with few experiences
        buffer = ReplayBuffer(capacity=10)
        buffer.push(np.array([1.0]), 0, 1.0, np.array([2.0]), False)

        # Act & Assert - Try to sample more than available
        with pytest.raises(
            ValueError, match="Cannot sample 2 experiences from buffer of size 1"
        ):
            buffer.sample(2)

    @pytest.mark.unit
    def test_buffer_invalid_capacity(self) -> None:
        """Test buffer raises error for invalid capacity."""
        # Arrange & Act & Assert - Test invalid capacity
        with pytest.raises(ValueError, match="capacity must be positive"):
            ReplayBuffer(capacity=0)


class TestDQNConfig:
    """Test the DQN configuration validation."""

    @pytest.mark.unit
    def test_config_valid_parameters(self) -> None:
        """Test DQNConfig accepts valid parameters."""
        # Arrange - Prepare configuration parameters

        # Act - Create config with valid parameters
        config = DQNConfig(state_size=4, action_size=2)

        # Assert - Verify config has correct values
        assert config.state_size == 4
        assert config.action_size == 2
        assert config.learning_rate == 0.001
        assert config.epsilon == 1.0

    @pytest.mark.unit
    def test_config_rejects_negative_state_size(self) -> None:
        """Test DQNConfig rejects negative state_size."""
        # Arrange - Prepare invalid state_size parameter

        # Act & Assert - Create config and verify validation error
        with pytest.raises(ValidationError, match="state_size"):
            DQNConfig(state_size=-1, action_size=4)

    @pytest.mark.unit
    def test_config_rejects_negative_action_size(self) -> None:
        """Test DQNConfig rejects negative action_size."""
        # Arrange - Prepare invalid action_size parameter

        # Act & Assert - Create config and verify validation error
        with pytest.raises(ValidationError, match="action_size"):
            DQNConfig(state_size=4, action_size=-1)

    @pytest.mark.unit
    def test_config_rejects_invalid_learning_rate(self) -> None:
        """Test DQNConfig rejects learning_rate > 1."""
        # Arrange - Prepare invalid learning_rate parameter

        # Act & Assert - Create config and verify validation error
        with pytest.raises(ValidationError):
            DQNConfig(state_size=4, action_size=2, learning_rate=1.5)

    @pytest.mark.unit
    def test_config_rejects_invalid_epsilon(self) -> None:
        """Test DQNConfig rejects epsilon > 1."""
        # Arrange - Prepare invalid epsilon parameter

        # Act & Assert - Create config and verify validation error
        with pytest.raises(ValidationError):
            DQNConfig(state_size=4, action_size=2, epsilon=1.5)

    @pytest.mark.unit
    def test_config_rejects_epsilon_min_greater_than_epsilon(self) -> None:
        """Test DQNConfig rejects epsilon_min > epsilon."""
        # Arrange - Prepare epsilon_min greater than epsilon

        # Act & Assert - Create config and verify validation error
        with pytest.raises(ValidationError, match="epsilon_min"):
            DQNConfig(state_size=4, action_size=2, epsilon=0.5, epsilon_min=0.8)

    @pytest.mark.unit
    def test_config_rejects_invalid_dqn_variant(self) -> None:
        """Test DQNConfig rejects invalid dqn_variant."""
        # Arrange - Prepare invalid dqn_variant parameter

        # Act & Assert - Create config and verify validation error
        with pytest.raises(ValidationError):
            DQNConfig(state_size=4, action_size=2, dqn_variant="invalid")

    @pytest.mark.unit
    def test_config_requires_decay_steps_for_linear_decay(self) -> None:
        """Test DQNConfig requires epsilon_decay_steps for linear decay."""
        # Arrange - Prepare linear decay without epsilon_decay_steps

        # Act & Assert - Create config and verify validation error
        with pytest.raises(ValidationError, match="epsilon_decay_steps"):
            DQNConfig(state_size=4, action_size=2, epsilon_decay_type="linear")

    @pytest.mark.unit
    def test_config_accepts_linear_decay_with_steps(self) -> None:
        """Test DQNConfig accepts linear decay with epsilon_decay_steps."""
        # Arrange - Prepare linear decay with epsilon_decay_steps

        # Act - Create config with valid linear decay parameters
        config = DQNConfig(
            state_size=4,
            action_size=2,
            epsilon_decay_type="linear",
            epsilon_decay_steps=100,
        )

        # Assert - Verify config has correct decay parameters
        assert config.epsilon_decay_type == "linear"
        assert config.epsilon_decay_steps == 100


class TestDQNAgent:
    """Test the DQN agent implementation."""

    @pytest.mark.unit
    def test_agent_initialization(self) -> None:
        """Test DQN agent initializes correctly."""
        # Arrange - Set up agent parameters
        state_size = 4
        action_size = 2

        # Act - Create DQN agent
        agent = DQNAgent(state_size=state_size, action_size=action_size)

        # Assert - Verify agent properties
        assert agent.state_size == state_size
        assert agent.action_size == action_size
        assert agent.epsilon == 1.0
        assert len(agent.memory) == 0

    @pytest.mark.unit
    def test_agent_config_object_initialization(self) -> None:
        """Test DQN agent accepts config object."""
        # Arrange - Create config object
        config = DQNConfig(state_size=4, action_size=2, learning_rate=0.002)

        # Act - Create agent with config
        agent = DQNAgent(config=config)

        # Assert - Verify agent initialized with config values
        assert agent.config == config
        assert agent.state_size == 4
        assert agent.action_size == 2
        assert agent.learning_rate == 0.002

    @pytest.mark.unit
    def test_agent_backwards_compatible_kwargs(self) -> None:
        """Test DQN agent accepts kwargs for backwards compatibility."""
        # Arrange - Prepare agent parameters

        # Act - Create agent with kwargs (old style)
        agent = DQNAgent(state_size=4, action_size=2, learning_rate=0.003)

        # Assert - Verify agent initialized correctly
        assert agent.state_size == 4
        assert agent.action_size == 2
        assert agent.learning_rate == 0.003

    @pytest.mark.unit
    def test_agent_initialization_with_seed(self) -> None:
        """Test DQN agent initializes with random seed."""
        # Arrange - Set up agent with random seed
        random_seed = 42

        # Act - Create DQN agent with seed
        agent = DQNAgent(state_size=4, action_size=2, random_seed=random_seed)

        # Assert - Verify agent is created without errors
        assert agent.state_size == 4
        assert agent.action_size == 2

    @pytest.mark.unit
    def test_agent_initialization_with_new_parameters(self) -> None:
        """Test DQN agent initializes with new parameters."""
        # Arrange & Act - Create agent with new parameters
        agent = DQNAgent(
            state_size=4,
            action_size=2,
            dqn_variant="double",
            use_huber_loss=True,
            gradient_clip_norm=2.0,
            tau=0.01,
        )

        # Assert - Verify new parameters are set correctly
        assert agent.dqn_variant == "double"
        assert agent.use_huber_loss is True
        assert agent.gradient_clip_norm == 2.0
        assert agent.tau == 0.01

    @pytest.mark.unit
    def test_set_seed_method(self) -> None:
        """Test set_seed method for reproducibility."""
        # Arrange - Create agent
        agent = DQNAgent(state_size=4, action_size=2)

        # Act - Set seed
        agent.set_seed(123)

        # Assert - Verify seed was set (no error raised)
        assert agent.state_size == 4
        assert agent.action_size == 2

    @pytest.mark.unit
    def test_invalid_initialization_parameters(self) -> None:
        """Test DQN agent raises errors for invalid parameters."""
        # Arrange & Act & Assert - Test invalid state size
        with pytest.raises(ValidationError):
            DQNAgent(state_size=0, action_size=2)

        # Arrange & Act & Assert - Test invalid action size
        with pytest.raises(ValidationError):
            DQNAgent(state_size=4, action_size=0)

        # Arrange & Act & Assert - Test invalid learning rate
        with pytest.raises(ValidationError):
            DQNAgent(state_size=4, action_size=2, learning_rate=1.5)

        # Arrange & Act & Assert - Test invalid discount factor
        with pytest.raises(ValidationError):
            DQNAgent(state_size=4, action_size=2, discount_factor=1.5)

        # Arrange & Act & Assert - Test invalid epsilon
        with pytest.raises(ValidationError):
            DQNAgent(state_size=4, action_size=2, epsilon=1.5)

        # Arrange & Act & Assert - Test invalid batch size
        with pytest.raises(ValidationError):
            DQNAgent(state_size=4, action_size=2, batch_size=0)

        # Arrange & Act & Assert - Test invalid memory size
        with pytest.raises(ValidationError):
            DQNAgent(state_size=4, action_size=2, memory_size=0)

        # Arrange & Act & Assert - Test invalid target update
        with pytest.raises(ValidationError):
            DQNAgent(state_size=4, action_size=2, target_update=0)

        # Arrange & Act & Assert - Test invalid DQN variant
        with pytest.raises(ValidationError):
            DQNAgent(state_size=4, action_size=2, dqn_variant="invalid")

        # Arrange & Act & Assert - Test invalid gradient clip norm
        with pytest.raises(ValidationError):
            DQNAgent(state_size=4, action_size=2, gradient_clip_norm=0)

        # Arrange & Act & Assert - Test invalid tau
        with pytest.raises(ValidationError):
            DQNAgent(state_size=4, action_size=2, tau=1.5)

    @pytest.mark.unit
    def test_get_action_epsilon_greedy(self) -> None:
        """Test action selection using epsilon-greedy policy."""
        # Arrange - Create agent with high epsilon for exploration
        agent = DQNAgent(
            state_size=4,
            action_size=2,
            epsilon=1.0,  # Always explore
            random_seed=42,
        )
        state = np.array([1.0, 2.0, 3.0, 4.0])

        # Act - Get action multiple times
        actions = [agent.get_action(state) for _ in range(10)]

        # Assert - Verify actions are valid
        assert all(0 <= action < 2 for action in actions)

    @pytest.mark.unit
    def test_get_action_invalid_state_shape(self) -> None:
        """Test get_action raises error for invalid state shape."""
        # Arrange - Create agent
        agent = DQNAgent(state_size=4, action_size=2)

        # Act & Assert - Test invalid state shape
        with pytest.raises(ValueError, match="State shape"):
            agent.get_action(np.array([1.0, 2.0]))  # Wrong shape

    @pytest.mark.unit
    def test_step_functionality(self) -> None:
        """Test step function saves experience and trains."""
        # Arrange - Create agent with small memory and batch size
        agent = DQNAgent(
            state_size=4,
            action_size=2,
            memory_size=100,
            batch_size=10,
            epsilon=0.1,  # Low exploration for testing
        )

        # Act - Fill memory to enable training
        for i in range(15):
            state = np.array([float(i), float(i + 1), float(i + 2), float(i + 3)])
            next_state = np.array(
                [float(i + 1), float(i + 2), float(i + 3), float(i + 4)]
            )
            agent.step(state, i % 2, 1.0, next_state, False)

        # Assert - Memory should contain experiences
        assert len(agent.memory) > 0

    @pytest.mark.unit
    def test_step_invalid_parameters(self) -> None:
        """Test step raises errors for invalid parameters."""
        # Arrange - Create agent
        agent = DQNAgent(state_size=4, action_size=2)

        state = np.array([1.0, 2.0, 3.0, 4.0])
        next_state = np.array([2.0, 3.0, 4.0, 5.0])

        # Act & Assert - Test invalid state shape
        with pytest.raises(ValueError, match="State shape"):
            agent.step(
                np.array([1.0, 2.0]), 0, 1.0, next_state, False
            )  # Wrong state shape

        # Act & Assert - Test invalid action
        with pytest.raises(ValueError, match="Action -1 is out of range"):
            agent.step(state, -1, 1.0, next_state, False)

        # Act & Assert - Test invalid next state shape
        with pytest.raises(ValueError, match="Next state shape"):
            agent.step(state, 0, 1.0, np.array([1.0, 2.0]), False)  # Wrong shape

    @pytest.mark.unit
    def test_decay_epsilon(self) -> None:
        """Test epsilon decay over time."""
        # Arrange - Create agent with epsilon decay
        agent = DQNAgent(
            state_size=4,
            action_size=2,
            epsilon=1.0,
            epsilon_decay=0.9,
            epsilon_min=0.01,
        )

        # Act - Decay epsilon multiple times
        initial_epsilon = agent.get_epsilon()
        agent.decay_epsilon()
        first_decay = agent.get_epsilon()
        agent.decay_epsilon()
        second_decay = agent.get_epsilon()

        # Assert - Verify epsilon decayed correctly
        assert first_decay == initial_epsilon * 0.9
        assert second_decay == initial_epsilon * 0.9 * 0.9

    @pytest.mark.unit
    def test_epsilon_minimum(self) -> None:
        """Test epsilon doesn't go below minimum value."""
        # Arrange - Create agent with epsilon decay
        agent = DQNAgent(
            state_size=4,
            action_size=2,
            epsilon=0.1,
            epsilon_decay=0.5,
            epsilon_min=0.05,
        )

        # Act - Decay epsilon until it should hit minimum
        for _ in range(10):  # Decay many times
            agent.decay_epsilon()

        # Assert - Verify epsilon is at minimum
        assert agent.get_epsilon() == 0.05

    @pytest.mark.unit
    def test_get_q_values(self) -> None:
        """Test getting Q-values for all actions."""
        # Arrange - Create agent
        agent = DQNAgent(state_size=4, action_size=2)
        state = np.array([1.0, 2.0, 3.0, 4.0])

        # Act - Get Q-values
        q_values = agent.get_q_values(state)

        # Assert - Verify Q-values shape and type
        assert q_values.shape == (2,)
        assert isinstance(q_values, np.ndarray)

    @pytest.mark.unit
    def test_get_q_values_invalid_state_shape(self) -> None:
        """Test get_q_values raises error for invalid state shape."""
        # Arrange - Create agent
        agent = DQNAgent(state_size=4, action_size=2)

        # Act & Assert - Test invalid state shape
        with pytest.raises(ValueError, match="State shape"):
            agent.get_q_values(np.array([1.0, 2.0]))  # Wrong shape

    @pytest.mark.unit
    def test_get_action_values(self) -> None:
        """Test getting Q-values as action dictionary."""
        # Arrange - Create agent
        agent = DQNAgent(state_size=4, action_size=2)
        state = np.array([1.0, 2.0, 3.0, 4.0])

        # Act - Get action values
        action_values = agent.get_action_values(state)

        # Assert - Verify action values format
        assert isinstance(action_values, dict)
        assert len(action_values) == 2
        assert all(isinstance(value, float) for value in action_values.values())

    @pytest.mark.unit
    def test_get_policy(self) -> None:
        """Test getting greedy policy for multiple states."""
        # Arrange - Create agent
        agent = DQNAgent(state_size=4, action_size=2)
        states = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
            ]
        )

        # Act - Get policy
        policy = agent.get_policy(states)

        # Assert - Verify policy shape and values
        assert policy.shape == (2,)
        assert all(0 <= action < 2 for action in policy)

    @pytest.mark.unit
    def test_get_policy_invalid_states_shape(self) -> None:
        """Test get_policy raises error for invalid states shape."""
        # Arrange - Create agent
        agent = DQNAgent(state_size=4, action_size=2)

        # Act & Assert - Test invalid states shape
        with pytest.raises(ValueError, match="States shape"):
            agent.get_policy(np.array([[1.0, 2.0], [3.0, 4.0]]))  # Wrong shape

    @pytest.mark.unit
    def test_reset_memory(self) -> None:
        """Test resetting memory buffer."""
        # Arrange - Create agent and add some experiences
        agent = DQNAgent(state_size=4, action_size=2, memory_size=100)

        # Add some experiences
        for i in range(5):
            state = np.array([float(i), float(i + 1), float(i + 2), float(i + 3)])
            next_state = np.array(
                [float(i + 1), float(i + 2), float(i + 3), float(i + 4)]
            )
            agent.step(state, i % 2, 1.0, next_state, False)

        # Act - Reset memory
        agent.reset_memory()

        # Assert - Verify memory is empty
        assert len(agent.memory) == 0

    @pytest.mark.unit
    def test_set_epsilon(self) -> None:
        """Test setting exploration rate."""
        # Arrange - Create agent
        agent = DQNAgent(state_size=4, action_size=2)

        # Act - Set new epsilon
        agent.set_epsilon(0.5)

        # Assert - Verify epsilon was set
        assert agent.get_epsilon() == 0.5

    @pytest.mark.unit
    def test_set_epsilon_invalid(self) -> None:
        """Test setting invalid epsilon raises error."""
        # Arrange - Create agent
        agent = DQNAgent(state_size=4, action_size=2)

        # Act & Assert - Test invalid epsilon
        with pytest.raises(ValueError, match="epsilon must be between 0 and 1"):
            agent.set_epsilon(1.5)

    @pytest.mark.unit
    def test_save_and_load_model(self, tmp_path) -> None:
        """Test saving and loading model."""
        # Arrange - Create agent and train it a bit
        agent = DQNAgent(state_size=4, action_size=2, memory_size=100, batch_size=10)

        # Add some experiences
        for i in range(15):
            state = np.array([float(i), float(i + 1), float(i + 2), float(i + 3)])
            next_state = np.array(
                [float(i + 1), float(i + 2), float(i + 3), float(i + 4)]
            )
            agent.step(state, i % 2, 1.0, next_state, False)

        filepath = tmp_path / "test_model.pth"

        # Act - Save model
        agent.save_model(str(filepath))

        # Create new agent and load model
        new_agent = DQNAgent(state_size=4, action_size=2)
        new_agent.load_model(str(filepath))

        # Assert - Verify model was loaded correctly
        assert new_agent.epsilon == agent.epsilon
        assert new_agent.step_count == agent.step_count
        assert new_agent.state_size == agent.state_size
        assert new_agent.action_size == agent.action_size

    @pytest.mark.unit
    def test_dqn_variants(self) -> None:
        """Test different DQN variants work correctly."""
        # Arrange - Create agents with different variants
        vanilla_agent = DQNAgent(state_size=4, action_size=2, dqn_variant="vanilla")
        double_agent = DQNAgent(state_size=4, action_size=2, dqn_variant="double")

        # Act - Get variant settings
        vanilla_variant = vanilla_agent.dqn_variant
        double_variant = double_agent.dqn_variant

        # Assert - Verify variants are set correctly
        assert vanilla_variant == "vanilla"
        assert double_variant == "double"

    @pytest.mark.unit
    def test_huber_loss_setting(self) -> None:
        """Test Huber loss setting works correctly."""
        # Arrange - Create agents with different loss functions
        mse_agent = DQNAgent(state_size=4, action_size=2, use_huber_loss=False)
        huber_agent = DQNAgent(state_size=4, action_size=2, use_huber_loss=True)

        # Act - Get loss settings
        mse_loss_setting = mse_agent.use_huber_loss
        huber_loss_setting = huber_agent.use_huber_loss

        # Assert - Verify loss settings are correct
        assert mse_loss_setting is False
        assert huber_loss_setting is True

    @pytest.mark.unit
    def test_soft_target_updates(self) -> None:
        """Test soft target updates work correctly."""
        # Arrange - Create agent with soft updates
        agent = DQNAgent(state_size=4, action_size=2, tau=0.01)

        # Act - Get tau value
        tau_value = agent.tau

        # Assert - Verify tau is set correctly
        assert tau_value == 0.01

    @pytest.mark.unit
    def test_gradient_clipping_setting(self) -> None:
        """Test gradient clipping setting works correctly."""
        # Arrange - Create agent with gradient clipping
        agent = DQNAgent(state_size=4, action_size=2, gradient_clip_norm=2.0)

        # Act - Get gradient clip norm
        clip_norm = agent.gradient_clip_norm

        # Assert - Verify gradient clip norm is set correctly
        assert clip_norm == 2.0

    @pytest.mark.unit
    def test_agent_repr(self) -> None:
        """Test string representation of agent."""
        # Arrange - Create agent
        agent = DQNAgent(
            state_size=4,
            action_size=2,
            learning_rate=0.001,
            discount_factor=0.95,
            epsilon=0.2,
            dqn_variant="double",
            use_huber_loss=True,
            tau=0.01,
        )

        # Act - Get string representation
        repr_str = repr(agent)

        # Assert - Verify representation contains key information
        assert "DQNAgent" in repr_str
        assert "state_size=4" in repr_str
        assert "action_size=2" in repr_str
        assert "learning_rate=0.001" in repr_str
        assert "discount_factor=0.95" in repr_str
        assert "epsilon=0.200" in repr_str
        assert "dqn_variant=double" in repr_str
        assert "use_huber_loss=True" in repr_str
        assert "tau=0.01" in repr_str

    @pytest.mark.unit
    def test_epsilon_decay_types(self) -> None:
        """Test different epsilon decay types work correctly."""
        # Arrange - Create agents with different decay types
        multiplicative_agent = DQNAgent(
            state_size=4,
            action_size=2,
            epsilon_decay_type="multiplicative",
            epsilon_decay=0.9,
        )
        linear_agent = DQNAgent(
            state_size=4,
            action_size=2,
            epsilon_decay_type="linear",
            epsilon_decay_steps=100,
        )

        # Act - Get decay types
        mult_type = multiplicative_agent.epsilon_decay_type
        linear_type = linear_agent.epsilon_decay_type

        # Assert - Verify decay types are set correctly
        assert mult_type == "multiplicative"
        assert linear_type == "linear"

    @pytest.mark.unit
    def test_linear_epsilon_decay(self) -> None:
        """Test linear epsilon decay works correctly."""
        # Arrange - Create agent with linear decay
        agent = DQNAgent(
            state_size=4,
            action_size=2,
            epsilon=1.0,
            epsilon_min=0.1,
            epsilon_decay_type="linear",
            epsilon_decay_steps=10,
        )

        # Act - Decay epsilon by step
        initial_epsilon = agent.get_epsilon()
        agent.decay_epsilon_by_step(0)  # Should be at initial
        step_0_epsilon = agent.get_epsilon()
        agent.decay_epsilon_by_step(5)  # Should be halfway
        step_5_epsilon = agent.get_epsilon()
        agent.decay_epsilon_by_step(10)  # Should be at minimum
        step_10_epsilon = agent.get_epsilon()

        # Assert - Verify linear decay progression
        assert step_0_epsilon == initial_epsilon
        assert step_5_epsilon < step_0_epsilon
        assert step_10_epsilon == 0.1  # epsilon_min

    @pytest.mark.unit
    def test_epsilon_decay_validation(self) -> None:
        """Test epsilon decay parameter validation."""
        # Arrange & Act & Assert - Test invalid decay type
        with pytest.raises(ValidationError):
            DQNAgent(
                state_size=4,
                action_size=2,
                epsilon_decay_type="invalid",
            )

        # Arrange & Act & Assert - Test linear decay without steps
        with pytest.raises(ValidationError, match="epsilon_decay_steps"):
            DQNAgent(
                state_size=4,
                action_size=2,
                epsilon_decay_type="linear",
            )

        # Arrange & Act & Assert - Test invalid decay steps
        with pytest.raises(ValidationError, match="epsilon_decay_steps"):
            DQNAgent(
                state_size=4,
                action_size=2,
                epsilon_decay_type="linear",
                epsilon_decay_steps=0,
            )

    @pytest.mark.unit
    def test_soft_target_updates_with_tau(self) -> None:
        """Test soft target updates work with tau > 0."""
        # Arrange - Create agent with soft updates
        agent = DQNAgent(state_size=4, action_size=2, tau=0.01)

        # Act - Get tau value and verify soft update is enabled
        tau_value = agent.tau

        # Assert - Verify tau is set correctly for soft updates
        assert tau_value == 0.01
        assert tau_value > 0.0
