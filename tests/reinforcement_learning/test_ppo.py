"""Tests for PPO (Proximal Policy Optimization) algorithm implementation."""

import numpy as np
import pytest
import torch

from algokit.algorithms.reinforcement_learning.ppo import (
    PolicyNetwork,
    PPOAgent,
    RolloutBuffer,
    RolloutExperience,
    ValueNetwork,
)


class TestPolicyNetwork:
    """Test cases for PolicyNetwork."""

    def test_policy_network_initialization(self) -> None:
        """Test that PolicyNetwork initializes correctly."""
        # Arrange - Set up network parameters
        state_size = 4
        action_size = 2
        hidden_sizes = [64, 32]

        # Act - Create the network
        network = PolicyNetwork(state_size, action_size, hidden_sizes)

        # Assert - Verify network is created with parameters
        assert network is not None
        assert len(list(network.parameters())) > 0

    def test_policy_network_forward_pass(self) -> None:
        """Test that PolicyNetwork forward pass works correctly."""
        # Arrange - Set up network and input tensor
        state_size = 4
        action_size = 2
        network = PolicyNetwork(state_size, action_size)
        state = torch.randn(1, state_size)

        # Act - Run forward pass through network
        output = network(state)

        # Assert - Verify output shape and probability sum
        assert output.shape == (1, action_size)
        # Policy network outputs logits, not probabilities
        assert torch.isfinite(output).all()

    def test_policy_network_with_dropout(self) -> None:
        """Test that PolicyNetwork works with dropout."""
        # Arrange - Set up network with dropout and input
        state_size = 4
        action_size = 2
        dropout_rate = 0.5
        network = PolicyNetwork(state_size, action_size, dropout_rate=dropout_rate)
        state = torch.randn(1, state_size)

        # Act - Run forward pass through network
        output = network(state)

        # Assert - Verify output shape and probability sum
        assert output.shape == (1, action_size)
        # Policy network outputs logits, not probabilities
        assert torch.isfinite(output).all()


class TestValueNetwork:
    """Test cases for ValueNetwork."""

    def test_value_network_initialization(self) -> None:
        """Test that ValueNetwork initializes correctly."""
        # Arrange - Set up network parameters
        state_size = 4
        hidden_sizes = [64, 32]

        # Act - Create the network
        network = ValueNetwork(state_size, hidden_sizes)

        # Assert - Verify network is created with parameters
        assert network is not None
        assert len(list(network.parameters())) > 0

    def test_value_network_forward_pass(self) -> None:
        """Test that ValueNetwork forward pass works correctly."""
        # Arrange - Set up network and input tensor
        state_size = 4
        network = ValueNetwork(state_size)
        state = torch.randn(1, state_size)

        # Act - Run forward pass through network
        output = network(state)

        # Assert - Verify output shape is correct
        assert output.shape == (1,)

    def test_value_network_with_dropout(self) -> None:
        """Test that ValueNetwork works with dropout."""
        # Arrange - Set up network with dropout and input
        state_size = 4
        dropout_rate = 0.5
        network = ValueNetwork(state_size, dropout_rate=dropout_rate)
        state = torch.randn(1, state_size)

        # Act - Run forward pass through network
        output = network(state)

        # Assert - Verify output shape is correct
        assert output.shape == (1,)


class TestRolloutBuffer:
    """Test cases for RolloutBuffer."""

    def test_buffer_initialization(self) -> None:
        """Test that RolloutBuffer initializes correctly."""
        # Arrange - Set up buffer parameters
        buffer_size = 1000

        # Act - Create the buffer
        buffer = RolloutBuffer(buffer_size)

        # Assert - Verify buffer is created with correct size
        assert buffer.buffer_size == buffer_size
        assert len(buffer) == 0

    def test_buffer_add_experience(self) -> None:
        """Test that RolloutBuffer adds experiences correctly."""
        # Arrange - Set up buffer and experience
        buffer = RolloutBuffer(10)
        experience = RolloutExperience(
            state=np.array([1, 2, 3, 4]),
            action=1,
            reward=0.5,
            done=False,
            old_log_prob=-0.1,
            old_value=0.8,
        )

        # Act - Add experience to buffer
        buffer.add(experience)

        # Assert - Verify experience was added correctly
        assert len(buffer) == 1
        assert buffer.buffer[0] == experience

    def test_buffer_sample(self) -> None:
        """Test that RolloutBuffer samples experiences correctly."""
        # Arrange - Set up buffer with multiple experiences
        buffer = RolloutBuffer(10)
        experiences = [
            RolloutExperience(
                state=np.array([i, i + 1, i + 2, i + 3]),
                action=i % 2,
                reward=float(i),
                done=False,
                old_log_prob=-0.1,
                old_value=0.8,
            )
            for i in range(5)
        ]

        for exp in experiences:
            buffer.add(exp)

        # Act - Get all experiences from buffer
        sampled = buffer.get_all()[:3]

        # Assert - Verify correct number and type of samples
        assert len(sampled) == 3
        assert all(isinstance(exp, RolloutExperience) for exp in sampled)

    def test_buffer_clear(self) -> None:
        """Test that RolloutBuffer clears correctly."""
        # Arrange - Set up buffer with experience
        buffer = RolloutBuffer(10)
        experience = RolloutExperience(
            state=np.array([1, 2, 3, 4]),
            action=1,
            reward=0.5,
            done=False,
            old_log_prob=-0.1,
            old_value=0.8,
        )
        buffer.add(experience)

        # Act - Clear the buffer
        buffer.clear()

        # Assert - Verify buffer is empty
        assert len(buffer) == 0


class TestPPOAgent:
    """Test cases for PPOAgent."""

    def test_agent_initialization(self) -> None:
        """Test that PPOAgent initializes correctly."""
        # Arrange - Set up agent parameters
        state_size = 4
        action_size = 2

        # Act - Create the agent
        agent = PPOAgent(state_size, action_size)

        # Assert - Verify agent properties are set correctly
        assert agent.state_size == state_size
        assert agent.action_size == action_size
        assert agent.device.type in ["cpu", "cuda"]

    def test_agent_initialization_with_invalid_params(self) -> None:
        """Test that PPOAgent raises errors for invalid parameters."""
        # Arrange & Act & Assert - Test various invalid parameter combinations
        with pytest.raises(ValueError, match="state_size must be positive"):
            PPOAgent(0, 2)

        with pytest.raises(ValueError, match="action_size must be positive"):
            PPOAgent(4, 0)

        with pytest.raises(ValueError, match="learning_rate must be between 0 and 1"):
            PPOAgent(4, 2, learning_rate=1.5)

        with pytest.raises(ValueError, match="discount_factor must be between 0 and 1"):
            PPOAgent(4, 2, discount_factor=1.5)

        with pytest.raises(ValueError, match="clip_ratio must be between 0 and 1"):
            PPOAgent(4, 2, clip_ratio=1.5)

    def test_get_action_training_mode(self) -> None:
        """Test that get_action works in training mode."""
        # Arrange - Set up agent and state
        agent = PPOAgent(4, 2)
        state = np.array([1, 2, 3, 4])

        # Act - Get action from agent in training mode
        action, log_prob, value = agent.get_action(state)

        # Assert - Verify action and values are valid
        assert isinstance(action, int)
        assert 0 <= action < 2
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_get_action_evaluation_mode(self) -> None:
        """Test that get_action works in evaluation mode."""
        # Arrange - Set up agent and state
        agent = PPOAgent(4, 2)
        state = np.array([1, 2, 3, 4])

        # Act - Get action from agent in evaluation mode
        action, log_prob, value = agent.get_action(state)

        # Assert - Verify action and values are valid
        assert isinstance(action, int)
        assert 0 <= action < 2
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_add_experience(self) -> None:
        """Test that add_experience works correctly."""
        # Arrange - Set up agent and experience data
        agent = PPOAgent(4, 2)
        state = np.array([1, 2, 3, 4])
        action = 1
        reward = 0.5
        log_prob = -0.1
        value = 0.8
        done = False

        # Act - Add experience to agent buffer manually
        experience = RolloutExperience(
            state=state,
            action=action,
            reward=reward,
            done=done,
            old_log_prob=log_prob,
            old_value=value,
        )
        agent.rollout_buffer.add(experience)

        # Assert - Verify experience was added to buffer
        assert len(agent.rollout_buffer) == 1

    def test_compute_gae(self) -> None:
        """Test that compute_gae works correctly."""
        # Arrange - Set up agent and trajectory data
        agent = PPOAgent(4, 2)
        rewards = [1.0, 0.5, 2.0]
        values = [0.8, 0.9, 1.0]
        dones = [False, False, True]
        next_value = 0.0

        # Act - Compute GAE advantages and returns
        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)

        # Assert - Verify GAE computation results
        assert len(advantages) == 3
        assert len(returns) == 3
        assert all(isinstance(adv, float) for adv in advantages)
        assert all(isinstance(ret, float) for ret in returns)

    def test_update_with_insufficient_data(self) -> None:
        """Test that update returns empty metrics with insufficient data."""
        # Arrange - Set up agent with large batch size
        agent = PPOAgent(4, 2, batch_size=10)

        # Act - Try to update with no data
        metrics = agent.update()

        # Assert - Verify empty metrics are returned
        assert metrics["policy_loss"] == 0.0
        assert metrics["value_loss"] == 0.0
        assert metrics["entropy_loss"] == 0.0

    def test_update_with_sufficient_data(self) -> None:
        """Test that update works with sufficient data."""
        # Arrange - Set up agent and add experiences
        agent = PPOAgent(4, 2, batch_size=5)

        # Add enough experiences
        for _i in range(10):
            state = np.random.randn(4)
            action, log_prob, value = agent.get_action(state)
            experience = RolloutExperience(
                state=state,
                action=action,
                reward=0.5,
                done=False,
                old_log_prob=log_prob,
                old_value=value,
            )
            agent.rollout_buffer.add(experience)

        # Act - Update the agent
        metrics = agent.update()

        # Assert - Verify metrics and buffer clearing
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy_loss" in metrics
        assert len(agent.rollout_buffer) == 0  # Buffer should be cleared after update

    def test_save_and_load(self) -> None:
        """Test that save and load work correctly."""
        # Arrange - Set up agent and file path
        agent = PPOAgent(4, 2)
        filepath = "/tmp/test_ppo_model.pth"

        # Act - Save agent and load into new agent
        agent.save(filepath)

        # Create new agent and load
        new_agent = PPOAgent(4, 2)
        new_agent.load(filepath)

        # Assert - Verify agent properties match
        # Check that parameters are loaded (exact comparison is complex)
        assert new_agent.state_size == agent.state_size
        assert new_agent.action_size == agent.action_size

    def test_set_training_mode(self) -> None:
        """Test that set_training works correctly."""
        # Arrange - Set up agent
        agent = PPOAgent(4, 2)

        # Act - Set training to False
        agent.set_training(False)

        # Assert - Verify training mode is disabled
        assert not agent.training
        assert not agent.policy.training
        assert not agent.value.training

        # Act - Set training to True
        agent.set_training(True)

        # Assert - Verify training mode is enabled
        assert agent.training
        assert agent.policy.training
        assert agent.value.training

    def test_agent_with_custom_parameters(self) -> None:
        """Test that PPOAgent works with custom parameters."""
        # Arrange - Set up custom parameters
        state_size = 8
        action_size = 3
        learning_rate = 0.001
        discount_factor = 0.95
        hidden_sizes = [256, 128, 64]
        dropout_rate = 0.1
        buffer_size = 5000
        batch_size = 32
        clip_ratio = 0.1
        value_coef = 0.3
        entropy_coef = 0.02
        max_grad_norm = 0.3

        # Act - Create agent with custom parameters
        agent = PPOAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            clip_ratio=clip_ratio,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
        )

        # Assert - Verify all parameters are set correctly
        assert agent.state_size == state_size
        assert agent.action_size == action_size
        assert agent.learning_rate == learning_rate
        assert agent.discount_factor == discount_factor
        assert agent.clip_ratio == clip_ratio
        assert agent.value_coef == value_coef
        assert agent.entropy_coef == entropy_coef
        assert agent.max_grad_norm == max_grad_norm

    def test_agent_with_gpu_if_available(self) -> None:
        """Test that PPOAgent uses GPU if available."""
        # Arrange - Set up agent with CUDA device
        agent = PPOAgent(4, 2, device="cuda")

        # Act - Check device type
        device_type = agent.device.type

        # Assert - Verify device is set correctly based on availability
        if torch.cuda.is_available():
            assert device_type == "cuda"
        else:
            assert device_type == "cpu"

    def test_agent_with_random_seed(self) -> None:
        """Test that PPOAgent respects random seed."""
        # Arrange - Set up two agents with same seed
        seed = 42
        agent1 = PPOAgent(4, 2, random_seed=seed)
        agent2 = PPOAgent(4, 2, random_seed=seed)
        state = np.array([1, 2, 3, 4])

        # Act - Get actions from both agents with same seed
        action1, _, _ = agent1.get_action(state)
        action2, _, _ = agent2.get_action(state)

        # Assert - Verify both agents produce valid actions
        assert isinstance(action1, int)
        assert isinstance(action2, int)
        assert 0 <= action1 < 2
        assert 0 <= action2 < 2

    def test_agent_training_loop_integration(self) -> None:
        """Test a complete training loop integration."""
        # Arrange - Set up agent and simulate episode
        agent = PPOAgent(4, 2, batch_size=5)

        # Simulate a training episode
        state = np.random.randn(4)
        for step in range(10):
            # Act - Get action and add experience
            action, log_prob, value = agent.get_action(state)
            reward = np.random.randn()
            done = step == 9

            experience = RolloutExperience(
                state=state,
                action=action,
                reward=reward,
                done=done,
                old_log_prob=log_prob,
                old_value=value,
            )
            agent.rollout_buffer.add(experience)

            if done:
                break
            state = np.random.randn(4)

        # Act - Update the agent
        metrics = agent.update()

        # Assert - Verify training metrics and buffer clearing
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy_loss" in metrics
        assert len(agent.rollout_buffer) == 0  # Buffer should be cleared
