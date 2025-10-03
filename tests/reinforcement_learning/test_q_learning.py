"""Tests for Q-Learning algorithm implementation."""

import numpy as np
import pytest

from algokit.algorithms.reinforcement_learning.q_learning import QLearningAgent


class TestQLearningAgent:
    """Test the Q-Learning agent implementation."""

    @pytest.mark.unit
    def test_agent_initialization(self) -> None:
        """Test Q-Learning agent initializes correctly."""
        # Arrange - Set up agent parameters
        state_space_size = 5
        action_space_size = 3
        learning_rate = 0.1
        discount_factor = 0.95
        epsilon = 0.1

        # Act - Create Q-Learning agent
        agent = QLearningAgent(
            state_space_size=state_space_size,
            action_space_size=action_space_size,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
        )

        # Assert - Verify agent properties
        assert agent.state_space_size == state_space_size
        assert agent.action_space_size == action_space_size
        assert agent.learning_rate == learning_rate
        assert agent.discount_factor == discount_factor
        assert agent.epsilon == epsilon
        assert agent.q_table.shape == (state_space_size, action_space_size)
        assert np.all(agent.q_table == 0)

    @pytest.mark.unit
    def test_agent_initialization_with_seed(self) -> None:
        """Test Q-Learning agent initializes with random seed."""
        # Arrange - Set up agent with random seed
        random_seed = 42

        # Act - Create Q-Learning agent with seed
        agent = QLearningAgent(
            state_space_size=3,
            action_space_size=2,
            random_seed=random_seed,
        )

        # Assert - Verify agent is created without errors
        assert agent.state_space_size == 3
        assert agent.action_space_size == 2

    @pytest.mark.unit
    def test_invalid_initialization_parameters(self) -> None:
        """Test Q-Learning agent raises errors for invalid parameters."""
        # Arrange & Act & Assert - Test invalid state space size
        with pytest.raises(ValueError, match="state_space_size must be positive"):
            QLearningAgent(state_space_size=0, action_space_size=2)

        # Arrange & Act & Assert - Test invalid action space size
        with pytest.raises(ValueError, match="action_space_size must be positive"):
            QLearningAgent(state_space_size=3, action_space_size=0)

        # Arrange & Act & Assert - Test invalid learning rate
        with pytest.raises(ValueError, match="learning_rate must be between 0 and 1"):
            QLearningAgent(state_space_size=3, action_space_size=2, learning_rate=1.5)

        # Arrange & Act & Assert - Test invalid discount factor
        with pytest.raises(ValueError, match="discount_factor must be between 0 and 1"):
            QLearningAgent(state_space_size=3, action_space_size=2, discount_factor=1.5)

        # Arrange & Act & Assert - Test invalid epsilon
        with pytest.raises(ValueError, match="epsilon must be between 0 and 1"):
            QLearningAgent(state_space_size=3, action_space_size=2, epsilon=1.5)

        # Arrange & Act & Assert - Test invalid epsilon min
        with pytest.raises(
            ValueError, match="epsilon_min must be between 0 and epsilon"
        ):
            QLearningAgent(
                state_space_size=3, action_space_size=2, epsilon=0.1, epsilon_min=0.2
            )

    @pytest.mark.unit
    def test_get_action_epsilon_greedy(self) -> None:
        """Test action selection using epsilon-greedy policy."""
        # Arrange - Create agent with high epsilon for exploration
        agent = QLearningAgent(
            state_space_size=3,
            action_space_size=2,
            epsilon=1.0,  # Always explore
            random_seed=42,
        )
        state = 1

        # Act - Get action multiple times
        actions = [agent.get_action(state) for _ in range(10)]

        # Assert - Verify actions are valid
        assert all(0 <= action < 2 for action in actions)

    @pytest.mark.unit
    def test_get_action_invalid_state(self) -> None:
        """Test get_action raises error for invalid state."""
        # Arrange - Create agent
        agent = QLearningAgent(state_space_size=3, action_space_size=2)

        # Act & Assert - Test invalid state
        with pytest.raises(ValueError, match="State -1 is out of range"):
            agent.get_action(-1)

        with pytest.raises(ValueError, match="State 3 is out of range"):
            agent.get_action(3)

    @pytest.mark.unit
    def test_update_q_value(self) -> None:
        """Test Q-value update using Q-Learning rule."""
        # Arrange - Create agent with specific parameters
        agent = QLearningAgent(
            state_space_size=3,
            action_space_size=2,
            learning_rate=0.1,
            discount_factor=0.9,
        )
        state = 0
        action = 1
        reward = 1.0
        next_state = 1

        # Act - Update Q-value
        agent.update_q_value(state, action, reward, next_state)

        # Assert - Verify Q-value was updated
        expected_q = 0.1 * (
            1.0 + 0.9 * 0 - 0
        )  # alpha * (reward + gamma * max_q - current_q)
        assert agent.get_q_value(state, action) == expected_q

    @pytest.mark.unit
    def test_update_q_value_episode_done(self) -> None:
        """Test Q-value update when episode is done."""
        # Arrange - Create agent
        agent = QLearningAgent(state_space_size=3, action_space_size=2)
        state = 0
        action = 1
        reward = 1.0
        next_state = 1

        # Act - Update Q-value with done=True
        agent.update_q_value(state, action, reward, next_state, done=True)

        # Assert - Verify Q-value was updated without future rewards
        expected_q = 0.1 * (
            1.0 + 0.95 * 0 - 0
        )  # alpha * (reward + gamma * 0 - current_q)
        assert agent.get_q_value(state, action) == expected_q

    @pytest.mark.unit
    def test_update_q_value_invalid_parameters(self) -> None:
        """Test update_q_value raises errors for invalid parameters."""
        # Arrange - Create agent
        agent = QLearningAgent(state_space_size=3, action_space_size=2)

        # Act & Assert - Test invalid state
        with pytest.raises(ValueError, match="State -1 is out of range"):
            agent.update_q_value(-1, 0, 1.0, 1)

        # Act & Assert - Test invalid action
        with pytest.raises(ValueError, match="Action -1 is out of range"):
            agent.update_q_value(0, -1, 1.0, 1)

        # Act & Assert - Test invalid next state
        with pytest.raises(ValueError, match="Next state -1 is out of range"):
            agent.update_q_value(0, 0, 1.0, -1)

    @pytest.mark.unit
    def test_decay_epsilon(self) -> None:
        """Test epsilon decay over time."""
        # Arrange - Create agent with epsilon decay
        agent = QLearningAgent(
            state_space_size=3,
            action_space_size=2,
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
        agent = QLearningAgent(
            state_space_size=3,
            action_space_size=2,
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
    def test_get_q_value(self) -> None:
        """Test getting Q-value for state-action pair."""
        # Arrange - Create agent and manually set Q-value
        agent = QLearningAgent(state_space_size=3, action_space_size=2)
        agent.q_table[1, 1] = 0.5

        # Act - Get Q-value
        q_value = agent.get_q_value(1, 1)

        # Assert - Verify Q-value is correct
        assert q_value == 0.5

    @pytest.mark.unit
    def test_get_q_value_invalid_parameters(self) -> None:
        """Test get_q_value raises errors for invalid parameters."""
        # Arrange - Create agent
        agent = QLearningAgent(state_space_size=3, action_space_size=2)

        # Act & Assert - Test invalid state
        with pytest.raises(ValueError, match="State -1 is out of range"):
            agent.get_q_value(-1, 0)

        # Act & Assert - Test invalid action
        with pytest.raises(ValueError, match="Action -1 is out of range"):
            agent.get_q_value(0, -1)

    @pytest.mark.unit
    def test_get_policy(self) -> None:
        """Test getting greedy policy."""
        # Arrange - Create agent and set Q-values
        agent = QLearningAgent(state_space_size=3, action_space_size=2)
        agent.q_table[0, 0] = 0.1
        agent.q_table[0, 1] = 0.5  # Best action for state 0
        agent.q_table[1, 0] = 0.8  # Best action for state 1
        agent.q_table[1, 1] = 0.3
        agent.q_table[2, 0] = 0.2
        agent.q_table[2, 1] = 0.7  # Best action for state 2

        # Act - Get policy
        policy = agent.get_policy()

        # Assert - Verify policy selects best actions
        assert policy == [1, 0, 1]

    @pytest.mark.unit
    def test_get_state_values(self) -> None:
        """Test getting state values."""
        # Arrange - Create agent and set Q-values
        agent = QLearningAgent(state_space_size=3, action_space_size=2)
        agent.q_table[0, 0] = 0.1
        agent.q_table[0, 1] = 0.5  # Max for state 0
        agent.q_table[1, 0] = 0.8  # Max for state 1
        agent.q_table[1, 1] = 0.3
        agent.q_table[2, 0] = 0.2
        agent.q_table[2, 1] = 0.7  # Max for state 2

        # Act - Get state values
        state_values = agent.get_state_values()

        # Assert - Verify state values are maximum Q-values
        assert state_values == [0.5, 0.8, 0.7]

    @pytest.mark.unit
    def test_reset_q_table(self) -> None:
        """Test resetting Q-table to zeros."""
        # Arrange - Create agent and modify Q-table
        agent = QLearningAgent(state_space_size=3, action_space_size=2)
        agent.q_table[0, 0] = 0.5
        agent.q_table[1, 1] = 0.8

        # Act - Reset Q-table
        agent.reset_q_table()

        # Assert - Verify Q-table is all zeros
        assert np.all(agent.q_table == 0)

    @pytest.mark.unit
    def test_set_epsilon(self) -> None:
        """Test setting exploration rate."""
        # Arrange - Create agent
        agent = QLearningAgent(state_space_size=3, action_space_size=2)

        # Act - Set new epsilon
        agent.set_epsilon(0.5)

        # Assert - Verify epsilon was set
        assert agent.get_epsilon() == 0.5

    @pytest.mark.unit
    def test_set_epsilon_invalid(self) -> None:
        """Test setting invalid epsilon raises error."""
        # Arrange - Create agent
        agent = QLearningAgent(state_space_size=3, action_space_size=2)

        # Act & Assert - Test invalid epsilon
        with pytest.raises(ValueError, match="epsilon must be between 0 and 1"):
            agent.set_epsilon(1.5)

    @pytest.mark.unit
    def test_get_q_table(self) -> None:
        """Test getting copy of Q-table."""
        # Arrange - Create agent and modify Q-table
        agent = QLearningAgent(state_space_size=3, action_space_size=2)
        agent.q_table[0, 0] = 0.5

        # Act - Get Q-table copy
        q_table_copy = agent.get_q_table()

        # Assert - Verify it's a copy (modifying copy doesn't affect original)
        q_table_copy[0, 0] = 0.8
        assert agent.q_table[0, 0] == 0.5
        assert q_table_copy[0, 0] == 0.8

    @pytest.mark.unit
    def test_agent_repr(self) -> None:
        """Test string representation of agent."""
        # Arrange - Create agent
        agent = QLearningAgent(
            state_space_size=5,
            action_space_size=3,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.2,
        )

        # Act - Get string representation
        repr_str = repr(agent)

        # Assert - Verify representation contains key information
        assert "QLearningAgent" in repr_str
        assert "state_space_size=5" in repr_str
        assert "action_space_size=3" in repr_str
        assert "learning_rate=0.1" in repr_str
        assert "discount_factor=0.95" in repr_str
        assert "epsilon=0.200" in repr_str
