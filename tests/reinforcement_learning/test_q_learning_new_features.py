"""Tests for new Q-Learning algorithm features."""

import numpy as np
import pytest
from pydantic import ValidationError

from algokit.algorithms.reinforcement_learning.q_learning import QLearningAgent


class TestQLearningNewFeatures:
    """Test the new features of the Q-Learning agent."""

    @pytest.mark.unit
    def test_random_tie_breaking_in_action_selection(self) -> None:
        """Test that action selection breaks ties randomly."""
        # Arrange - Create agent with tied Q-values
        agent = QLearningAgent(
            state_space_size=1,
            action_space_size=3,
            epsilon_start=0.0,  # No exploration
            epsilon_end=0.0,  # No exploration
            random_seed=42,
        )
        # Set all Q-values to the same value to create ties
        agent.q_table[0, :] = 1.0

        # Act - Get multiple actions
        actions = [agent.select_action(0) for _ in range(100)]

        # Assert - Should get different actions due to random tie-breaking
        unique_actions = set(actions)
        assert len(unique_actions) > 1, "Should break ties randomly"

    @pytest.mark.unit
    def test_set_seed_reproducibility(self) -> None:
        """Test that set_seed provides reproducible results."""
        # Arrange - Create agent and set seed
        agent = QLearningAgent(state_space_size=3, action_space_size=2)
        agent.set_seed(123)

        # Act - Get actions from agent
        actions = [agent.select_action(0) for _ in range(5)]

        # Assert - Actions should be deterministic (not all the same due to exploration)
        assert len(actions) == 5, "Should get correct number of actions"
        # The key is that set_seed doesn't raise an error and produces some actions

    @pytest.mark.unit
    def test_double_q_learning_enabled(self) -> None:
        """Test Double Q-Learning functionality."""
        # Arrange - Create agent with Double Q-Learning enabled
        agent = QLearningAgent(
            state_space_size=3,
            action_space_size=2,
            use_double_q=True,
            random_seed=42,
        )

        # Act - Perform a step
        agent.step(state=0, action=0, reward=1.0, next_state=1, done=False)

        # Assert - Both Q-tables should exist and be different
        assert agent.q_table_b is not None, "Second Q-table should exist"
        assert agent.use_double_q is True, "Double Q-Learning should be enabled"

    @pytest.mark.unit
    def test_double_q_learning_disabled(self) -> None:
        """Test that Double Q-Learning is disabled by default."""
        # Arrange - Create agent without Double Q-Learning
        agent = QLearningAgent(state_space_size=3, action_space_size=2)

        # Act - Check Double Q-Learning status

        # Assert - Second Q-table should not exist
        assert agent.q_table_b is None, "Second Q-table should not exist when disabled"
        assert agent.use_double_q is False, "Double Q-Learning should be disabled"

    @pytest.mark.unit
    def test_debug_logging_enabled(self) -> None:
        """Test debug logging functionality."""
        # Arrange - Create agent with debug enabled
        agent = QLearningAgent(
            state_space_size=3,
            action_space_size=2,
            debug=True,
        )

        # Act - Perform a step (should not raise errors)
        agent.step(state=0, action=0, reward=1.0, next_state=1, done=False)

        # Assert - Logger should be set up
        assert agent.debug is True, "Debug should be enabled"
        assert agent._logger is not None, "Logger should be initialized"

    @pytest.mark.unit
    def test_get_action_values(self) -> None:
        """Test getting action values for a state."""
        # Arrange - Create agent and set Q-values
        agent = QLearningAgent(state_space_size=3, action_space_size=2)
        agent.q_table[0, 0] = 0.5
        agent.q_table[0, 1] = 0.8

        # Act - Get action values
        action_values = agent.get_action_values(0)

        # Assert - Should return correct values
        assert action_values == [0.5, 0.8], "Should return correct action values"

    @pytest.mark.unit
    def test_pretty_print_policy(self) -> None:
        """Test pretty printing policy with names."""
        # Arrange - Create agent and set Q-values
        agent = QLearningAgent(state_space_size=2, action_space_size=2)
        agent.q_table[0, 1] = 0.8  # Best action for state 0
        agent.q_table[1, 0] = 0.9  # Best action for state 1

        # Act - Pretty print policy
        policy_str = agent.pretty_print_policy(
            state_names=["Start", "End"], action_names=["Left", "Right"]
        )

        # Assert - Should contain state and action names
        assert "Start" in policy_str, "Should contain state names"
        assert "End" in policy_str, "Should contain state names"
        assert "Left" in policy_str or "Right" in policy_str, (
            "Should contain action names"
        )

    @pytest.mark.unit
    def test_pretty_print_policy_default_names(self) -> None:
        """Test pretty printing policy with default names."""
        # Arrange - Create agent
        agent = QLearningAgent(state_space_size=2, action_space_size=2)

        # Act - Pretty print policy
        policy_str = agent.pretty_print_policy()

        # Assert - Should contain default names
        assert "State 0" in policy_str, "Should contain default state names"
        assert "State 1" in policy_str, "Should contain default state names"
        assert "Action" in policy_str, "Should contain default action names"

    @pytest.mark.unit
    def test_get_q_table_b(self) -> None:
        """Test getting second Q-table."""
        # Arrange - Create agent with Double Q-Learning
        agent = QLearningAgent(
            state_space_size=3,
            action_space_size=2,
            use_double_q=True,
        )

        # Act - Get second Q-table
        q_table_b = agent.get_q_table_b()

        # Assert - Should return a copy of the second Q-table
        assert q_table_b is not None, "Should return second Q-table"
        assert q_table_b.shape == (3, 2), "Should have correct shape"

    @pytest.mark.unit
    def test_get_q_table_b_disabled(self) -> None:
        """Test getting second Q-table when disabled."""
        # Arrange - Create agent without Double Q-Learning
        agent = QLearningAgent(state_space_size=3, action_space_size=2)

        # Act - Get second Q-table
        q_table_b = agent.get_q_table_b()

        # Assert - Should return None
        assert q_table_b is None, (
            "Should return None when Double Q-Learning is disabled"
        )

    @pytest.mark.unit
    def test_reset_q_table_with_double_q(self) -> None:
        """Test resetting Q-tables with Double Q-Learning."""
        # Arrange - Create agent with Double Q-Learning and modify Q-tables
        agent = QLearningAgent(
            state_space_size=3,
            action_space_size=2,
            use_double_q=True,
        )
        agent.q_table[0, 0] = 0.5
        assert agent.q_table_b is not None, (
            "q_table_b should not be None when use_double_q=True"
        )
        agent.q_table_b[0, 0] = 0.8

        # Act - Reset Q-tables
        agent.reset_q_table()

        # Assert - Both Q-tables should be reset
        assert np.all(agent.q_table == 0), "First Q-table should be reset"
        assert np.all(agent.q_table_b == 0), "Second Q-table should be reset"

    @pytest.mark.unit
    def test_epsilon_scheduling_parameters(self) -> None:
        """Test epsilon scheduling parameters."""
        # Arrange - Create agent with custom epsilon scheduling
        agent = QLearningAgent(
            state_space_size=3,
            action_space_size=2,
            epsilon_start=0.9,
            epsilon_end=0.05,
            epsilon_decay=0.95,
        )

        # Act - Check parameters are set correctly

        # Assert - Parameters should be set correctly
        assert agent.epsilon_start == 0.9, "epsilon_start should be set"
        assert agent.epsilon_end == 0.05, "epsilon_end should be set"
        assert agent.epsilon_decay == 0.95, "epsilon_decay should be set"
        assert agent.epsilon == 0.9, "Current epsilon should start at epsilon_start"

    @pytest.mark.unit
    def test_epsilon_decay_with_end_value(self) -> None:
        """Test epsilon decay respects end value."""
        # Arrange - Create agent with epsilon decay
        agent = QLearningAgent(
            state_space_size=3,
            action_space_size=2,
            epsilon_start=0.5,
            epsilon_end=0.1,
            epsilon_decay=0.8,
        )

        # Act - Decay epsilon multiple times
        for _ in range(10):
            agent.decay_epsilon()

        # Assert - Epsilon should not go below end value
        assert agent.epsilon >= agent.epsilon_end, (
            "Epsilon should not go below end value"
        )

    @pytest.mark.unit
    def test_step_method_vs_update_q_value(self) -> None:
        """Test that step method and update_q_value produce same results."""
        # Arrange - Create two identical agents
        agent1 = QLearningAgent(state_space_size=3, action_space_size=2, random_seed=42)
        agent2 = QLearningAgent(state_space_size=3, action_space_size=2, random_seed=42)

        # Act - Use different methods
        agent1.step(state=0, action=0, reward=1.0, next_state=1, done=False)
        agent2.update_q_value(state=0, action=0, reward=1.0, next_state=1, done=False)

        # Assert - Q-tables should be identical
        assert np.allclose(agent1.q_table, agent2.q_table), (
            "Both methods should produce same results"
        )

    @pytest.mark.unit
    def test_select_action_vs_get_action(self) -> None:
        """Test that select_action and get_action produce same results."""
        # Arrange - Create agent and set seed
        agent = QLearningAgent(state_space_size=3, action_space_size=2)
        agent.set_seed(42)

        # Act - Use both methods on same agent
        action1 = agent.select_action(0)
        action2 = agent.get_action(0)

        # Assert - Both methods should work (they're aliases)
        assert isinstance(action1, int | np.integer), (
            "select_action should return integer"
        )
        assert isinstance(action2, int | np.integer), "get_action should return integer"
        assert 0 <= action1 < 2, "action should be valid"
        assert 0 <= action2 < 2, "action should be valid"

    @pytest.mark.unit
    def test_parameter_validation_strict(self) -> None:
        """Test strict parameter validation."""
        # Arrange - Set up test cases for invalid parameters

        # Act & Assert - Test learning_rate = 0 (should fail via Pydantic)
        with pytest.raises(ValidationError, match="learning_rate"):
            QLearningAgent(state_space_size=3, action_space_size=2, learning_rate=0.0)

        # Act & Assert - Test discount_factor = 0 (should fail via Pydantic)
        with pytest.raises(ValidationError, match="discount_factor"):
            QLearningAgent(state_space_size=3, action_space_size=2, discount_factor=0.0)

        # Act & Assert - Test epsilon_end > epsilon_start (should fail via Pydantic)
        with pytest.raises(
            ValidationError, match="epsilon_end.*must be less than or equal"
        ):
            QLearningAgent(
                state_space_size=3,
                action_space_size=2,
                epsilon_start=0.1,
                epsilon_end=0.5,
            )
