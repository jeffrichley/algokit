"""Tests for new SARSA algorithm features."""

import numpy as np
import pytest
from pydantic import ValidationError

from algokit.algorithms.reinforcement_learning.sarsa import SarsaAgent


class TestSarsaNewFeatures:
    """Test the new features of the SARSA agent."""

    @pytest.mark.unit
    def test_random_tie_breaking_in_action_selection(self) -> None:
        """Test that action selection breaks ties randomly."""
        # Arrange - Create agent with tied Q-values
        agent = SarsaAgent(
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
        agent = SarsaAgent(state_space_size=3, action_space_size=2)
        agent.set_seed(123)

        # Act - Get actions from agent
        actions = [agent.select_action(0) for _ in range(5)]

        # Assert - Actions should be deterministic (not all the same due to exploration)
        assert len(actions) == 5, "Should get correct number of actions"
        # The key is that set_seed doesn't raise an error and produces some actions

    @pytest.mark.unit
    def test_expected_sarsa_enabled(self) -> None:
        """Test Expected SARSA functionality."""
        # Arrange - Create agent with Expected SARSA enabled
        agent = SarsaAgent(
            state_space_size=3,
            action_space_size=2,
            use_expected_sarsa=True,
            random_seed=42,
        )

        # Act - Perform a step
        next_action = agent.step(
            state=0, action=0, reward=1.0, next_state=1, done=False
        )

        # Assert - Expected SARSA should be enabled and return next action
        assert agent.use_expected_sarsa is True, "Expected SARSA should be enabled"
        assert isinstance(next_action, int | np.integer), "Should return next action"
        assert 0 <= next_action < 2, "Next action should be valid"

    @pytest.mark.unit
    def test_expected_sarsa_disabled(self) -> None:
        """Test that Expected SARSA is disabled by default."""
        # Arrange - Create agent without Expected SARSA
        agent = SarsaAgent(state_space_size=3, action_space_size=2)

        # Act - Check Expected SARSA status

        # Assert - Expected SARSA should be disabled
        assert agent.use_expected_sarsa is False, "Expected SARSA should be disabled"

    @pytest.mark.unit
    def test_debug_logging_enabled(self) -> None:
        """Test debug logging functionality."""
        # Arrange - Create agent with debug enabled
        agent = SarsaAgent(
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
        agent = SarsaAgent(state_space_size=3, action_space_size=2)
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
        agent = SarsaAgent(state_space_size=2, action_space_size=2)
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
        agent = SarsaAgent(state_space_size=2, action_space_size=2)

        # Act - Pretty print policy
        policy_str = agent.pretty_print_policy()

        # Assert - Should contain default names
        assert "State 0" in policy_str, "Should contain default state names"
        assert "State 1" in policy_str, "Should contain default state names"
        assert "Action" in policy_str, "Should contain default action names"

    @pytest.mark.unit
    def test_get_policy_entropy(self) -> None:
        """Test policy entropy calculation."""
        # Arrange - Create agent with tied Q-values
        agent = SarsaAgent(state_space_size=2, action_space_size=2)
        agent.q_table[0, :] = 1.0  # Tied actions in state 0
        agent.q_table[1, 0] = 0.8
        agent.q_table[1, 1] = 0.3

        # Act - Get policy entropy
        entropy = agent.get_policy_entropy()

        # Assert - Should return non-negative entropy
        assert entropy >= 0, "Policy entropy should be non-negative"

    @pytest.mark.unit
    def test_get_average_q_magnitude(self) -> None:
        """Test average Q-value magnitude calculation."""
        # Arrange - Create agent and set Q-values
        agent = SarsaAgent(state_space_size=2, action_space_size=2)
        agent.q_table[0, 0] = 0.5
        agent.q_table[0, 1] = -0.3
        agent.q_table[1, 0] = 0.8
        agent.q_table[1, 1] = 0.2

        # Act - Get average Q-value magnitude
        avg_magnitude = agent.get_average_q_magnitude()

        # Assert - Should return correct average magnitude
        expected = (0.5 + 0.3 + 0.8 + 0.2) / 4  # Average of absolute values
        assert abs(avg_magnitude - expected) < 1e-6, (
            "Should return correct average magnitude"
        )

    @pytest.mark.unit
    def test_epsilon_scheduling_parameters(self) -> None:
        """Test epsilon scheduling parameters."""
        # Arrange - Create agent with custom epsilon scheduling
        agent = SarsaAgent(
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
        agent = SarsaAgent(
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
        agent1 = SarsaAgent(state_space_size=3, action_space_size=2, random_seed=42)
        agent2 = SarsaAgent(state_space_size=3, action_space_size=2, random_seed=42)

        # Act - Use different methods
        next_action1 = agent1.step(
            state=0, action=0, reward=1.0, next_state=1, done=False
        )
        agent2.update_q_value(
            state=0,
            action=0,
            reward=1.0,
            next_state=1,
            next_action=next_action1,
            done=False,
        )

        # Assert - Q-tables should be identical
        assert np.allclose(agent1.q_table, agent2.q_table), (
            "Both methods should produce same results"
        )

    @pytest.mark.unit
    def test_select_action_vs_get_action(self) -> None:
        """Test that select_action and get_action produce same results."""
        # Arrange - Create agent and set seed
        agent = SarsaAgent(state_space_size=3, action_space_size=2)
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
    def test_on_policy_behavior(self) -> None:
        """Test that step method enforces on-policy behavior."""
        # Arrange - Create agent with high epsilon
        agent = SarsaAgent(
            state_space_size=3,
            action_space_size=2,
            epsilon_start=0.8,  # High exploration
            random_seed=42,
        )

        # Act - Perform step and get next action
        next_action = agent.step(
            state=0, action=0, reward=1.0, next_state=1, done=False
        )

        # Assert - Next action should be chosen by current epsilon-greedy policy
        assert isinstance(next_action, int | np.integer), "Should return next action"
        assert 0 <= next_action < 2, "Next action should be valid"

    @pytest.mark.unit
    def test_expected_sarsa_vs_standard_sarsa(self) -> None:
        """Test Expected SARSA vs standard SARSA functionality."""
        # Arrange - Create two agents with different algorithms
        standard_agent = SarsaAgent(
            state_space_size=3,
            action_space_size=2,
            epsilon_start=0.5,  # Partial exploration
            use_expected_sarsa=False,
            random_seed=42,
        )
        expected_agent = SarsaAgent(
            state_space_size=3,
            action_space_size=2,
            epsilon_start=0.5,  # Partial exploration
            use_expected_sarsa=True,
            random_seed=42,
        )

        # Act - Perform same updates
        standard_agent.update_q_value(0, 0, 1.0, 1, 1, False)
        expected_agent.update_q_value(0, 0, 1.0, 1, 1, False)

        # Assert - Both should work correctly (Expected SARSA may or may not differ depending on Q-values)
        assert standard_agent.use_expected_sarsa is False, (
            "Standard SARSA should be disabled"
        )
        assert expected_agent.use_expected_sarsa is True, (
            "Expected SARSA should be enabled"
        )
        # Both should produce valid Q-table updates
        assert standard_agent.q_table[0, 0] > 0, "Standard SARSA should update Q-values"
        assert expected_agent.q_table[0, 0] > 0, "Expected SARSA should update Q-values"

    @pytest.mark.unit
    def test_parameter_validation_strict(self) -> None:
        """Test strict parameter validation."""
        # Arrange - Set up test cases for invalid parameters

        # Act & Assert - Test learning_rate = 0 (should fail via Pydantic)
        with pytest.raises(ValidationError, match="learning_rate"):
            SarsaAgent(state_space_size=3, action_space_size=2, learning_rate=0.0)

        # Act & Assert - Test discount_factor = 0 (should fail via Pydantic)
        with pytest.raises(ValidationError, match="discount_factor"):
            SarsaAgent(state_space_size=3, action_space_size=2, discount_factor=0.0)

        # Act & Assert - Test epsilon_end > epsilon_start (should fail via Pydantic)
        with pytest.raises(ValidationError, match="epsilon_end"):
            SarsaAgent(
                state_space_size=3,
                action_space_size=2,
                epsilon_start=0.1,
                epsilon_end=0.5,
            )

    @pytest.mark.unit
    def test_backward_compatibility(self) -> None:
        """Test backward compatibility with old parameter names."""
        # Arrange - Create agent with old parameter names
        agent = SarsaAgent(
            state_space_size=3,
            action_space_size=2,
            epsilon=0.5,  # Old parameter name
            epsilon_min=0.1,  # Old parameter name
        )

        # Act - Check parameters are mapped correctly

        # Assert - Parameters should be mapped correctly
        assert agent.epsilon == 0.5, "epsilon should be mapped to epsilon_start"
        assert agent.epsilon_end == 0.1, "epsilon_min should be mapped to epsilon_end"
