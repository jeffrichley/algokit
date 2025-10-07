"""Tests for Options Framework with advanced features.

Tests cover:
1. Dynamic Q-network resizing when adding new options
2. Learnable termination functions β(s)
3. Option policy exploration (softmax/epsilon-greedy)
4. Eligibility traces and n-step updates
5. Configurable termination for primitive actions
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from algokit.algorithms.hierarchical_rl.options_framework import (
    IntraOptionQLearning,
    Option,
    OptionsAgent,
    TerminationNetwork,
)


@pytest.mark.unit
class TestOption:
    """Test Option dataclass and functionality."""

    def test_option_creation_with_defaults(self) -> None:
        """Test that Option can be created with default values."""
        # Arrange - Set up option parameters with defaults
        name = "test"

        # Act - Create option with default values
        option = Option(
            name=name,
            initiation_set=lambda s: True,
            policy=lambda s: 0,
        )

        # Assert - Verify default values are set correctly
        assert option.name == "test"
        assert option.termination is None
        assert option.is_primitive is False
        assert option.temperature == 0.0
        assert option.epsilon == 0.0

    def test_option_creation_with_exploration(self) -> None:
        """Test that Option can be created with exploration parameters."""
        # Arrange - Set up option parameters with exploration
        name = "explore"
        temperature = 0.5
        epsilon = 0.2

        # Act - Create option with exploration parameters
        option = Option(
            name=name,
            initiation_set=lambda s: True,
            policy=lambda s: 1,
            temperature=temperature,
            epsilon=epsilon,
        )

        # Assert - Verify exploration parameters are set
        assert option.temperature == 0.5
        assert option.epsilon == 0.2

    def test_option_primitive_flag(self) -> None:
        """Test that primitive flag is correctly set."""
        # Arrange - Set up primitive option parameters
        name = "primitive"

        # Act - Create primitive option
        primitive = Option(
            name=name,
            initiation_set=lambda s: True,
            policy=lambda s: 0,
            is_primitive=True,
        )

        # Assert - Verify primitive flag is True
        assert primitive.is_primitive is True


@pytest.mark.unit
class TestTerminationNetwork:
    """Test learnable termination network β(s)."""

    def test_termination_network_initialization(self) -> None:
        """Test that TerminationNetwork initializes correctly."""
        # Arrange - Set up network parameters
        state_size = 4
        n_options = 3

        # Act - Create termination network
        net = TerminationNetwork(state_size=state_size, n_options=n_options)

        # Assert - Verify network structure has correct layers
        assert len(net.network) == 6  # 3 linear + 2 relu + 1 sigmoid

    def test_termination_network_output_shape(self) -> None:
        """Test that TerminationNetwork outputs correct shape."""
        # Arrange - Create network and random state
        net = TerminationNetwork(state_size=4, n_options=3)
        state = torch.randn(4)

        # Act - Forward pass through network
        output = net(state)

        # Assert - Verify output shape matches number of options
        assert output.shape == (3,)

    def test_termination_network_output_range(self) -> None:
        """Test that TerminationNetwork outputs probabilities in [0, 1]."""
        # Arrange - Create network and random state
        net = TerminationNetwork(state_size=4, n_options=3)
        state = torch.randn(4)

        # Act - Forward pass through network
        output = net(state)

        # Assert - Verify all probabilities are in valid range
        assert torch.all(output >= 0.0)
        assert torch.all(output <= 1.0)


@pytest.mark.unit
class TestIntraOptionQLearning:
    """Test intra-option Q-learning with advanced features."""

    def test_initialization_without_traces(self) -> None:
        """Test initialization without eligibility traces."""
        # Arrange - Set up Q-learner parameters without traces
        state_size = 4
        n_options = 2

        # Act - Create Q-learner without traces
        q_learner = IntraOptionQLearning(
            state_size=state_size, n_options=n_options, use_traces=False
        )

        # Assert - Verify traces are disabled
        assert q_learner.n_options == 2
        assert q_learner.use_traces is False
        assert len(q_learner.traces) == 0

    def test_initialization_with_traces(self) -> None:
        """Test initialization with eligibility traces."""
        # Arrange - Set up Q-learner parameters with traces
        state_size = 4
        n_options = 2

        # Act - Create Q-learner with traces enabled
        q_learner = IntraOptionQLearning(
            state_size=state_size, n_options=n_options, use_traces=True
        )

        # Assert - Verify traces are initialized for each option
        assert q_learner.use_traces is True
        assert len(q_learner.traces) == 2  # One per option

    def test_dynamic_network_resizing(self) -> None:
        """Test that Q-network can be dynamically resized."""
        # Arrange - Create Q-learner and store initial option count
        q_learner = IntraOptionQLearning(state_size=4, n_options=2)
        initial_options = q_learner.n_options

        # Act - Resize network to accommodate more options
        q_learner.resize_network(new_n_options=5)

        # Assert - Verify network size increased correctly
        assert q_learner.n_options == 5
        assert q_learner.q_network[-1].out_features == 5
        assert q_learner.n_options > initial_options

    def test_resizing_preserves_weights(self) -> None:
        """Test that resizing preserves learned weights for existing options."""
        # Arrange - Create Q-learner and capture initial Q-values
        q_learner = IntraOptionQLearning(state_size=4, n_options=2)
        state = torch.randn(4)
        with torch.no_grad():
            q_before = q_learner.get_option_values(state).clone()

        # Act - Resize network to add new options
        q_learner.resize_network(new_n_options=4)

        # Assert - Verify existing options retain similar Q-values
        with torch.no_grad():
            q_after = q_learner.get_option_values(state)
            # First 2 options should have similar values
            assert torch.allclose(q_before, q_after[:2], atol=1e-5)

    def test_n_step_buffer_initialization(self) -> None:
        """Test that n-step buffers are initialized correctly."""
        # Arrange - Set up Q-learner parameters with n-step
        state_size = 4
        n_options = 2
        n_step = 5

        # Act - Create Q-learner with n-step enabled
        q_learner = IntraOptionQLearning(
            state_size=state_size, n_options=n_options, n_step=n_step
        )

        # Assert - Verify n-step parameter is set
        assert q_learner.n_step == 5

    def test_update_with_n_step(self) -> None:
        """Test Q-value update with n-step returns."""
        # Arrange - Create Q-learner with n-step and random states
        q_learner = IntraOptionQLearning(
            state_size=4, n_options=2, n_step=3, use_traces=False
        )
        state = torch.randn(4)
        next_state = torch.randn(4)

        # Act - Perform Q-value update with n-step
        loss = q_learner.update(
            state=state,
            option=0,
            reward=1.0,
            next_state=next_state,
            done=False,
            next_option=0,
            use_n_step=True,
        )

        # Assert - Verify loss is valid and non-negative
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_update_with_eligibility_traces(self) -> None:
        """Test Q-value update using eligibility traces."""
        # Arrange - Create Q-learner with traces and random states
        q_learner = IntraOptionQLearning(
            state_size=4, n_options=2, use_traces=True, lambda_trace=0.9
        )
        state = torch.randn(4)
        next_state = torch.randn(4)

        # Act - Perform Q-value update using traces
        loss = q_learner.update(
            state=state,
            option=0,
            reward=1.0,
            next_state=next_state,
            done=False,
            next_option=0,
        )

        # Assert - Verify loss is valid and non-negative
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_reset_traces(self) -> None:
        """Test that eligibility traces can be reset."""
        # Arrange - Create Q-learner with traces and modify them
        q_learner = IntraOptionQLearning(state_size=4, n_options=2, use_traces=True)
        for name in q_learner.traces[0]:
            q_learner.traces[0][name] += 1.0

        # Act - Reset all eligibility traces
        q_learner.reset_traces()

        # Assert - Verify all traces are zero
        for name in q_learner.traces[0]:
            assert torch.all(q_learner.traces[0][name] == 0.0)


@pytest.mark.unit
class TestOptionsAgent:
    """Test OptionsAgent with advanced features."""

    def test_agent_initialization_with_primitives(self) -> None:
        """Test agent initialization with primitive options."""
        # Arrange - Set up test data
        # Act - Perform action
        agent = OptionsAgent(
            state_size=4,
            action_size=2,
            learning_rate=0.001,
            gamma=0.99,
        )

        # Assert - Verify expected behavior
        assert agent.n_options == 2  # One per action
        assert all(opt.is_primitive for opt in agent.options)

    def test_agent_with_learnable_termination(self) -> None:
        """Test agent initialization with learnable termination."""
        # Arrange - Set up test data
        # Act - Perform action
        agent = OptionsAgent(
            state_size=4,
            action_size=2,
            learn_termination=True,
        )

        # Assert - Verify expected behavior
        assert agent.termination_network is not None
        assert agent.termination_optimizer is not None
        assert agent.learn_termination_enabled is True

    def test_agent_without_learnable_termination(self) -> None:
        """Test agent initialization without learnable termination."""
        # Arrange - Set up test data
        # Act - Perform action
        agent = OptionsAgent(
            state_size=4,
            action_size=2,
            learn_termination=False,
        )

        # Assert - Verify expected behavior
        assert agent.termination_network is None
        assert agent.termination_optimizer is None

    def test_configurable_primitive_termination(self) -> None:
        """Test configurable termination probability for primitives."""
        # Arrange - Set up test data
        # Act - Perform action
        agent = OptionsAgent(
            state_size=4,
            action_size=2,
            learn_termination=False,
            primitive_termination_prob=0.5,
        )

        # Assert - Verify expected behavior
        assert agent.primitive_termination_prob == 0.5

    def test_add_option_resizes_network(self) -> None:
        """Test that adding option resizes Q-network."""
        # Arrange - Set up test data
        agent = OptionsAgent(state_size=4, action_size=2)
        initial_n_options = agent.n_options

        new_option = Option(
            name="new_option",
            initiation_set=lambda s: True,
            policy=lambda s: 0,
        )

        # Act - Perform action under test
        agent.add_option(new_option)

        # Assert - Verify expected behavior
        assert agent.n_options == initial_n_options + 1
        assert agent.q_learner.n_options == agent.n_options

    def test_add_option_resizes_termination_network(self) -> None:
        """Test that adding option resizes termination network."""
        # Arrange - Set up test data
        agent = OptionsAgent(
            state_size=4,
            action_size=2,
            learn_termination=True,
        )
        initial_n_options = agent.n_options

        new_option = Option(
            name="new_option",
            initiation_set=lambda s: True,
            policy=lambda s: 0,
        )

        # Act - Perform action under test
        agent.add_option(new_option)

        # Assert - Verify expected behavior
        assert agent.termination_network is not None
        # Check output layer size
        output_layer = agent.termination_network.network[-2]  # Before sigmoid
        assert output_layer.out_features == initial_n_options + 1

    def test_option_selection_epsilon_greedy(self) -> None:
        """Test epsilon-greedy option selection."""
        # Arrange - Set up test data
        agent = OptionsAgent(
            state_size=4,
            action_size=2,
            epsilon=1.0,  # Always explore
            seed=42,
        )
        state = torch.randn(4)

        # Act - Perform action under test
        option = agent.select_option(state)

        # Assert - Verify expected behavior
        assert 0 <= option < agent.n_options

    def test_option_selection_greedy(self) -> None:
        """Test greedy option selection."""
        # Arrange - Set up test data
        agent = OptionsAgent(
            state_size=4,
            action_size=2,
            epsilon=0.0,  # Always exploit
            seed=42,
        )
        state = torch.randn(4)

        # Act - Perform action under test
        option = agent.select_option(state)

        # Assert - Verify expected behavior
        assert 0 <= option < agent.n_options

    def test_get_action_with_epsilon_exploration(self) -> None:
        """Test action selection with epsilon exploration in option."""
        # Arrange - Set up test data
        option = Option(
            name="explore",
            initiation_set=lambda s: True,
            policy=lambda s: 0,
            epsilon=1.0,  # Always explore
        )
        agent = OptionsAgent(
            state_size=4,
            action_size=4,
            options=[option],
            seed=42,
        )
        state = np.array([0.1, 0.2, 0.3, 0.4])

        # Act - Perform action under test
        action = agent.get_action(state, 0)

        # Assert - Verify expected behavior
        assert 0 <= action < agent.action_size

    def test_get_action_with_softmax_exploration(self) -> None:
        """Test action selection with softmax exploration in option."""
        # Arrange - Set up test data
        option = Option(
            name="softmax",
            initiation_set=lambda s: True,
            policy=lambda s: 0,
            temperature=0.5,
        )
        agent = OptionsAgent(
            state_size=4,
            action_size=4,
            options=[option],
            seed=42,
        )
        state = np.array([0.1, 0.2, 0.3, 0.4])

        # Act - Perform action under test
        action = agent.get_action(state, 0)

        # Assert - Verify expected behavior
        assert 0 <= action < agent.action_size

    def test_should_terminate_with_learnable(self) -> None:
        """Test termination decision with learnable termination."""
        # Arrange - Set up test data
        agent = OptionsAgent(
            state_size=4,
            action_size=2,
            learn_termination=True,
            seed=42,
        )
        state = np.array([0.1, 0.2, 0.3, 0.4])

        # Act - Perform action under test
        terminates = agent.should_terminate(state, 0)

        # Assert - Verify expected behavior
        assert isinstance(terminates, bool)

    def test_should_terminate_with_fixed(self) -> None:
        """Test termination decision with fixed termination."""
        # Arrange - Set up test data
        agent = OptionsAgent(
            state_size=4,
            action_size=2,
            learn_termination=False,
            primitive_termination_prob=1.0,
            seed=42,
        )
        state = np.array([0.1, 0.2, 0.3, 0.4])

        # Act - Perform action under test
        terminates = agent.should_terminate(state, 0)

        # Assert - Verify expected behavior
        assert isinstance(terminates, bool)

    def test_learn_termination_function(self) -> None:
        """Test learning of termination function with entropy regularization."""
        # Arrange - Set up test data
        agent = OptionsAgent(
            state_size=4,
            action_size=2,
            learn_termination=True,
            termination_entropy_weight=0.01,
        )
        state = torch.randn(4)

        # Act - Perform action under test
        loss, entropy = agent.learn_termination(
            state=state,
            option=0,
            should_terminate=True,
            advantage=-0.5,  # Negative advantage
        )

        # Assert - Verify loss and entropy are valid floats
        assert isinstance(loss, float)
        assert not np.isnan(loss)
        assert not np.isinf(loss)
        assert isinstance(entropy, float)
        assert entropy >= 0.0  # Entropy is always non-negative
        assert not np.isnan(entropy)
        assert not np.isinf(entropy)

    def test_learn_termination_with_option_critic_gradient(self) -> None:
        """Test termination learning with option-critic style gradient."""
        # Arrange - Set up agent with option-critic termination
        agent = OptionsAgent(
            state_size=4,
            action_size=2,
            learn_termination=True,
            use_option_critic_termination=True,
        )
        state = torch.randn(4)

        # Act - Learn termination with positive advantage
        loss_oc, entropy_oc = agent.learn_termination(
            state=state,
            option=0,
            should_terminate=True,
            advantage=0.5,  # Positive advantage
        )

        # Assert - Verify gradient sign reversal works
        assert isinstance(loss_oc, float)
        assert isinstance(entropy_oc, float)
        assert entropy_oc >= 0.0

    def test_termination_entropy_prevents_collapse(self) -> None:
        """Test that entropy regularization prevents β(s) from collapsing to extremes."""
        # Arrange - Set up agent with high entropy weight
        agent = OptionsAgent(
            state_size=4,
            action_size=2,
            learn_termination=True,
            termination_entropy_weight=0.1,  # High entropy weight
        )
        state = torch.randn(4)

        # Act - Perform multiple updates
        entropies = []
        for _ in range(10):
            _, entropy = agent.learn_termination(
                state=state,
                option=0,
                should_terminate=True,
                advantage=-1.0,  # Strong signal to terminate
            )
            entropies.append(entropy)

        # Assert - Entropy should remain non-zero (not collapsed)
        assert all(e > 0.0 for e in entropies), "Entropy should remain positive"
        assert np.mean(entropies) > 0.1, "Average entropy should be significant"

    def test_learn_updates_q_values(self) -> None:
        """Test that learn updates Q-values."""
        # Arrange - Set up test data
        agent = OptionsAgent(state_size=4, action_size=2)
        state = torch.randn(4)
        next_state = torch.randn(4)

        # Act - Perform action under test
        metrics = agent.learn(
            state=state,
            option=0,
            reward=1.0,
            next_state=next_state,
            done=False,
            next_option=0,
        )

        # Assert - Verify expected behavior
        assert "loss" in metrics
        assert "epsilon" in metrics
        assert isinstance(metrics["loss"], float)

    def test_learn_with_termination_learning(self) -> None:
        """Test learning with termination function updates."""
        # Arrange - Set up test data
        agent = OptionsAgent(
            state_size=4,
            action_size=2,
            learn_termination=True,
        )
        state = torch.randn(4)
        next_state = torch.randn(4)

        # Act - Perform action under test
        metrics = agent.learn(
            state=state,
            option=0,
            reward=1.0,
            next_state=next_state,
            done=False,
            next_option=1,
            terminated=True,
        )

        # Assert - Verify expected behavior
        assert "loss" in metrics
        assert "termination_loss" in metrics
        assert "epsilon" in metrics

    def test_train_episode_mock_env(self) -> None:
        """Test training episode with mock environment."""
        # Arrange - Set up test data
        agent = OptionsAgent(state_size=4, action_size=2, seed=42)

        # Create mock environment
        env = MagicMock()
        env.reset.return_value = (np.array([0.0, 0.0, 0.0, 0.0]), {})
        env.step.return_value = (
            np.array([0.1, 0.1, 0.1, 0.1]),  # next_state
            1.0,  # reward
            True,  # done
            False,  # truncated
            {},  # info
        )

        # Act - Perform action under test
        metrics = agent.train_episode(env, max_steps=10)

        # Assert - Verify expected behavior
        assert "reward" in metrics
        assert "steps" in metrics
        assert "avg_loss" in metrics
        assert "epsilon" in metrics
        assert "option_changes" in metrics

    def test_get_statistics(self) -> None:
        """Test statistics gathering."""
        # Arrange - Set up test data
        agent = OptionsAgent(state_size=4, action_size=2)
        agent.episode_rewards = [1.0, 2.0, 3.0]

        # Act - Perform action under test
        stats = agent.get_statistics()

        # Assert - Verify expected behavior
        assert "total_episodes" in stats
        assert "avg_reward" in stats
        assert "n_options" in stats
        assert "epsilon" in stats
        assert stats["total_episodes"] == 3

    def test_statistics_with_termination_losses(self) -> None:
        """Test statistics include termination losses when available."""
        # Arrange - Set up test data
        agent = OptionsAgent(
            state_size=4,
            action_size=2,
            learn_termination=True,
        )
        agent.termination_losses = [0.1, 0.2, 0.3]

        # Act - Perform action under test
        stats = agent.get_statistics()

        # Assert - Verify expected behavior
        assert "avg_termination_loss" in stats

    def test_statistics_include_option_success_rates(self) -> None:
        """Test that statistics include per-option success rates."""
        # Arrange - Set up agent and track option performance
        agent = OptionsAgent(state_size=4, action_size=2)
        agent.option_frequencies["primitive_0"] = 10
        agent.option_successes["primitive_0"] = 7
        agent.option_failures["primitive_0"] = 3

        # Act - Get statistics
        stats = agent.get_statistics()

        # Assert - Verify success rate calculation
        assert "option_success_rates" in stats
        assert "primitive_0" in stats["option_success_rates"]
        assert stats["option_success_rates"]["primitive_0"] == 0.7  # 7/10

    def test_statistics_include_option_rewards(self) -> None:
        """Test that statistics include average rewards per option."""
        # Arrange - Set up agent with option reward data
        agent = OptionsAgent(state_size=4, action_size=2)
        agent.option_total_rewards["primitive_0"] = [1.0, 2.0, 3.0]
        agent.option_total_rewards["primitive_1"] = [0.5, 1.5]

        # Act - Get statistics
        stats = agent.get_statistics()

        # Assert - Verify average rewards are calculated
        assert "avg_option_rewards" in stats
        assert "primitive_0" in stats["avg_option_rewards"]
        assert stats["avg_option_rewards"]["primitive_0"] == 2.0
        assert stats["avg_option_rewards"]["primitive_1"] == 1.0

    def test_statistics_include_termination_entropy(self) -> None:
        """Test that statistics include termination entropy metrics."""
        # Arrange - Set up agent with termination entropy data
        agent = OptionsAgent(state_size=4, action_size=2, learn_termination=True)
        agent.termination_entropy = [0.5, 0.6, 0.7, 0.4, 0.8]

        # Act - Get statistics
        stats = agent.get_statistics()

        # Assert - Verify entropy statistics are included
        assert "avg_termination_entropy" in stats
        assert "min_termination_entropy" in stats
        assert "max_termination_entropy" in stats
        assert stats["avg_termination_entropy"] == pytest.approx(0.6)
        assert stats["min_termination_entropy"] == 0.4
        assert stats["max_termination_entropy"] == 0.8

    def test_train_episode_tracks_option_performance(self) -> None:
        """Test that train_episode tracks per-option success rates and rewards."""
        # Arrange - Set up agent and mock environment
        agent = OptionsAgent(
            state_size=4, action_size=2, epsilon=0.0, learn_termination=False
        )
        env = MagicMock()
        env.reset.return_value = (np.zeros(4), {})

        # Simulate episode with mixed rewards
        rewards = [1.0, -0.5, 1.0, 2.0]  # Total: 3.5
        states = [np.ones(4) * i for i in range(len(rewards) + 1)]

        call_count = [0]

        def step_side_effect(
            _action: int,
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, str]]:
            idx = call_count[0]
            call_count[0] += 1
            done = idx == len(rewards) - 1
            return states[idx + 1], rewards[idx], done, False, {}

        env.step.side_effect = step_side_effect

        # Act - Train for one episode
        agent.train_episode(env, max_steps=10)

        # Assert - Verify option performance is tracked
        stats = agent.get_statistics()
        assert "option_success_rates" in stats
        assert "avg_option_rewards" in stats
        assert len(agent.option_total_rewards) > 0

    def test_train_episode_returns_entropy_metrics(self) -> None:
        """Test that train_episode returns termination entropy in metrics."""
        # Arrange - Set up agent with learnable termination
        agent = OptionsAgent(
            state_size=4,
            action_size=2,
            learn_termination=True,
            termination_entropy_weight=0.01,
        )
        env = MagicMock()
        env.reset.return_value = (np.zeros(4), {})
        env.step.return_value = (
            np.ones(4) * 0.1,
            1.0,
            True,  # Episode done after one step
            False,
            {},
        )

        # Act - Train for one episode
        metrics = agent.train_episode(env, max_steps=10)

        # Assert - Verify entropy metrics are included
        assert "avg_term_entropy" in metrics
        assert isinstance(metrics["avg_term_entropy"], float)


@pytest.mark.integration
class TestOptionsFrameworkIntegration:
    """Integration tests for Options Framework."""

    def test_full_workflow_with_all_features(self) -> None:
        """Test complete workflow with all advanced features enabled."""
        # Arrange - Set up test data
        agent = OptionsAgent(
            state_size=4,
            action_size=2,
            learning_rate=0.001,
            termination_lr=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.99,
            lambda_trace=0.9,
            n_step=3,
            use_traces=True,
            learn_termination=True,
            primitive_termination_prob=0.8,
            seed=42,
        )

        # Create mock environment
        env = MagicMock()
        env.reset.return_value = (np.zeros(4), {})
        env.step.return_value = (
            np.ones(4) * 0.1,
            1.0,
            False,
            False,
            {},
        )

        # Act - train for a few steps
        metrics = agent.train_episode(env, max_steps=5)

        # Assert - Verify expected behavior
        assert metrics is not None
        assert "reward" in metrics
        assert "avg_loss" in metrics

    def test_dynamic_option_addition_workflow(self) -> None:
        """Test workflow of dynamically adding options."""
        # Arrange - Set up test data
        agent = OptionsAgent(
            state_size=4,
            action_size=2,
            learn_termination=True,
            seed=42,
        )
        initial_options = agent.n_options

        # Act - Add multiple options
        for i in range(3):
            new_option = Option(
                name=f"option_{i}",
                initiation_set=lambda s: True,
                policy=lambda s: 0,
                temperature=0.2,
            )
            agent.add_option(new_option)

        # Assert - Verify expected behavior
        assert agent.n_options == initial_options + 3
        assert agent.q_learner.n_options == agent.n_options
        assert agent.termination_network is not None


@pytest.mark.unit
class TestIntraOptionQLearningConfig:
    """Test Pydantic configuration validation for IntraOptionQLearning."""

    def test_config_validates_negative_state_size(self) -> None:
        """Test that IntraOptionQLearningConfig rejects negative state_size."""
        # Arrange - Import validation error and config classes
        from pydantic import ValidationError

        from algokit.algorithms.hierarchical_rl.options_framework import (
            IntraOptionQLearningConfig,
        )

        # Act & Assert - Verify validation error is raised for invalid parameter
        with pytest.raises(ValidationError, match="state_size"):
            IntraOptionQLearningConfig(state_size=-1, n_options=4)

    def test_config_validates_zero_state_size(self) -> None:
        """Test that IntraOptionQLearningConfig rejects zero state_size."""
        # Arrange - Import validation error and config classes
        from pydantic import ValidationError

        from algokit.algorithms.hierarchical_rl.options_framework import (
            IntraOptionQLearningConfig,
        )

        # Act & Assert - Verify validation error is raised for invalid parameter
        with pytest.raises(ValidationError, match="state_size"):
            IntraOptionQLearningConfig(state_size=0, n_options=4)

    def test_config_validates_negative_n_options(self) -> None:
        """Test that IntraOptionQLearningConfig rejects negative n_options."""
        # Arrange - Import validation error and config classes
        from pydantic import ValidationError

        from algokit.algorithms.hierarchical_rl.options_framework import (
            IntraOptionQLearningConfig,
        )

        # Act & Assert - Verify validation error is raised for invalid parameter
        with pytest.raises(ValidationError, match="n_options"):
            IntraOptionQLearningConfig(state_size=4, n_options=-1)

    def test_config_validates_learning_rate_too_high(self) -> None:
        """Test that IntraOptionQLearningConfig rejects learning_rate > 1."""
        # Arrange - Import validation error and config classes
        from pydantic import ValidationError

        from algokit.algorithms.hierarchical_rl.options_framework import (
            IntraOptionQLearningConfig,
        )

        # Act & Assert - Verify validation error is raised for invalid parameter
        with pytest.raises(ValidationError, match="learning_rate"):
            IntraOptionQLearningConfig(state_size=4, n_options=2, learning_rate=1.5)

    def test_config_validates_gamma_out_of_range(self) -> None:
        """Test that IntraOptionQLearningConfig rejects gamma outside [0, 1]."""
        # Arrange - Import validation error and config classes
        from pydantic import ValidationError

        from algokit.algorithms.hierarchical_rl.options_framework import (
            IntraOptionQLearningConfig,
        )

        # Act & Assert - Verify validation error is raised for invalid parameter
        with pytest.raises(ValidationError, match="gamma"):
            IntraOptionQLearningConfig(state_size=4, n_options=2, gamma=1.5)

    def test_config_accepts_valid_parameters(self) -> None:
        """Test that IntraOptionQLearningConfig accepts valid parameters."""
        # Arrange - Import config class for testing
        from algokit.algorithms.hierarchical_rl.options_framework import (
            IntraOptionQLearningConfig,
        )

        # Act - Create config with test parameters
        config = IntraOptionQLearningConfig(
            state_size=4,
            n_options=2,
            learning_rate=0.001,
            gamma=0.99,
        )

        # Assert - Verify parameters are correctly set
        assert config.state_size == 4
        assert config.n_options == 2
        assert config.learning_rate == 0.001
        assert config.gamma == 0.99

    def test_backwards_compatible_kwargs(self) -> None:
        """Test that IntraOptionQLearning accepts kwargs for backwards compatibility."""
        # Arrange - No setup required for backwards compatibility test

        # Act - Create learner with kwargs
        learner = IntraOptionQLearning(state_size=4, n_options=2)

        # Assert - Verify parameters are correctly set
        assert learner.state_size == 4
        assert learner.n_options == 2

    def test_config_object_initialization(self) -> None:
        """Test that IntraOptionQLearning accepts config object."""
        # Arrange - Import config class for testing
        from algokit.algorithms.hierarchical_rl.options_framework import (
            IntraOptionQLearningConfig,
        )

        config = IntraOptionQLearningConfig(state_size=4, n_options=2)

        # Act - Create learner with kwargs
        learner = IntraOptionQLearning(config=config)

        # Assert - Verify parameters are correctly set
        assert learner.config == config
        assert learner.state_size == 4
        assert learner.n_options == 2


@pytest.mark.unit
class TestOptionsAgentConfig:
    """Test Pydantic configuration validation for OptionsAgent."""

    def test_config_validates_negative_state_size(self) -> None:
        """Test that OptionsAgentConfig rejects negative state_size."""
        # Arrange - Import validation error and config classes
        from pydantic import ValidationError

        from algokit.algorithms.hierarchical_rl.options_framework import (
            OptionsAgentConfig,
        )

        # Act & Assert - Verify validation error is raised for invalid parameter
        with pytest.raises(ValidationError, match="state_size"):
            OptionsAgentConfig(state_size=-1, action_size=2)

    def test_config_validates_negative_action_size(self) -> None:
        """Test that OptionsAgentConfig rejects negative action_size."""
        # Arrange - Import validation error and config classes
        from pydantic import ValidationError

        from algokit.algorithms.hierarchical_rl.options_framework import (
            OptionsAgentConfig,
        )

        # Act & Assert - Verify validation error is raised for invalid parameter
        with pytest.raises(ValidationError, match="action_size"):
            OptionsAgentConfig(state_size=4, action_size=-1)

    def test_config_validates_epsilon_min_greater_than_epsilon(self) -> None:
        """Test that OptionsAgentConfig rejects epsilon_min > epsilon."""
        # Arrange - Import validation error and config classes
        from pydantic import ValidationError

        from algokit.algorithms.hierarchical_rl.options_framework import (
            OptionsAgentConfig,
        )

        # Act & Assert - Verify validation error is raised for invalid parameter
        with pytest.raises(ValidationError, match="epsilon_min.*epsilon"):
            OptionsAgentConfig(
                state_size=4,
                action_size=2,
                epsilon=0.1,
                epsilon_min=0.5,
            )

    def test_config_validates_epsilon_out_of_range(self) -> None:
        """Test that OptionsAgentConfig rejects epsilon outside [0, 1]."""
        # Arrange - Import validation error and config classes
        from pydantic import ValidationError

        from algokit.algorithms.hierarchical_rl.options_framework import (
            OptionsAgentConfig,
        )

        # Act & Assert - Verify validation error is raised for invalid parameter
        with pytest.raises(ValidationError, match="epsilon"):
            OptionsAgentConfig(state_size=4, action_size=2, epsilon=1.5)

    def test_config_accepts_valid_parameters(self) -> None:
        """Test that OptionsAgentConfig accepts valid parameters."""
        # Arrange - Import config class for testing
        from algokit.algorithms.hierarchical_rl.options_framework import (
            OptionsAgentConfig,
        )

        # Act - Create config with test parameters
        config = OptionsAgentConfig(
            state_size=4,
            action_size=2,
            learning_rate=0.001,
            gamma=0.99,
        )

        # Assert - Verify parameters are correctly set
        assert config.state_size == 4
        assert config.action_size == 2
        assert config.learning_rate == 0.001
        assert config.gamma == 0.99

    def test_backwards_compatible_kwargs(self) -> None:
        """Test that OptionsAgent accepts kwargs for backwards compatibility."""
        # Arrange - No setup required for backwards compatibility test

        # Act - Create agent with kwargs
        agent = OptionsAgent(state_size=4, action_size=2)

        # Assert - Verify parameters are correctly set
        assert agent.state_size == 4
        assert agent.action_size == 2

    def test_config_object_initialization(self) -> None:
        """Test that OptionsAgent accepts config object."""
        # Arrange - Import config class for testing
        from algokit.algorithms.hierarchical_rl.options_framework import (
            OptionsAgentConfig,
        )

        config = OptionsAgentConfig(state_size=4, action_size=2)

        # Act - Create agent with config object
        agent = OptionsAgent(config=config)

        # Assert - Verify parameters are correctly set
        assert agent.config == config
        assert agent.state_size == 4
        assert agent.action_size == 2
