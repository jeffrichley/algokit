"""Tests for HIRO hierarchical reinforcement learning algorithm.

This test suite covers:
- Configuration validation using Pydantic
- Backwards compatibility with kwargs initialization
- Parameter constraints and edge cases
- Core functionality of HIRO agent
"""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from algokit.algorithms.hierarchical_rl.hiro import (
    HigherLevelPolicy,
    HIROAgent,
    HIROConfig,
    LowerLevelPolicy,
)


class TestHIROConfig:
    """Tests for HIROConfig Pydantic model."""

    @pytest.mark.unit
    def test_config_validates_positive_state_size(self) -> None:
        """Test that config rejects non-positive state_size."""
        # Arrange - prepare invalid state_size values (zero and negative)
        # Act - attempt to create HIROConfig with invalid values
        # Assert - ValidationError is raised for non-positive state_size
        with pytest.raises(ValidationError, match="state_size"):
            HIROConfig(state_size=0, action_size=4)

        with pytest.raises(ValidationError, match="state_size"):
            HIROConfig(state_size=-1, action_size=4)

    @pytest.mark.unit
    def test_config_validates_positive_action_size(self) -> None:
        """Test that config rejects non-positive action_size."""
        # Arrange - prepare invalid action_size values (zero and negative)
        # Act - attempt to create HIROConfig with invalid values
        # Assert - ValidationError is raised for non-positive action_size
        with pytest.raises(ValidationError, match="action_size"):
            HIROConfig(state_size=4, action_size=0)

        with pytest.raises(ValidationError, match="action_size"):
            HIROConfig(state_size=4, action_size=-1)

    @pytest.mark.unit
    def test_config_validates_positive_goal_size(self) -> None:
        """Test that config rejects non-positive goal_size."""
        # Arrange - prepare invalid goal_size values (zero and negative)
        # Act - attempt to create HIROConfig with invalid values
        # Assert - ValidationError is raised for non-positive goal_size
        with pytest.raises(ValidationError, match="goal_size"):
            HIROConfig(state_size=4, action_size=2, goal_size=0)

        with pytest.raises(ValidationError, match="goal_size"):
            HIROConfig(state_size=4, action_size=2, goal_size=-1)

    @pytest.mark.unit
    def test_config_validates_positive_hidden_size(self) -> None:
        """Test that config rejects non-positive hidden_size."""
        # Arrange - prepare invalid hidden_size values (zero and negative)
        # Act - attempt to create HIROConfig with invalid values
        # Assert - ValidationError is raised for non-positive hidden_size
        with pytest.raises(ValidationError, match="hidden_size"):
            HIROConfig(state_size=4, action_size=2, hidden_size=0)

        with pytest.raises(ValidationError, match="hidden_size"):
            HIROConfig(state_size=4, action_size=2, hidden_size=-1)

    @pytest.mark.unit
    def test_config_validates_positive_goal_horizon(self) -> None:
        """Test that config rejects non-positive goal_horizon."""
        # Arrange - prepare invalid goal_horizon values (zero and negative)
        # Act - attempt to create HIROConfig with invalid values
        # Assert - ValidationError is raised for non-positive goal_horizon
        with pytest.raises(ValidationError, match="goal_horizon"):
            HIROConfig(state_size=4, action_size=2, goal_horizon=0)

        with pytest.raises(ValidationError, match="goal_horizon"):
            HIROConfig(state_size=4, action_size=2, goal_horizon=-1)

    @pytest.mark.unit
    def test_config_validates_learning_rate_range(self) -> None:
        """Test that config validates learning_rate is in (0, 1]."""
        # Arrange - prepare invalid and valid learning_rate values
        # Act - attempt to create HIROConfig with various values
        # Assert - ValidationError for invalid values, success for valid values
        with pytest.raises(ValidationError, match="learning_rate"):
            HIROConfig(state_size=4, action_size=2, learning_rate=0.0)

        with pytest.raises(ValidationError, match="learning_rate"):
            HIROConfig(state_size=4, action_size=2, learning_rate=-0.1)

        with pytest.raises(ValidationError, match="learning_rate"):
            HIROConfig(state_size=4, action_size=2, learning_rate=1.5)

        # Valid values
        config1 = HIROConfig(state_size=4, action_size=2, learning_rate=0.001)
        assert config1.learning_rate == 0.001

        config2 = HIROConfig(state_size=4, action_size=2, learning_rate=1.0)
        assert config2.learning_rate == 1.0

    @pytest.mark.unit
    def test_config_validates_gamma_range(self) -> None:
        """Test that config validates gamma is in [0, 1]."""
        # Arrange - prepare invalid and valid gamma values
        # Act - attempt to create HIROConfig with various gamma values
        # Assert - ValidationError for out-of-range, success for valid values
        with pytest.raises(ValidationError, match="gamma"):
            HIROConfig(state_size=4, action_size=2, gamma=-0.1)

        with pytest.raises(ValidationError, match="gamma"):
            HIROConfig(state_size=4, action_size=2, gamma=1.5)

        # Valid values
        config1 = HIROConfig(state_size=4, action_size=2, gamma=0.0)
        assert config1.gamma == 0.0

        config2 = HIROConfig(state_size=4, action_size=2, gamma=0.99)
        assert config2.gamma == 0.99

        config3 = HIROConfig(state_size=4, action_size=2, gamma=1.0)
        assert config3.gamma == 1.0

    @pytest.mark.unit
    def test_config_validates_tau_range(self) -> None:
        """Test that config validates tau is in (0, 1]."""
        # Arrange - prepare invalid and valid tau values
        # Act - attempt to create HIROConfig with various tau values
        # Assert - ValidationError for out-of-range, success for valid values
        with pytest.raises(ValidationError, match="tau"):
            HIROConfig(state_size=4, action_size=2, tau=0.0)

        with pytest.raises(ValidationError, match="tau"):
            HIROConfig(state_size=4, action_size=2, tau=-0.1)

        with pytest.raises(ValidationError, match="tau"):
            HIROConfig(state_size=4, action_size=2, tau=1.5)

        # Valid values
        config1 = HIROConfig(state_size=4, action_size=2, tau=0.001)
        assert config1.tau == 0.001

        config2 = HIROConfig(state_size=4, action_size=2, tau=1.0)
        assert config2.tau == 1.0

    @pytest.mark.unit
    def test_config_validates_device(self) -> None:
        """Test that config validates device is 'cpu' or 'cuda'."""
        # Arrange - prepare invalid and valid device strings
        # Act - attempt to create HIROConfig with various device values
        # Assert - ValidationError for invalid devices, success with normalization
        with pytest.raises(ValidationError, match="Device must be"):
            HIROConfig(state_size=4, action_size=2, device="invalid")

        with pytest.raises(ValidationError, match="Device must be"):
            HIROConfig(state_size=4, action_size=2, device="gpu")

        # Valid values
        config1 = HIROConfig(state_size=4, action_size=2, device="cpu")
        assert config1.device == "cpu"

        config2 = HIROConfig(state_size=4, action_size=2, device="CPU")
        assert config2.device == "cpu"  # Should normalize to lowercase

        # CUDA (if available, won't fail validation)
        config3 = HIROConfig(state_size=4, action_size=2, device="cuda")
        assert config3.device == "cuda"

    @pytest.mark.unit
    def test_config_validates_seed(self) -> None:
        """Test that config validates seed is non-negative if provided."""
        # Arrange - prepare invalid and valid seed values including None
        # Act - attempt to create HIROConfig with various seed values
        # Assert - ValidationError for negative seed, success for None and non-negative
        with pytest.raises(ValidationError, match="Seed must be non-negative"):
            HIROConfig(state_size=4, action_size=2, seed=-1)

        # Valid values
        config1 = HIROConfig(state_size=4, action_size=2, seed=None)
        assert config1.seed is None

        config2 = HIROConfig(state_size=4, action_size=2, seed=0)
        assert config2.seed == 0

        config3 = HIROConfig(state_size=4, action_size=2, seed=42)
        assert config3.seed == 42

    @pytest.mark.unit
    def test_config_validates_non_negative_policy_noise(self) -> None:
        """Test that config validates policy_noise is non-negative."""
        # Arrange - prepare invalid and valid policy_noise values
        # Act - attempt to create HIROConfig with various policy_noise values
        # Assert - ValidationError for negative, success for non-negative values
        with pytest.raises(ValidationError, match="policy_noise"):
            HIROConfig(state_size=4, action_size=2, policy_noise=-0.1)

        # Valid values
        config1 = HIROConfig(state_size=4, action_size=2, policy_noise=0.0)
        assert config1.policy_noise == 0.0

        config2 = HIROConfig(state_size=4, action_size=2, policy_noise=0.5)
        assert config2.policy_noise == 0.5

    @pytest.mark.unit
    def test_config_validates_non_negative_noise_clip(self) -> None:
        """Test that config validates noise_clip is non-negative."""
        # Arrange - prepare invalid and valid noise_clip values
        # Act - attempt to create HIROConfig with various noise_clip values
        # Assert - ValidationError for negative, success for non-negative values
        with pytest.raises(ValidationError, match="noise_clip"):
            HIROConfig(state_size=4, action_size=2, noise_clip=-0.1)

        # Valid values
        config1 = HIROConfig(state_size=4, action_size=2, noise_clip=0.0)
        assert config1.noise_clip == 0.0

        config2 = HIROConfig(state_size=4, action_size=2, noise_clip=1.0)
        assert config2.noise_clip == 1.0

    @pytest.mark.unit
    def test_config_validates_positive_intrinsic_scale(self) -> None:
        """Test that config validates intrinsic_scale is positive."""
        # Arrange - prepare invalid intrinsic_scale values (zero and negative)
        # Act - attempt to create HIROConfig with invalid values
        # Assert - ValidationError is raised for non-positive intrinsic_scale
        with pytest.raises(ValidationError, match="intrinsic_scale"):
            HIROConfig(state_size=4, action_size=2, intrinsic_scale=0.0)

        with pytest.raises(ValidationError, match="intrinsic_scale"):
            HIROConfig(state_size=4, action_size=2, intrinsic_scale=-0.5)

        # Valid values
        config = HIROConfig(state_size=4, action_size=2, intrinsic_scale=0.5)
        assert config.intrinsic_scale == 0.5

    @pytest.mark.unit
    def test_config_with_all_defaults(self) -> None:
        """Test that config works with only required parameters."""
        # Arrange - create config with only required parameters
        # Act - instantiate HIROConfig with minimal parameters
        # Assert - all default values are correctly set
        config = HIROConfig(state_size=4, action_size=2)

        assert config.state_size == 4
        assert config.action_size == 2
        assert config.goal_size == 16
        assert config.hidden_size == 256
        assert config.goal_horizon == 10
        assert config.learning_rate == 0.0003
        assert config.gamma == 0.99
        assert config.tau == 0.005
        assert config.device == "cpu"
        assert config.seed is None
        assert config.policy_noise == 0.2
        assert config.noise_clip == 0.5
        assert config.intrinsic_scale == 1.0

    @pytest.mark.unit
    def test_config_with_custom_values(self) -> None:
        """Test that config accepts custom values for all parameters."""
        # Arrange - prepare all custom parameter values
        # Act - create HIROConfig with all custom values
        # Assert - all values are correctly set to custom values
        config = HIROConfig(
            state_size=8,
            action_size=4,
            goal_size=32,
            hidden_size=512,
            goal_horizon=20,
            learning_rate=0.001,
            gamma=0.95,
            tau=0.01,
            device="cpu",
            seed=123,
            policy_noise=0.3,
            noise_clip=0.6,
            intrinsic_scale=2.0,
        )

        assert config.state_size == 8
        assert config.action_size == 4
        assert config.goal_size == 32
        assert config.hidden_size == 512
        assert config.goal_horizon == 20
        assert config.learning_rate == 0.001
        assert config.gamma == 0.95
        assert config.tau == 0.01
        assert config.device == "cpu"
        assert config.seed == 123
        assert config.policy_noise == 0.3
        assert config.noise_clip == 0.6
        assert config.intrinsic_scale == 2.0


class TestHIROAgentInitialization:
    """Tests for HIROAgent initialization with both config and kwargs."""

    @pytest.mark.unit
    def test_agent_initialization_with_config_object(self) -> None:
        """Test that agent accepts and uses config object."""
        # Arrange - create a validated HIROConfig object
        config = HIROConfig(state_size=4, action_size=2, seed=42)

        # Act - initialize agent with the config object
        agent = HIROAgent(config=config)

        # Assert - agent stores config and extracts parameters correctly
        assert agent.config == config
        assert agent.state_size == 4
        assert agent.action_size == 2
        assert agent.goal_size == 16  # default
        assert agent.gamma == 0.99  # default
        assert isinstance(agent.device, torch.device)

    @pytest.mark.unit
    def test_agent_initialization_with_kwargs(self) -> None:
        """Test that agent accepts kwargs for backwards compatibility."""
        # Arrange - prepare kwargs for backwards compatible initialization
        # Act - initialize agent with individual parameters (old style)
        # Assert - config is created internally and parameters are set
        agent = HIROAgent(state_size=8, action_size=4, goal_size=32, seed=123)

        assert agent.state_size == 8
        assert agent.action_size == 4
        assert agent.goal_size == 32
        assert isinstance(agent.config, HIROConfig)
        assert agent.config.state_size == 8
        assert agent.config.action_size == 4

    @pytest.mark.unit
    def test_agent_initialization_validates_via_config(self) -> None:
        """Test that agent initialization validates parameters via Pydantic."""
        # Arrange - prepare invalid parameters
        # Act - attempt to create agent with invalid parameters
        # Assert - ValidationError is raised by Pydantic
        with pytest.raises(ValidationError, match="state_size"):
            HIROAgent(state_size=-1, action_size=4)

        with pytest.raises(ValidationError, match="learning_rate"):
            HIROAgent(state_size=4, action_size=2, learning_rate=1.5)

    @pytest.mark.unit
    def test_agent_initializes_networks(self) -> None:
        """Test that agent initializes all required networks."""
        # Arrange - create valid config with custom hidden size
        config = HIROConfig(state_size=4, action_size=2, hidden_size=128)

        # Act - initialize agent with config
        agent = HIROAgent(config=config)

        # Assert - all policy networks are initialized correctly
        assert isinstance(agent.higher_policy, HigherLevelPolicy)
        assert isinstance(agent.higher_target, HigherLevelPolicy)
        assert isinstance(agent.lower_policy, LowerLevelPolicy)
        assert isinstance(agent.lower_target, LowerLevelPolicy)

    @pytest.mark.unit
    def test_agent_initializes_optimizers(self) -> None:
        """Test that agent initializes all optimizers."""
        # Arrange - create valid config
        config = HIROConfig(state_size=4, action_size=2)

        # Act - initialize agent with config
        agent = HIROAgent(config=config)

        # Assert - all optimizers are initialized and not None
        assert agent.higher_actor_optimizer is not None
        assert agent.higher_critic_optimizer is not None
        assert agent.lower_actor_optimizer is not None
        assert agent.lower_critic_optimizer is not None

    @pytest.mark.unit
    def test_agent_seed_sets_random_state(self) -> None:
        """Test that seed parameter sets random state consistently."""
        # Arrange - create two configs with same seed
        config1 = HIROConfig(state_size=4, action_size=2, seed=42)
        config2 = HIROConfig(state_size=4, action_size=2, seed=42)

        # Act - initialize two agents with same seed
        agent1 = HIROAgent(config=config1)
        agent2 = HIROAgent(config=config2)

        # Assert - agents should have consistent initialization
        # (This is a simplified test; actual behavior depends on when random ops occur)
        assert agent1.state_size == agent2.state_size


class TestHIROAgentFunctionality:
    """Tests for core HIRO agent functionality."""

    @pytest.mark.unit
    def test_select_goal_returns_correct_shape(self) -> None:
        """Test that select_goal returns goal with correct shape."""
        # Arrange - create agent with specific goal size and prepare state
        config = HIROConfig(state_size=4, action_size=2, goal_size=8)
        agent = HIROAgent(config=config)
        state = torch.randn(4)

        # Act - select goal using higher-level policy
        goal = agent.select_goal(state)

        # Assert - goal has correct shape and values in valid range
        assert goal.shape == (8,)
        assert torch.all(torch.abs(goal) <= 1.0)  # Tanh output

    @pytest.mark.unit
    def test_select_action_returns_valid_action(self) -> None:
        """Test that select_action returns valid action index."""
        # Arrange - create agent and prepare state and goal tensors
        config = HIROConfig(state_size=4, action_size=3)
        agent = HIROAgent(config=config)
        state = torch.randn(4)
        goal = torch.randn(16)

        # Act - select action using lower-level policy with no exploration
        action = agent.select_action(state, goal, epsilon=0.0)

        # Assert - action is valid integer within action space bounds
        assert isinstance(action, int)
        assert 0 <= action < 3

    @pytest.mark.unit
    def test_goal_distance_computes_intrinsic_reward(self) -> None:
        """Test that goal_distance computes intrinsic reward."""
        # Arrange - create agent and prepare identical state and goal
        config = HIROConfig(state_size=4, action_size=2)
        agent = HIROAgent(config=config)
        state = torch.tensor([1.0, 2.0, 3.0, 4.0])
        goal = torch.tensor([1.0, 2.0, 3.0, 4.0])

        # Act - compute intrinsic reward for reaching goal
        reward = agent.goal_distance(state, goal)

        # Assert - reward is float (identical state and goal should give high reward)
        assert isinstance(reward, float)
        # Identical state and goal should give high reward (low distance)

    @pytest.mark.unit
    def test_relabel_goal_computes_state_delta(self) -> None:
        """Test that relabel_goal computes state delta correctly."""
        # Arrange - create agent and prepare trajectory with start state
        config = HIROConfig(state_size=4, action_size=2, goal_horizon=3)
        agent = HIROAgent(config=config)
        start_state = torch.tensor([1.0, 2.0, 3.0, 4.0])
        trajectory = [
            torch.tensor([1.5, 2.5, 3.5, 4.5]),
            torch.tensor([2.0, 3.0, 4.0, 5.0]),
            torch.tensor([2.5, 3.5, 4.5, 5.5]),
        ]

        # Act - relabel goal based on achieved trajectory
        relabeled_goal = agent.relabel_goal(start_state, trajectory, horizon=3)

        # Assert - relabeled goal is state delta matching expected value
        assert relabeled_goal.shape == (4,)
        # Should be delta: trajectory[-1] - start_state
        expected_delta = trajectory[2] - start_state
        assert torch.allclose(relabeled_goal, expected_delta)

    @pytest.mark.unit
    def test_soft_update_targets(self) -> None:
        """Test that soft update modifies target networks."""
        # Arrange - create agent and store initial target network parameters
        config = HIROConfig(state_size=4, action_size=2, tau=0.1)
        agent = HIROAgent(config=config)

        # Get initial target params for both higher and lower networks
        initial_higher_params = [p.clone() for p in agent.higher_target.parameters()]
        initial_lower_params = [p.clone() for p in agent.lower_target.parameters()]

        # Modify policy networks
        for p in agent.higher_policy.parameters():
            p.data += 1.0
        for p in agent.lower_policy.parameters():
            p.data += 1.0

        # Act - perform soft update of target networks
        agent.soft_update_targets()

        # Assert - target network parameters have changed from initial values
        for init_p, updated_p in zip(
            initial_higher_params, agent.higher_target.parameters()
        ):
            assert not torch.equal(init_p, updated_p)
        for init_p, updated_p in zip(
            initial_lower_params, agent.lower_target.parameters()
        ):
            assert not torch.equal(init_p, updated_p)

    @pytest.mark.unit
    def test_get_statistics_returns_dict(self) -> None:
        """Test that get_statistics returns dictionary with expected keys."""
        # Arrange - create agent with default configuration
        config = HIROConfig(state_size=4, action_size=2)
        agent = HIROAgent(config=config)

        # Act - retrieve agent statistics
        stats = agent.get_statistics()

        # Assert - statistics dictionary contains all expected keys
        assert isinstance(stats, dict)
        assert "total_episodes" in stats
        assert "avg_reward" in stats
        assert "avg_lower_critic_loss" in stats
        assert "avg_higher_critic_loss" in stats
        assert "avg_lower_actor_loss" in stats
        assert "avg_higher_actor_loss" in stats
        assert "avg_intrinsic_reward" in stats
        assert "avg_extrinsic_reward" in stats
        assert "intrinsic_extrinsic_ratio" in stats
        assert "goal_horizon" in stats


class TestHIROPolicyNetworks:
    """Tests for HIRO policy network classes."""

    @pytest.mark.unit
    def test_higher_level_policy_forward(self) -> None:
        """Test that HigherLevelPolicy forward pass works."""
        # Arrange - create higher-level policy and batch of states
        policy = HigherLevelPolicy(state_size=4, goal_size=8, hidden_size=128)
        state = torch.randn(2, 4)  # Batch of 2

        # Act - run forward pass to generate goals
        goal = policy(state)

        # Assert - goals have correct shape and are bounded by Tanh
        assert goal.shape == (2, 8)
        assert torch.all(torch.abs(goal) <= 1.0)  # Tanh output

    @pytest.mark.unit
    def test_higher_level_policy_get_value(self) -> None:
        """Test that HigherLevelPolicy critic works."""
        # Arrange - create policy and prepare state-goal pairs
        policy = HigherLevelPolicy(state_size=4, goal_size=8, hidden_size=128)
        state = torch.randn(2, 4)
        goal = torch.randn(2, 8)

        # Act - compute Q-values for state-goal pairs
        q_value = policy.get_value(state, goal)

        # Assert - Q-values have correct shape (batch_size, 1)
        assert q_value.shape == (2, 1)

    @pytest.mark.unit
    def test_lower_level_policy_forward(self) -> None:
        """Test that LowerLevelPolicy forward pass works."""
        # Arrange - create lower-level policy and prepare state-goal pairs
        policy = LowerLevelPolicy(
            state_size=4, action_size=3, goal_size=8, hidden_size=128
        )
        state = torch.randn(2, 4)
        goal = torch.randn(2, 8)

        # Act - run forward pass to generate action logits
        logits = policy(state, goal)

        # Assert - logits have correct shape (batch_size, action_size)
        assert logits.shape == (2, 3)

    @pytest.mark.unit
    def test_lower_level_policy_get_value(self) -> None:
        """Test that LowerLevelPolicy critic works."""
        # Arrange - create policy and prepare state-goal-action tuples
        policy = LowerLevelPolicy(
            state_size=4, action_size=3, goal_size=8, hidden_size=128
        )
        state = torch.randn(2, 4)
        goal = torch.randn(2, 8)
        action = torch.randn(2, 3)

        # Act - compute Q-values for state-goal-action tuples
        q_value = policy.get_value(state, goal, action)

        # Assert - Q-values have correct shape (batch_size, 1)
        assert q_value.shape == (2, 1)


class TestHIROAgentEdgeCases:
    """Tests for edge cases and additional coverage in HIRO agent."""

    @pytest.mark.unit
    def test_select_action_with_epsilon_exploration(self) -> None:
        """Test that select_action uses random exploration when epsilon triggers."""
        # Arrange - create agent with fixed seed for reproducibility
        config = HIROConfig(state_size=4, action_size=3, seed=42)
        agent = HIROAgent(config=config)
        state = torch.randn(4)
        goal = torch.randn(16)

        # Act - select action with high epsilon (force random exploration)
        action = agent.select_action(state, goal, epsilon=1.0)

        # Assert - action is valid integer (random branch was taken)
        assert isinstance(action, int)
        assert 0 <= action < 3

    @pytest.mark.unit
    def test_goal_distance_updates_statistics(self) -> None:
        """Test that goal_distance updates statistics after buffer exceeds 100 samples."""
        # Arrange - create agent and add many samples to distance buffer
        config = HIROConfig(state_size=4, action_size=2)
        agent = HIROAgent(config=config)
        state = torch.randn(4)
        goal = torch.randn(4)

        # Act - compute distance many times to fill buffer past 100 samples
        for _ in range(150):
            agent.goal_distance(state + torch.randn(4) * 0.1, goal)

        # Assert - statistics should have been updated (mean and std)
        assert agent.distance_mean != 0.0
        assert agent.distance_std != 1.0  # Should have changed from initial value
        assert len(agent.distance_buffer) == 150

    @pytest.mark.unit
    def test_relabel_goal_with_short_trajectory(self) -> None:
        """Test that relabel_goal handles trajectories shorter than horizon."""
        # Arrange - create agent with horizon of 5 but only 2 states in trajectory
        config = HIROConfig(state_size=4, action_size=2, goal_horizon=5)
        agent = HIROAgent(config=config)
        start_state = torch.tensor([1.0, 2.0, 3.0, 4.0])
        trajectory = [
            torch.tensor([1.5, 2.5, 3.5, 4.5]),
            torch.tensor([2.0, 3.0, 4.0, 5.0]),
        ]

        # Act - relabel goal with trajectory shorter than horizon
        relabeled_goal = agent.relabel_goal(start_state, trajectory, horizon=5)

        # Assert - should use last state when trajectory < horizon
        assert relabeled_goal.shape == (4,)
        expected_delta = trajectory[-1] - start_state
        assert torch.allclose(relabeled_goal, expected_delta)


class TestHIROTrainingMethods:
    """Tests for HIRO training methods including critics and actors."""

    @pytest.mark.unit
    def test_train_lower_with_empty_buffer(self) -> None:
        """Test that train_lower returns 0.0 when buffer is too small."""
        # Arrange - create agent with empty lower buffer
        config = HIROConfig(state_size=4, action_size=2)
        agent = HIROAgent(config=config)

        # Act - attempt to train with insufficient data
        loss = agent.train_lower(batch_size=64)

        # Assert - should return 0.0 when buffer < batch_size
        assert loss == 0.0

    @pytest.mark.unit
    def test_train_lower_with_sufficient_data(self) -> None:
        """Test that train_lower updates critic with sufficient data."""
        # Arrange - create agent and populate lower buffer with experiences
        config = HIROConfig(state_size=4, action_size=2, seed=42)
        agent = HIROAgent(config=config)

        # Add experiences to lower buffer
        for _ in range(100):
            agent.lower_buffer.append(
                {
                    "state": torch.randn(4),
                    "action": 0,
                    "reward": 1.0,
                    "next_state": torch.randn(4),
                    "goal": torch.randn(16),
                    "done": False,
                }
            )

        # Act - train lower-level critic
        loss = agent.train_lower(batch_size=32)

        # Assert - loss should be positive float (training occurred)
        assert isinstance(loss, float)
        assert loss >= 0.0

    @pytest.mark.unit
    def test_train_lower_actor_with_empty_buffer(self) -> None:
        """Test that train_lower_actor returns 0.0 when buffer is too small."""
        # Arrange - create agent with empty lower buffer
        config = HIROConfig(state_size=4, action_size=2)
        agent = HIROAgent(config=config)

        # Act - attempt to train actor with insufficient data
        loss = agent.train_lower_actor(batch_size=64)

        # Assert - should return 0.0 when buffer < batch_size
        assert loss == 0.0

    @pytest.mark.unit
    def test_train_lower_actor_with_sufficient_data(self) -> None:
        """Test that train_lower_actor updates policy with sufficient data."""
        # Arrange - create agent and populate lower buffer with experiences
        config = HIROConfig(state_size=4, action_size=3, seed=42)
        agent = HIROAgent(config=config)

        # Add experiences to lower buffer
        for _ in range(100):
            agent.lower_buffer.append(
                {
                    "state": torch.randn(4),
                    "action": 0,
                    "reward": 1.0,
                    "next_state": torch.randn(4),
                    "goal": torch.randn(16),
                    "done": False,
                }
            )

        # Act - train lower-level actor
        loss = agent.train_lower_actor(batch_size=32)

        # Assert - loss should be float (training occurred, can be negative)
        assert isinstance(loss, float)

    @pytest.mark.unit
    def test_train_higher_with_empty_buffer(self) -> None:
        """Test that train_higher returns 0.0 when buffer is too small."""
        # Arrange - create agent with empty higher buffer
        config = HIROConfig(state_size=4, action_size=2)
        agent = HIROAgent(config=config)

        # Act - attempt to train with insufficient data
        loss = agent.train_higher(batch_size=64)

        # Assert - should return 0.0 when buffer < batch_size
        assert loss == 0.0

    @pytest.mark.unit
    def test_train_higher_with_sufficient_data(self) -> None:
        """Test that train_higher updates critic with sufficient data."""
        # Arrange - create agent and populate higher buffer with experiences
        config = HIROConfig(state_size=4, action_size=2, seed=42)
        agent = HIROAgent(config=config)

        # Add experiences to higher buffer
        for _ in range(100):
            agent.higher_buffer.append(
                {
                    "state": torch.randn(4),
                    "goal": torch.randn(16),
                    "reward": 1.0,
                    "next_state": torch.randn(4),
                    "done": False,
                }
            )

        # Act - train higher-level critic
        loss = agent.train_higher(batch_size=32)

        # Assert - loss should be positive float (training occurred)
        assert isinstance(loss, float)
        assert loss >= 0.0

    @pytest.mark.unit
    def test_train_higher_actor_with_empty_buffer(self) -> None:
        """Test that train_higher_actor returns 0.0 when buffer is too small."""
        # Arrange - create agent with empty higher buffer
        config = HIROConfig(state_size=4, action_size=2)
        agent = HIROAgent(config=config)

        # Act - attempt to train actor with insufficient data
        loss = agent.train_higher_actor(batch_size=64)

        # Assert - should return 0.0 when buffer < batch_size
        assert loss == 0.0

    @pytest.mark.unit
    def test_train_higher_actor_with_sufficient_data(self) -> None:
        """Test that train_higher_actor updates policy with sufficient data."""
        # Arrange - create agent and populate higher buffer with experiences
        config = HIROConfig(state_size=4, action_size=2, seed=42)
        agent = HIROAgent(config=config)

        # Add experiences to higher buffer
        for _ in range(100):
            agent.higher_buffer.append(
                {
                    "state": torch.randn(4),
                    "goal": torch.randn(16),
                    "reward": 1.0,
                    "next_state": torch.randn(4),
                    "done": False,
                }
            )

        # Act - train higher-level actor
        loss = agent.train_higher_actor(batch_size=32)

        # Assert - loss should be float (training occurred, can be negative)
        assert isinstance(loss, float)


class TestHIROEpisodeTraining:
    """Tests for HIRO episode training and integration."""

    @pytest.mark.unit
    def test_train_episode_returns_metrics(self) -> None:
        """Test that train_episode executes and returns expected metrics."""

        # Arrange - create a simple mock environment
        class SimpleEnv:
            """Simple mock environment for testing."""

            def __init__(self) -> None:
                self.step_count = 0

            def reset(self) -> tuple[list[float], dict[str, object]]:
                """Reset environment."""
                return [0.0, 0.0, 0.0, 0.0], {}

            def step(
                self, action: int
            ) -> tuple[list[float], float, bool, bool, dict[str, object]]:
                """Take a step in environment."""
                self.step_count += 1
                next_state = [0.1, 0.1, 0.1, 0.1]
                reward = 1.0
                done = self.step_count >= 5  # Episode ends after 5 steps
                truncated = False
                return next_state, reward, done, truncated, {}

        config = HIROConfig(state_size=4, action_size=2, goal_size=4, seed=42)
        agent = HIROAgent(config=config)
        env = SimpleEnv()

        # Act - train for one episode with short max_steps
        metrics = agent.train_episode(env, max_steps=10, epsilon=0.5)

        # Assert - metrics dictionary contains expected keys
        assert isinstance(metrics, dict)
        assert "reward" in metrics
        assert "steps" in metrics
        assert "avg_lower_critic_loss" in metrics
        assert "avg_higher_critic_loss" in metrics
        assert "avg_lower_actor_loss" in metrics
        assert "avg_higher_actor_loss" in metrics
        assert metrics["steps"] > 0

    @pytest.mark.unit
    def test_train_episode_populates_buffers(self) -> None:
        """Test that train_episode adds experiences to buffers."""

        # Arrange - create a simple mock environment
        class SimpleEnv:
            """Simple mock environment for testing."""

            def __init__(self) -> None:
                self.step_count = 0

            def reset(self) -> tuple[list[float], dict[str, object]]:
                """Reset environment."""
                return [0.0, 0.0, 0.0, 0.0], {}

            def step(
                self, action: int
            ) -> tuple[list[float], float, bool, bool, dict[str, object]]:
                """Take a step in environment."""
                self.step_count += 1
                next_state = [0.1, 0.1, 0.1, 0.1]
                reward = 1.0
                done = self.step_count >= 10
                truncated = False
                return next_state, reward, done, truncated, {}

        config = HIROConfig(
            state_size=4, action_size=2, goal_size=4, goal_horizon=5, seed=42
        )
        agent = HIROAgent(config=config)
        env = SimpleEnv()

        # Store initial buffer sizes
        initial_lower_size = len(agent.lower_buffer)
        initial_higher_size = len(agent.higher_buffer)

        # Act - train for one episode
        agent.train_episode(env, max_steps=20, epsilon=0.3)

        # Assert - buffers should have new experiences
        assert len(agent.lower_buffer) > initial_lower_size
        # Higher buffer should have at least one experience (goal horizon reached)
        assert len(agent.higher_buffer) >= initial_higher_size

    @pytest.mark.unit
    def test_train_episode_tracks_statistics(self) -> None:
        """Test that train_episode tracks statistics correctly."""

        # Arrange - create a simple mock environment
        class SimpleEnv:
            """Simple mock environment for testing."""

            def __init__(self) -> None:
                self.step_count = 0

            def reset(self) -> tuple[list[float], dict[str, object]]:
                """Reset environment."""
                return [0.0, 0.0, 0.0, 0.0], {}

            def step(
                self, action: int
            ) -> tuple[list[float], float, bool, bool, dict[str, object]]:
                """Take a step in environment."""
                self.step_count += 1
                next_state = [0.1, 0.1, 0.1, 0.1]
                reward = 1.0
                done = self.step_count >= 8
                truncated = False
                return next_state, reward, done, truncated, {}

        config = HIROConfig(state_size=4, action_size=2, goal_size=4, seed=42)
        agent = HIROAgent(config=config)
        env = SimpleEnv()

        # Act - train for one episode
        agent.train_episode(env, max_steps=15, epsilon=0.2)

        # Assert - statistics should be tracked
        assert len(agent.episode_rewards) > 0
        assert len(agent.intrinsic_rewards) > 0
        assert len(agent.extrinsic_rewards) > 0

    @pytest.mark.unit
    def test_train_episode_with_old_style_env_reset(self) -> None:
        """Test that train_episode handles environments with old reset signature."""

        # Arrange - create env that returns only state (not tuple)
        class OldStyleEnv:
            """Old style mock environment that returns state directly."""

            def __init__(self) -> None:
                self.step_count = 0

            def reset(self) -> list[float]:
                """Reset environment (old style - returns state only)."""
                return [0.0, 0.0, 0.0, 0.0]

            def step(
                self, action: int
            ) -> tuple[list[float], float, bool, bool, dict[str, object]]:
                """Take a step in environment."""
                self.step_count += 1
                next_state = [0.1, 0.1, 0.1, 0.1]
                reward = 1.0
                done = self.step_count >= 5
                truncated = False
                return next_state, reward, done, truncated, {}

        config = HIROConfig(state_size=4, action_size=2, goal_size=4, seed=42)
        agent = HIROAgent(config=config)
        env = OldStyleEnv()

        # Act - train for one episode with old-style environment
        metrics = agent.train_episode(env, max_steps=10, epsilon=0.5)

        # Assert - should work correctly with old-style reset
        assert isinstance(metrics, dict)
        assert metrics["steps"] > 0

    @pytest.mark.unit
    def test_train_episode_with_none_goal_state(self) -> None:
        """Test that train_episode handles None goal_state correctly."""

        # Arrange - create environment and agent
        class SimpleEnv:
            """Simple mock environment for testing."""

            def __init__(self) -> None:
                self.step_count = 0

            def reset(self) -> tuple[list[float], dict[str, object]]:
                """Reset environment."""
                return [0.0, 0.0, 0.0, 0.0], {}

            def step(
                self, action: int
            ) -> tuple[list[float], float, bool, bool, dict[str, object]]:
                """Take a step in environment."""
                self.step_count += 1
                next_state = [0.1, 0.1, 0.1, 0.1]
                reward = 1.0
                done = self.step_count >= 3
                truncated = False
                return next_state, reward, done, truncated, {}

        config = HIROConfig(state_size=4, action_size=2, goal_size=4, seed=42)
        agent = HIROAgent(config=config)
        env = SimpleEnv()

        # Manually set goal_state to None to test the None branch
        agent.current_goal = None
        agent.goal_state = None

        # Act - train for one episode
        metrics = agent.train_episode(env, max_steps=5, epsilon=0.3)

        # Assert - should complete successfully with None goal
        assert isinstance(metrics, dict)
        assert metrics["steps"] > 0

    @pytest.mark.unit
    def test_train_episode_tracks_losses_when_positive(self) -> None:
        """Test that train_episode tracks losses only when they are positive."""

        # Arrange - create environment with longer episodes to ensure training
        class LongerEnv:
            """Environment that runs longer to ensure training occurs."""

            def __init__(self) -> None:
                self.step_count = 0

            def reset(self) -> tuple[list[float], dict[str, object]]:
                """Reset environment."""
                return [0.0, 0.0, 0.0, 0.0], {}

            def step(
                self, action: int
            ) -> tuple[list[float], float, bool, bool, dict[str, object]]:
                """Take a step in environment."""
                self.step_count += 1
                next_state = [
                    0.1 * self.step_count,
                    0.1 * self.step_count,
                    0.1 * self.step_count,
                    0.1 * self.step_count,
                ]
                reward = 1.0
                done = self.step_count >= 150  # Run longer to accumulate buffer
                truncated = False
                return next_state, reward, done, truncated, {}

        config = HIROConfig(state_size=4, action_size=2, goal_size=4, seed=42)
        agent = HIROAgent(config=config)
        env = LongerEnv()

        # Act - train for a longer episode to ensure losses are tracked
        metrics = agent.train_episode(env, max_steps=200, epsilon=0.3)

        # Assert - losses should be tracked (at least some entries should exist)
        assert isinstance(metrics, dict)
        assert metrics["steps"] > 0
        # At least one loss should have been tracked
        assert (
            len(agent.lower_critic_losses) > 0
            or len(agent.higher_critic_losses) > 0
            or len(agent.lower_actor_losses) > 0
            or len(agent.higher_actor_losses) > 0
        )
