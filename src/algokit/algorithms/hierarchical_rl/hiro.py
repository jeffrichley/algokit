"""HIRO: Data-Efficient Hierarchical Reinforcement Learning.

This module implements HIRO (Nachum et al., 2018), which uses goal-conditioned
hierarchical RL with off-policy correction. HIRO addresses the non-stationarity
problem in hierarchical RL through goal relabeling and off-policy learning.

Key features:
- Higher-level policy proposes goals (as state deltas) for lower-level policy
- Lower-level policy learns goal-conditioned actions
- Off-policy correction through goal relabeling using state deltas (s_{t+k} - s_t)
- Hindsight experience replay for data efficiency
- TD3-style target policy smoothing for stability
- Normalized and scaled intrinsic rewards to prevent instability
- Diverse goal sampling (mix of recent and older experiences)

Implementation improvements:
1. Goals are state deltas (relative displacements) not absolute states
2. Intrinsic rewards are normalized by running statistics and scaled by state dimensionality
3. Target policy smoothing with clipped Gaussian noise (TD3-style)
4. Diverse experience sampling: 50% recent, 50% diverse from entire buffer
5. Proper Q-value computation for both hierarchical levels
6. Explicit actor updates using deterministic policy gradient for both levels
7. Delayed policy updates (TD3-style) - actors updated less frequently than critics
8. Separate optimizers for actor and critic networks

References:
    Nachum, O., Gu, S. S., Lee, H., & Levine, S. (2018).
    Data-Efficient Hierarchical Reinforcement Learning. NeurIPS 2018.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pydantic import BaseModel, Field, field_validator


class HigherLevelPolicy(nn.Module):
    """Higher-level policy that proposes goals for the lower-level.

    The higher-level policy operates at a slower timescale and provides
    goals in the state space for the lower-level policy to achieve.
    """

    def __init__(self, state_size: int, goal_size: int, hidden_size: int = 256) -> None:
        """Initialize higher-level policy.

        Args:
            state_size: Dimension of state space
            goal_size: Dimension of goal space (subgoal representation)
            hidden_size: Size of hidden layers
        """
        super().__init__()

        self.state_size = state_size
        self.goal_size = goal_size

        # Policy network
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, goal_size),
            nn.Tanh(),  # Bounded goals
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_size + goal_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Propose a goal given the current state.

        Args:
            state: Current state

        Returns:
            Proposed goal
        """
        return self.network(state)

    def get_value(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Get Q-value for state-goal pair.

        Args:
            state: Current state
            goal: Proposed goal

        Returns:
            Q-value
        """
        combined = torch.cat([state, goal], dim=-1)
        return self.critic(combined)


class LowerLevelPolicy(nn.Module):
    """Lower-level goal-conditioned policy.

    The lower-level policy learns to achieve goals proposed by the
    higher-level policy through primitive actions.
    """

    def __init__(
        self, state_size: int, action_size: int, goal_size: int, hidden_size: int = 256
    ) -> None:
        """Initialize lower-level policy.

        Args:
            state_size: Dimension of state space
            action_size: Number of primitive actions
            goal_size: Dimension of goal space
            hidden_size: Size of hidden layers
        """
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.goal_size = goal_size

        # Goal-conditioned policy
        self.policy = nn.Sequential(
            nn.Linear(state_size + goal_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

        # Goal-conditioned critic
        self.critic = nn.Sequential(
            nn.Linear(state_size + goal_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Get action logits given state and goal.

        Args:
            state: Current state
            goal: Goal to achieve

        Returns:
            Action logits
        """
        combined = torch.cat([state, goal], dim=-1)
        return self.policy(combined)

    def get_value(
        self, state: torch.Tensor, goal: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Get Q-value for state-goal-action triple.

        Args:
            state: Current state
            goal: Current goal
            action: Action taken

        Returns:
            Q-value
        """
        combined = torch.cat([state, goal, action], dim=-1)
        return self.critic(combined)


class HIROConfig(BaseModel):
    """Configuration parameters for HIRO with automatic validation.

    This model uses Pydantic for declarative parameter validation,
    reducing complexity while maintaining strict type safety and
    comprehensive validation.

    Attributes:
        state_size: Dimension of state space (must be positive)
        action_size: Number of primitive actions (must be positive)
        goal_size: Dimension of goal space (must be positive)
        hidden_size: Size of hidden layers in networks (must be positive)
        goal_horizon: Steps between higher-level decisions (must be positive)
        learning_rate: Learning rate for networks (must be in (0, 1])
        gamma: Discount factor (must be in [0, 1])
        tau: Soft update coefficient for target networks (must be in (0, 1])
        device: Device for computation ('cpu' or 'cuda')
        seed: Random seed for reproducibility (optional, must be non-negative if set)
        policy_noise: Noise std for target policy smoothing (must be non-negative)
        noise_clip: Maximum absolute value for policy noise (must be non-negative)
        intrinsic_scale: Scaling factor for intrinsic rewards (must be positive)
    """

    # Required parameters
    state_size: int = Field(..., gt=0, description="Dimension of state space")
    action_size: int = Field(..., gt=0, description="Number of primitive actions")

    # Optional parameters with defaults
    goal_size: int = Field(default=16, gt=0, description="Dimension of goal space")
    hidden_size: int = Field(default=256, gt=0, description="Size of hidden layers")
    goal_horizon: int = Field(
        default=10, gt=0, description="Steps between higher-level decisions"
    )
    learning_rate: float = Field(
        default=0.0003, gt=0.0, le=1.0, description="Learning rate for networks"
    )
    gamma: float = Field(default=0.99, ge=0.0, le=1.0, description="Discount factor")
    tau: float = Field(
        default=0.005,
        gt=0.0,
        le=1.0,
        description="Soft update coefficient for target networks",
    )
    device: str = Field(
        default="cpu", description="Device for computation ('cpu' or 'cuda')"
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )
    policy_noise: float = Field(
        default=0.2,
        ge=0.0,
        description="Noise std for target policy smoothing (TD3-style)",
    )
    noise_clip: float = Field(
        default=0.5, ge=0.0, description="Maximum absolute value for policy noise"
    )
    intrinsic_scale: float = Field(
        default=1.0, gt=0.0, description="Scaling factor for intrinsic rewards"
    )

    model_config = {"arbitrary_types_allowed": True}  # For torch.device

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device string.

        Args:
            v: Device string to validate

        Returns:
            Validated device string

        Raises:
            ValueError: If device is not 'cpu' or 'cuda'
        """
        if v.lower() not in ["cpu", "cuda"]:
            raise ValueError(f"Device must be 'cpu' or 'cuda', got '{v}'")
        return v.lower()

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, v: int | None) -> int | None:
        """Validate seed is non-negative if provided.

        Args:
            v: Seed value to validate

        Returns:
            Validated seed value

        Raises:
            ValueError: If seed is negative
        """
        if v is not None and v < 0:
            raise ValueError(f"Seed must be non-negative, got {v}")
        return v


class HIROAgent:
    """HIRO agent with hierarchical goal-conditioned RL.

    The agent consists of:
    - Higher-level policy: Proposes goals at a slower timescale
    - Lower-level policy: Achieves goals through primitive actions
    - Off-policy correction: Relabels goals for data efficiency
    """

    def __init__(self, config: HIROConfig | None = None, **kwargs: Any) -> None:
        """Initialize HIRO agent.

        Args:
            config: Pre-validated configuration object (recommended)
            **kwargs: Individual parameters for backwards compatibility

        Examples:
            # New style (recommended)
            >>> config = HIROConfig(state_size=4, action_size=2)
            >>> agent = HIROAgent(config=config)

            # Old style (backwards compatible)
            >>> agent = HIROAgent(state_size=4, action_size=2)

        Raises:
            ValidationError: If parameters are invalid (via Pydantic)
        """
        # Validate parameters (automatic via Pydantic)
        if config is None:
            config = HIROConfig(**kwargs)

        # Store config
        self.config = config

        # Set random seeds if provided
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            random.seed(config.seed)

        # Extract all parameters
        self.state_size = config.state_size
        self.action_size = config.action_size
        self.goal_size = config.goal_size
        self.goal_horizon = config.goal_horizon
        self.gamma = config.gamma
        self.tau = config.tau
        self.device = torch.device(config.device)
        self.policy_noise = config.policy_noise
        self.noise_clip = config.noise_clip
        self.intrinsic_scale = config.intrinsic_scale

        # Track distance statistics for normalization
        self.distance_mean = 0.0
        self.distance_std = 1.0
        self.distance_buffer: deque[float] = deque(maxlen=10000)

        # Initialize higher-level policy
        self.higher_policy = HigherLevelPolicy(
            state_size=config.state_size,
            goal_size=config.goal_size,
            hidden_size=config.hidden_size,
        ).to(self.device)

        self.higher_target = HigherLevelPolicy(
            state_size=config.state_size,
            goal_size=config.goal_size,
            hidden_size=config.hidden_size,
        ).to(self.device)
        self.higher_target.load_state_dict(self.higher_policy.state_dict())

        # Initialize lower-level policy
        self.lower_policy = LowerLevelPolicy(
            state_size=config.state_size,
            action_size=config.action_size,
            goal_size=config.goal_size,
            hidden_size=config.hidden_size,
        ).to(self.device)

        self.lower_target = LowerLevelPolicy(
            state_size=config.state_size,
            action_size=config.action_size,
            goal_size=config.goal_size,
            hidden_size=config.hidden_size,
        ).to(self.device)
        self.lower_target.load_state_dict(self.lower_policy.state_dict())

        # Optimizers - separate for actor and critic
        self.higher_actor_optimizer = optim.Adam(
            self.higher_policy.network.parameters(), lr=config.learning_rate
        )
        self.higher_critic_optimizer = optim.Adam(
            self.higher_policy.critic.parameters(), lr=config.learning_rate
        )
        self.lower_actor_optimizer = optim.Adam(
            self.lower_policy.policy.parameters(), lr=config.learning_rate
        )
        self.lower_critic_optimizer = optim.Adam(
            self.lower_policy.critic.parameters(), lr=config.learning_rate
        )

        # Experience buffers
        self.higher_buffer: deque[dict[str, Any]] = deque(maxlen=100000)
        self.lower_buffer: deque[dict[str, Any]] = deque(maxlen=100000)

        # Current goal tracking
        self.current_goal: torch.Tensor | None = None
        self.goal_state: torch.Tensor | None = None  # State when goal was proposed
        self.steps_since_goal = 0

        # Statistics
        self.episode_rewards: list[float] = []
        self.higher_critic_losses: list[float] = []
        self.lower_critic_losses: list[float] = []
        self.higher_actor_losses: list[float] = []
        self.lower_actor_losses: list[float] = []
        self.intrinsic_rewards: list[float] = []
        self.extrinsic_rewards: list[float] = []

    def select_goal(self, state: torch.Tensor) -> torch.Tensor:
        """Select goal using higher-level policy.

        Args:
            state: Current state

        Returns:
            Selected goal
        """
        with torch.no_grad():
            goal = self.higher_policy(state.unsqueeze(0)).squeeze(0)
        return goal

    def select_action(
        self, state: torch.Tensor, goal: torch.Tensor, epsilon: float = 0.1
    ) -> int:
        """Select action using lower-level policy.

        Args:
            state: Current state
            goal: Current goal
            epsilon: Exploration parameter

        Returns:
            Selected action
        """
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            logits = self.lower_policy(state.unsqueeze(0), goal.unsqueeze(0))
            probs = F.softmax(logits, dim=-1).squeeze(0)
            action = int(torch.multinomial(probs, 1).item())

        return action

    def goal_distance(self, state: torch.Tensor, goal: torch.Tensor) -> float:
        """Compute distance to goal (for intrinsic reward).

        Args:
            state: Current state
            goal: Target goal

        Returns:
            Normalized negative distance (higher is better)
        """
        # Use negative L2 distance as intrinsic reward
        raw_dist = torch.norm(state - goal).item()

        # Update distance statistics
        self.distance_buffer.append(raw_dist)
        if len(self.distance_buffer) > 100:
            self.distance_mean = float(np.mean(self.distance_buffer))
            self.distance_std = float(np.std(self.distance_buffer)) + 1e-8

        # Normalize and scale the distance
        # Divide by state dimensionality to keep scale reasonable
        normalized_dist = (raw_dist - self.distance_mean) / self.distance_std
        scaled_reward = (
            -normalized_dist * self.intrinsic_scale / np.sqrt(self.state_size)
        )

        return scaled_reward

    def relabel_goal(
        self, start_state: torch.Tensor, trajectory: list[torch.Tensor], horizon: int
    ) -> torch.Tensor:
        """Relabel goal using hindsight (off-policy correction).

        HIRO relabels goals as state deltas: g = s_{t+k} - s_t
        This represents the relative displacement the agent should achieve.

        Args:
            start_state: State at the start of the goal horizon
            trajectory: List of states in trajectory (relative to start)
            horizon: Goal horizon

        Returns:
            Relabeled goal as state delta
        """
        # Use the delta between achieved and start states (s_{t+k} - s_t)
        if len(trajectory) >= horizon:
            achieved_state = trajectory[horizon - 1]
        else:
            achieved_state = trajectory[-1]

        # Goal is the relative displacement, not absolute position
        relabeled_goal = achieved_state - start_state
        return relabeled_goal

    def train_lower(self, batch_size: int = 64) -> float:
        """Train lower-level policy with diverse goal sampling.

        Args:
            batch_size: Batch size for training

        Returns:
            Loss value
        """
        if len(self.lower_buffer) < batch_size:
            return 0.0

        # Sample batch with diversity: mix of recent and older experiences
        # This ensures balanced training between near and far subgoals
        recent_size = batch_size // 2
        diverse_size = batch_size - recent_size

        # Recent experiences (last 20% of buffer)
        recent_start = max(0, len(self.lower_buffer) - len(self.lower_buffer) // 5)
        recent_batch = random.sample(
            list(self.lower_buffer)[recent_start:],
            min(recent_size, len(self.lower_buffer) - recent_start),
        )

        # Diverse sampling from entire buffer
        diverse_batch = random.sample(
            list(self.lower_buffer), min(diverse_size, len(self.lower_buffer))
        )

        batch = recent_batch + diverse_batch

        states = torch.stack([exp["state"] for exp in batch])
        actions = torch.tensor([exp["action"] for exp in batch], device=self.device)
        rewards = torch.tensor(
            [exp["reward"] for exp in batch], device=self.device, dtype=torch.float32
        )
        next_states = torch.stack([exp["next_state"] for exp in batch])
        goals = torch.stack([exp["goal"] for exp in batch])
        dones = torch.tensor(
            [exp["done"] for exp in batch], device=self.device, dtype=torch.float32
        )

        # Convert actions to one-hot for critic
        actions_onehot = F.one_hot(actions, num_classes=self.action_size).float()

        # Current Q-value
        current_q = self.lower_policy.get_value(states, goals, actions_onehot)

        # Target Q-value
        with torch.no_grad():
            next_logits = self.lower_target(next_states, goals)
            next_probs = F.softmax(next_logits, dim=-1)
            next_actions_onehot = (
                next_probs  # Expected Q-value over action distribution
            )
            next_q = self.lower_target.get_value(
                next_states, goals, next_actions_onehot
            )
            target_q = rewards.unsqueeze(1) + self.gamma * next_q * (
                1 - dones.unsqueeze(1)
            )

        # Compute loss
        loss = F.mse_loss(current_q, target_q)

        # Update critic
        self.lower_critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lower_policy.critic.parameters(), 1.0)
        self.lower_critic_optimizer.step()

        return loss.item()

    def train_lower_actor(self, batch_size: int = 64) -> float:
        """Train lower-level actor using policy gradient.

        Uses expected Q-value maximization for discrete action spaces.

        Args:
            batch_size: Batch size for training

        Returns:
            Loss value
        """
        if len(self.lower_buffer) < batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(list(self.lower_buffer), batch_size)

        states = torch.stack([exp["state"] for exp in batch])
        goals = torch.stack([exp["goal"] for exp in batch])

        # Get action probabilities from policy
        logits = self.lower_policy(states, goals)
        probs = F.softmax(logits, dim=-1)

        # Compute expected Q-value: E[Q(s, g, a)] = Σ π(a|s,g) * Q(s, g, a)
        # We want to maximize this, so minimize the negative
        q_values_per_action = []
        for a in range(self.action_size):
            action_onehot = F.one_hot(
                torch.tensor([a] * batch_size, device=self.device),
                num_classes=self.action_size,
            ).float()
            q_a = self.lower_policy.get_value(states, goals, action_onehot)
            q_values_per_action.append(q_a)

        q_values = torch.cat(
            q_values_per_action, dim=1
        )  # Shape: (batch_size, action_size)
        expected_q = (probs * q_values).sum(dim=1, keepdim=True)

        # Policy loss: maximize expected Q-value (minimize negative)
        actor_loss = -expected_q.mean()

        # Update actor
        self.lower_actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lower_policy.policy.parameters(), 1.0)
        self.lower_actor_optimizer.step()

        return actor_loss.item()

    def train_higher(self, batch_size: int = 64) -> float:
        """Train higher-level policy with TD3-style target smoothing and diverse sampling.

        Args:
            batch_size: Batch size for training

        Returns:
            Loss value
        """
        if len(self.higher_buffer) < batch_size:
            return 0.0

        # Sample batch with diversity: mix of recent and older experiences
        recent_size = batch_size // 2
        diverse_size = batch_size - recent_size

        # Recent experiences (last 20% of buffer)
        recent_start = max(0, len(self.higher_buffer) - len(self.higher_buffer) // 5)
        recent_batch = random.sample(
            list(self.higher_buffer)[recent_start:],
            min(recent_size, len(self.higher_buffer) - recent_start),
        )

        # Diverse sampling from entire buffer
        diverse_batch = random.sample(
            list(self.higher_buffer), min(diverse_size, len(self.higher_buffer))
        )

        batch = recent_batch + diverse_batch

        states = torch.stack([exp["state"] for exp in batch])
        goals = torch.stack([exp["goal"] for exp in batch])
        rewards = torch.tensor(
            [exp["reward"] for exp in batch], device=self.device, dtype=torch.float32
        )
        next_states = torch.stack([exp["next_state"] for exp in batch])
        dones = torch.tensor(
            [exp["done"] for exp in batch], device=self.device, dtype=torch.float32
        )

        # Current Q-value
        current_q = self.higher_policy.get_value(states, goals)

        # Target Q-value with policy smoothing (TD3-style)
        with torch.no_grad():
            # Get target policy action (next subgoal)
            next_goals = self.higher_target(next_states)

            # Add clipped Gaussian noise for target policy smoothing
            noise = torch.randn_like(next_goals) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_goals_noisy = next_goals + noise
            next_goals_noisy = torch.clamp(
                next_goals_noisy, -1.0, 1.0
            )  # Keep in valid range

            next_q = self.higher_target.get_value(next_states, next_goals_noisy)
            target_q = rewards.unsqueeze(1) + self.gamma * next_q * (
                1 - dones.unsqueeze(1)
            )

        # Compute loss
        loss = F.mse_loss(current_q, target_q)

        # Update critic
        self.higher_critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.higher_policy.critic.parameters(), 1.0)
        self.higher_critic_optimizer.step()

        return loss.item()

    def train_higher_actor(self, batch_size: int = 64) -> float:
        """Train higher-level actor using deterministic policy gradient.

        Args:
            batch_size: Batch size for training

        Returns:
            Loss value
        """
        if len(self.higher_buffer) < batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(list(self.higher_buffer), batch_size)

        states = torch.stack([exp["state"] for exp in batch])

        # Get goals from policy
        goals = self.higher_policy(states)

        # Compute Q-value for the proposed goals
        q_value = self.higher_policy.get_value(states, goals)

        # Policy loss: maximize Q-value (minimize negative)
        actor_loss = -q_value.mean()

        # Update actor
        self.higher_actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.higher_policy.network.parameters(), 1.0)
        self.higher_actor_optimizer.step()

        return actor_loss.item()

    def soft_update_targets(self) -> None:
        """Soft update of target networks."""
        for target_param, param in zip(
            self.higher_target.parameters(), self.higher_policy.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.lower_target.parameters(), self.lower_policy.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def train_episode(
        self, env: Any, max_steps: int = 1000, epsilon: float = 0.1
    ) -> dict[str, float]:
        """Train for one episode.

        Args:
            env: Environment to train in
            max_steps: Maximum steps per episode
            epsilon: Exploration parameter

        Returns:
            Dictionary with episode metrics
        """
        state, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        state = torch.FloatTensor(state).to(self.device)

        total_reward = 0.0
        steps = 0
        trajectory = [state]

        # Reset goal
        self.current_goal = None
        self.steps_since_goal = 0

        for step in range(max_steps):
            # Update goal if needed
            if self.steps_since_goal == 0:
                self.goal_state = state.clone()
                self.current_goal = self.select_goal(state)
                goal_start_step = step

            # Select and execute action
            action = self.select_action(
                state,
                self.current_goal
                if self.current_goal is not None
                else torch.zeros(self.goal_size, device=self.device),
                epsilon,
            )

            result = env.step(action)
            next_state, reward, done = result[0], result[1], result[2]
            next_state = torch.FloatTensor(next_state).to(self.device)

            total_reward += reward
            steps += 1
            trajectory.append(next_state)

            # Compute intrinsic reward for lower-level
            # Goals are deltas, so compare achieved delta to goal delta
            if self.goal_state is not None and self.current_goal is not None:
                achieved_delta = next_state - self.goal_state
                intrinsic = self.goal_distance(achieved_delta, self.current_goal)
            else:
                intrinsic = 0.0

            # Track intrinsic and extrinsic rewards for statistics
            self.intrinsic_rewards.append(intrinsic)
            self.extrinsic_rewards.append(reward)

            # Store lower-level experience
            self.lower_buffer.append(
                {
                    "state": state,
                    "action": action,
                    "reward": intrinsic,
                    "next_state": next_state,
                    "goal": self.current_goal
                    if self.current_goal is not None
                    else torch.zeros(self.goal_size, device=self.device),
                    "done": done,
                }
            )

            # Off-policy correction: also store with relabeled goal
            if len(trajectory) > goal_start_step + 1:
                relabeled_goal = self.relabel_goal(
                    self.goal_state if self.goal_state is not None else state,
                    trajectory[goal_start_step:],
                    self.goal_horizon,
                )
                # For relabeled goals, the intrinsic reward is based on
                # how close we got to the achieved delta
                current_delta = next_state - (
                    self.goal_state if self.goal_state is not None else state
                )
                relabeled_intrinsic = self.goal_distance(current_delta, relabeled_goal)
                self.lower_buffer.append(
                    {
                        "state": state,
                        "action": action,
                        "reward": relabeled_intrinsic,
                        "next_state": next_state,
                        "goal": relabeled_goal,
                        "done": done,
                    }
                )

            # Store higher-level experience (at goal horizon)
            self.steps_since_goal += 1
            if self.steps_since_goal == self.goal_horizon or done:
                if self.goal_state is not None and self.current_goal is not None:
                    self.higher_buffer.append(
                        {
                            "state": self.goal_state,
                            "goal": self.current_goal,
                            "reward": reward,
                            "next_state": next_state,
                            "done": done,
                        }
                    )
                self.steps_since_goal = 0

            # Train both levels
            # Train critics every step
            lower_critic_loss = self.train_lower(batch_size=64)
            higher_critic_loss = self.train_higher(batch_size=64)

            if lower_critic_loss > 0:
                self.lower_critic_losses.append(lower_critic_loss)
            if higher_critic_loss > 0:
                self.higher_critic_losses.append(higher_critic_loss)

            # Train actors less frequently (TD3-style delayed policy updates)
            if step % 2 == 0:
                lower_actor_loss = self.train_lower_actor(batch_size=64)
                higher_actor_loss = self.train_higher_actor(batch_size=64)

                if lower_actor_loss > 0:
                    self.lower_actor_losses.append(lower_actor_loss)
                if higher_actor_loss > 0:
                    self.higher_actor_losses.append(higher_actor_loss)

                # Soft update targets after actor updates
                self.soft_update_targets()

            state = next_state

            if done:
                break

        self.episode_rewards.append(total_reward)

        return {
            "reward": total_reward,
            "steps": steps,
            "avg_lower_critic_loss": (
                float(np.mean(self.lower_critic_losses[-100:]))
                if self.lower_critic_losses
                else 0.0
            ),
            "avg_higher_critic_loss": (
                float(np.mean(self.higher_critic_losses[-100:]))
                if self.higher_critic_losses
                else 0.0
            ),
            "avg_lower_actor_loss": (
                float(np.mean(self.lower_actor_losses[-100:]))
                if self.lower_actor_losses
                else 0.0
            ),
            "avg_higher_actor_loss": (
                float(np.mean(self.higher_actor_losses[-100:]))
                if self.higher_actor_losses
                else 0.0
            ),
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get agent statistics.

        Returns:
            Dictionary with statistics
        """
        # Compute intrinsic/extrinsic reward ratio
        recent_intrinsic: float = (
            float(np.mean(self.intrinsic_rewards[-1000:]))
            if self.intrinsic_rewards
            else 0.0
        )
        recent_extrinsic: float = (
            float(np.mean(self.extrinsic_rewards[-1000:]))
            if self.extrinsic_rewards
            else 0.0
        )
        reward_ratio = (
            abs(recent_intrinsic) / (abs(recent_extrinsic) + 1e-8)
            if recent_extrinsic != 0
            else 0.0
        )

        return {
            "total_episodes": len(self.episode_rewards),
            "avg_reward": (
                np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
            ),
            "avg_lower_critic_loss": (
                np.mean(self.lower_critic_losses[-100:])
                if self.lower_critic_losses
                else 0.0
            ),
            "avg_higher_critic_loss": (
                np.mean(self.higher_critic_losses[-100:])
                if self.higher_critic_losses
                else 0.0
            ),
            "avg_lower_actor_loss": (
                np.mean(self.lower_actor_losses[-100:])
                if self.lower_actor_losses
                else 0.0
            ),
            "avg_higher_actor_loss": (
                np.mean(self.higher_actor_losses[-100:])
                if self.higher_actor_losses
                else 0.0
            ),
            "avg_intrinsic_reward": recent_intrinsic,
            "avg_extrinsic_reward": recent_extrinsic,
            "intrinsic_extrinsic_ratio": reward_ratio,
            "goal_horizon": self.goal_horizon,
        }
