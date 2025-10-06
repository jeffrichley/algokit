"""HIRO: Data-Efficient Hierarchical Reinforcement Learning.

This module implements HIRO (Nachum et al., 2018), which uses goal-conditioned
hierarchical RL with off-policy correction. HIRO addresses the non-stationarity
problem in hierarchical RL through goal relabeling and off-policy learning.

Key features:
- Higher-level policy proposes goals for lower-level policy
- Lower-level policy learns goal-conditioned actions
- Off-policy correction through goal relabeling
- Hindsight experience replay for data efficiency

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


class HIROAgent:
    """HIRO agent with hierarchical goal-conditioned RL.

    The agent consists of:
    - Higher-level policy: Proposes goals at a slower timescale
    - Lower-level policy: Achieves goals through primitive actions
    - Off-policy correction: Relabels goals for data efficiency
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        goal_size: int = 16,
        hidden_size: int = 256,
        goal_horizon: int = 10,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        """Initialize HIRO agent.

        Args:
            state_size: Dimension of state space
            action_size: Number of primitive actions
            goal_size: Dimension of goal space
            hidden_size: Size of hidden layers
            goal_horizon: Steps between higher-level decisions
            learning_rate: Learning rate for networks
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            device: Device for computation
            seed: Random seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.state_size = state_size
        self.action_size = action_size
        self.goal_size = goal_size
        self.goal_horizon = goal_horizon
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)

        # Initialize higher-level policy
        self.higher_policy = HigherLevelPolicy(
            state_size=state_size, goal_size=goal_size, hidden_size=hidden_size
        ).to(self.device)

        self.higher_target = HigherLevelPolicy(
            state_size=state_size, goal_size=goal_size, hidden_size=hidden_size
        ).to(self.device)
        self.higher_target.load_state_dict(self.higher_policy.state_dict())

        # Initialize lower-level policy
        self.lower_policy = LowerLevelPolicy(
            state_size=state_size,
            action_size=action_size,
            goal_size=goal_size,
            hidden_size=hidden_size,
        ).to(self.device)

        self.lower_target = LowerLevelPolicy(
            state_size=state_size,
            action_size=action_size,
            goal_size=goal_size,
            hidden_size=hidden_size,
        ).to(self.device)
        self.lower_target.load_state_dict(self.lower_policy.state_dict())

        # Optimizers
        self.higher_optimizer = optim.Adam(
            self.higher_policy.parameters(), lr=learning_rate
        )
        self.lower_optimizer = optim.Adam(
            self.lower_policy.parameters(), lr=learning_rate
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
        self.higher_losses: list[float] = []
        self.lower_losses: list[float] = []

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
            Negative distance (higher is better)
        """
        # Use negative L2 distance as intrinsic reward
        dist = -torch.norm(state - goal).item()
        return dist

    def relabel_goal(
        self, trajectory: list[torch.Tensor], horizon: int
    ) -> torch.Tensor:
        """Relabel goal using hindsight (off-policy correction).

        Args:
            trajectory: List of states in trajectory
            horizon: Goal horizon

        Returns:
            Relabeled goal
        """
        # Use the actual achieved state as the relabeled goal
        if len(trajectory) >= horizon:
            return trajectory[horizon - 1]
        else:
            return trajectory[-1]

    def train_lower(self, batch_size: int = 64) -> float:
        """Train lower-level policy.

        Args:
            batch_size: Batch size for training

        Returns:
            Loss value
        """
        if len(self.lower_buffer) < batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(self.lower_buffer, batch_size)

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

        # Update
        self.lower_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lower_policy.parameters(), 1.0)
        self.lower_optimizer.step()

        return loss.item()

    def train_higher(self, batch_size: int = 64) -> float:
        """Train higher-level policy.

        Args:
            batch_size: Batch size for training

        Returns:
            Loss value
        """
        if len(self.higher_buffer) < batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(self.higher_buffer, batch_size)

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

        # Target Q-value
        with torch.no_grad():
            next_goals = self.higher_target(next_states)
            next_q = self.higher_target.get_value(next_states, next_goals)
            target_q = rewards.unsqueeze(1) + self.gamma * next_q * (
                1 - dones.unsqueeze(1)
            )

        # Compute loss
        loss = F.mse_loss(current_q, target_q)

        # Update
        self.higher_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.higher_policy.parameters(), 1.0)
        self.higher_optimizer.step()

        return loss.item()

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
            intrinsic = self.goal_distance(
                next_state,
                self.current_goal if self.current_goal is not None else state,
            )

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
            if len(trajectory) > 1:
                relabeled_goal = self.relabel_goal(
                    trajectory[goal_start_step:], self.goal_horizon
                )
                relabeled_intrinsic = self.goal_distance(next_state, relabeled_goal)
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
            lower_loss = self.train_lower(batch_size=64)
            higher_loss = self.train_higher(batch_size=64)

            if lower_loss > 0:
                self.lower_losses.append(lower_loss)
            if higher_loss > 0:
                self.higher_losses.append(higher_loss)

            # Soft update targets
            self.soft_update_targets()

            state = next_state

            if done:
                break

        self.episode_rewards.append(total_reward)

        return {
            "reward": total_reward,
            "steps": steps,
            "avg_lower_loss": (
                float(np.mean(self.lower_losses[-100:])) if self.lower_losses else 0.0
            ),
            "avg_higher_loss": (
                float(np.mean(self.higher_losses[-100:])) if self.higher_losses else 0.0
            ),
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get agent statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_episodes": len(self.episode_rewards),
            "avg_reward": (
                np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
            ),
            "avg_lower_loss": (
                np.mean(self.lower_losses[-100:]) if self.lower_losses else 0.0
            ),
            "avg_higher_loss": (
                np.mean(self.higher_losses[-100:]) if self.higher_losses else 0.0
            ),
            "goal_horizon": self.goal_horizon,
        }
