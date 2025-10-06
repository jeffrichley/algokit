"""Feudal Reinforcement Learning (FeudalNet).

This module implements Feudal Reinforcement Learning, which uses a hierarchical
structure with a Manager network that sets sub-goals and a Worker network that
learns to achieve those sub-goals.

The key idea is temporal abstraction through goal-setting: the Manager operates
at a slower timescale and provides direction, while the Worker operates at a
faster timescale to execute primitive actions.

References:
    Dayan, P., & Hinton, G. E. (1993). Feudal reinforcement learning.
    Vezhnevets, A. S., et al. (2017). FeUdal Networks for Hierarchical
    Reinforcement Learning. ICML 2017.
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


class ManagerNetwork(nn.Module):
    """Manager network that sets sub-goals for the worker.

    The manager perceives states at a lower resolution/frequency and
    outputs goal vectors in a latent space.
    """

    def __init__(self, state_size: int, goal_size: int, hidden_size: int = 256) -> None:
        """Initialize manager network.

        Args:
            state_size: Dimension of state space
            goal_size: Dimension of goal space (latent representation)
            hidden_size: Size of hidden layers
        """
        super().__init__()

        self.state_size = state_size
        self.goal_size = goal_size

        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Goal generator (outputs direction in latent space)
        self.goal_generator = nn.Linear(hidden_size, goal_size)

        # Value network for manager
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through manager network.

        Args:
            state: Current state

        Returns:
            Tuple of (goal vector, state value)
        """
        encoded = self.encoder(state)

        # Generate goal (normalized direction in latent space)
        goal = self.goal_generator(encoded)
        goal = F.normalize(goal, p=2, dim=-1)  # L2 normalization

        # Compute value
        value = self.value(encoded)

        return goal, value


class WorkerNetwork(nn.Module):
    """Worker network that executes actions to achieve manager's goals.

    The worker receives the current state and the manager's goal, and
    outputs a distribution over primitive actions.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        goal_size: int,
        hidden_size: int = 256,
    ) -> None:
        """Initialize worker network.

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

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
        )

        # Goal conditioned policy
        self.policy = nn.Sequential(
            nn.Linear(hidden_size + goal_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

        # Value network for worker (conditioned on goal)
        self.value = nn.Sequential(
            nn.Linear(hidden_size + goal_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self, state: torch.Tensor, goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through worker network.

        Args:
            state: Current state
            goal: Goal from manager

        Returns:
            Tuple of (action logits, state-goal value)
        """
        state_encoded = self.state_encoder(state)

        # Concatenate state and goal
        combined = torch.cat([state_encoded, goal], dim=-1)

        # Get action logits and value
        logits = self.policy(combined)
        value = self.value(combined)

        return logits, value


class FeudalAgent:
    """Feudal RL agent with hierarchical manager-worker structure.

    The agent consists of:
    - Manager: Sets abstract goals at a slower timescale
    - Worker: Achieves goals through primitive actions at a faster timescale
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        goal_size: int = 16,
        hidden_size: int = 256,
        manager_horizon: int = 10,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        """Initialize Feudal RL agent.

        Args:
            state_size: Dimension of state space
            action_size: Number of primitive actions
            goal_size: Dimension of goal/latent space
            hidden_size: Size of hidden layers
            manager_horizon: Number of steps between manager decisions
            learning_rate: Learning rate for both networks
            gamma: Discount factor
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
        self.manager_horizon = manager_horizon
        self.gamma = gamma
        self.device = torch.device(device)

        # Initialize networks
        self.manager = ManagerNetwork(
            state_size=state_size, goal_size=goal_size, hidden_size=hidden_size
        ).to(self.device)

        self.worker = WorkerNetwork(
            state_size=state_size,
            action_size=action_size,
            goal_size=goal_size,
            hidden_size=hidden_size,
        ).to(self.device)

        # Optimizers
        self.manager_optimizer = optim.Adam(self.manager.parameters(), lr=learning_rate)
        self.worker_optimizer = optim.Adam(self.worker.parameters(), lr=learning_rate)

        # Experience buffers
        self.manager_buffer: deque[dict[str, Any]] = deque(maxlen=10000)
        self.worker_buffer: deque[dict[str, Any]] = deque(maxlen=10000)

        # Current goal
        self.current_goal: torch.Tensor | None = None
        self.steps_since_goal_update = 0

        # Statistics
        self.episode_rewards: list[float] = []
        self.manager_losses: list[float] = []
        self.worker_losses: list[float] = []

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> int:
        """Select action using worker policy conditioned on current goal.

        Args:
            state: Current state
            deterministic: Whether to select deterministically

        Returns:
            Selected action
        """
        # Update goal if needed
        if (
            self.current_goal is None
            or self.steps_since_goal_update >= self.manager_horizon
        ):
            with torch.no_grad():
                self.current_goal, _ = self.manager(state.unsqueeze(0))
                self.current_goal = self.current_goal.squeeze(0)
                self.steps_since_goal_update = 0

        # Get action from worker
        with torch.no_grad():
            logits, _ = self.worker(state.unsqueeze(0), self.current_goal.unsqueeze(0))
            probs = F.softmax(logits, dim=-1).squeeze(0)

        if deterministic:
            action = int(probs.argmax().item())
        else:
            action = int(torch.multinomial(probs, 1).item())

        self.steps_since_goal_update += 1

        return action

    def intrinsic_reward(
        self, state: torch.Tensor, next_state: torch.Tensor, goal: torch.Tensor
    ) -> float:
        """Compute intrinsic reward for worker based on goal achievement.

        The intrinsic reward is the cosine similarity between the
        state transition direction and the goal direction.

        Args:
            state: Current state
            next_state: Next state
            goal: Current goal

        Returns:
            Intrinsic reward
        """
        # Compute state difference (direction of transition)
        state_diff = next_state - state

        # Normalize
        state_diff_norm = F.normalize(state_diff.unsqueeze(0), p=2, dim=-1).squeeze(0)
        goal_norm = F.normalize(goal.unsqueeze(0), p=2, dim=-1).squeeze(0)

        # Cosine similarity
        intrinsic = float(torch.dot(state_diff_norm, goal_norm).item())

        return intrinsic

    def train_worker(self, batch_size: int = 32) -> float:
        """Train worker using stored experiences.

        Args:
            batch_size: Batch size for training

        Returns:
            Average loss
        """
        if len(self.worker_buffer) < batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(self.worker_buffer, batch_size)

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

        # Get current Q-values
        logits, values = self.worker(states, goals)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get target values
        with torch.no_grad():
            _, next_values = self.worker(next_states, goals)
            targets = rewards + self.gamma * next_values.squeeze(1) * (1 - dones)

        # Compute advantages
        advantages = targets - values.squeeze(1)

        # Policy loss (actor)
        policy_loss = -(action_log_probs * advantages.detach()).mean()

        # Value loss (critic)
        value_loss = F.mse_loss(values.squeeze(1), targets)

        # Total loss
        loss = policy_loss + 0.5 * value_loss

        # Update worker
        self.worker_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.worker.parameters(), 0.5)
        self.worker_optimizer.step()

        return loss.item()

    def train_manager(self, batch_size: int = 32) -> float:
        """Train manager using stored experiences.

        Args:
            batch_size: Batch size for training

        Returns:
            Average loss
        """
        if len(self.manager_buffer) < batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(self.manager_buffer, batch_size)

        states = torch.stack([exp["state"] for exp in batch])
        rewards = torch.tensor(
            [exp["reward"] for exp in batch], device=self.device, dtype=torch.float32
        )
        next_states = torch.stack([exp["next_state"] for exp in batch])
        dones = torch.tensor(
            [exp["done"] for exp in batch], device=self.device, dtype=torch.float32
        )

        # Get current values and goals
        goals, values = self.manager(states)

        # Get target values
        with torch.no_grad():
            _, next_values = self.manager(next_states)
            targets = rewards + self.gamma * next_values.squeeze(1) * (1 - dones)

        # Value loss
        loss = F.mse_loss(values.squeeze(1), targets)

        # Update manager
        self.manager_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.manager.parameters(), 0.5)
        self.manager_optimizer.step()

        return loss.item()

    def train_episode(self, env: Any, max_steps: int = 1000) -> dict[str, float]:
        """Train for one episode.

        Args:
            env: Environment to train in
            max_steps: Maximum steps per episode

        Returns:
            Dictionary with episode metrics
        """
        state, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        state = torch.FloatTensor(state).to(self.device)

        total_reward = 0.0
        steps = 0
        manager_updates = 0

        # Reset goal
        self.current_goal = None
        self.steps_since_goal_update = 0

        for _ in range(max_steps):
            # Store state for manager if this is a goal update step
            if self.steps_since_goal_update == 0:
                manager_state = state.clone()

            # Select and execute action
            action = self.select_action(state)

            result = env.step(action)
            next_state, reward, done = result[0], result[1], result[2]
            next_state = torch.FloatTensor(next_state).to(self.device)

            total_reward += reward
            steps += 1

            # Compute intrinsic reward for worker
            intrinsic = self.intrinsic_reward(
                state,
                next_state,
                self.current_goal
                if self.current_goal is not None
                else torch.zeros(self.goal_size, device=self.device),
            )

            # Store experience for worker
            self.worker_buffer.append(
                {
                    "state": state,
                    "action": action,
                    "reward": intrinsic,  # Use intrinsic reward
                    "next_state": next_state,
                    "goal": self.current_goal
                    if self.current_goal is not None
                    else torch.zeros(self.goal_size, device=self.device),
                    "done": done,
                }
            )

            # Store experience for manager (at goal horizon intervals)
            if self.steps_since_goal_update == self.manager_horizon or done:
                self.manager_buffer.append(
                    {
                        "state": manager_state,
                        "reward": reward,  # Use extrinsic reward
                        "next_state": next_state,
                        "done": done,
                    }
                )
                manager_updates += 1

            # Train both networks
            worker_loss = self.train_worker(batch_size=32)
            manager_loss = self.train_manager(batch_size=32)

            if worker_loss > 0:
                self.worker_losses.append(worker_loss)
            if manager_loss > 0:
                self.manager_losses.append(manager_loss)

            state = next_state

            if done:
                break

        self.episode_rewards.append(total_reward)

        return {
            "reward": total_reward,
            "steps": steps,
            "manager_updates": manager_updates,
            "avg_worker_loss": (
                float(np.mean(self.worker_losses[-100:])) if self.worker_losses else 0.0
            ),
            "avg_manager_loss": (
                float(np.mean(self.manager_losses[-100:]))
                if self.manager_losses
                else 0.0
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
            "avg_worker_loss": (
                np.mean(self.worker_losses[-100:]) if self.worker_losses else 0.0
            ),
            "avg_manager_loss": (
                np.mean(self.manager_losses[-100:]) if self.manager_losses else 0.0
            ),
            "manager_horizon": self.manager_horizon,
        }
