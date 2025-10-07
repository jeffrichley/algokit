"""Feudal Reinforcement Learning (FeudalNet).

This module implements Feudal Reinforcement Learning, which uses a hierarchical
structure with a Manager network that sets sub-goals and a Worker network that
learns to achieve those sub-goals.

The key idea is temporal abstraction through goal-setting: the Manager operates
at a slower timescale and provides direction, while the Worker operates at a
faster timescale to execute primitive actions.

Production-Quality Improvements:
- Shared state encoder for consistent latent space representations
- Proper temporal coordination (manager updates only at horizon intervals)
- Advantage normalization and entropy regularization for stability
- N-step bootstrapping for manager value targets
- Consistent device handling for all tensors
- Separate learning rates for manager (1e-4) and worker (3e-4) for optimal training
- KL divergence monitoring between consecutive goals for interpretability
- Gradient norm tracking for both manager and worker networks

Interpretability Features:
- Goal KL divergence: Tracks how much manager policy changes between updates
- Gradient norms: Monitors training dynamics and detects optimization issues
- Separate learning rates: Manager learns slower (stability), worker faster (adaptation)

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


class StateEncoder(nn.Module):
    """Shared state encoder for both manager and worker.

    Encodes raw states into a latent representation that can be used
    for goal-setting and intrinsic reward computation.
    """

    def __init__(
        self, state_size: int, latent_size: int, hidden_size: int = 256
    ) -> None:
        """Initialize state encoder.

        Args:
            state_size: Dimension of state space
            latent_size: Dimension of latent space
            hidden_size: Size of hidden layers
        """
        super().__init__()

        self.state_size = state_size
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state to latent representation.

        Args:
            state: Raw state

        Returns:
            Latent representation
        """
        return self.encoder(state)


class ManagerNetwork(nn.Module):
    """Manager network that sets sub-goals for the worker.

    The manager perceives states at a lower resolution/frequency and
    outputs goal vectors in a latent space.
    """

    def __init__(
        self, latent_size: int, goal_size: int, hidden_size: int = 256
    ) -> None:
        """Initialize manager network.

        Args:
            latent_size: Dimension of latent state representation
            goal_size: Dimension of goal space (latent representation)
            hidden_size: Size of hidden layers
        """
        super().__init__()

        self.latent_size = latent_size
        self.goal_size = goal_size

        # Goal generator from latent state
        self.goal_generator = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, goal_size),
        )

        # Value network for manager
        self.value = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, latent_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through manager network.

        Args:
            latent_state: Encoded state in latent space

        Returns:
            Tuple of (goal vector, state value)
        """
        # Generate goal (normalized direction in latent space)
        goal = self.goal_generator(latent_state)
        goal = F.normalize(goal, p=2, dim=-1)  # L2 normalization

        # Compute value
        value = self.value(latent_state)

        return goal, value


class WorkerNetwork(nn.Module):
    """Worker network that executes actions to achieve manager's goals.

    The worker receives the latent state and the manager's goal, and
    outputs a distribution over primitive actions.
    """

    def __init__(
        self,
        latent_size: int,
        action_size: int,
        goal_size: int,
        hidden_size: int = 256,
    ) -> None:
        """Initialize worker network.

        Args:
            latent_size: Dimension of latent state space
            action_size: Number of primitive actions
            goal_size: Dimension of goal space
            hidden_size: Size of hidden layers
        """
        super().__init__()

        self.latent_size = latent_size
        self.action_size = action_size
        self.goal_size = goal_size

        # Goal conditioned policy
        self.policy = nn.Sequential(
            nn.Linear(latent_size + goal_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

        # Value network for worker (conditioned on goal)
        self.value = nn.Sequential(
            nn.Linear(latent_size + goal_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self, latent_state: torch.Tensor, goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through worker network.

        Args:
            latent_state: Encoded state in latent space
            goal: Goal from manager

        Returns:
            Tuple of (action logits, state-goal value)
        """
        # Concatenate latent state and goal
        combined = torch.cat([latent_state, goal], dim=-1)

        # Get action logits and value
        logits = self.policy(combined)
        value = self.value(combined)

        return logits, value


class FeudalAgent:
    """Production-quality Feudal RL agent with hierarchical manager-worker structure.

    The agent consists of:
    - Shared state encoder: Maps raw states to latent representations
    - Manager: Sets abstract goals at a slower timescale
    - Worker: Achieves goals through primitive actions at a faster timescale

    Research-Grade Improvements:
    - Shared encoder for consistent intrinsic reward computation
    - Proper temporal coordination (manager updates at horizons only)
    - Advantage normalization and entropy regularization
    - N-step returns for manager training
    - Differential learning rates (manager: 1e-4, worker: 3e-4)
    - KL divergence tracking between consecutive goals
    - Gradient norm monitoring for interpretability

    The implementation follows the FeUdal Networks (FuN) architecture from
    Vezhnevets et al. (2017) with modern best practices for stable training.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        latent_size: int = 64,
        goal_size: int | None = None,
        hidden_size: int = 256,
        manager_horizon: int = 10,
        learning_rate: float = 0.0001,
        manager_lr: float | None = None,
        worker_lr: float | None = None,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        """Initialize Feudal RL agent.

        Args:
            state_size: Dimension of state space
            action_size: Number of primitive actions
            latent_size: Dimension of latent state encoding
            goal_size: Dimension of goal space (defaults to latent_size for intrinsic reward computation)
            hidden_size: Size of hidden layers
            manager_horizon: Number of steps between manager decisions
            learning_rate: Base learning rate (used if manager_lr/worker_lr not specified)
            manager_lr: Manager-specific learning rate (defaults to 1e-4 for stability)
            worker_lr: Worker-specific learning rate (defaults to 3e-4 for faster adaptation)
            gamma: Discount factor
            entropy_coef: Entropy regularization coefficient
            device: Device for computation
            seed: Random seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.state_size = state_size
        self.action_size = action_size
        self.latent_size = latent_size
        # Default goal_size to latent_size for intrinsic reward computation
        self.goal_size = goal_size if goal_size is not None else latent_size
        self.manager_horizon = manager_horizon
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = torch.device(device)

        # Set learning rates with recommended defaults
        self.manager_lr = manager_lr if manager_lr is not None else 1e-4
        self.worker_lr = worker_lr if worker_lr is not None else 3e-4

        # Initialize shared state encoder
        self.state_encoder = StateEncoder(
            state_size=state_size, latent_size=latent_size, hidden_size=hidden_size
        ).to(self.device)

        # Initialize manager
        self.manager = ManagerNetwork(
            latent_size=latent_size, goal_size=self.goal_size, hidden_size=hidden_size
        ).to(self.device)

        # Initialize worker
        self.worker = WorkerNetwork(
            latent_size=latent_size,
            action_size=action_size,
            goal_size=self.goal_size,
            hidden_size=hidden_size,
        ).to(self.device)

        # Goal projection for intrinsic reward (if goal_size != latent_size)
        self.goal_projection: nn.Linear | None
        if self.goal_size != latent_size:
            self.goal_projection = nn.Linear(self.goal_size, latent_size).to(
                self.device
            )
        else:
            self.goal_projection = None

        # Optimizers with separate learning rates for manager/worker
        # Manager learns slower for stability, worker learns faster for adaptation
        self.encoder_optimizer = optim.Adam(
            self.state_encoder.parameters(), lr=learning_rate
        )
        self.manager_optimizer = optim.Adam(
            self.manager.parameters(), lr=self.manager_lr
        )
        self.worker_optimizer = optim.Adam(self.worker.parameters(), lr=self.worker_lr)

        # Experience buffers
        self.manager_buffer: deque[dict[str, Any]] = deque(maxlen=10000)
        self.worker_buffer: deque[dict[str, Any]] = deque(maxlen=10000)

        # Current goal tracking
        self.current_goal: torch.Tensor | None = None
        self.previous_goal: torch.Tensor | None = None
        self.steps_since_goal_update = 0

        # Statistics
        self.episode_rewards: list[float] = []
        self.manager_losses: list[float] = []
        self.worker_losses: list[float] = []

        # Interpretability metrics
        self.goal_kl_divergences: list[float] = []
        self.manager_grad_norms: list[float] = []
        self.worker_grad_norms: list[float] = []

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state to latent representation.

        Args:
            state: Raw state tensor

        Returns:
            Latent state representation
        """
        return self.state_encoder(state)

    def compute_goal_kl_divergence(
        self, goal1: torch.Tensor, goal2: torch.Tensor
    ) -> float:
        """Compute KL divergence between two normalized goal vectors.

        Treats goals as pseudo-probability distributions over latent directions
        for interpretability of goal changes.

        Args:
            goal1: First goal vector (normalized)
            goal2: Second goal vector (normalized)

        Returns:
            KL divergence (symmetric)
        """
        # Convert to probability-like distributions using softmax
        # Add small epsilon to avoid numerical issues
        eps = 1e-8
        p1 = F.softmax(goal1 / 0.1, dim=-1) + eps  # temperature 0.1 for sharpness
        p2 = F.softmax(goal2 / 0.1, dim=-1) + eps

        # Compute symmetric KL divergence
        kl_div = float(
            (
                F.kl_div(p1.log(), p2, reduction="sum")
                + F.kl_div(p2.log(), p1, reduction="sum")
            ).item()
            / 2.0
        )

        return kl_div

    def select_action(
        self, latent_state: torch.Tensor, deterministic: bool = False
    ) -> int:
        """Select action using worker policy conditioned on current goal.

        Args:
            latent_state: Encoded latent state
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
                new_goal, _ = self.manager(latent_state.unsqueeze(0))
                new_goal = new_goal.squeeze(0)

                # Compute KL divergence between consecutive goals for interpretability
                if self.current_goal is not None:
                    kl_div = self.compute_goal_kl_divergence(
                        self.current_goal, new_goal
                    )
                    self.goal_kl_divergences.append(kl_div)

                # Update goal tracking
                self.previous_goal = self.current_goal
                self.current_goal = new_goal
                self.steps_since_goal_update = 0

        # Get action from worker
        with torch.no_grad():
            logits, _ = self.worker(
                latent_state.unsqueeze(0), self.current_goal.unsqueeze(0)
            )
            probs = F.softmax(logits, dim=-1).squeeze(0)

        if deterministic:
            action = int(probs.argmax().item())
        else:
            action = int(torch.multinomial(probs, 1).item())

        self.steps_since_goal_update += 1

        return action

    def intrinsic_reward(
        self,
        latent_state: torch.Tensor,
        next_latent_state: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Compute intrinsic reward for worker based on goal achievement.

        The intrinsic reward is the cosine similarity between the
        latent state transition direction and the goal direction.

        Args:
            latent_state: Current latent state
            next_latent_state: Next latent state
            goal: Current goal

        Returns:
            Intrinsic reward
        """
        # Compute latent state difference (direction of transition)
        latent_diff = next_latent_state - latent_state

        # Project goal to latent space if needed
        if self.goal_projection is not None:
            goal_projected = self.goal_projection(goal)
        else:
            goal_projected = goal

        # Normalize both vectors
        latent_diff_norm = F.normalize(latent_diff.unsqueeze(0), p=2, dim=-1).squeeze(0)
        goal_norm = F.normalize(goal_projected.unsqueeze(0), p=2, dim=-1).squeeze(0)

        # Cosine similarity
        intrinsic = float(torch.dot(latent_diff_norm, goal_norm).item())

        return intrinsic

    def train_worker(self, batch_size: int = 32) -> float:
        """Train worker using stored experiences with advantage normalization and entropy.

        Args:
            batch_size: Batch size for training

        Returns:
            Average loss
        """
        if len(self.worker_buffer) < batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(self.worker_buffer, batch_size)

        latent_states = torch.stack([exp["latent_state"] for exp in batch])
        actions = torch.tensor([exp["action"] for exp in batch], device=self.device)
        rewards = torch.tensor(
            [exp["reward"] for exp in batch], device=self.device, dtype=torch.float32
        )
        next_latent_states = torch.stack([exp["next_latent_state"] for exp in batch])
        goals = torch.stack([exp["goal"] for exp in batch])
        dones = torch.tensor(
            [exp["done"] for exp in batch], device=self.device, dtype=torch.float32
        )

        # Get current Q-values and action probabilities
        logits, values = self.worker(latent_states, goals)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute entropy for regularization
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # Get target values
        with torch.no_grad():
            _, next_values = self.worker(next_latent_states, goals)
            targets = rewards + self.gamma * next_values.squeeze(1) * (1 - dones)

        # Compute advantages with normalization
        advantages = targets - values.squeeze(1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss (actor) with entropy bonus
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        policy_loss = policy_loss - self.entropy_coef * entropy

        # Value loss (critic)
        value_loss = F.mse_loss(values.squeeze(1), targets)

        # Total loss
        loss = policy_loss + 0.5 * value_loss

        # Update worker and encoder
        self.worker_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        loss.backward()

        # Track gradient norms for interpretability (before clipping)
        worker_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.worker.parameters(), 0.5).item()
        )
        self.worker_grad_norms.append(worker_grad_norm)

        torch.nn.utils.clip_grad_norm_(self.state_encoder.parameters(), 0.5)
        self.worker_optimizer.step()
        self.encoder_optimizer.step()

        return loss.item()

    def train_manager(self, batch_size: int = 32) -> float:
        """Train manager using n-step returns.

        Args:
            batch_size: Batch size for training

        Returns:
            Average loss
        """
        if len(self.manager_buffer) < batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(self.manager_buffer, batch_size)

        latent_states = torch.stack([exp["latent_state"] for exp in batch])
        n_step_returns = torch.tensor(
            [exp["n_step_return"] for exp in batch],
            device=self.device,
            dtype=torch.float32,
        )
        next_latent_states = torch.stack([exp["next_latent_state"] for exp in batch])
        dones = torch.tensor(
            [exp["done"] for exp in batch], device=self.device, dtype=torch.float32
        )

        # Get current values and goals
        goals, values = self.manager(latent_states)

        # Get target values using n-step returns
        with torch.no_grad():
            _, next_values = self.manager(next_latent_states)
            # N-step return already computed, just add bootstrapped value
            targets = n_step_returns + (
                self.gamma**self.manager_horizon * next_values.squeeze(1) * (1 - dones)
            )

        # Value loss
        loss = F.mse_loss(values.squeeze(1), targets)

        # Update manager
        self.manager_optimizer.zero_grad()
        loss.backward()

        # Track gradient norms for interpretability (before clipping)
        manager_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.manager.parameters(), 0.5).item()
        )
        self.manager_grad_norms.append(manager_grad_norm)

        self.manager_optimizer.step()

        return loss.item()

    def compute_n_step_return(self, rewards: list[float], gamma: float) -> float:
        """Compute n-step discounted return.

        Args:
            rewards: List of rewards over the horizon
            gamma: Discount factor

        Returns:
            N-step discounted return
        """
        n_step_return = 0.0
        for i, reward in enumerate(rewards):
            n_step_return += (gamma**i) * reward
        return n_step_return

    def train_episode(self, env: Any, max_steps: int = 1000) -> dict[str, float]:
        """Train for one episode with proper temporal coordination.

        Args:
            env: Environment to train in
            max_steps: Maximum steps per episode

        Returns:
            Dictionary with episode metrics
        """
        # Reset environment and ensure tensor is on correct device
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        state = torch.FloatTensor(state).to(self.device)

        total_reward = 0.0
        steps = 0
        manager_updates = 0

        # Reset goal
        self.current_goal = None
        self.steps_since_goal_update = 0

        # Track horizon rewards for n-step returns
        horizon_rewards: list[float] = []
        horizon_start_latent_state: torch.Tensor | None = None

        for _ in range(max_steps):
            # Encode state
            with torch.no_grad():
                latent_state = self.encode_state(state)

            # Store latent state at start of horizon
            if self.steps_since_goal_update == 0:
                horizon_start_latent_state = latent_state.clone()
                horizon_rewards = []

            # Select and execute action
            action = self.select_action(latent_state)

            result = env.step(action)
            next_state, reward, done = result[0], result[1], result[2]
            next_state = torch.FloatTensor(next_state).to(self.device)

            total_reward += reward
            steps += 1
            horizon_rewards.append(reward)

            # Encode next state
            with torch.no_grad():
                next_latent_state = self.encode_state(next_state)

            # Compute intrinsic reward for worker
            intrinsic = self.intrinsic_reward(
                latent_state,
                next_latent_state,
                self.current_goal
                if self.current_goal is not None
                else torch.zeros(self.goal_size, device=self.device),
            )

            # Store experience for worker
            self.worker_buffer.append(
                {
                    "latent_state": latent_state,
                    "action": action,
                    "reward": intrinsic,  # Use intrinsic reward
                    "next_latent_state": next_latent_state,
                    "goal": (
                        self.current_goal
                        if self.current_goal is not None
                        else torch.zeros(self.goal_size, device=self.device)
                    ),
                    "done": done,
                }
            )

            # Store experience for manager at horizon intervals
            if (
                self.steps_since_goal_update == self.manager_horizon or done
            ) and horizon_start_latent_state is not None:
                # Compute n-step return
                n_step_return = self.compute_n_step_return(horizon_rewards, self.gamma)

                self.manager_buffer.append(
                    {
                        "latent_state": horizon_start_latent_state,
                        "n_step_return": n_step_return,
                        "next_latent_state": next_latent_state,
                        "done": done,
                    }
                )
                manager_updates += 1

            # Train worker every step
            worker_loss = self.train_worker(batch_size=32)
            if worker_loss > 0:
                self.worker_losses.append(worker_loss)

            # Train manager only at horizon intervals
            if self.steps_since_goal_update == 0 and manager_updates > 0:
                manager_loss = self.train_manager(batch_size=32)
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
            # Interpretability metrics from this episode
            "avg_goal_kl_divergence": (
                float(np.mean(self.goal_kl_divergences[-steps:]))
                if len(self.goal_kl_divergences) >= steps
                else 0.0
            ),
            "avg_worker_grad_norm": (
                float(np.mean(self.worker_grad_norms[-steps:]))
                if len(self.worker_grad_norms) >= steps
                else 0.0
            ),
            "avg_manager_grad_norm": (
                float(np.mean(self.manager_grad_norms[-manager_updates:]))
                if len(self.manager_grad_norms) >= manager_updates
                else 0.0
            ),
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get agent statistics including interpretability metrics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_episodes": len(self.episode_rewards),
            "avg_reward": (
                float(np.mean(self.episode_rewards[-100:]))
                if self.episode_rewards
                else 0.0
            ),
            "avg_worker_loss": (
                float(np.mean(self.worker_losses[-100:])) if self.worker_losses else 0.0
            ),
            "avg_manager_loss": (
                float(np.mean(self.manager_losses[-100:]))
                if self.manager_losses
                else 0.0
            ),
            "manager_horizon": self.manager_horizon,
            "manager_lr": self.manager_lr,
            "worker_lr": self.worker_lr,
            # Interpretability metrics
            "avg_goal_kl_divergence": (
                float(np.mean(self.goal_kl_divergences[-100:]))
                if self.goal_kl_divergences
                else 0.0
            ),
            "avg_manager_grad_norm": (
                float(np.mean(self.manager_grad_norms[-100:]))
                if self.manager_grad_norms
                else 0.0
            ),
            "avg_worker_grad_norm": (
                float(np.mean(self.worker_grad_norms[-100:]))
                if self.worker_grad_norms
                else 0.0
            ),
        }
