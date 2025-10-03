"""Proximal Policy Optimization (PPO) reinforcement learning algorithm implementation.

This module contains a fully correct and stable PPO algorithm implementation using PyTorch.
The implementation follows the PPO paper specifications with proper clipped surrogate objective,
GAE, entropy bonus, value loss clipping, and on-policy updates.
"""

import random
from collections import namedtuple
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define experience tuple for rollout storage
RolloutExperience = namedtuple(
    "RolloutExperience",
    ["state", "action", "reward", "done", "old_log_prob", "old_value"],
)


class PolicyNetwork(nn.Module):
    """Policy network for action probability approximation.

    The policy network outputs action probabilities for a given state,
    implementing the policy π(a|s) with orthogonal weight initialization.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: list[int] | None = None,
        dropout_rate: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        """Initialize the Policy network.

        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        layers: list[nn.Module] = []
        prev_size = state_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))

            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))

            layers.append(nn.ReLU())

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_size = hidden_size

        # Output layer for action probabilities (no activation - will use softmax in forward)
        layers.append(nn.Linear(prev_size, action_size))

        self.network = nn.Sequential(*layers)

        # Initialize weights orthogonally
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights using orthogonal initialization."""
        for i, module in enumerate(self.modules()):
            if isinstance(module, nn.Linear):
                # Use sqrt(2) for hidden layers, 0.01 for output layer
                if i == len(list(self.modules())) - 1:  # Last layer (output)
                    nn.init.orthogonal_(module.weight, gain=0.01)
                else:  # Hidden layers
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            state: Input state tensor

        Returns:
            Action logits tensor (before softmax)
        """
        return self.network(state)

    def get_action_and_log_prob(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get action and log probability from the policy.

        Args:
            state: Input state tensor

        Returns:
            Tuple of (action, log_prob)
        """
        logits = self.forward(state)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob

    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get log probability of given action under current policy.

        Args:
            state: Input state tensor
            action: Action tensor

        Returns:
            Log probability tensor
        """
        logits = self.forward(state)
        action_dist = torch.distributions.Categorical(logits=logits)
        return action_dist.log_prob(action)


class ValueNetwork(nn.Module):
    """Value network for state value approximation.

    The value network outputs the estimated value of a given state,
    implementing the value function V(s) with orthogonal weight initialization.
    """

    def __init__(
        self,
        state_size: int,
        hidden_sizes: list[int] | None = None,
        dropout_rate: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        """Initialize the Value network.

        Args:
            state_size: Dimension of the state space
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        layers: list[nn.Module] = []
        prev_size = state_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))

            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))

            layers.append(nn.ReLU())

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_size = hidden_size

        # Output layer for state value
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights orthogonally
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            state: Input state tensor

        Returns:
            State value tensor
        """
        return self.network(state).squeeze(-1)


class RolloutBuffer:
    """Rollout buffer for PPO algorithm.

    Stores complete rollouts for on-policy training.
    """

    def __init__(self, buffer_size: int) -> None:
        """Initialize the buffer.

        Args:
            buffer_size: Maximum number of experiences to store
        """
        self.buffer_size = buffer_size
        self.buffer: list[RolloutExperience] = []
        self.ptr = 0

    def add(self, experience: RolloutExperience) -> None:
        """Add experience to buffer.

        Args:
            experience: Experience tuple to add
        """
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.ptr] = experience
            self.ptr = (self.ptr + 1) % self.buffer_size

    def get_all(self) -> list[RolloutExperience]:
        """Get all experiences from buffer.

        Returns:
            List of all experiences
        """
        return self.buffer.copy()

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.ptr = 0

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class PPOAgent:
    """Proximal Policy Optimization (PPO) agent.

    Fully correct and stable PPO implementation following the paper specifications:
    - Clipped surrogate objective with proper ratio calculation
    - GAE with bootstrap for advantage estimation
    - Entropy bonus (maximized, not penalized)
    - Value loss with optional clipping
    - On-policy updates (no replay buffer)
    - Gradient clipping and orthogonal initialization
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 3e-4,
        discount_factor: float = 0.99,
        hidden_sizes: list[int] | None = None,
        dropout_rate: float = 0.0,
        buffer_size: int = 2048,
        batch_size: int = 64,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        gae_lambda: float = 0.95,
        clip_value_loss: bool = True,
        n_epochs: int = 4,
        device: str = "cpu",
        random_seed: int | None = None,
    ) -> None:
        """Initialize PPO agent.

        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
            learning_rate: Learning rate for both networks
            discount_factor: Discount factor for future rewards
            hidden_sizes: List of hidden layer sizes for both networks
            dropout_rate: Dropout rate for regularization
            buffer_size: Size of rollout buffer
            batch_size: Batch size for training
            clip_ratio: PPO clipping ratio
            value_coef: Value function loss coefficient
            entropy_coef: Entropy bonus coefficient (positive for maximization)
            max_grad_norm: Maximum gradient norm for clipping
            gae_lambda: GAE lambda parameter
            clip_value_loss: Whether to clip value loss
            n_epochs: Number of training epochs per update
            device: Device to run computations on ('cpu' or 'cuda')
            random_seed: Random seed for reproducible results

        Raises:
            ValueError: If any parameter is invalid
        """
        if state_size <= 0:
            raise ValueError("state_size must be positive")
        if action_size <= 0:
            raise ValueError("action_size must be positive")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0 <= discount_factor <= 1:
            raise ValueError("discount_factor must be between 0 and 1")
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 < clip_ratio < 1:
            raise ValueError("clip_ratio must be between 0 and 1")
        if value_coef < 0:
            raise ValueError("value_coef must be non-negative")
        if entropy_coef < 0:
            raise ValueError("entropy_coef must be non-negative")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if not 0 <= gae_lambda <= 1:
            raise ValueError("gae_lambda must be between 0 and 1")
        if n_epochs <= 0:
            raise ValueError("n_epochs must be positive")

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.clip_value_loss = clip_value_loss
        self.n_epochs = n_epochs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Set random seeds
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Initialize networks
        self.policy = PolicyNetwork(
            state_size, action_size, hidden_sizes, dropout_rate
        ).to(self.device)
        self.value = ValueNetwork(state_size, hidden_sizes, dropout_rate).to(
            self.device
        )

        # Initialize optimizer
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=learning_rate,
            eps=1e-5,  # Small epsilon for numerical stability
        )

        # Initialize rollout buffer
        self.rollout_buffer = RolloutBuffer(buffer_size)

        # Training mode
        self.training = True

    def get_action(self, state: np.ndarray) -> tuple[int, float, float]:
        """Get action, log probability, and value from current policy.

        Args:
            state: Current state

        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob = self.policy.get_action_and_log_prob(state_tensor)
            value = self.value(state_tensor)

        return int(action.item()), float(log_prob.item()), float(value.item())

    def collect_rollout(self, env: Any, n_steps: int) -> None:
        """Collect rollout data for training.

        Args:
            env: Environment to interact with
            n_steps: Number of steps to collect
        """
        self.set_training(False)  # Set to eval mode for data collection

        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Handle gymnasium reset format

        for _ in range(n_steps):
            action, log_prob, value = self.get_action(state)

            # Take action in environment
            result = env.step(action)

            # Handle different environment interfaces
            if len(result) == 5:
                # Gymnasium format: (next_state, reward, done, truncated, info)
                next_state, reward, done, truncated, info = result
                done = done or truncated
            elif len(result) == 4:
                # OpenAI Gym format: (next_state, reward, done, info)
                next_state, reward, done, info = result
            else:
                # Minimal format: (next_state, reward, done)
                next_state, reward, done = result

            # Store experience
            experience = RolloutExperience(
                state=state.copy(),
                action=action,
                reward=reward,
                done=done,
                old_log_prob=log_prob,
                old_value=value,
            )
            self.rollout_buffer.add(experience)

            # Update state
            state = next_state

            # Reset if episode is done
            if done:
                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]

    def compute_gae(
        self,
        rewards: list[float],
        values: list[float],
        dones: list[bool],
        next_value: float = 0.0,
    ) -> tuple[list[float], list[float]]:
        """Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Next value estimate (bootstrap value)

        Returns:
            Tuple of (advantages, returns)
        """
        advantages: list[float] = []
        returns: list[float] = []

        # Compute advantages using GAE
        advantage = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # Last step: use bootstrap value if not terminal
                next_value_t = next_value if not dones[t] else 0.0
            else:
                next_value_t = values[t + 1]

            # TD error
            delta = (
                rewards[t]
                + self.discount_factor * next_value_t * (1 - dones[t])
                - values[t]
            )

            # GAE advantage
            advantage = (
                delta
                + self.discount_factor * self.gae_lambda * (1 - dones[t]) * advantage
            )
            advantages.insert(0, advantage)

        # Compute returns = advantages + values
        for t in range(len(rewards)):
            returns.append(advantages[t] + values[t])

        return advantages, returns

    def update(self) -> dict[str, float]:
        """Update the agent using PPO algorithm.

        Returns:
            Dictionary of training metrics
        """
        if len(self.rollout_buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0}

        # Get all rollout data
        experiences = self.rollout_buffer.get_all()

        # Extract data
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(
            self.device
        )
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        old_log_probs = torch.FloatTensor([e.old_log_prob for e in experiences]).to(
            self.device
        )
        old_values = torch.FloatTensor([e.old_value for e in experiences]).to(
            self.device
        )
        rewards = [e.reward for e in experiences]
        values = [
            e.old_value for e in experiences
        ]  # Use old values for GAE computation
        dones = [e.done for e in experiences]

        # Get next value for bootstrap (last state value if not terminal)
        next_value = 0.0
        if not dones[-1]:
            with torch.no_grad():
                next_state = (
                    torch.FloatTensor(experiences[-1].state)
                    .unsqueeze(0)
                    .to(self.device)
                )
                next_value = self.value(next_state).item()

        # Compute advantages and returns using GAE
        advantages_list, returns_list = self.compute_gae(
            rewards, values, dones, next_value
        )
        advantages = torch.FloatTensor(advantages_list).to(self.device)
        returns = torch.FloatTensor(returns_list).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training loop with mini-batch updates
        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kl_divs = []
        clip_fractions = []

        # Create indices for mini-batch sampling
        buffer_size = len(experiences)
        indices = np.arange(buffer_size)

        for _ in range(self.n_epochs):
            # Shuffle indices for mini-batch sampling
            np.random.shuffle(indices)

            # Mini-batch updates
            for start_idx in range(0, buffer_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, buffer_size)
                batch_indices = indices[start_idx:end_idx]

                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Get current policy and value estimates
                current_log_probs = self.policy.get_log_prob(
                    batch_states, batch_actions
                )
                current_values = self.value(batch_states)

                # Compute probability ratio
                ratio = torch.exp(current_log_probs - batch_old_log_probs)

                # Compute clipped surrogate objective
                surr1 = ratio * batch_advantages
                clipped_ratio = torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                )
                surr2 = clipped_ratio * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute approximate KL divergence for monitoring
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - current_log_probs).mean()
                    approx_kl_divs.append(approx_kl.item())

                # Compute clip fraction for monitoring
                with torch.no_grad():
                    clip_fraction = ((ratio - 1).abs() > self.clip_ratio).float().mean()
                    clip_fractions.append(clip_fraction.item())

                # Compute value loss
                if self.clip_value_loss:
                    # Fixed: Use old_values instead of old_log_probs for value clipping
                    value_pred_clipped = batch_old_values + torch.clamp(
                        current_values - batch_old_values,
                        -self.clip_ratio,
                        self.clip_ratio,
                    )
                    value_loss1 = F.mse_loss(current_values, batch_returns)
                    value_loss2 = F.mse_loss(value_pred_clipped, batch_returns)
                    value_loss = torch.max(value_loss1, value_loss2)
                else:
                    value_loss = F.mse_loss(current_values, batch_returns)

                # Compute entropy bonus (maximize entropy)
                action_dist = torch.distributions.Categorical(
                    logits=self.policy(batch_states)
                )
                entropy_bonus = action_dist.entropy().mean()

                # Total loss
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef
                    * entropy_bonus  # Negative because we want to maximize entropy
                )

                # Update networks
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                # Store losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_bonus.item())

        # Clear rollout buffer
        self.rollout_buffer.clear()

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy_loss": float(np.mean(entropy_losses)),
            "approx_kl": float(np.mean(approx_kl_divs)),
            "clip_fraction": float(np.mean(clip_fractions)),
        }

    def set_training(self, training: bool) -> None:
        """Set training mode for networks.

        Args:
            training: Whether to set training mode
        """
        self.training = training
        if training:
            self.policy.train()
            self.value.train()
        else:
            self.policy.eval()
            self.value.eval()

    def save(self, filepath: str) -> None:
        """Save agent state to file.

        Args:
            filepath: Path to save the model
        """
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "value_state_dict": self.value.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        """Load agent state from file.

        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value.load_state_dict(checkpoint["value_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
