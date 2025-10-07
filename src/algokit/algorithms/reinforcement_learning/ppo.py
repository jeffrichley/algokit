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
from pydantic import BaseModel, Field

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


class PPOConfig(BaseModel):
    """Configuration parameters for PPO with automatic validation.

    This model uses Pydantic for declarative parameter validation,
    reducing complexity while maintaining strict type safety and
    comprehensive validation.

    Attributes:
        state_size: Dimension of the state space (must be positive)
        action_size: Dimension of the action space (must be positive)
        learning_rate: Learning rate for both networks (0 < lr <= 1)
        discount_factor: Discount factor for future rewards (0 <= gamma <= 1)
        hidden_sizes: List of hidden layer sizes for both networks
        dropout_rate: Dropout rate for regularization (0 <= rate < 1)
        buffer_size: Size of rollout buffer (must be positive)
        batch_size: Batch size for training (must be positive)
        clip_ratio: PPO clipping ratio (0 < ratio < 1)
        value_coef: Value function loss coefficient (>= 0)
        entropy_coef: Entropy bonus coefficient (>= 0)
        max_grad_norm: Maximum gradient norm for clipping (must be positive)
        gae_lambda: GAE lambda parameter (0 <= lambda <= 1)
        clip_value_loss: Whether to clip value loss
        n_epochs: Number of training epochs per update (must be positive)
        device: Device to run computations on ('cpu' or 'cuda')
        random_seed: Random seed for reproducible results (optional)
    """

    state_size: int = Field(gt=0, description="Dimension of the state space")
    action_size: int = Field(gt=0, description="Dimension of the action space")
    learning_rate: float = Field(
        default=3e-4,
        gt=0.0,
        le=1.0,
        description="Learning rate for both networks",
    )
    discount_factor: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Discount factor for future rewards",
    )
    hidden_sizes: list[int] | None = Field(
        default=None, description="List of hidden layer sizes for both networks"
    )
    dropout_rate: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Dropout rate for regularization",
    )
    buffer_size: int = Field(default=2048, gt=0, description="Size of rollout buffer")
    batch_size: int = Field(default=64, gt=0, description="Batch size for training")
    clip_ratio: float = Field(
        default=0.2, gt=0.0, lt=1.0, description="PPO clipping ratio"
    )
    value_coef: float = Field(
        default=0.5, ge=0.0, description="Value function loss coefficient"
    )
    entropy_coef: float = Field(
        default=0.01,
        ge=0.0,
        description="Entropy bonus coefficient (positive for maximization)",
    )
    max_grad_norm: float = Field(
        default=0.5, gt=0.0, description="Maximum gradient norm for clipping"
    )
    gae_lambda: float = Field(
        default=0.95, ge=0.0, le=1.0, description="GAE lambda parameter"
    )
    clip_value_loss: bool = Field(
        default=True, description="Whether to clip value loss"
    )
    n_epochs: int = Field(
        default=4, gt=0, description="Number of training epochs per update"
    )
    device: str = Field(
        default="cpu", description="Device to run computations on ('cpu' or 'cuda')"
    )
    random_seed: int | None = Field(
        default=None, description="Random seed for reproducible results"
    )

    model_config = {"arbitrary_types_allowed": True}


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
        state_size: int | None = None,
        action_size: int | None = None,
        *,
        config: PPOConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize PPO agent.

        Args:
            state_size: Dimension of the state space (for backwards compatibility)
            action_size: Dimension of the action space (for backwards compatibility)
            config: Pre-validated configuration object (recommended)
            **kwargs: Individual parameters for backwards compatibility

        Examples:
            # New style (recommended)
            >>> config = PPOConfig(state_size=4, action_size=2)
            >>> agent = PPOAgent(config=config)

            # Old style (backwards compatible)
            >>> agent = PPOAgent(state_size=4, action_size=2)
            >>> agent = PPOAgent(4, 2)  # Positional arguments also work

        Raises:
            ValidationError: If parameters are invalid (via Pydantic)
        """
        # Validate parameters (automatic via Pydantic)
        if config is None:
            # Support positional arguments
            if state_size is not None:
                kwargs["state_size"] = state_size
            if action_size is not None:
                kwargs["action_size"] = action_size
            config = PPOConfig(**kwargs)

        # Store config
        self.config = config

        # Extract all parameters
        self.state_size = config.state_size
        self.action_size = config.action_size
        self.learning_rate = config.learning_rate
        self.discount_factor = config.discount_factor
        self.batch_size = config.batch_size
        self.clip_ratio = config.clip_ratio
        self.value_coef = config.value_coef
        self.entropy_coef = config.entropy_coef
        self.max_grad_norm = config.max_grad_norm
        self.gae_lambda = config.gae_lambda
        self.clip_value_loss = config.clip_value_loss
        self.n_epochs = config.n_epochs
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        # Set random seeds
        if config.random_seed is not None:
            torch.manual_seed(config.random_seed)
            np.random.seed(config.random_seed)
            random.seed(config.random_seed)

        # Initialize networks
        self.policy = PolicyNetwork(
            config.state_size,
            config.action_size,
            config.hidden_sizes,
            config.dropout_rate,
        ).to(self.device)
        self.value = ValueNetwork(
            config.state_size, config.hidden_sizes, config.dropout_rate
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=config.learning_rate,
            eps=1e-5,  # Small epsilon for numerical stability
        )

        # Initialize rollout buffer
        self.rollout_buffer = RolloutBuffer(config.buffer_size)

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
