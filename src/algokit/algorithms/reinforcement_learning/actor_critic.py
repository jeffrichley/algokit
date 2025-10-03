"""Actor-Critic reinforcement learning algorithm implementation.

This module contains the Actor-Critic algorithm implementation using PyTorch neural networks.
Actor-Critic combines policy gradient methods (Actor) with value function approximation (Critic)
to provide stable learning with reduced variance compared to pure policy gradient methods.

This implementation is strictly on-policy, collecting full episodes or fixed-length rollouts
and updating the policy using only the actions actually taken during the rollout.
"""

import random
from collections import namedtuple
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define experience tuple for on-policy rollouts
RolloutExperience = namedtuple(
    "RolloutExperience", ["state", "action", "reward", "log_prob", "value", "done"]
)


class ActorNetwork(nn.Module):
    """Actor network for policy approximation.

    The actor network outputs action probabilities for a given state,
    implementing the policy π(a|s).
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: list[int] | None = None,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize the Actor network.

        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 128]

        if not hidden_sizes:
            raise ValueError("hidden_sizes cannot be empty")

        # Build network layers
        layers = []
        prev_size = state_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
            prev_size = hidden_size

        # Output layer for action probabilities
        layers.append(nn.Linear(prev_size, action_size))
        layers.append(nn.Softmax(dim=-1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the actor network.

        Args:
            state: Input state tensor

        Returns:
            Action probabilities tensor
        """
        return self.network(state)


class CriticNetwork(nn.Module):
    """Critic network for value function approximation.

    The critic network estimates the state value function V(s),
    providing baseline estimates to reduce variance in policy gradients.
    """

    def __init__(
        self,
        state_size: int,
        hidden_sizes: list[int] | None = None,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize the Critic network.

        Args:
            state_size: Dimension of the state space
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 128]

        if not hidden_sizes:
            raise ValueError("hidden_sizes cannot be empty")

        # Build network layers
        layers = []
        prev_size = state_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
            prev_size = hidden_size

        # Output layer for state value
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the critic network.

        Args:
            state: Input state tensor

        Returns:
            State value tensor
        """
        return self.network(state)


class ActorCriticAgent:
    """Actor-Critic reinforcement learning agent.

    Strictly on-policy implementation that collects full episodes or fixed-length
    rollouts and updates the policy using only the actions actually taken.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate_actor: float = 0.001,
        learning_rate_critic: float = 0.001,
        discount_factor: float = 0.99,
        hidden_sizes: list[int] | None = None,
        dropout_rate: float = 0.0,
        entropy_coefficient: float = 0.01,
        gae_lambda: float = 0.95,
        normalize_advantages: bool = True,
        gradient_clip_norm: float = 0.5,
        device: str = "cpu",
        random_seed: int | None = None,
    ) -> None:
        """Initialize Actor-Critic agent.

        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
            learning_rate_actor: Learning rate for actor network
            learning_rate_critic: Learning rate for critic network
            discount_factor: Discount factor for future rewards
            hidden_sizes: List of hidden layer sizes for both networks
            dropout_rate: Dropout rate for regularization
            entropy_coefficient: Coefficient for entropy bonus in actor loss
            gae_lambda: Lambda parameter for Generalized Advantage Estimation
            normalize_advantages: Whether to normalize advantages for better conditioning
            gradient_clip_norm: Maximum norm for gradient clipping (0 to disable)
            device: Device to run computations on ('cpu' or 'cuda')
            random_seed: Random seed for reproducible results

        Raises:
            ValueError: If any parameter is invalid
        """
        if state_size <= 0:
            raise ValueError("state_size must be positive")
        if action_size <= 0:
            raise ValueError("action_size must be positive")
        if not 0 < learning_rate_actor <= 1:
            raise ValueError("learning_rate_actor must be between 0 and 1")
        if not 0 < learning_rate_critic <= 1:
            raise ValueError("learning_rate_critic must be between 0 and 1")
        if not 0 <= discount_factor <= 1:
            raise ValueError("discount_factor must be between 0 and 1")
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        if entropy_coefficient < 0:
            raise ValueError("entropy_coefficient must be non-negative")
        if not 0 <= gae_lambda <= 1:
            raise ValueError("gae_lambda must be between 0 and 1")
        if gradient_clip_norm < 0:
            raise ValueError("gradient_clip_norm must be non-negative")

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.discount_factor = discount_factor
        self.entropy_coefficient = entropy_coefficient
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.gradient_clip_norm = gradient_clip_norm
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Set random seeds
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Initialize networks
        self.actor = ActorNetwork(
            state_size, action_size, hidden_sizes, dropout_rate
        ).to(self.device)
        self.critic = CriticNetwork(state_size, hidden_sizes, dropout_rate).to(
            self.device
        )

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=learning_rate_actor
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=learning_rate_critic
        )

        # Training mode
        self.training = True

    def get_action(
        self, state: np.ndarray, training: bool = True
    ) -> tuple[int, float, float]:
        """Select action using current policy.

        Args:
            state: Current state
            training: Whether in training mode (affects exploration)

        Returns:
            Tuple of (selected_action, log_probability, value_estimate)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            # Get value estimate from critic
            value = self.critic(state_tensor)

        return action.item(), log_prob.item(), value.item()

    def collect_rollout(
        self, env: Any, n_steps: int | None = None, max_episode_length: int = 1000
    ) -> list[RolloutExperience]:
        """Collect on-policy rollout data from environment.

        Args:
            env: Environment with reset() and step() methods
            n_steps: Number of steps to collect (None for full episodes)
            max_episode_length: Maximum steps per episode

        Returns:
            List of rollout experiences
        """
        rollout_data = []
        state = env.reset()
        episode_length = 0

        while True:
            # Get action, log_prob, and value from current policy
            action, log_prob, value = self.get_action(state, training=True)

            # Take step in environment
            next_state, reward, done, _ = env.step(action)

            # Store experience
            experience = RolloutExperience(
                state=state.copy(),
                action=action,
                reward=reward,
                log_prob=log_prob,
                value=value,
                done=done,
            )
            rollout_data.append(experience)

            # Check termination conditions
            if done or episode_length >= max_episode_length:
                if n_steps is None:  # Collect full episodes
                    break
                else:  # Collect fixed number of steps
                    state = env.reset()
                    episode_length = 0
                    if len(rollout_data) >= n_steps:
                        break
            else:
                state = next_state
                episode_length += 1

            # Check if we've collected enough steps
            if n_steps is not None and len(rollout_data) >= n_steps:
                break

        return rollout_data[:n_steps] if n_steps is not None else rollout_data

    def compute_returns(
        self, rewards: list[float], values: list[float], dones: list[bool]
    ) -> list[float]:
        """Compute returns using value function as baseline.

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags

        Returns:
            List of computed returns
        """
        returns: list[float] = []
        running_return = 0.0

        # Compute returns backwards
        for i in reversed(range(len(rewards))):
            if dones[i]:
                running_return = 0.0
            running_return = rewards[i] + self.discount_factor * running_return
            returns.insert(0, running_return)

        return returns

    def compute_gae_advantages(
        self,
        rewards: list[float],
        values: list[float],
        dones: list[bool],
        next_value: float = 0.0,
    ) -> tuple[list[float], list[float]]:
        """Compute Generalized Advantage Estimation (GAE) advantages and returns.

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for the next state (for bootstrap)

        Returns:
            Tuple of (advantages, returns)
        """
        advantages: list[float] = []
        returns: list[float] = []

        # Add next_value to values for bootstrap
        extended_values = values + [next_value]

        # Compute GAE advantages backwards
        gae = 0.0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                gae = 0.0

            # Compute TD error
            delta = (
                rewards[i] + self.discount_factor * extended_values[i + 1] - values[i]
            )

            # Update GAE
            gae = delta + self.discount_factor * self.gae_lambda * gae
            advantages.insert(0, gae)

        # Compute returns from advantages and values
        for i in range(len(advantages)):
            returns.append(advantages[i] + values[i])

        return advantages, returns

    def learn(self, rollout_data: list[RolloutExperience]) -> dict[str, float]:
        """Update networks using on-policy rollout data.

        Args:
            rollout_data: List of rollout experiences

        Returns:
            Dictionary containing loss information
        """
        if not rollout_data:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy_loss": 0.0}

        # Extract data from rollout
        states = torch.FloatTensor(np.array([exp.state for exp in rollout_data])).to(
            self.device
        )
        actions = torch.LongTensor([exp.action for exp in rollout_data]).to(self.device)
        rewards = [exp.reward for exp in rollout_data]
        dones = [exp.done for exp in rollout_data]

        # Get current value estimates from critic for bootstrap
        with torch.no_grad():
            current_values = self.critic(states).squeeze()
            # Use last state's value for bootstrap if not done
            next_value = current_values[-1].item() if not dones[-1] else 0.0

        # Compute GAE advantages and returns
        advantages, returns = self.compute_gae_advantages(
            rewards, [exp.value for exp in rollout_data], dones, next_value
        )
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages if enabled
        if self.normalize_advantages:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
                advantages_tensor.std() + 1e-8
            )

        # Get current value estimates from critic
        current_values = self.critic(states).squeeze()

        # Critic loss (MSE between predicted values and returns)
        critic_loss = F.mse_loss(current_values, returns_tensor)

        # Update critic with gradient clipping
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), max_norm=self.gradient_clip_norm
            )
        self.critic_optimizer.step()

        # Actor loss (policy gradient with advantages)
        # Recompute action probabilities for gradient computation
        action_probs = self.actor(states)
        action_dist = torch.distributions.Categorical(action_probs)
        new_log_probs = action_dist.log_prob(actions)

        # Policy gradient loss
        policy_loss = -(new_log_probs * advantages_tensor.detach()).mean()

        # Entropy bonus for exploration
        entropy = action_dist.entropy().mean()
        entropy_loss = -self.entropy_coefficient * entropy

        # Total actor loss
        actor_loss = policy_loss + entropy_loss

        # Update actor with gradient clipping
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=self.gradient_clip_norm
            )
        self.actor_optimizer.step()

        return {
            "actor_loss": policy_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "entropy": entropy.item(),
            "mean_advantage": advantages_tensor.mean().item(),
            "advantage_std": advantages_tensor.std().item(),
        }

    def save(self, filepath: str) -> None:
        """Save agent state to file.

        Args:
            filepath: Path to save the model
        """
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "entropy_coefficient": self.entropy_coefficient,
                "gae_lambda": self.gae_lambda,
                "normalize_advantages": self.normalize_advantages,
                "gradient_clip_norm": self.gradient_clip_norm,
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        """Load agent state from file.

        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        # Restore hyperparameters if available
        if "entropy_coefficient" in checkpoint:
            self.entropy_coefficient = checkpoint["entropy_coefficient"]
        if "gae_lambda" in checkpoint:
            self.gae_lambda = checkpoint["gae_lambda"]
        if "normalize_advantages" in checkpoint:
            self.normalize_advantages = checkpoint["normalize_advantages"]
        if "gradient_clip_norm" in checkpoint:
            self.gradient_clip_norm = checkpoint["gradient_clip_norm"]

    def set_training(self, training: bool) -> None:
        """Set training mode for networks.

        Args:
            training: Whether to set training mode
        """
        self.training = training
        if training:
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()
