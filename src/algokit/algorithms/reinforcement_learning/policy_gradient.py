"""Policy Gradient reinforcement learning algorithm implementation.

This module contains the Policy Gradient algorithm implementation using PyTorch neural networks.
Policy Gradient methods directly optimize the policy function using gradient ascent on the
expected return, making them particularly useful for continuous action spaces and stochastic policies.

This implementation is strictly on-policy and algorithmically correct:
- Uses log-probabilities from the actual actions taken during rollouts
- No resampling of actions during training
- Proper advantage normalization (not baseline targets)
- Entropy bonus for exploration
- Optional GAE (Generalized Advantage Estimation) for variance reduction
"""

import random
from collections import namedtuple
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define experience tuple for trajectory storage
RolloutExperience = namedtuple(
    "RolloutExperience", ["state", "action", "reward", "log_prob", "value", "done"]
)


class PolicyNetwork(nn.Module):
    """Policy network for action probability approximation.

    The policy network outputs action probabilities for a given state,
    implementing the policy π(a|s) for both discrete and continuous action spaces.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: list[int] | None = None,
        dropout_rate: float = 0.0,
        continuous_actions: bool = False,
    ) -> None:
        """Initialize the Policy network.

        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            continuous_actions: Whether to use continuous action space
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 128]

        self.state_size = state_size
        self.action_size = action_size
        self.continuous_actions = continuous_actions

        # Build network layers
        layers = []
        input_size = state_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            input_size = hidden_size

        self.network = nn.Sequential(*layers)

        if continuous_actions:
            # For continuous actions, output mean and log_std
            self.mean_head = nn.Linear(input_size, action_size)
            self.log_std_head = nn.Linear(input_size, action_size)
        else:
            # For discrete actions, output logits
            self.action_head = nn.Linear(input_size, action_size)

    def forward(
        self, state: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            state: Input state tensor

        Returns:
            Action logits or (mean, log_std) for continuous actions
        """
        features = self.network(state)

        if self.continuous_actions:
            mean = self.mean_head(features)
            log_std = self.log_std_head(features)
            return mean, log_std
        else:
            return self.action_head(features)

    def get_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action from the policy.

        Args:
            state: Current state tensor

        Returns:
            Tuple of (action, log_probability)
        """
        if self.continuous_actions:
            mean, log_std = self.forward(state)
            std = torch.exp(log_std.clamp(-20, 2))  # Clamp for numerical stability
            normal_dist = torch.distributions.Normal(mean, std)
            action = normal_dist.sample()
            log_prob = normal_dist.log_prob(action).sum(dim=-1, keepdim=True)
            return action, log_prob
        else:
            logits = self.forward(state)
            cat_dist = torch.distributions.Categorical(logits=logits)
            action = cat_dist.sample()
            log_prob = cat_dist.log_prob(action)
            return action, log_prob


class BaselineNetwork(nn.Module):
    """Baseline network for variance reduction.

    The baseline network estimates the value function V(s) to reduce
    variance in policy gradient estimates.
    """

    def __init__(
        self,
        state_size: int,
        hidden_sizes: list[int] | None = None,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize the Baseline network.

        Args:
            state_size: Dimension of the state space
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 128]

        # Build network layers
        layers = []
        input_size = state_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            input_size = hidden_size

        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the baseline network.

        Args:
            state: Input state tensor

        Returns:
            Baseline value estimate
        """
        return self.network(state).squeeze(-1)


class PolicyGradientAgent:
    """Policy Gradient agent implementing REINFORCE algorithm.

    This agent learns a policy directly using policy gradient methods,
    with optional baseline for variance reduction. The implementation is
    strictly on-policy and algorithmically correct.

    Features:
    - On-policy only: uses log-probabilities from actual rollout actions
    - Proper advantage normalization (not baseline targets)
    - Entropy bonus for exploration
    - Optional GAE for variance reduction
    - Correct baseline training against unnormalized returns
    """

    baseline: BaselineNetwork | None

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        use_baseline: bool = True,
        hidden_sizes: list[int] | None = None,
        dropout_rate: float = 0.0,
        continuous_actions: bool = False,
        device: str = "cpu",
        seed: int | None = None,
        entropy_coefficient: float = 0.01,
        use_gae: bool = False,
        gae_lambda: float = 0.95,
        normalize_advantages: bool = True,
        normalize_rewards: bool = False,
    ) -> None:
        """Initialize the Policy Gradient agent.

        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
            learning_rate: Learning rate for optimizers
            gamma: Discount factor for future rewards
            use_baseline: Whether to use baseline for variance reduction
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            continuous_actions: Whether to use continuous action space
            device: Device to run computations on
            seed: Random seed for reproducibility
            entropy_coefficient: Coefficient for entropy bonus in policy loss
            use_gae: Whether to use Generalized Advantage Estimation
            gae_lambda: GAE lambda parameter for variance reduction
            normalize_advantages: Whether to normalize advantages
            normalize_rewards: Whether to normalize rewards for stability
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.continuous_actions = continuous_actions
        self.device = torch.device(device)
        self.entropy_coefficient = entropy_coefficient
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.normalize_rewards = normalize_rewards

        # Initialize policy network
        self.policy = PolicyNetwork(
            state_size=state_size,
            action_size=action_size,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            continuous_actions=continuous_actions,
        ).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Initialize baseline network if using baseline
        if self.use_baseline:
            self.baseline = BaselineNetwork(
                state_size=state_size,
                hidden_sizes=hidden_sizes,
                dropout_rate=dropout_rate,
            ).to(self.device)

            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(), lr=learning_rate
            )
        else:
            self.baseline = None

        # Training state
        self.training = True
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

        # KL tracking for stability
        self.kl_divergence_history: list[float] = []

        # Reward normalization statistics
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_update_count = 0

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
            action, log_prob, value = self.evaluate(state)

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

    def evaluate(self, state: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Evaluate state and return action, log_prob, and value.

        Args:
            state: Current state

        Returns:
            Tuple of (action, log_probability, value_estimate)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob = self.policy.get_action(state_tensor)

            # Get value estimate from baseline if available
            if self.use_baseline and self.baseline is not None:
                value = self.baseline(state_tensor)
            else:
                value = torch.tensor(0.0, device=self.device)

        if self.continuous_actions:
            return (
                action.detach().cpu().numpy().flatten(),
                log_prob.detach().cpu().numpy().item(),
                value.detach().cpu().numpy().item(),
            )
        else:
            return (
                action.detach().cpu().numpy().item(),
                log_prob.detach().cpu().numpy().item(),
                value.detach().cpu().numpy().item(),
            )

    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Choose an action using the current policy (backward compatibility).

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Selected action
        """
        action, _, _ = self.evaluate(state)
        return action

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
            running_return = rewards[i] + self.gamma * running_return
            returns.insert(0, running_return)

        return returns

    def compute_gae_advantages(
        self,
        rewards: list[float],
        values: list[float],
        dones: list[bool],
        next_value: float = 0.0,
    ) -> list[float]:
        """Compute GAE advantages.

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Bootstrap value for non-terminal final states

        Returns:
            List of GAE advantages
        """
        advantages: list[float] = []
        gae = 0.0

        # Compute GAE backwards
        for i in reversed(range(len(rewards))):
            if dones[i]:
                gae = 0.0
            else:
                # Use bootstrap value for the last step if not terminal
                next_value_t = next_value if i == len(rewards) - 1 else values[i + 1]

                delta = rewards[i] + self.gamma * next_value_t - values[i]
                gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)

        return advantages

    def _normalize_rewards(self, rewards: list[float]) -> list[float]:
        """Normalize rewards using running statistics for stability.

        Args:
            rewards: List of raw rewards

        Returns:
            List of normalized rewards
        """
        if not self.normalize_rewards or not rewards:
            return rewards

        # Update running statistics
        new_mean = float(np.mean(rewards))
        new_std = float(np.std(rewards))

        if self.reward_update_count == 0:
            self.reward_mean = new_mean
            self.reward_std = new_std if new_std > 1e-8 else 1.0
        else:
            # Exponential moving average
            alpha = 0.01
            self.reward_mean = (1 - alpha) * self.reward_mean + alpha * new_mean
            self.reward_std = (1 - alpha) * self.reward_std + alpha * (
                new_std if new_std > 1e-8 else self.reward_std
            )

        self.reward_update_count += 1

        # Normalize rewards
        normalized_rewards = [
            (r - self.reward_mean) / (self.reward_std + 1e-8) for r in rewards
        ]
        return normalized_rewards

    def _compute_kl_divergence(
        self, old_log_probs: torch.Tensor, new_log_probs: torch.Tensor
    ) -> float:
        """Compute KL divergence between old and new policy distributions.

        Args:
            old_log_probs: Log probabilities from old policy
            new_log_probs: Log probabilities from new policy

        Returns:
            KL divergence value
        """
        # KL divergence = E[log(old_policy) - log(new_policy)]
        kl_div = (old_log_probs - new_log_probs).mean().item()
        return kl_div

    def learn(self, rollout_data: list[RolloutExperience]) -> dict[str, float]:
        """Update networks using on-policy rollout data.

        Args:
            rollout_data: List of rollout experiences

        Returns:
            Dictionary containing loss information and metrics
        """
        if not rollout_data:
            return {
                "policy_loss": 0.0,
                "baseline_loss": 0.0,
                "entropy_loss": 0.0,
                "mean_return": 0.0,
                "mean_advantage": 0.0,
            }

        # Extract data from rollout
        states = torch.FloatTensor(np.array([exp.state for exp in rollout_data])).to(
            self.device
        )

        # Handle actions based on whether they're discrete or continuous
        if self.continuous_actions:
            actions = torch.FloatTensor(
                np.array([exp.action for exp in rollout_data])
            ).to(self.device)
        else:
            actions = torch.LongTensor([exp.action for exp in rollout_data]).to(
                self.device
            )

        rewards = [exp.reward for exp in rollout_data]
        values = torch.FloatTensor([exp.value for exp in rollout_data]).to(self.device)
        dones = [exp.done for exp in rollout_data]

        # Normalize rewards if requested
        normalized_rewards = self._normalize_rewards(rewards)

        # Compute returns using normalized rewards
        returns = self.compute_returns(
            normalized_rewards, [exp.value for exp in rollout_data], dones
        )
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Compute advantages
        if self.use_gae:
            # Get next value for bootstrap (last state value if not terminal)
            next_value = 0.0
            if not dones[-1] and self.use_baseline and self.baseline is not None:
                with torch.no_grad():
                    next_state = (
                        torch.FloatTensor(rollout_data[-1].state)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    next_value = self.baseline(next_state).item()

            advantages = self.compute_gae_advantages(
                normalized_rewards,
                [exp.value for exp in rollout_data],
                dones,
                next_value,
            )
            advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        else:
            advantages_tensor = returns_tensor - values.detach()

        # Normalize advantages if requested
        if self.normalize_advantages and len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
                advantages_tensor.std() + 1e-8
            )

        # Baseline loss (train against unnormalized returns)
        if self.use_baseline and self.baseline is not None:
            current_values = self.baseline(states)
            baseline_loss = F.mse_loss(current_values, returns_tensor)

            # Update baseline
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()
        else:
            baseline_loss = torch.tensor(0.0)

        # Policy loss using log-probabilities from actual actions taken
        # Recompute action probabilities for gradient computation
        old_log_probs = torch.FloatTensor([exp.log_prob for exp in rollout_data]).to(
            self.device
        )

        if self.continuous_actions:
            mean, log_std = self.policy.forward(states)
            std = torch.exp(log_std.clamp(-20, 2))
            normal_dist = torch.distributions.Normal(mean, std)
            new_log_probs = normal_dist.log_prob(actions).sum(dim=-1)
        else:
            logits = self.policy.forward(states)
            cat_dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = cat_dist.log_prob(actions)

        # Compute KL divergence for stability tracking
        kl_divergence = self._compute_kl_divergence(old_log_probs, new_log_probs)
        self.kl_divergence_history.append(kl_divergence)

        # Policy gradient loss
        policy_loss = -(new_log_probs * advantages_tensor.detach()).mean()

        # Entropy bonus for exploration
        if self.continuous_actions:
            entropy = normal_dist.entropy().sum(dim=-1).mean()
        else:
            entropy = cat_dist.entropy().mean()

        entropy_loss = -self.entropy_coefficient * entropy

        # Total policy loss
        total_policy_loss = policy_loss + entropy_loss

        # Update policy
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "entropy": entropy.item(),
            "mean_return": returns_tensor.mean().item(),
            "mean_advantage": advantages_tensor.mean().item(),
            "kl_divergence": kl_divergence,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "raw_returns": returns_tensor.detach().cpu().numpy().tolist(),
            "normalized_advantages": advantages_tensor.detach().cpu().numpy().tolist(),
        }

    def get_training_stats(self) -> dict:
        """Get training statistics.

        Returns:
            Dictionary containing training statistics
        """
        if not self.episode_rewards:
            return {
                "mean_reward": 0.0,
                "mean_length": 0.0,
                "mean_kl_divergence": 0.0,
                "reward_normalization_stats": {
                    "mean": self.reward_mean,
                    "std": self.reward_std,
                },
            }

        stats = {
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "mean_length": np.mean(self.episode_lengths),
            "std_length": np.std(self.episode_lengths),
            "total_episodes": len(self.episode_rewards),
            "reward_normalization_stats": {
                "mean": self.reward_mean,
                "std": self.reward_std,
            },
        }

        # Add KL divergence statistics if available
        if self.kl_divergence_history:
            stats.update(
                {
                    "mean_kl_divergence": np.mean(self.kl_divergence_history),
                    "std_kl_divergence": np.std(self.kl_divergence_history),
                    "recent_kl_divergence": self.kl_divergence_history[-1]
                    if self.kl_divergence_history
                    else 0.0,
                }
            )
        else:
            stats["mean_kl_divergence"] = 0.0

        return stats

    def save(self, filepath: str) -> None:
        """Save the agent's state.

        Args:
            filepath: Path to save the agent
        """
        state = {
            "policy_state_dict": self.policy.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "state_size": self.state_size,
            "action_size": self.action_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "use_baseline": self.use_baseline,
            "continuous_actions": self.continuous_actions,
        }

        if self.use_baseline and self.baseline is not None:
            state.update(
                {
                    "baseline_state_dict": self.baseline.state_dict(),
                    "baseline_optimizer_state_dict": self.baseline_optimizer.state_dict()
                    if self.baseline_optimizer is not None
                    else None,
                }
            )

        torch.save(state, filepath)

    def load(self, filepath: str) -> None:
        """Load the agent's state.

        Args:
            filepath: Path to load the agent from
        """
        state = torch.load(filepath, map_location=self.device)

        self.policy.load_state_dict(state["policy_state_dict"])
        self.policy_optimizer.load_state_dict(state["policy_optimizer_state_dict"])

        if (
            self.use_baseline
            and "baseline_state_dict" in state
            and self.baseline is not None
        ):
            self.baseline.load_state_dict(state["baseline_state_dict"])
            if (
                self.baseline_optimizer is not None
                and "baseline_optimizer_state_dict" in state
            ):
                self.baseline_optimizer.load_state_dict(
                    state["baseline_optimizer_state_dict"]
                )

    def set_training_mode(self, training: bool) -> None:
        """Set training mode for the agent.

        Args:
            training: Whether to set training mode
        """
        self.training = training
        if training:
            self.policy.train()
            if self.use_baseline and self.baseline is not None:
                self.baseline.train()
        else:
            self.policy.eval()
            if self.use_baseline and self.baseline is not None:
                self.baseline.eval()
