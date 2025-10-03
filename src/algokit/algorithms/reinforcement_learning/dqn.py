"""Deep Q-Network (DQN) reinforcement learning algorithm implementation.

This module contains the DQN algorithm implementation using PyTorch neural networks.
DQN extends Q-Learning to handle high-dimensional state spaces by using deep neural
networks to approximate the Q-function, with experience replay and target networks
for stable learning.

This implementation supports:
- Vanilla DQN: Standard DQN with target network
- Double DQN: Uses online network for action selection, target network for evaluation
- Huber loss: More robust learning against outliers
- Gradient clipping: Prevents exploding gradients
- Soft target updates: Polyak averaging for smoother target network updates
"""

import random
from collections import deque, namedtuple
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define experience tuple for replay buffer
Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


class DQNNetwork(nn.Module):
    """Deep Q-Network neural network architecture.

    A fully connected neural network that takes state as input and outputs
    Q-values for each action.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: list[int] | None = None,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize the DQN network.

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
        layers: list[nn.Module] = []
        input_size = state_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size

        # Output layer
        layers.append(nn.Linear(input_size, action_size))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            state: Input state tensor

        Returns:
            Q-values for each action
        """
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN training.

    Stores experiences and provides random sampling for training,
    helping to break correlation between consecutive experiences.
    """

    def __init__(self, capacity: int) -> None:
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.buffer: deque[Experience] = deque(maxlen=capacity)
        self.capacity = capacity

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add experience to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        """Sample random batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of sampled experiences
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} experiences from buffer of size {len(self.buffer)}"
            )

        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent for reinforcement learning.

    DQN uses a deep neural network to approximate the Q-function, enabling
    learning in high-dimensional state spaces. It includes experience replay
    and target networks for stable learning.

    This implementation supports multiple DQN variants:
    - Vanilla DQN: Standard DQN with target network
    - Double DQN: Uses online network for action selection, target network for evaluation
    - Huber loss: More robust learning against outliers
    - Gradient clipping: Prevents exploding gradients
    - Soft target updates: Polyak averaging for smoother target network updates

    Attributes:
        state_size: Dimension of the state space
        action_size: Dimension of the action space
        learning_rate: Learning rate for neural network optimizer
        discount_factor: Discount factor for future rewards
        epsilon: Exploration rate for epsilon-greedy policy
        epsilon_decay: Rate of epsilon decay over time
        epsilon_min: Minimum exploration rate
        batch_size: Batch size for neural network training
        memory_size: Size of experience replay buffer
        target_update: Frequency of target network updates
        device: PyTorch device for computations
        q_network: Main Q-network
        target_network: Target Q-network
        optimizer: Neural network optimizer
        memory: Experience replay buffer
        step_count: Counter for target network updates
        dqn_variant: Type of DQN algorithm (vanilla or double)
        use_huber_loss: Whether to use Huber loss instead of MSE
        gradient_clip_norm: Maximum gradient norm for clipping
        tau: Soft update parameter for target network (0 = hard update)
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: list[int] | None = None,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 64,
        memory_size: int = 10000,
        target_update: int = 100,
        dropout_rate: float = 0.0,
        device: torch.device | None = None,
        random_seed: int | None = None,
        dqn_variant: Literal["vanilla", "double"] = "double",
        use_huber_loss: bool = True,
        gradient_clip_norm: float = 1.0,
        tau: float = 0.0,
        epsilon_decay_type: Literal["multiplicative", "linear"] = "multiplicative",
        epsilon_decay_steps: int | None = None,
    ) -> None:
        """Initialize DQN agent.

        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
            hidden_sizes: List of hidden layer sizes
            learning_rate: Learning rate for neural network optimizer
            discount_factor: Discount factor for future rewards
            epsilon: Initial exploration rate for epsilon-greedy policy
            epsilon_decay: Rate of epsilon decay over time
            epsilon_min: Minimum exploration rate
            batch_size: Batch size for neural network training
            memory_size: Size of experience replay buffer
            target_update: Frequency of target network updates
            dropout_rate: Dropout rate for regularization
            device: PyTorch device for computations
            random_seed: Random seed for reproducible results
            dqn_variant: Type of DQN algorithm ("vanilla" or "double")
            use_huber_loss: Whether to use Huber loss instead of MSE
            gradient_clip_norm: Maximum gradient norm for clipping
            tau: Soft update parameter for target network (0 = hard update)
            epsilon_decay_type: Type of epsilon decay ("multiplicative" or "linear")
            epsilon_decay_steps: Number of steps for linear decay (None for multiplicative)

        Raises:
            ValueError: If any parameter is invalid
        """
        if state_size <= 0:
            raise ValueError("state_size must be positive")
        if action_size <= 0:
            raise ValueError("action_size must be positive")
        if not 0 <= learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0 <= discount_factor <= 1:
            raise ValueError("discount_factor must be between 0 and 1")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be between 0 and 1")
        if not 0 <= epsilon_min <= epsilon:
            raise ValueError("epsilon_min must be between 0 and epsilon")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if memory_size <= 0:
            raise ValueError("memory_size must be positive")
        if target_update <= 0:
            raise ValueError("target_update must be positive")
        if dqn_variant not in ["vanilla", "double"]:
            raise ValueError("dqn_variant must be 'vanilla' or 'double'")
        if gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if not 0 <= tau <= 1:
            raise ValueError("tau must be between 0 and 1")
        if epsilon_decay_type not in ["multiplicative", "linear"]:
            raise ValueError("epsilon_decay_type must be 'multiplicative' or 'linear'")
        if epsilon_decay_type == "linear" and epsilon_decay_steps is None:
            raise ValueError("epsilon_decay_steps must be provided for linear decay")
        if epsilon_decay_steps is not None and epsilon_decay_steps <= 0:
            raise ValueError("epsilon_decay_steps must be positive")

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.target_update = target_update
        self.dqn_variant = dqn_variant
        self.use_huber_loss = use_huber_loss
        self.gradient_clip_norm = gradient_clip_norm
        self.tau = tau
        self.epsilon_decay_type = epsilon_decay_type
        self.epsilon_decay_steps = epsilon_decay_steps

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Set random seeds
        if random_seed is not None:
            self.set_seed(random_seed)

        # Initialize networks
        if hidden_sizes is None:
            hidden_sizes = [128, 128]
        self.q_network = DQNNetwork(
            state_size, action_size, hidden_sizes, dropout_rate
        ).to(self.device)
        self.target_network = DQNNetwork(
            state_size, action_size, hidden_sizes, dropout_rate
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Initialize experience replay buffer
        self.memory = ReplayBuffer(memory_size)

        # Initialize step counter
        self.step_count = 0

        # Initialize target network
        self._update_target_network()

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible experiments.

        Args:
            seed: Random seed value
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _update_target_network(self) -> None:
        """Update target network with current Q-network weights.

        Uses hard update (copy weights) if tau=0, otherwise soft update (Polyak averaging).
        """
        if self.tau == 0.0:
            # Hard update: copy weights directly
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            # Soft update: Polyak averaging
            for target_param, local_param in zip(
                self.target_network.parameters(), self.q_network.parameters()
            ):
                target_param.data.copy_(
                    self.tau * local_param.data + (1.0 - self.tau) * target_param.data
                )

    def get_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Selected action

        Raises:
            ValueError: If state has wrong shape
        """
        if state.shape != (self.state_size,):
            raise ValueError(
                f"State shape {state.shape} does not match expected {(self.state_size,)}"
            )

        if np.random.random() < self.epsilon:
            # Explore: choose random action
            return np.random.choice(self.action_size)
        else:
            # Exploit: choose best action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return int(q_values.argmax().item())

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Save experience and potentially train the network.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after taking action
            done: Whether episode is finished

        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate state shape
        if state.shape != (self.state_size,):
            raise ValueError(
                f"State shape {state.shape} does not match expected {(self.state_size,)}"
            )

        # Validate action
        if not 0 <= action < self.action_size:
            raise ValueError(f"Action {action} is out of range [0, {self.action_size})")

        # Validate next_state shape
        if next_state.shape != (self.state_size,):
            raise ValueError(
                f"Next state shape {next_state.shape} does not match expected {(self.state_size,)}"
            )

        # Save experience
        self.memory.push(state, action, reward, next_state, done)

        # Train if we have enough experiences
        if len(self.memory) >= self.batch_size:
            self._train()

        # Update target network
        self.step_count += 1
        if self.tau > 0.0:
            # Soft update: update target network every step
            self._update_target_network()
        elif self.step_count % self.target_update == 0:
            # Hard update: update target network periodically
            self._update_target_network()

    def _train(self) -> None:
        """Train the Q-network using experience replay."""
        # Sample batch from memory
        experiences = self.memory.sample(self.batch_size)

        # Convert to tensors - use numpy arrays for better performance
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(
            self.device
        )
        actions = torch.LongTensor(np.array([e.action for e in experiences])).to(
            self.device
        )
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences])).to(
            self.device
        )
        next_states = torch.FloatTensor(
            np.array([e.next_state for e in experiences])
        ).to(self.device)
        dones = torch.BoolTensor(np.array([e.done for e in experiences])).to(
            self.device
        )

        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Compute next Q-values based on DQN variant
        with torch.no_grad():
            if self.dqn_variant == "double":
                # Double DQN: use online network to select action, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states).gather(
                    1, next_actions.unsqueeze(1)
                )
            else:
                # Vanilla DQN: use target network for both selection and evaluation
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)

            target_q_values = rewards.unsqueeze(1) + (
                self.discount_factor * next_q_values * ~dones.unsqueeze(1)
            )

        # Compute loss
        if self.use_huber_loss:
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
        else:
            loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.gradient_clip_norm
        )
        self.optimizer.step()

    def decay_epsilon(self) -> None:
        """Decay epsilon for exploration schedule."""
        if self.epsilon > self.epsilon_min:
            if self.epsilon_decay_type == "multiplicative":
                self.epsilon *= self.epsilon_decay
            elif (
                self.epsilon_decay_type == "linear"
                and self.epsilon_decay_steps is not None
            ):
                # Linear decay: epsilon decreases linearly from initial to minimum
                decay_rate = (
                    self.epsilon - self.epsilon_min
                ) / self.epsilon_decay_steps
                self.epsilon = max(self.epsilon_min, self.epsilon - decay_rate)

    def decay_epsilon_by_step(self, step: int) -> None:
        """Decay epsilon based on step count for more flexible exploration schedules.

        Args:
            step: Current step number
        """
        if self.epsilon_decay_type == "linear" and self.epsilon_decay_steps is not None:
            # Linear decay based on step count
            if step < self.epsilon_decay_steps:
                self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * (
                    1.0 - step / self.epsilon_decay_steps
                )
            else:
                self.epsilon = self.epsilon_min
        else:
            # Fall back to regular decay
            self.decay_epsilon()

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given a state.

        Args:
            state: Input state

        Returns:
            Q-values for all actions

        Raises:
            ValueError: If state has wrong shape
        """
        if state.shape != (self.state_size,):
            raise ValueError(
                f"State shape {state.shape} does not match expected {(self.state_size,)}"
            )

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().detach().numpy().flatten()

    def get_action_values(self, state: np.ndarray) -> dict[int, float]:
        """Get Q-values for all actions as a dictionary.

        Args:
            state: Input state

        Returns:
            Dictionary mapping actions to Q-values
        """
        q_values = self.get_q_values(state)
        return {action: float(q_values[action]) for action in range(self.action_size)}

    def get_policy(self, states: np.ndarray) -> np.ndarray:
        """Get greedy policy for multiple states.

        Args:
            states: Array of states (batch_size, state_size)

        Returns:
            Array of best actions for each state

        Raises:
            ValueError: If states have wrong shape
        """
        if states.shape[1] != self.state_size:
            raise ValueError(
                f"States shape {states.shape} does not match expected (batch_size, {self.state_size})"
            )

        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            q_values = self.q_network(states_tensor)
            return q_values.argmax(1).cpu().detach().numpy()

    def reset_memory(self) -> None:
        """Reset experience replay buffer."""
        self.memory = ReplayBuffer(self.memory_size)

    def set_epsilon(self, epsilon: float) -> None:
        """Set exploration rate.

        Args:
            epsilon: New exploration rate

        Raises:
            ValueError: If epsilon is invalid
        """
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")
        self.epsilon = epsilon

    def get_epsilon(self) -> float:
        """Get current exploration rate.

        Returns:
            Current exploration rate
        """
        return self.epsilon

    def save_model(self, filepath: str) -> None:
        """Save the trained model to file.

        Args:
            filepath: Path to save the model
        """
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "step_count": self.step_count,
                "state_size": self.state_size,
                "action_size": self.action_size,
                "dqn_variant": self.dqn_variant,
                "use_huber_loss": self.use_huber_loss,
                "gradient_clip_norm": self.gradient_clip_norm,
                "tau": self.tau,
                "epsilon_decay_type": self.epsilon_decay_type,
                "epsilon_decay_steps": self.epsilon_decay_steps,
            },
            filepath,
        )

    def load_model(self, filepath: str) -> None:
        """Load a trained model from file.

        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.step_count = checkpoint["step_count"]

        # Load new parameters if available
        if "dqn_variant" in checkpoint:
            self.dqn_variant = checkpoint["dqn_variant"]
        if "use_huber_loss" in checkpoint:
            self.use_huber_loss = checkpoint["use_huber_loss"]
        if "gradient_clip_norm" in checkpoint:
            self.gradient_clip_norm = checkpoint["gradient_clip_norm"]
        if "tau" in checkpoint:
            self.tau = checkpoint["tau"]
        if "epsilon_decay_type" in checkpoint:
            self.epsilon_decay_type = checkpoint["epsilon_decay_type"]
        if "epsilon_decay_steps" in checkpoint:
            self.epsilon_decay_steps = checkpoint["epsilon_decay_steps"]

    def __repr__(self) -> str:
        """String representation of DQN agent."""
        return (
            f"DQNAgent(state_size={self.state_size}, "
            f"action_size={self.action_size}, "
            f"learning_rate={self.learning_rate}, "
            f"discount_factor={self.discount_factor}, "
            f"epsilon={self.epsilon:.3f}, "
            f"dqn_variant={self.dqn_variant}, "
            f"use_huber_loss={self.use_huber_loss}, "
            f"tau={self.tau}, "
            f"epsilon_decay_type={self.epsilon_decay_type}, "
            f"device={self.device})"
        )
