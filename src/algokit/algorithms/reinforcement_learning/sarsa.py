"""SARSA (State-Action-Reward-State-Action) reinforcement learning algorithm implementation.

This module contains the SARSA algorithm implementation for model-free,
on-policy reinforcement learning. SARSA learns the action-value function
while following the current policy, making it more conservative than Q-Learning.
"""

import numpy as np


class SarsaAgent:
    """SARSA agent for reinforcement learning.

    SARSA (State-Action-Reward-State-Action) is a model-free, on-policy
    reinforcement learning algorithm that learns the action-value function
    while following the current policy. Unlike Q-Learning, SARSA updates
    Q-values based on the action actually taken in the next state.

    Attributes:
        learning_rate: Learning rate (alpha) for Q-value updates
        discount_factor: Discount factor (gamma) for future rewards
        epsilon: Exploration rate for epsilon-greedy policy
        epsilon_decay: Rate of epsilon decay over time
        epsilon_min: Minimum exploration rate
        q_table: Q-table storing state-action values
        actions: List of possible actions
        state_space_size: Size of the state space
        action_space_size: Size of the action space
    """

    def __init__(
        self,
        state_space_size: int,
        action_space_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        random_seed: int | None = None,
    ) -> None:
        """Initialize SARSA agent.

        Args:
            state_space_size: Number of possible states
            action_space_size: Number of possible actions
            learning_rate: Learning rate (alpha) for Q-value updates
            discount_factor: Discount factor (gamma) for future rewards
            epsilon: Initial exploration rate for epsilon-greedy policy
            epsilon_decay: Rate of epsilon decay over time
            epsilon_min: Minimum exploration rate
            random_seed: Random seed for reproducible results

        Raises:
            ValueError: If any parameter is invalid
        """
        if state_space_size <= 0:
            raise ValueError("state_space_size must be positive")
        if action_space_size <= 0:
            raise ValueError("action_space_size must be positive")
        if not 0 <= learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0 <= discount_factor <= 1:
            raise ValueError("discount_factor must be between 0 and 1")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")
        if not 0 <= epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be between 0 and 1")
        if not 0 <= epsilon_min <= epsilon:
            raise ValueError("epsilon_min must be between 0 and epsilon")

        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_space_size, action_space_size))

        # Initialize actions list
        self.actions = list(range(action_space_size))

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

    def get_action(self, state: int) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Selected action

        Raises:
            ValueError: If state is invalid
        """
        if not 0 <= state < self.state_space_size:
            raise ValueError(
                f"State {state} is out of range [0, {self.state_space_size})"
            )

        if np.random.random() < self.epsilon:
            # Explore: choose random action
            return np.random.choice(self.actions)
        else:
            # Exploit: choose best action
            return int(np.argmax(self.q_table[state]))

    def update_q_value(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        done: bool = False,
    ) -> None:
        """Update Q-value using SARSA update rule.

        The SARSA update rule is:
        Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after taking action
            next_action: Action to be taken in next state
            done: Whether episode is finished

        Raises:
            ValueError: If any parameter is invalid
        """
        if not 0 <= state < self.state_space_size:
            raise ValueError(
                f"State {state} is out of range [0, {self.state_space_size})"
            )
        if not 0 <= action < self.action_space_size:
            raise ValueError(
                f"Action {action} is out of range [0, {self.action_space_size})"
            )
        if not 0 <= next_state < self.state_space_size:
            raise ValueError(
                f"Next state {next_state} is out of range [0, {self.state_space_size})"
            )
        if not 0 <= next_action < self.action_space_size:
            raise ValueError(
                f"Next action {next_action} is out of range [0, {self.action_space_size})"
            )

        # Current Q-value
        current_q = self.q_table[state, action]

        # Q-value for next state-action pair (if not done)
        next_q = 0 if done else self.q_table[next_state, next_action]

        # SARSA update
        target_q = reward + self.discount_factor * next_q
        self.q_table[state, action] += self.learning_rate * (target_q - current_q)

    def decay_epsilon(self) -> None:
        """Decay epsilon for exploration schedule."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_q_value(self, state: int, action: int) -> float:
        """Get Q-value for state-action pair.

        Args:
            state: State
            action: Action

        Returns:
            Q-value for state-action pair

        Raises:
            ValueError: If state or action is invalid
        """
        if not 0 <= state < self.state_space_size:
            raise ValueError(
                f"State {state} is out of range [0, {self.state_space_size})"
            )
        if not 0 <= action < self.action_space_size:
            raise ValueError(
                f"Action {action} is out of range [0, {self.action_space_size})"
            )

        return float(self.q_table[state, action])

    def get_policy(self) -> list[int]:
        """Get greedy policy (best action for each state).

        Returns:
            List of best actions for each state
        """
        return [
            int(np.argmax(self.q_table[state]))
            for state in range(self.state_space_size)
        ]

    def get_state_values(self) -> list[float]:
        """Get state values (maximum Q-value for each state).

        Returns:
            List of state values
        """
        return [
            float(np.max(self.q_table[state])) for state in range(self.state_space_size)
        ]

    def reset_q_table(self) -> None:
        """Reset Q-table to zeros."""
        self.q_table.fill(0)

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

    def get_q_table(self) -> np.ndarray:
        """Get copy of Q-table.

        Returns:
            Copy of Q-table
        """
        return self.q_table.copy()

    def __repr__(self) -> str:
        """String representation of SARSA agent."""
        return (
            f"SarsaAgent(state_space_size={self.state_space_size}, "
            f"action_space_size={self.action_space_size}, "
            f"learning_rate={self.learning_rate}, "
            f"discount_factor={self.discount_factor}, "
            f"epsilon={self.epsilon:.3f})"
        )
