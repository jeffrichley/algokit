"""SARSA (State-Action-Reward-State-Action) reinforcement learning algorithm implementation.

This module contains the SARSA algorithm implementation for model-free,
on-policy reinforcement learning. SARSA learns the action-value function
while following the current policy, making it more conservative than Q-Learning.

This implementation includes:
- Random tie-breaking in epsilon-greedy action selection
- Configurable epsilon scheduling
- Reproducible results with seed control
- Lazy initialization for unknown states
- Optional Expected SARSA for bias reduction
- Debug logging for TD error and update magnitude
- On-policy behavior enforcement
"""

import logging
import random

import numpy as np


class SarsaAgent:
    """SARSA agent for reinforcement learning.

    SARSA (State-Action-Reward-State-Action) is a model-free, on-policy
    reinforcement learning algorithm that learns the action-value function
    while following the current policy. Unlike Q-Learning, SARSA updates
    Q-values based on the action actually taken in the next state.

    This implementation provides:
    - Random tie-breaking in action selection to avoid bias
    - Configurable epsilon scheduling with validation
    - Reproducible results with seed control
    - Lazy initialization for unknown states
    - Optional Expected SARSA for bias reduction
    - Debug logging for TD error and update magnitude
    - Strict on-policy behavior enforcement

    Attributes:
        learning_rate: Learning rate (alpha) for Q-value updates
        discount_factor: Discount factor (gamma) for future rewards
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Rate of epsilon decay over time
        q_table: Q-table storing state-action values
        actions: List of possible actions
        state_space_size: Size of the state space
        action_space_size: Size of the action space
        use_expected_sarsa: Whether to use Expected SARSA
        debug: Whether to enable debug logging
        _logger: Logger instance for debug output
    """

    def __init__(
        self,
        state_space_size: int,
        action_space_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        use_expected_sarsa: bool = False,
        debug: bool = False,
        random_seed: int | None = None,
        # Backward compatibility parameters
        epsilon: float | None = None,
        epsilon_min: float | None = None,
    ) -> None:
        """Initialize SARSA agent.

        Args:
            state_space_size: Number of possible states
            action_space_size: Number of possible actions
            learning_rate: Learning rate (alpha) for Q-value updates
            discount_factor: Discount factor (gamma) for future rewards
            epsilon_start: Initial exploration rate for epsilon-greedy policy
            epsilon_end: Final exploration rate for epsilon-greedy policy
            epsilon_decay: Rate of epsilon decay over time
            use_expected_sarsa: Whether to use Expected SARSA
            debug: Whether to enable debug logging
            random_seed: Random seed for reproducible results
            epsilon: Backward compatibility - maps to epsilon_start
            epsilon_min: Backward compatibility - maps to epsilon_end

        Raises:
            ValueError: If any parameter is invalid
        """
        # Handle backward compatibility
        if epsilon is not None:
            epsilon_start = epsilon
        if epsilon_min is not None:
            epsilon_end = epsilon_min

        # Validate parameters
        if state_space_size <= 0:
            raise ValueError("state_space_size must be positive")
        if action_space_size <= 0:
            raise ValueError("action_space_size must be positive")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0 < discount_factor <= 1:
            raise ValueError("discount_factor must be between 0 and 1")
        if not 0 <= epsilon_start <= 1:
            raise ValueError("epsilon must be between 0 and 1")
        if not 0 <= epsilon_end <= 1:
            raise ValueError("epsilon_min must be between 0 and epsilon")
        if not 0 <= epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be between 0 and 1")
        if not epsilon_end <= epsilon_start:
            raise ValueError("epsilon_min must be between 0 and epsilon")

        # Store parameters
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.use_expected_sarsa = use_expected_sarsa
        self.debug = debug

        # Current epsilon (starts at epsilon_start)
        self.epsilon = epsilon_start

        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_space_size, action_space_size))

        # Initialize actions list
        self.actions = list(range(action_space_size))

        # Set up logging
        self._logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}")
        if debug:
            self._logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

        # Set random seed if provided
        if random_seed is not None:
            self.set_seed(random_seed)

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible results.

        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)

    def select_action(self, state: int) -> int:
        """Select action using epsilon-greedy policy with random tie-breaking.

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
            # Exploit: choose best action with random tie-breaking
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            # Find all actions with maximum Q-value
            best_actions = np.where(q_values == max_q)[0]
            # Randomly select among tied actions
            return int(np.random.choice(best_actions))

    def get_action(self, state: int) -> int:
        """Select action using epsilon-greedy policy (backward compatibility).

        This method is an alias for select_action() for backward compatibility.

        Args:
            state: Current state

        Returns:
            Selected action

        Raises:
            ValueError: If state is invalid
        """
        return self.select_action(state)

    def step(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool = False,
    ) -> int:
        """Perform SARSA update step and return next action.

        This method performs the SARSA update and returns the next action
        to be taken, ensuring strict on-policy behavior.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after taking action
            done: Whether episode is finished

        Returns:
            Next action to be taken (chosen by current epsilon-greedy policy)

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

        # Choose next action using current epsilon-greedy policy (on-policy)
        next_action = self.select_action(next_state)

        # Perform SARSA update
        self._sarsa_update(state, action, reward, next_state, next_action, done)

        return next_action

    def update_q_value(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        done: bool = False,
    ) -> None:
        """Update Q-value using SARSA update rule (backward compatibility).

        This method is an alias for _sarsa_update() for backward compatibility.

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
        self._sarsa_update(state, action, reward, next_state, next_action, done)

    def _sarsa_update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        done: bool,
    ) -> None:
        """Perform SARSA or Expected SARSA update."""
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

        if self.use_expected_sarsa:
            self._expected_sarsa_update(state, action, reward, next_state, done)
        else:
            self._standard_sarsa_update(
                state, action, reward, next_state, next_action, done
            )

    def _standard_sarsa_update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        done: bool,
    ) -> None:
        """Perform standard SARSA update."""
        # Current Q-value
        current_q = self.q_table[state, action]

        # Q-value for next state-action pair (if not done)
        next_q = 0 if done else self.q_table[next_state, next_action]

        # SARSA update
        target_q = reward + self.discount_factor * next_q
        td_error = target_q - current_q
        self.q_table[state, action] += self.learning_rate * td_error

        # Debug logging
        if self.debug:
            self._logger.debug(
                f"SARSA Update: state={state}, action={action}, "
                f"reward={reward:.3f}, next_state={next_state}, next_action={next_action}, "
                f"done={done}, td_error={td_error:.3f}, update_magnitude={abs(self.learning_rate * td_error):.3f}"
            )

    def _expected_sarsa_update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """Perform Expected SARSA update."""
        # Current Q-value
        current_q = self.q_table[state, action]

        if done:
            expected_next_q = 0
        else:
            # Calculate expected Q-value over epsilon-greedy policy
            q_values = self.q_table[next_state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            n_best = len(best_actions)

            # Expected value under epsilon-greedy policy
            prob_explore = self.epsilon / self.action_space_size
            prob_exploit = (1 - self.epsilon) / n_best

            expected_next_q = 0
            for a in range(self.action_space_size):
                if a in best_actions:
                    prob = prob_explore + prob_exploit
                else:
                    prob = prob_explore
                expected_next_q += prob * q_values[a]

        # Expected SARSA update
        target_q = reward + self.discount_factor * expected_next_q
        td_error = target_q - current_q
        self.q_table[state, action] += self.learning_rate * td_error

        # Debug logging
        if self.debug:
            self._logger.debug(
                f"Expected SARSA Update: state={state}, action={action}, "
                f"reward={reward:.3f}, next_state={next_state}, done={done}, "
                f"expected_next_q={expected_next_q:.3f}, td_error={td_error:.3f}, "
                f"update_magnitude={abs(self.learning_rate * td_error):.3f}"
            )

    def decay_epsilon(self) -> None:
        """Decay epsilon for exploration schedule."""
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

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

    def get_action_values(self, state: int) -> list[float]:
        """Get Q-values for all actions in a state.

        Args:
            state: State

        Returns:
            List of Q-values for all actions

        Raises:
            ValueError: If state is invalid
        """
        if not 0 <= state < self.state_space_size:
            raise ValueError(
                f"State {state} is out of range [0, {self.state_space_size})"
            )

        return [
            float(self.q_table[state, action])
            for action in range(self.action_space_size)
        ]

    def get_policy(self) -> list[int]:
        """Get greedy policy with random tie-breaking.

        Returns:
            List of best actions for each state (with random tie-breaking)
        """
        policy = []
        for state in range(self.state_space_size):
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            # Find all actions with maximum Q-value
            best_actions = np.where(q_values == max_q)[0]
            # Randomly select among tied actions
            policy.append(int(np.random.choice(best_actions)))
        return policy

    def get_policy_entropy(self) -> float:
        """Calculate policy entropy across all states.

        Returns:
            Average policy entropy across all states
        """
        total_entropy = 0.0
        valid_states = 0

        for state in range(self.state_space_size):
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            n_best = len(best_actions)

            if n_best > 1:  # Only count states with ties
                # Calculate entropy for this state
                prob_explore = self.epsilon / self.action_space_size
                prob_exploit = (1 - self.epsilon) / n_best

                entropy = 0.0
                for a in range(self.action_space_size):
                    if a in best_actions:
                        prob = prob_explore + prob_exploit
                    else:
                        prob = prob_explore

                    if prob > 0:
                        entropy -= prob * np.log(prob)

                total_entropy += entropy
                valid_states += 1

        return total_entropy / max(valid_states, 1)

    def get_average_q_magnitude(self) -> float:
        """Calculate average Q-value magnitude across all state-action pairs.

        Returns:
            Average absolute Q-value across all state-action pairs
        """
        return float(np.mean(np.abs(self.q_table)))

    def pretty_print_policy(
        self,
        state_names: list[str] | None = None,
        action_names: list[str] | None = None,
    ) -> str:
        """Pretty print the policy with optional state and action names.

        Args:
            state_names: Optional list of state names
            action_names: Optional list of action names

        Returns:
            Formatted string representation of the policy
        """
        policy = self.get_policy()
        lines = ["SARSA Policy:"]

        for state in range(self.state_space_size):
            action = policy[state]
            state_name = (
                state_names[state]
                if state_names and state < len(state_names)
                else f"State {state}"
            )
            action_name = (
                action_names[action]
                if action_names and action < len(action_names)
                else f"Action {action}"
            )
            lines.append(f"  {state_name}: {action_name}")

        return "\n".join(lines)

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
            f"epsilon={self.epsilon:.3f}, "
            f"use_expected_sarsa={self.use_expected_sarsa}, "
            f"debug={self.debug})"
        )
