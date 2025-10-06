"""Options Framework for Hierarchical Reinforcement Learning.

This module implements the Options Framework (Sutton, Precup, Singh 1999),
which provides temporal abstraction through "options" - closed-loop policies
that can be executed over multiple time steps.

An option consists of:
- Initiation set I: States where the option can be initiated
- Policy π: The behavior policy while executing the option
- Termination condition β: Probability of terminating in each state

References:
    Sutton, R. S., Precup, D., & Singh, S. (1999). Between MDPs and semi-MDPs:
    A framework for temporal abstraction in reinforcement learning.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class Option:
    """Represents an option (temporally extended action).

    Attributes:
        name: Identifier for the option
        initiation_set: Function that returns True if option can be initiated in state
        policy: Function mapping states to actions (can be learned or fixed)
        termination: Function returning probability of termination in state
        is_primitive: Whether this is a primitive action (single step)
    """

    name: str
    initiation_set: Callable[[Any], bool]
    policy: Callable[[Any], int | torch.Tensor]
    termination: Callable[[Any], float]
    is_primitive: bool = False


class IntraOptionQLearning:
    """Intra-option Q-learning for learning option values.

    This enables learning while executing options, not just at termination,
    which improves data efficiency.
    """

    def __init__(
        self,
        state_size: int,
        n_options: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        device: str = "cpu",
    ) -> None:
        """Initialize intra-option Q-learning.

        Args:
            state_size: Dimension of state space
            n_options: Number of options available
            learning_rate: Learning rate for Q-function
            gamma: Discount factor
            device: Device for computation
        """
        self.state_size = state_size
        self.n_options = n_options
        self.gamma = gamma
        self.device = torch.device(device)

        # Q-values over options
        self.q_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_options),
        ).to(self.device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def get_option_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get Q-values for all options in given state.

        Args:
            state: Current state

        Returns:
            Q-values for each option
        """
        return self.q_network(state)

    def update(
        self,
        state: torch.Tensor,
        option: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        next_option: int | None = None,
    ) -> float:
        """Update Q-values using intra-option learning.

        Args:
            state: Current state
            option: Current option being executed
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            next_option: Next option to execute (if any)

        Returns:
            TD error (loss value)
        """
        # Current Q-value
        q_current = self.q_network(state)[option]

        # Compute target
        with torch.no_grad():
            if done:
                q_target = torch.tensor(reward, device=self.device)
            else:
                q_next = self.q_network(next_state)
                if next_option is not None:
                    # Option continues
                    q_target = reward + self.gamma * q_next[next_option]
                else:
                    # Option terminates, use max over available options
                    q_target = reward + self.gamma * q_next.max()

        # Compute loss and update
        loss = nn.functional.mse_loss(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class OptionsAgent:
    """Options Framework agent with temporal abstraction.

    This agent learns to select among a set of options (temporally extended
    actions) rather than primitive actions at each step. Options can be
    pre-defined skills or learned behaviors.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        options: list[Option] | None = None,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        """Initialize Options Framework agent.

        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space (for primitive actions)
            options: List of available options (if None, creates primitive options)
            learning_rate: Learning rate for Q-learning
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            device: Device for computation
            seed: Random seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = torch.device(device)

        # Create default primitive options if none provided
        if options is None:
            self.options = self._create_primitive_options()
        else:
            self.options = options

        self.n_options = len(self.options)

        # Initialize intra-option Q-learning
        self.q_learner = IntraOptionQLearning(
            state_size=state_size,
            n_options=self.n_options,
            learning_rate=learning_rate,
            gamma=gamma,
            device=device,
        )

        # Track current option
        self.current_option: int | None = None
        self.option_start_state: Any = None

        # Statistics
        self.episode_rewards: list[float] = []
        self.option_durations: dict[str, list[int]] = defaultdict(list)
        self.option_frequencies: dict[str, int] = defaultdict(int)

    def _create_primitive_options(self) -> list[Option]:
        """Create primitive options (one per action).

        Returns:
            List of primitive options
        """
        options = []
        for action in range(self.action_size):

            def make_policy(a: int) -> Callable[[Any], int]:
                return lambda s: a

            option = Option(
                name=f"primitive_{action}",
                initiation_set=lambda s: True,  # Can always be initiated
                policy=make_policy(action),  # Always returns the same action
                termination=lambda s: 1.0,  # Always terminates after one step
                is_primitive=True,
            )
            options.append(option)
        return options

    def select_option(
        self, state: torch.Tensor, available_options: list[int] | None = None
    ) -> int:
        """Select an option using epsilon-greedy policy.

        Args:
            state: Current state
            available_options: Indices of options that can be initiated (if None, checks initiation sets)

        Returns:
            Index of selected option
        """
        # Determine available options
        if available_options is None:
            available_options = [
                i for i, opt in enumerate(self.options) if opt.initiation_set(state)
            ]

        if not available_options:
            # If no options available, fallback to random primitive action
            available_options = [
                i for i, opt in enumerate(self.options) if opt.is_primitive
            ]

        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            return random.choice(available_options)
        else:
            with torch.no_grad():
                q_values = self.q_learner.get_option_values(state)
                # Mask unavailable options
                masked_q = torch.full_like(q_values, float("-inf"))
                masked_q[available_options] = q_values[available_options]
                return int(masked_q.argmax().item())

    def get_action(self, state: Any, option_idx: int) -> int:
        """Get primitive action from option policy.

        Args:
            state: Current state
            option_idx: Index of option to execute

        Returns:
            Primitive action to take
        """
        option = self.options[option_idx]
        action = option.policy(state)

        # Handle both int and tensor returns
        if isinstance(action, torch.Tensor):
            return int(action.item())
        return action

    def should_terminate(self, state: Any, option_idx: int) -> bool:
        """Check if current option should terminate.

        Args:
            state: Current state
            option_idx: Index of current option

        Returns:
            True if option should terminate
        """
        option = self.options[option_idx]
        termination_prob = option.termination(state)
        return random.random() < termination_prob

    def learn(
        self,
        state: torch.Tensor,
        option: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        next_option: int | None = None,
    ) -> dict[str, float]:
        """Learn from experience using intra-option Q-learning.

        Args:
            state: Current state
            option: Option that was executed
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            next_option: Next option (if option continues)

        Returns:
            Dictionary with learning metrics
        """
        loss = self.q_learner.update(
            state, option, reward, next_state, done, next_option
        )

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return {"loss": loss, "epsilon": self.epsilon}

    def train_episode(
        self, env: Any, max_steps: int = 1000
    ) -> dict[str, float | list[float]]:
        """Train for one episode using options.

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
        losses = []
        option_changes = 0

        # Select initial option
        current_option = self.select_option(state)
        option_start_step = 0

        for step in range(max_steps):
            # Get action from current option
            action = self.get_action(state.cpu().numpy(), current_option)

            # Take action in environment
            result = env.step(action)
            next_state, reward, done = result[0], result[1], result[2]
            next_state = torch.FloatTensor(next_state).to(self.device)

            total_reward += reward
            steps += 1

            # Check if option should terminate
            option_terminates = self.should_terminate(
                next_state.cpu().numpy(), current_option
            )

            # Determine next option
            next_option = None
            if not done and not option_terminates:
                next_option = current_option  # Continue with same option
            elif not done:
                # Option terminates, select new option
                next_option = self.select_option(next_state)
                option_changes += 1

                # Record option duration
                duration = step - option_start_step + 1
                option_name = self.options[current_option].name
                self.option_durations[option_name].append(duration)
                self.option_frequencies[option_name] += 1
                option_start_step = step + 1

            # Learn from experience
            metrics = self.learn(
                state, current_option, reward, next_state, done, next_option
            )
            losses.append(metrics["loss"])

            # Update state and option
            state = next_state
            if next_option is not None:
                current_option = next_option

            if done:
                break

        self.episode_rewards.append(total_reward)

        return {
            "reward": total_reward,
            "steps": steps,
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
            "epsilon": self.epsilon,
            "option_changes": option_changes,
            "avg_option_duration": steps / max(option_changes, 1),
        }

    def add_option(self, option: Option) -> None:
        """Add a new option to the agent's repertoire.

        Args:
            option: Option to add
        """
        self.options.append(option)
        self.n_options = len(self.options)

        # Note: This would require resizing the Q-network in practice
        # For now, this is a placeholder for the API

    def get_statistics(self) -> dict[str, Any]:
        """Get agent statistics.

        Returns:
            Dictionary with agent statistics
        """
        return {
            "total_episodes": len(self.episode_rewards),
            "avg_reward": (
                np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
            ),
            "n_options": self.n_options,
            "option_frequencies": dict(self.option_frequencies),
            "avg_option_durations": {
                name: np.mean(durations)
                for name, durations in self.option_durations.items()
            },
            "epsilon": self.epsilon,
        }
