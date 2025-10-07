"""Options Framework for Hierarchical Reinforcement Learning.

This module implements the Options Framework (Sutton, Precup, Singh 1999),
which provides temporal abstraction through "options" - closed-loop policies
that can be executed over multiple time steps.

An option consists of:
- Initiation set I: States where the option can be initiated
- Policy π: The behavior policy while executing the option
- Termination condition β: Probability of terminating in each state

This implementation includes advanced features:
- Dynamic Q-network resizing for adding new options
- Learnable termination functions β(s)
- Exploration in option policies (softmax/epsilon-greedy)
- Eligibility traces and n-step updates
- Configurable termination for primitive actions

References:
    Sutton, R. S., Precup, D., & Singh, S. (1999). Between MDPs and semi-MDPs:
    A framework for temporal abstraction in reinforcement learning.
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


@dataclass
class Option:
    """Represents an option (temporally extended action).

    Attributes:
        name: Identifier for the option
        initiation_set: Function that returns True if option can be initiated in state
        policy: Function mapping states to actions (can be learned or fixed)
        termination: Function returning probability of termination in state
        is_primitive: Whether this is a primitive action (single step)
        temperature: Temperature for softmax exploration (1.0 = uniform, 0.0 = greedy)
        epsilon: Epsilon for epsilon-greedy exploration in option policy
    """

    name: str
    initiation_set: Callable[[Any], bool]
    policy: Callable[[Any], int | torch.Tensor]
    termination: Callable[[Any], float] | None = None
    is_primitive: bool = False
    temperature: float = 0.0  # For softmax exploration
    epsilon: float = 0.0  # For epsilon-greedy exploration


class TerminationNetwork(nn.Module):
    """Learnable termination function β(s) for options.

    This network learns when options should terminate based on the state,
    rather than using fixed termination conditions.
    """

    def __init__(self, state_size: int, n_options: int, hidden_size: int = 64) -> None:
        """Initialize termination network.

        Args:
            state_size: Dimension of state space
            n_options: Number of options
            hidden_size: Size of hidden layers
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_options),
            nn.Sigmoid(),  # Output termination probabilities [0, 1]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute termination probabilities for all options.

        Args:
            state: Current state

        Returns:
            Termination probabilities for each option
        """
        return self.network(state)


class IntraOptionQLearningConfig(BaseModel):
    """Configuration parameters for IntraOptionQLearning with automatic validation.

    This model uses Pydantic for declarative parameter validation,
    reducing complexity while maintaining strict type safety and
    comprehensive validation.

    Attributes:
        state_size: Dimension of state space (must be positive)
        n_options: Number of options available (must be positive)
        learning_rate: Learning rate for Q-function (0 < lr <= 1)
        gamma: Discount factor (0 <= gamma <= 1)
        lambda_trace: Trace decay parameter (0 <= lambda <= 1)
        n_step: Number of steps for n-step returns (must be positive)
        use_traces: Whether to use eligibility traces
        device: Device for computation
    """

    state_size: int = Field(gt=0, description="Dimension of state space")
    n_options: int = Field(gt=0, description="Number of options available")
    learning_rate: float = Field(
        default=0.001, gt=0.0, le=1.0, description="Learning rate for Q-function"
    )
    gamma: float = Field(default=0.99, ge=0.0, le=1.0, description="Discount factor")
    lambda_trace: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Trace decay parameter (0 = no traces, 1 = full traces)",
    )
    n_step: int = Field(
        default=5, gt=0, description="Number of steps for n-step returns"
    )
    use_traces: bool = Field(
        default=True, description="Whether to use eligibility traces"
    )
    device: str = Field(default="cpu", description="Device for computation")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class IntraOptionQLearning:
    """Intra-option Q-learning with eligibility traces and n-step updates.

    This enables learning while executing options, not just at termination,
    which improves data efficiency. Includes support for:
    - Dynamic network resizing when adding new options
    - Eligibility traces (λ-returns)
    - N-step updates for faster value propagation
    """

    def __init__(
        self,
        config: IntraOptionQLearningConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize intra-option Q-learning.

        Args:
            config: Pre-validated configuration object (recommended)
            **kwargs: Individual parameters for backwards compatibility

        Examples:
            # New style (recommended)
            >>> config = IntraOptionQLearningConfig(state_size=4, n_options=2)
            >>> learner = IntraOptionQLearning(config=config)

            # Old style (backwards compatible)
            >>> learner = IntraOptionQLearning(state_size=4, n_options=2)

        Raises:
            ValidationError: If parameters are invalid (via Pydantic)
        """
        # Validate parameters (automatic via Pydantic)
        if config is None:
            config = IntraOptionQLearningConfig(**kwargs)

        # Store config
        self.config = config

        # Extract all parameters
        self.state_size = config.state_size
        self.n_options = config.n_options
        self.gamma = config.gamma
        self.lambda_trace = config.lambda_trace
        self.n_step = config.n_step
        self.use_traces = config.use_traces
        self.device = torch.device(config.device)
        self.learning_rate = config.learning_rate

        # Q-values over options
        self.q_network = self._create_q_network(self.n_options).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Eligibility traces (one per option)
        self.traces: dict[int, dict[str, torch.Tensor]] = {}
        if self.use_traces:
            self._initialize_traces()

        # N-step buffer for each option
        self.n_step_buffers: dict[int, deque[tuple[torch.Tensor, float]]] = defaultdict(
            lambda: deque(maxlen=self.n_step)
        )

    def _create_q_network(self, n_options: int) -> nn.Sequential:
        """Create Q-network architecture.

        Args:
            n_options: Number of options

        Returns:
            Q-network module
        """
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_options),
        )

    def _initialize_traces(self) -> None:
        """Initialize eligibility traces for all network parameters."""
        for i in range(self.n_options):
            self.traces[i] = {}
            for name, param in self.q_network.named_parameters():
                self.traces[i][name] = torch.zeros_like(param)

    def resize_network(self, new_n_options: int) -> None:
        """Dynamically resize Q-network to accommodate new options.

        This preserves learned weights for existing options and initializes
        new weights for added options.

        Args:
            new_n_options: New total number of options
        """
        if new_n_options <= self.n_options:
            return  # No resize needed

        # Create new network
        new_network = self._create_q_network(new_n_options).to(self.device)

        # Transfer weights for existing options
        old_final_layer = self.q_network[-1]
        new_final_layer = new_network[-1]

        # Copy weights for existing options
        with torch.no_grad():
            new_final_layer.weight[: self.n_options] = old_final_layer.weight
            new_final_layer.bias[: self.n_options] = old_final_layer.bias

            # Copy all other layers
            for i in range(len(self.q_network) - 1):
                if hasattr(self.q_network[i], "weight"):
                    new_network[i].weight.data = self.q_network[i].weight.data.clone()
                    new_network[i].bias.data = self.q_network[i].bias.data.clone()

        # Update network and optimizer
        self.q_network = new_network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.n_options = new_n_options

        # Reinitialize traces if using them
        if self.use_traces:
            self._initialize_traces()

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
        use_n_step: bool = True,
    ) -> float:
        """Update Q-values using intra-option learning with traces/n-step.

        Args:
            state: Current state
            option: Current option being executed
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            next_option: Next option to execute (if any)
            use_n_step: Whether to use n-step updates

        Returns:
            TD error (loss value)
        """
        # Add to n-step buffer
        if use_n_step and self.n_step > 1:
            self.n_step_buffers[option].append((state, reward))

            # Only update when buffer is full or episode ends
            if len(self.n_step_buffers[option]) < self.n_step and not done:
                return 0.0

        # Compute n-step return
        if use_n_step and self.n_step > 1 and self.n_step_buffers[option]:
            n_step_state, n_step_return = self._compute_n_step_return(
                option, next_state, done, next_option
            )
            state_to_use = n_step_state
            reward_to_use = n_step_return
        else:
            state_to_use = state
            reward_to_use = reward

        # Current Q-value
        q_current = self.q_network(state_to_use)[option]

        # Compute target
        with torch.no_grad():
            if done:
                q_target = torch.tensor(reward_to_use, device=self.device)
            else:
                q_next = self.q_network(next_state)
                if next_option is not None:
                    # Option continues
                    q_target = (
                        reward_to_use
                        + (self.gamma ** len(self.n_step_buffers[option]))
                        * q_next[next_option]
                    )
                else:
                    # Option terminates, use max over available options
                    q_target = (
                        reward_to_use
                        + (self.gamma ** len(self.n_step_buffers[option]))
                        * q_next.max()
                    )

        # Compute TD error
        td_error = q_target - q_current

        # Update with eligibility traces or standard gradient
        loss: float | torch.Tensor
        if self.use_traces:
            loss = self._update_with_traces(state_to_use, option, td_error)
        else:
            loss = nn.functional.mse_loss(q_current, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear n-step buffer on termination
        if done or next_option != option:
            self.n_step_buffers[option].clear()

        return loss.item() if isinstance(loss, torch.Tensor) else loss

    def _compute_n_step_return(
        self,
        option: int,
        final_state: torch.Tensor,
        done: bool,
        next_option: int | None,
    ) -> tuple[torch.Tensor, float]:
        """Compute n-step return from buffer.

        Args:
            option: Current option
            final_state: Final state after n steps
            done: Whether episode is done
            next_option: Next option (if any)

        Returns:
            Tuple of (initial state, n-step return)
        """
        buffer = list(self.n_step_buffers[option])
        if not buffer:
            return final_state, 0.0

        initial_state, _ = buffer[0]
        n_step_return = 0.0

        # Accumulate discounted rewards
        for i, (_, reward) in enumerate(buffer):
            n_step_return += (self.gamma**i) * reward

        # Add bootstrap value if not done
        if not done:
            with torch.no_grad():
                q_next = self.q_network(final_state)
                if next_option is not None:
                    bootstrap = q_next[next_option]
                else:
                    bootstrap = q_next.max()
                n_step_return += (self.gamma ** len(buffer)) * bootstrap.item()

        return initial_state, n_step_return

    def _update_with_traces(
        self, state: torch.Tensor, option: int, td_error: torch.Tensor
    ) -> float:
        """Update Q-network using eligibility traces.

        Args:
            state: Current state
            option: Current option
            td_error: TD error

        Returns:
            Loss value
        """
        # Compute gradients
        self.optimizer.zero_grad()
        q_value = self.q_network(state)[option]
        q_value.backward()

        # Update traces and parameters
        with torch.no_grad():
            for name, param in self.q_network.named_parameters():
                if param.grad is not None:
                    # Update trace: z = gamma * lambda * z + grad
                    self.traces[option][name] = (
                        self.gamma * self.lambda_trace * self.traces[option][name]
                        + param.grad
                    )

                    # Update parameter: theta += alpha * delta * z
                    param.data += (
                        self.learning_rate * td_error * self.traces[option][name]
                    )

        return abs(td_error.item())

    def reset_traces(self) -> None:
        """Reset eligibility traces (typically at episode start)."""
        if self.use_traces:
            self._initialize_traces()


class OptionsAgentConfig(BaseModel):
    """Configuration parameters for OptionsAgent with automatic validation.

    This model uses Pydantic for declarative parameter validation,
    reducing complexity while maintaining strict type safety and
    comprehensive validation.

    Attributes:
        state_size: Dimension of state space (must be positive)
        action_size: Dimension of action space (must be positive)
        options: List of available options (if None, creates primitive options)
        learning_rate: Learning rate for Q-learning (0 < lr <= 1)
        termination_lr: Learning rate for termination function (0 < lr <= 1)
        gamma: Discount factor (0 <= gamma <= 1)
        epsilon: Initial exploration rate (0 <= epsilon <= 1)
        epsilon_min: Minimum exploration rate (0 <= epsilon_min <= epsilon)
        epsilon_decay: Decay rate for exploration (0 < decay <= 1)
        lambda_trace: Trace decay parameter (0 <= lambda <= 1)
        n_step: Number of steps for n-step returns (must be positive)
        use_traces: Whether to use eligibility traces
        learn_termination: Whether to learn termination functions
        primitive_termination_prob: Termination probability for primitive options (0 <= prob <= 1)
        termination_entropy_weight: Weight for entropy regularization (>= 0)
        use_option_critic_termination: Use option-critic style termination gradient
        device: Device for computation
        seed: Random seed for reproducibility
    """

    state_size: int = Field(gt=0, description="Dimension of state space")
    action_size: int = Field(gt=0, description="Dimension of action space")
    options: list[Option] | None = Field(
        default=None,
        description="List of available options (if None, creates primitive options)",
    )
    learning_rate: float = Field(
        default=0.001, gt=0.0, le=1.0, description="Learning rate for Q-learning"
    )
    termination_lr: float = Field(
        default=0.001,
        gt=0.0,
        le=1.0,
        description="Learning rate for termination function",
    )
    gamma: float = Field(default=0.99, ge=0.0, le=1.0, description="Discount factor")
    epsilon: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Initial exploration rate"
    )
    epsilon_min: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Minimum exploration rate"
    )
    epsilon_decay: float = Field(
        default=0.995, gt=0.0, le=1.0, description="Decay rate for exploration"
    )
    lambda_trace: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Trace decay parameter for eligibility traces",
    )
    n_step: int = Field(
        default=5, gt=0, description="Number of steps for n-step returns"
    )
    use_traces: bool = Field(
        default=True, description="Whether to use eligibility traces"
    )
    learn_termination: bool = Field(
        default=True, description="Whether to learn termination functions"
    )
    primitive_termination_prob: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Termination probability for primitive options",
    )
    termination_entropy_weight: float = Field(
        default=0.01,
        ge=0.0,
        description="Weight for entropy regularization in termination loss",
    )
    use_option_critic_termination: bool = Field(
        default=False, description="Use option-critic style termination gradient"
    )
    device: str = Field(default="cpu", description="Device for computation")
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("epsilon_min")
    @classmethod
    def validate_epsilon_min(cls, v: float, info: ValidationInfo) -> float:
        """Validate that epsilon_min is not greater than epsilon."""
        epsilon = info.data.get("epsilon", 1.0)
        if v > epsilon:
            raise ValueError(f"epsilon_min ({v}) must be <= epsilon ({epsilon})")
        return v


class OptionsAgent:
    """Options Framework agent with temporal abstraction.

    This agent learns to select among a set of options (temporally extended
    actions) rather than primitive actions at each step. Options can be
    pre-defined skills or learned behaviors.

    Advanced features:
    - Dynamic option addition with Q-network resizing
    - Learnable termination functions
    - Option policy exploration (softmax/epsilon-greedy)
    - Eligibility traces and n-step updates
    """

    def __init__(
        self,
        config: OptionsAgentConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Options Framework agent.

        Args:
            config: Pre-validated configuration object (recommended)
            **kwargs: Individual parameters for backwards compatibility

        Examples:
            # New style (recommended)
            >>> config = OptionsAgentConfig(state_size=4, action_size=2)
            >>> agent = OptionsAgent(config=config)

            # Old style (backwards compatible)
            >>> agent = OptionsAgent(state_size=4, action_size=2)

        Raises:
            ValidationError: If parameters are invalid (via Pydantic)
        """
        # Validate parameters (automatic via Pydantic)
        if config is None:
            config = OptionsAgentConfig(**kwargs)

        # Store config
        self.config = config

        # Set random seed if provided
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            random.seed(config.seed)

        # Extract all parameters
        self.state_size = config.state_size
        self.action_size = config.action_size
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.learn_termination_enabled = config.learn_termination
        self.primitive_termination_prob = config.primitive_termination_prob
        self.termination_entropy_weight = config.termination_entropy_weight
        self.use_option_critic_termination = config.use_option_critic_termination
        self.device = torch.device(config.device)

        # Create default primitive options if none provided
        if config.options is None:
            self.options = self._create_primitive_options()
        else:
            self.options = config.options

        self.n_options = len(self.options)

        # Initialize intra-option Q-learning with advanced features
        self.q_learner = IntraOptionQLearning(
            state_size=self.state_size,
            n_options=self.n_options,
            learning_rate=config.learning_rate,
            gamma=self.gamma,
            lambda_trace=config.lambda_trace,
            n_step=config.n_step,
            use_traces=config.use_traces,
            device=config.device,
        )

        # Initialize learnable termination functions
        self.termination_network: TerminationNetwork | None
        self.termination_optimizer: optim.Adam | None
        if self.learn_termination_enabled:
            self.termination_network = TerminationNetwork(
                self.state_size, self.n_options
            ).to(self.device)
            self.termination_optimizer = optim.Adam(
                self.termination_network.parameters(), lr=config.termination_lr
            )
        else:
            self.termination_network = None
            self.termination_optimizer = None

        # Track current option
        self.current_option: int | None = None
        self.option_start_state: Any = None

        # Statistics
        self.episode_rewards: list[float] = []
        self.option_durations: dict[str, list[int]] = defaultdict(list)
        self.option_frequencies: dict[str, int] = defaultdict(int)
        self.option_successes: dict[str, int] = defaultdict(int)
        self.option_failures: dict[str, int] = defaultdict(int)
        self.option_total_rewards: dict[str, list[float]] = defaultdict(list)
        self.termination_losses: list[float] = []
        self.termination_entropy: list[float] = []

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
                termination=None
                if self.learn_termination_enabled
                else lambda s: self.primitive_termination_prob,
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
        """Get primitive action from option policy with exploration.

        Args:
            state: Current state
            option_idx: Index of option to execute

        Returns:
            Primitive action to take
        """
        option = self.options[option_idx]

        # Apply option-level exploration
        if option.epsilon > 0 and random.random() < option.epsilon:
            # Epsilon-greedy exploration within option
            return random.randint(0, self.action_size - 1)

        action = option.policy(state)

        # Handle both int and tensor returns
        if isinstance(action, torch.Tensor):
            action_value = int(action.item())
        else:
            action_value = action

        # Apply softmax exploration if temperature > 0
        if option.temperature > 0:
            # Create distribution over actions centered on policy action
            logits = torch.zeros(self.action_size)
            logits[action_value] = 1.0 / option.temperature
            probs = torch.softmax(logits, dim=0)
            action_value = int(torch.multinomial(probs, 1).item())

        return action_value

    def should_terminate(self, state: Any, option_idx: int) -> bool:
        """Check if current option should terminate.

        Args:
            state: Current state
            option_idx: Index of current option

        Returns:
            True if option should terminate
        """
        option = self.options[option_idx]

        # Use learned termination if available
        if self.termination_network is not None:
            state_tensor = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                termination_probs = self.termination_network(state_tensor)
                termination_prob = termination_probs[option_idx].item()
        elif option.termination is not None:
            termination_prob = option.termination(state)
        else:
            # Default termination for primitives
            termination_prob = self.primitive_termination_prob

        return random.random() < termination_prob

    def learn_termination(
        self,
        state: torch.Tensor,
        option: int,
        should_terminate: bool,
        advantage: float,
    ) -> tuple[float, float]:
        """Learn termination function using policy gradient with entropy regularization.

        The termination function is trained to maximize advantage:
        - Standard: Terminate when advantage is negative (better options available)
        - Option-critic: Sign-reversed for alignment with option-critic gradient convention

        Includes entropy penalty to prevent premature collapse to 0 or 1.

        Args:
            state: Current state
            option: Current option
            should_terminate: Whether option actually terminated
            advantage: Advantage of current option

        Returns:
            Tuple of (total loss, entropy)
        """
        if self.termination_network is None or self.termination_optimizer is None:
            return 0.0, 0.0

        # Get termination probability
        term_probs = self.termination_network(state)
        term_prob = term_probs[option]

        # Apply option-critic sign reversal if enabled
        effective_advantage = (
            -advantage if self.use_option_critic_termination else advantage
        )

        # Policy gradient loss: -log(π) * A
        # If advantage is negative, we want higher termination probability
        if should_terminate:
            pg_loss = -torch.log(term_prob + 1e-8) * effective_advantage
        else:
            pg_loss = -torch.log(1 - term_prob + 1e-8) * effective_advantage

        # Entropy regularization: -H(β) = -[β*log(β) + (1-β)*log(1-β)]
        # This encourages diversity and prevents collapse to extremes
        entropy = -(
            term_prob * torch.log(term_prob + 1e-8)
            + (1 - term_prob) * torch.log(1 - term_prob + 1e-8)
        )

        # Total loss: policy gradient - entropy bonus
        total_loss = pg_loss - self.termination_entropy_weight * entropy

        # Update termination network
        self.termination_optimizer.zero_grad()
        total_loss.backward()
        self.termination_optimizer.step()

        # Track entropy for monitoring
        self.termination_entropy.append(entropy.item())

        return total_loss.item(), entropy.item()

    def learn(
        self,
        state: torch.Tensor,
        option: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        next_option: int | None = None,
        terminated: bool = False,
    ) -> dict[str, float]:
        """Learn from experience using intra-option Q-learning.

        Args:
            state: Current state
            option: Option that was executed
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            next_option: Next option (if option continues)
            terminated: Whether option terminated

        Returns:
            Dictionary with learning metrics
        """
        # Update Q-values
        loss = self.q_learner.update(
            state, option, reward, next_state, done, next_option
        )

        # Update termination function if learning
        term_loss = 0.0
        term_entropy = 0.0
        if self.termination_network is not None and terminated:
            # Compute advantage for termination learning
            with torch.no_grad():
                q_values = self.q_learner.get_option_values(next_state)
                current_value = q_values[option]
                best_value = q_values.max()
                advantage = (best_value - current_value).item()

            term_loss, term_entropy = self.learn_termination(
                state, option, terminated, advantage
            )
            self.termination_losses.append(term_loss)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return {
            "loss": loss,
            "termination_loss": term_loss,
            "termination_entropy": term_entropy,
            "epsilon": self.epsilon,
        }

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

        # Reset eligibility traces at episode start
        self.q_learner.reset_traces()

        total_reward = 0.0
        steps = 0
        losses = []
        term_losses = []
        term_entropies = []
        option_changes = 0

        # Select initial option
        current_option = self.select_option(state)
        option_start_step = 0
        option_cumulative_reward = 0.0

        for step in range(max_steps):
            # Get action from current option
            action = self.get_action(state.cpu().numpy(), current_option)

            # Take action in environment
            result = env.step(action)
            next_state, reward, done = result[0], result[1], result[2]
            next_state = torch.FloatTensor(next_state).to(self.device)

            total_reward += reward
            option_cumulative_reward += reward
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

                # Record option statistics
                duration = step - option_start_step + 1
                option_name = self.options[current_option].name
                self.option_durations[option_name].append(duration)
                self.option_frequencies[option_name] += 1
                self.option_total_rewards[option_name].append(option_cumulative_reward)

                # Track success/failure based on positive/negative reward
                if option_cumulative_reward > 0:
                    self.option_successes[option_name] += 1
                else:
                    self.option_failures[option_name] += 1

                option_start_step = step + 1
                option_cumulative_reward = 0.0

            # Learn from experience
            metrics = self.learn(
                state,
                current_option,
                reward,
                next_state,
                done,
                next_option,
                terminated=option_terminates,
            )
            losses.append(metrics["loss"])
            if metrics["termination_loss"] > 0:
                term_losses.append(metrics["termination_loss"])
            if metrics["termination_entropy"] > 0:
                term_entropies.append(metrics["termination_entropy"])

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
            "avg_term_loss": float(np.mean(term_losses)) if term_losses else 0.0,
            "avg_term_entropy": float(np.mean(term_entropies))
            if term_entropies
            else 0.0,
            "epsilon": self.epsilon,
            "option_changes": option_changes,
            "avg_option_duration": steps / max(option_changes, 1),
        }

    def add_option(self, option: Option) -> None:
        """Add a new option to the agent's repertoire.

        This dynamically resizes the Q-network to accommodate the new option,
        preserving learned values for existing options.

        Args:
            option: Option to add
        """
        self.options.append(option)
        self.n_options = len(self.options)

        # Resize Q-network to accommodate new option
        self.q_learner.resize_network(self.n_options)

        # Resize termination network if learning termination
        if self.termination_network is not None:
            # Create new termination network
            new_term_network = TerminationNetwork(self.state_size, self.n_options).to(
                self.device
            )

            # Transfer weights for existing options
            with torch.no_grad():
                # Copy shared layers
                for i in range(len(self.termination_network.network) - 1):
                    if hasattr(self.termination_network.network[i], "weight"):
                        new_term_network.network[
                            i
                        ].weight.data = self.termination_network.network[
                            i
                        ].weight.data.clone()
                        new_term_network.network[
                            i
                        ].bias.data = self.termination_network.network[
                            i
                        ].bias.data.clone()

                # Copy output layer for existing options
                old_output = self.termination_network.network[-2]  # Before sigmoid
                new_output = new_term_network.network[-2]
                new_output.weight[: self.n_options - 1] = old_output.weight
                new_output.bias[: self.n_options - 1] = old_output.bias

            self.termination_network = new_term_network
            self.termination_optimizer = optim.Adam(
                self.termination_network.parameters(), lr=0.001
            )

    def get_statistics(self) -> dict[str, Any]:
        """Get agent statistics including per-option success rates and rewards.

        Returns:
            Dictionary with comprehensive agent statistics
        """
        # Compute per-option success rates
        option_success_rates = {}
        for name in self.option_frequencies:
            total = self.option_successes[name] + self.option_failures[name]
            if total > 0:
                option_success_rates[name] = self.option_successes[name] / total
            else:
                option_success_rates[name] = 0.0

        stats = {
            "total_episodes": len(self.episode_rewards),
            "avg_reward": (
                np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
            ),
            "n_options": self.n_options,
            "option_frequencies": dict(self.option_frequencies),
            "avg_option_durations": {
                name: float(np.mean(durations))
                for name, durations in self.option_durations.items()
                if durations
            },
            "avg_option_rewards": {
                name: float(np.mean(rewards))
                for name, rewards in self.option_total_rewards.items()
                if rewards
            },
            "option_success_rates": option_success_rates,
            "option_successes": dict(self.option_successes),
            "option_failures": dict(self.option_failures),
            "epsilon": self.epsilon,
        }

        if self.termination_losses:
            stats["avg_termination_loss"] = float(
                np.mean(self.termination_losses[-100:])
            )

        if self.termination_entropy:
            stats["avg_termination_entropy"] = float(
                np.mean(self.termination_entropy[-100:])
            )
            stats["min_termination_entropy"] = float(
                min(self.termination_entropy[-100:])
            )
            stats["max_termination_entropy"] = float(
                max(self.termination_entropy[-100:])
            )

        return stats
