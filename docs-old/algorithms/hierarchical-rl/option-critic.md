---
algorithm_key: "option-critic"
tags: [hierarchical-rl, algorithms, option-critic, temporal-abstraction, options, termination]
title: "Option-Critic"
family: "hierarchical-rl"
---

# Option-Critic

{{ algorithm_card("option-critic") }}

!!! abstract "Overview"
    Option-Critic is a hierarchical reinforcement learning algorithm that learns options (temporally extended actions) end-to-end using policy gradient methods. The algorithm automatically discovers useful options that can be reused across different tasks, enabling temporal abstraction and improved sample efficiency.

    This approach learns three components simultaneously: an option policy that selects actions given an option, an option selection policy that chooses which option to execute, and termination functions that determine when to end options. Option-Critic is particularly powerful in domains where tasks have natural temporal structure, such as robotics manipulation, navigation, and game playing.

## Mathematical Formulation

!!! math "Option-Critic Framework"
    An option consists of three components:

    - **Option Policy**: $\pi_\omega(a|s)$ - selects actions given option $\omega$ and state $s$
    - **Option Selection Policy**: $\pi_\Omega(\omega|s)$ - selects options given state $s$
    - **Termination Function**: $\beta_\omega(s)$ - probability of terminating option $\omega$ in state $s$

    The option-critic policy gradient theorem states:

    $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\Omega(\omega_t|s_t) A_\Omega(s_t, \omega_t) + \nabla_\theta \log \pi_\omega(a_t|s_t, \omega_t) A_\omega(s_t, \omega_t, a_t) \right]$$

    Where $A_\Omega$ and $A_\omega$ are advantage functions for option selection and action selection respectively.

!!! success "Key Properties"
    - **End-to-End Learning**: All components learned simultaneously
    - **Automatic Option Discovery**: Useful options emerge from learning
    - **Temporal Abstraction**: Options operate over extended time horizons
    - **Reusability**: Learned options can be applied to new tasks
    - **Sample Efficiency**: Better exploration and learning in complex environments

## Implementation Approaches

=== "Basic Option-Critic (Recommended)"
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import numpy as np

    class OptionPolicyNetwork(nn.Module):
        """Option policy network that selects actions given an option."""

        def __init__(self, state_size: int, option_size: int, action_size: int, hidden_size: int = 128):
            super(OptionPolicyNetwork, self).__init__()
            self.fc1 = nn.Linear(state_size + option_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, action_size)

        def forward(self, state: torch.Tensor, option: torch.Tensor) -> torch.Tensor:
            # Concatenate state and option
            combined = torch.cat([state, option], dim=-1)
            x = F.relu(self.fc1(combined))
            x = F.relu(self.fc2(x))
            action_probs = F.softmax(self.fc3(x), dim=-1)
            return action_probs

    class OptionSelectionNetwork(nn.Module):
        """Network that selects which option to execute."""

        def __init__(self, state_size: int, option_size: int, hidden_size: int = 128):
            super(OptionSelectionNetwork, self).__init__()
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, option_size)

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            option_probs = F.softmax(self.fc3(x), dim=-1)
            return option_probs

    class TerminationNetwork(nn.Module):
        """Network that determines when to terminate options."""

        def __init__(self, state_size: int, option_size: int, hidden_size: int = 128):
            super(TerminationNetwork, self).__init__()
            self.fc1 = nn.Linear(state_size + option_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, 1)

        def forward(self, state: torch.Tensor, option: torch.Tensor) -> torch.Tensor:
            # Concatenate state and option
            combined = torch.cat([state, option], dim=-1)
            x = F.relu(self.fc1(combined))
            x = F.relu(self.fc2(x))
            termination_prob = torch.sigmoid(self.fc3(x))
            return termination_prob

    class OptionCriticAgent:
        """
        Option-Critic agent implementation.

        Args:
            state_size: Dimension of state space
            option_size: Number of possible options
            action_size: Number of possible actions
            option_lr: Learning rate for option policy (default: 0.001)
            selection_lr: Learning rate for option selection (default: 0.001)
            termination_lr: Learning rate for termination function (default: 0.001)
            discount_factor: Discount factor gamma (default: 0.99)
            option_timeout: Maximum steps per option (default: 100)
        """

        def __init__(self, state_size: int, option_size: int, action_size: int,
                     option_lr: float = 0.001, selection_lr: float = 0.001, termination_lr: float = 0.001,
                     discount_factor: float = 0.99, option_timeout: int = 100):

            self.state_size = state_size
            self.option_size = option_size
            self.action_size = action_size
            self.gamma = discount_factor
            self.option_timeout = option_timeout

            # Networks
            self.option_policy = OptionPolicyNetwork(state_size, option_size, action_size)
            self.option_selection = OptionSelectionNetwork(state_size, option_size)
            self.termination = TerminationNetwork(state_size, option_size)

            # Optimizers
            self.option_optimizer = optim.Adam(self.option_policy.parameters(), lr=option_lr)
            self.selection_optimizer = optim.Adam(self.option_selection.parameters(), lr=selection_lr)
            self.termination_optimizer = optim.Adam(self.termination.parameters(), lr=termination_lr)

            # Experience buffers
            self.option_buffer = []
            self.selection_buffer = []
            self.termination_buffer = []

            # Current option tracking
            self.current_option = None
            self.option_steps = 0

        def get_option(self, state: np.ndarray) -> tuple[int, float]:
            """Select option using option selection network."""
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            option_probs = self.option_selection(state_tensor)

            # Sample option
            dist = torch.distributions.Categorical(option_probs)
            option = dist.sample()
            log_prob = dist.log_prob(option)

            return option.item(), log_prob.item()

        def get_action(self, state: np.ndarray, option: int) -> tuple[int, float]:
            """Get action using option policy given option."""
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            option_tensor = torch.FloatTensor([option]).unsqueeze(0)

            action_probs = self.option_policy(state_tensor, option_tensor)

            # Sample action
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return action.item(), log_prob.item()

        def should_terminate(self, state: np.ndarray, option: int) -> bool:
            """Determine if current option should terminate."""
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            option_tensor = torch.FloatTensor([option]).unsqueeze(0)

            termination_prob = self.termination(state_tensor, option_tensor)

            # Sample termination decision
            return torch.rand(1) < termination_prob

        def step(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, done: bool) -> None:
            """Process one step and potentially update option."""
            # Store option transition
            if self.current_option is not None:
                self.option_buffer.append((state, self.current_option, action, reward, next_state, done))

            # Check if option should terminate
            self.option_steps += 1
            if (self.option_steps >= self.option_timeout or
                self.should_terminate(next_state, self.current_option) or done):

                # Store selection transition
                if len(self.option_buffer) > 0:
                    total_reward = sum(r for _, _, _, r, _, _ in self.option_buffer)
                    self.selection_buffer.append((state, self.current_option, total_reward, next_state, done))

                # Select new option
                if not done:
                    new_option, _ = self.get_option(next_state)
                    self.current_option = new_option
                    self.option_steps = 0

                # Clear option buffer
                self.option_buffer.clear()

        def update_networks(self) -> None:
            """Update all networks."""
            self.update_option_policy()
            self.update_option_selection()
            self.update_termination()

        def update_option_policy(self) -> None:
            """Update option policy using stored experience."""
            if len(self.option_buffer) < 10:
                return

            # Sample batch from option buffer
            batch = np.random.choice(len(self.option_buffer), min(32, len(self.option_buffer)), replace=False)

            states, options, actions, rewards, next_states, dones = zip(*[self.option_buffer[i] for i in batch])

            # Convert to tensors
            states = torch.FloatTensor(states)
            options = torch.LongTensor(options)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.BoolTensor(dones)

            # Compute advantages (simplified)
            advantages = rewards + self.gamma * ~dones - 0.5  # Baseline of 0.5

            # Get current action probabilities
            action_probs = self.option_policy(states, F.one_hot(options, self.option_size).float())
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)

            # Policy gradient loss
            policy_loss = -(log_probs * advantages.detach()).mean()

            # Update option policy
            self.option_optimizer.zero_grad()
            policy_loss.backward()
            self.option_optimizer.step()

        def update_option_selection(self) -> None:
            """Update option selection policy."""
            if len(self.selection_buffer) < 10:
                return

            # Sample batch from selection buffer
            batch = np.random.choice(len(self.selection_buffer), min(32, len(self.selection_buffer)), replace=False)

            states, options, rewards, next_states, dones = zip(*[self.selection_buffer[i] for i in batch])

            # Convert to tensors
            states = torch.FloatTensor(states)
            options = torch.LongTensor(options)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.BoolTensor(dones)

            # Compute advantages (simplified)
            advantages = rewards + self.gamma * ~dones - 0.5  # Baseline of 0.5

            # Get current option probabilities
            option_probs = self.option_selection(states)
            dist = torch.distributions.Categorical(option_probs)
            log_probs = dist.log_prob(options)

            # Policy gradient loss
            policy_loss = -(log_probs * advantages.detach()).mean()

            # Update option selection
            self.selection_optimizer.zero_grad()
            policy_loss.backward()
            self.selection_optimizer.step()

        def update_termination(self) -> None:
            """Update termination function."""
            if len(self.option_buffer) < 10:
                return

            # Sample batch from option buffer
            batch = np.random.choice(len(self.option_buffer), min(32, len(self.option_buffer)), replace=False)

            states, options, actions, rewards, next_states, dones = zip(*[self.option_buffer[i] for i in batch])

            # Convert to tensors
            states = torch.FloatTensor(states)
            options = torch.LongTensor(options)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.BoolTensor(dones)

            # Termination target: 1 if done, 0 otherwise
            termination_targets = dones.float()

            # Get current termination probabilities
            termination_probs = self.termination(states, F.one_hot(options, self.option_size).float()).squeeze()

            # Binary cross entropy loss
            termination_loss = F.binary_cross_entropy(termination_probs, termination_targets)

            # Update termination network
            self.termination_optimizer.zero_grad()
            termination_loss.backward()
            self.termination_optimizer.step()
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/hierarchical_rl/option_critic.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/hierarchical_rl/option_critic.py)
    - **Tests**: [`tests/unit/hierarchical_rl/test_option_critic.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/hierarchical_rl/test_option_critic.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Basic Option-Critic** | O(batch_size × (policy_params + option_params + termination_params)) | O(batch_size × (state_size + option_size)) | Three-network architecture |
    **Option-Critic + Experience Replay** | O(batch_size × (policy_params + option_params + termination_params)) | O(batch_size × (state_size + option_size) + buffer_size) | Better sample efficiency |
    **Option-Critic + Value Networks** | O(batch_size × (policy_params + option_params + termination_params + value_params)) | O(batch_size × (state_size + option_size)) | Value function estimation |

!!! warning "Performance Considerations"
    - **Three networks** require careful coordination
    - **Option timeout** affects exploration and learning efficiency
    - **Termination function** is crucial for proper option learning
    - **Option selection** depends on option policy performance

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Robotics & Control"
        - **Robot Manipulation**: Complex manipulation tasks with options
        - **Autonomous Navigation**: Multi-level navigation planning
        - **Industrial Automation**: Process optimization with temporal abstraction
        - **Swarm Robotics**: Coordinated multi-agent behavior

    !!! grid-item "Game AI & Entertainment"
        - **Strategy Games**: Multi-level decision making and planning
        - **Open-World Games**: Complex task decomposition and execution
        - **Simulation Games**: Resource management with hierarchical goals
        - **Virtual Environments**: NPC behavior with long-term objectives

    !!! grid-item "Real-World Applications"
        - **Autonomous Vehicles**: Multi-level driving behavior and navigation
        - **Healthcare**: Treatment planning with hierarchical objectives
        - **Finance**: Portfolio management with temporal abstraction
        - **Network Control**: Traffic management with hierarchical policies

    !!! grid-item "Educational Value"
        - **Option Learning**: Understanding temporal abstraction in RL
        - **Automatic Discovery**: Learning useful options automatically
        - **Termination Functions**: Understanding when to end options
        - **Reusability**: Learning skills that can be reused

!!! success "Educational Value"
    - **Temporal Abstraction**: Perfect example of learning extended actions
    - **Option Discovery**: Shows how useful abstractions emerge from learning
    - **Termination Learning**: Demonstrates learning when to end options
    - **Skill Reuse**: Illustrates how options can be applied to new tasks

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Papers"
        1. **Bacon, P. L., et al.** (2017). The Option-Critic Architecture. *AAAI*, 31.

    !!! grid-item "Hierarchical RL Textbooks"
        2. **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction*. MIT Press.
        3. **Kaelbling, L. P., et al.** (1998). Hierarchical reinforcement learning with the MAXQ value function decomposition. *Journal of Artificial Intelligence Research*, 13.

    !!! grid-item "Online Resources"
        4. [Option-Critic - Wikipedia](https://en.wikipedia.org/wiki/Option-critic)
        5. [Option-Critic Implementation Guide](https://github.com/jeanfrancoisgagnon/option_critic)
        6. [Hierarchical RL Tutorial](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

    !!! grid-item "Implementation & Practice"
        7. [PyTorch Documentation](https://pytorch.org/docs/)
        8. [OpenAI Gym](https://www.gymlibrary.dev/) - RL environments
        9. [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - RL implementations

!!! tip "Interactive Learning"
    Try implementing Option-Critic yourself! Start with simple environments that have natural temporal structure, like navigation tasks with waypoints. Implement the basic three-network architecture first, then add experience replay for better sample efficiency. Experiment with different option timeouts and termination functions to see their impact on learning. Compare with flat approaches to see the benefits of temporal abstraction. This will give you deep insight into the power of learning useful options automatically in reinforcement learning.

## Navigation

{{ nav_grid(current_algorithm="option-critic", current_family="hierarchical-rl", max_related=5) }}
