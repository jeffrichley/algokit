---
algorithm_key: "sarsa"
tags: [reinforcement-learning, algorithms, sarsa, temporal-difference, on-policy, markov-decision-process]
title: "SARSA (State-Action-Reward-State-Action)"
family: "reinforcement-learning"
---

# SARSA (State-Action-Reward-State-Action)

{{ algorithm_card("sarsa") }}

!!! abstract "Overview"
    SARSA is an on-policy, model-free reinforcement learning algorithm that learns the action-value function (Q-function) for a Markov Decision Process (MDP). Unlike Q-Learning, which is off-policy and learns the optimal policy regardless of the behavior policy, SARSA learns the Q-values for the policy it's currently following.

    The name SARSA comes from the sequence of elements used in the update: (State, Action, Reward, State, Action). This on-policy nature makes SARSA more conservative and potentially safer in scenarios where exploration can be dangerous, as it learns the value of the policy it's actually executing rather than an optimal policy that might require risky exploration.

## Mathematical Formulation

!!! math "SARSA Update Rule"
    The core SARSA update rule follows the temporal difference learning equation:

    $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]$$

    Where:
    - $Q(s, a)$ is the Q-value for state $s$ and action $a$
    - $\alpha$ is the learning rate (controls update magnitude)
    - $r_t$ is the immediate reward at time $t$
    - $\gamma$ is the discount factor (future reward importance)
    - $Q(s_{t+1}, a_{t+1})$ is the Q-value for the next state-action pair

!!! success "Key Properties"
    - **On-Policy**: Learns the value of the policy being followed
    - **Model-Free**: Doesn't require knowledge of environment dynamics
    - **Temporal Difference**: Updates estimates based on difference between predicted and actual values
    - **Conservative**: Generally safer than off-policy methods in risky environments

## Implementation Approaches

=== "Standard SARSA (Recommended)"
    ```python
    import numpy as np
    from typing import Dict, Tuple, Any

    class SARSAAgent:
        """
        SARSA agent implementation.

        Args:
            state_size: Number of possible states
            action_size: Number of possible actions
            learning_rate: Learning rate alpha (default: 0.1)
            discount_factor: Discount factor gamma (default: 0.95)
            epsilon: Exploration rate for epsilon-greedy (default: 0.1)
        """

        def __init__(self, state_size: int, action_size: int,
                     learning_rate: float = 0.1, discount_factor: float = 0.95,
                     epsilon: float = 0.1):
            self.state_size = state_size
            self.action_size = action_size
            self.alpha = learning_rate
            self.gamma = discount_factor
            self.epsilon = epsilon

            # Initialize Q-table with zeros
            self.q_table = np.zeros((state_size, action_size))

            # Store current state and action for SARSA update
            self.current_state = None
            self.current_action = None

        def get_action(self, state: int) -> int:
            """Choose action using epsilon-greedy policy."""
            if np.random.random() < self.epsilon:
                # Exploration: random action
                return np.random.randint(self.action_size)
            else:
                # Exploitation: best action according to Q-table
                return np.argmax(self.q_table[state])

        def start_episode(self, state: int) -> int:
            """Start a new episode and return first action."""
            self.current_state = state
            self.current_action = self.get_action(state)
            return self.current_action

        def step(self, reward: float, next_state: int, done: bool) -> int:
            """Take a step in the environment and return next action."""
            if self.current_state is not None and self.current_action is not None:
                # SARSA update using current (s,a) and next (s',a')
                if done:
                    # Terminal state: no future rewards
                    next_q = 0
                else:
                    # Choose next action for SARSA update
                    next_action = self.get_action(next_state)
                    next_q = self.q_table[next_state, next_action]

                # SARSA update rule
                current_q = self.q_table[self.current_state, self.current_action]
                new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
                self.q_table[self.current_state, self.current_action] = new_q

                # Update current state and action
                self.current_state = next_state
                self.current_action = next_action if not done else None

                return next_action if not done else None

            return None

        def get_policy(self) -> np.ndarray:
            """Extract greedy policy from Q-table."""
            return np.argmax(self.q_table, axis=1)
    ```

=== "Expected SARSA (Advanced)"
    ```python
    class ExpectedSARSAAgent:
        """
        Expected SARSA agent that uses expected value of next state.
        """

        def __init__(self, state_size: int, action_size: int,
                     learning_rate: float = 0.1, discount_factor: float = 0.95,
                     epsilon: float = 0.1):
            self.state_size = state_size
            self.action_size = action_size
            self.alpha = learning_rate
            self.gamma = discount_factor
            self.epsilon = epsilon

            # Initialize Q-table with zeros
            self.q_table = np.zeros((state_size, action_size))

        def get_action(self, state: int) -> int:
            """Choose action using epsilon-greedy policy."""
            if np.random.random() < self.epsilon:
                return np.random.randint(self.action_size)
            else:
                return np.argmax(self.q_table[state])

        def update(self, state: int, action: int, reward: float,
                  next_state: int, done: bool) -> None:
            """Update Q-value using Expected SARSA."""
            current_q = self.q_table[state, action]

            if done:
                next_q = 0
            else:
                # Calculate expected Q-value of next state
                next_q_values = self.q_table[next_state]
                max_q = np.max(next_q_values)

                # Expected value considering epsilon-greedy policy
                next_q = (1 - self.epsilon) * max_q + (self.epsilon / self.action_size) * np.sum(next_q_values)

            # Expected SARSA update
            new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
            self.q_table[state, action] = new_q
    ```

=== "SARSA with Eligibility Traces (SARSA(λ))"
    ```python
    class SARSALambdaAgent:
        """
        SARSA with eligibility traces for better credit assignment.
        """

        def __init__(self, state_size: int, action_size: int,
                     learning_rate: float = 0.1, discount_factor: float = 0.95,
                     epsilon: float = 0.1, lambda_param: float = 0.9):
            self.state_size = state_size
            self.action_size = action_size
            self.alpha = learning_rate
            self.gamma = discount_factor
            self.epsilon = epsilon
            self.lambda_param = lambda_param

            # Initialize Q-table and eligibility traces
            self.q_table = np.zeros((state_size, action_size))
            self.eligibility_traces = np.zeros((state_size, action_size))

        def get_action(self, state: int) -> int:
            """Choose action using epsilon-greedy policy."""
            if np.random.random() < self.epsilon:
                return np.random.randint(self.action_size)
            else:
                return np.argmax(self.q_table[state])

        def update(self, state: int, action: int, reward: float,
                  next_state: int, next_action: int, done: bool) -> None:
            """Update Q-values using SARSA(λ)."""
            # Calculate TD error
            if done:
                target = reward
            else:
                target = reward + self.gamma * self.q_table[next_state, next_action]

            td_error = target - self.q_table[state, action]

            # Update eligibility traces
            self.eligibility_traces[state, action] += 1

            # Update all Q-values based on eligibility traces
            for s in range(self.state_size):
                for a in range(self.action_size):
                    if self.eligibility_traces[s, a] > 0:
                        self.q_table[s, a] += self.alpha * td_error * self.eligibility_traces[s, a]
                        self.eligibility_traces[s, a] *= self.gamma * self.lambda_param
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/reinforcement_learning/sarsa.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/reinforcement_learning/sarsa.py)
    - **Tests**: [`tests/unit/reinforcement_learning/test_sarsa.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/reinforcement_learning/test_sarsa.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Standard SARSA** | O(|S| × |A|) per update | O(|S| × |A|) | On-policy, safe exploration |
    **Expected SARSA** | O(|S| × |A|) per update | O(|S| × |A|) | Better convergence, more stable |
    **SARSA(λ)** | O(|S| × |A|) per update | O(|S| × |A|) | Better credit assignment, more complex |

!!! warning "Performance Considerations"
    - **On-policy nature** makes SARSA more conservative than Q-Learning
    - **Expected SARSA** generally provides better convergence properties
    - **Eligibility traces** improve credit assignment but increase complexity
    - **Exploration strategy** significantly affects learning performance

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Safe Learning Scenarios"
        - **Robotics**: Safe exploration in physical environments
        - **Autonomous Vehicles**: Conservative policy learning
        - **Healthcare**: Safe treatment policy optimization
        - **Finance**: Risk-averse trading strategies

    !!! grid-item "Game AI & Simulation"
        - **Strategy Games**: Learning from actual gameplay
        - **Simulation Environments**: Safe policy exploration
        - **Multi-agent Systems**: Coordinated policy learning
        - **Educational Games**: Adaptive difficulty adjustment

    !!! grid-item "Real-World Applications"
        - **Resource Management**: Conservative allocation policies
        - **Process Control**: Safe industrial process optimization
        - **Network Routing**: Stable routing policy learning
        - **Energy Management**: Safe consumption optimization

    !!! grid-item "Educational Value"
        - **Reinforcement Learning**: Understanding on-policy vs off-policy methods
        - **Temporal Difference**: Learning from sequential experience
        - **Policy Evaluation**: Understanding how to assess current policies
        - **Safe Exploration**: Learning importance of conservative approaches

!!! success "Educational Value"
    - **Reinforcement Learning**: Perfect example of on-policy temporal difference learning
    - **Policy Evaluation**: Shows how to learn the value of current policies
    - **Safe Exploration**: Demonstrates importance of conservative learning approaches
    - **Algorithm Comparison**: Illustrates differences between SARSA and Q-Learning

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Textbooks"
        1. **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction*. MIT Press. ISBN 978-0-262-03924-7.
        2. **Szepesvári, C.** (2010). *Algorithms for Reinforcement Learning*. Morgan & Claypool. ISBN 978-1-60845-492-1.

    !!! grid-item "SARSA Specific"
        3. **Rummery, G. A., & Niranjan, M.** (1994). On-line Q-learning using connectionist systems. *Technical Report CUED/F-INFENG/TR 166*.
        4. **Sutton, R. S.** (1996). Generalization in reinforcement learning: Successful examples using sparse coarse coding. *Advances in Neural Information Processing Systems*, 8.

    !!! grid-item "Online Resources"
        5. [SARSA - Wikipedia](https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action)
        6. [Reinforcement Learning Tutorial](https://www.gymlibrary.dev/)
        7. [SARSA vs Q-Learning Comparison](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

    !!! grid-item "Implementation & Practice"
        8. [Python Official Documentation](https://docs.python.org/3/)
        9. [NumPy Documentation](https://numpy.org/doc/)
        10. [OpenAI Gym](https://www.gymlibrary.dev/) - RL environments for testing

!!! tip "Interactive Learning"
    Try implementing SARSA yourself! Compare it with Q-Learning on the same environment to see the differences in behavior. Implement Expected SARSA to see how it improves convergence. Experiment with different lambda values in SARSA(λ) to understand how eligibility traces affect learning. This will give you deep insight into the trade-offs between on-policy and off-policy methods.

## Navigation

{{ nav_grid(current_algorithm="sarsa", current_family="reinforcement-learning", max_related=5) }}
