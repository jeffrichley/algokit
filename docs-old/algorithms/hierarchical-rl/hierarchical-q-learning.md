---
algorithm_key: "hierarchical-q-learning"
tags: [hierarchical-rl, algorithms, hierarchical-q-learning, temporal-abstraction, subgoals, value-based]
title: "Hierarchical Q-Learning"
family: "hierarchical-rl"
---

# Hierarchical Q-Learning

{{ algorithm_card("hierarchical-q-learning") }}

!!! abstract "Overview"
    Hierarchical Q-Learning extends the traditional Q-Learning framework to handle temporal abstraction and hierarchical task decomposition. The algorithm learns Q-functions at multiple levels: a high-level Q-function that estimates the value of subgoals, and low-level Q-functions that estimate the value of actions given specific subgoals.

    This hierarchical approach enables the agent to solve complex, long-horizon tasks by breaking them down into manageable subproblems. The high-level Q-function learns to sequence subgoals effectively, while the low-level Q-functions learn to achieve specific subgoals efficiently. Hierarchical Q-Learning is particularly powerful in domains where tasks have natural hierarchical structure, such as robotics manipulation, navigation, and game playing.

## Mathematical Formulation

!!! math "Hierarchical Q-Function Decomposition"
    The hierarchical Q-function can be decomposed into:

    $$Q_h(s_t, g_t, a_t) = Q_{meta}(s_t, g_t) + Q_{low}(s_t, g_t, a_t)$$

    Where:
    - $Q_h$ is the hierarchical Q-function
    - $Q_{meta}$ is the meta Q-function that estimates subgoal values
    - $Q_{low}$ is the low-level Q-function that estimates action values given subgoals
    - $s_t$ is the current state
    - $g_t$ is the current subgoal
    - $a_t$ is the action taken

    The hierarchical Q-Learning update rule is:

    $$Q_h(s_t, g_t, a_t) \leftarrow Q_h(s_t, g_t, a_t) + \alpha \left[ r_t + \gamma \max_{g'} Q_{meta}(s_{t+1}, g') - Q_h(s_t, g_t, a_t) \right]$$

    Where $\alpha$ is the learning rate and $\gamma$ is the discount factor.

!!! success "Key Properties"
    - **Temporal Abstraction**: High-level Q-functions operate over longer time horizons
    - **Subgoal Decomposition**: Complex tasks broken into manageable subproblems
    - **Hierarchical Learning**: Q-functions at different levels learn simultaneously
    - **Transfer Learning**: Low-level Q-functions can be reused across different tasks
    - **Sample Efficiency**: Better exploration and learning in complex environments

## Implementation Approaches

=== "Basic Hierarchical Q-Learning (Recommended)"
    ```python
    import numpy as np
    from typing import Dict, Tuple, Any

    class HierarchicalQLearningAgent:
        """
        Hierarchical Q-Learning agent implementation.

        Args:
            state_size: Number of possible states
            subgoal_size: Number of possible subgoals
            action_size: Number of possible actions
            learning_rate: Learning rate alpha (default: 0.1)
            discount_factor: Discount factor gamma (default: 0.95)
            epsilon: Exploration rate for epsilon-greedy (default: 0.1)
            subgoal_horizon: Maximum steps to achieve subgoal (default: 50)
        """

        def __init__(self, state_size: int, subgoal_size: int, action_size: int,
                     learning_rate: float = 0.1, discount_factor: float = 0.95,
                     epsilon: float = 0.1, subgoal_horizon: int = 50):

            self.state_size = state_size
            self.subgoal_size = subgoal_size
            self.action_size = action_size
            self.alpha = learning_rate
            self.gamma = discount_factor
            self.epsilon = epsilon
            self.subgoal_horizon = subgoal_horizon

            # Q-tables
            self.meta_q_table = np.zeros((state_size, subgoal_size))
            self.low_q_table = np.zeros((state_size, subgoal_size, action_size))

            # Current subgoal tracking
            self.current_subgoal = None
            self.subgoal_steps = 0

        def get_subgoal(self, state: int) -> int:
            """Select subgoal using epsilon-greedy policy on meta Q-function."""
            if np.random.random() < self.epsilon:
                # Exploration: random subgoal
                return np.random.randint(self.subgoal_size)
            else:
                # Exploitation: best subgoal according to meta Q-table
                return np.argmax(self.meta_q_table[state])

        def get_action(self, state: int, subgoal: int) -> int:
            """Choose action using epsilon-greedy policy on low-level Q-function."""
            if np.random.random() < self.epsilon:
                # Exploration: random action
                return np.random.randint(self.action_size)
            else:
                # Exploitation: best action according to low-level Q-table
                return np.argmax(self.low_q_table[state, subgoal])

        def step(self, state: int, action: int, reward: float,
                next_state: int, done: bool) -> None:
            """Process one step and potentially update subgoal."""
            # Update low-level Q-function
            if self.current_subgoal is not None:
                self.update_low_level_q(state, self.current_subgoal, action, reward, next_state, done)

            # Check if subgoal should be updated
            self.subgoal_steps += 1
            if (self.subgoal_steps >= self.subgoal_horizon or
                self.is_subgoal_achieved(state, next_state) or done):

                # Update meta Q-function
                if self.current_subgoal is not None:
                    total_reward = reward  # Simplified - in practice, sum rewards over subgoal period
                    self.update_meta_q(state, self.current_subgoal, total_reward, next_state, done)

                # Select new subgoal
                if not done:
                    new_subgoal = self.get_subgoal(next_state)
                    self.current_subgoal = new_subgoal
                    self.subgoal_steps = 0

                # Reset subgoal tracking
                self.subgoal_steps = 0

        def is_subgoal_achieved(self, state: int, next_state: int) -> bool:
            """Check if current subgoal has been achieved."""
            # Simple state-based subgoal achievement
            # In practice, this would be domain-specific
            return state != next_state

        def update_low_level_q(self, state: int, subgoal: int, action: int,
                              reward: float, next_state: int, done: bool) -> None:
            """Update low-level Q-function using Q-Learning update rule."""
            current_q = self.low_q_table[state, subgoal, action]

            if done:
                # Terminal state: no future rewards
                max_next_q = 0
            else:
                # Non-terminal state: maximum Q-value of next state for current subgoal
                max_next_q = np.max(self.low_q_table[next_state, subgoal])

            # Q-Learning update rule
            new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
            self.low_q_table[state, subgoal, action] = new_q

        def update_meta_q(self, state: int, subgoal: int, reward: float,
                         next_state: int, done: bool) -> None:
            """Update meta Q-function using Q-Learning update rule."""
            current_q = self.meta_q_table[state, subgoal]

            if done:
                # Terminal state: no future rewards
                max_next_q = 0
            else:
                # Non-terminal state: maximum Q-value of next state for any subgoal
                max_next_q = np.max(self.meta_q_table[next_state])

            # Q-Learning update rule
            new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
            self.meta_q_table[state, subgoal] = new_q

        def get_policy(self) -> Tuple[np.ndarray, np.ndarray]:
            """Extract greedy policies from Q-tables."""
            meta_policy = np.argmax(self.meta_q_table, axis=1)
            low_policy = np.argmax(self.low_q_table, axis=2)
            return meta_policy, low_policy
    ```

=== "HQL with Experience Replay (Advanced)"
    ```python
    class HQLWithReplay(HierarchicalQLearningAgent):
        """
        Hierarchical Q-Learning agent with experience replay.
        """

        def __init__(self, state_size: int, subgoal_size: int, action_size: int, **kwargs):
            super().__init__(state_size, subgoal_size, action_size, **kwargs)

            # Experience replay buffers
            self.meta_buffer = []
            self.low_buffer = []
            self.batch_size = 32

        def store_meta_experience(self, state: int, subgoal: int, reward: float,
                                next_state: int, done: bool) -> None:
            """Store meta-level experience in replay buffer."""
            self.meta_buffer.append((state, subgoal, reward, next_state, done))

            # Update when buffer is full
            if len(self.meta_buffer) >= self.batch_size:
                self.replay_meta()

        def store_low_experience(self, state: int, subgoal: int, action: int,
                               reward: float, next_state: int, done: bool) -> None:
            """Store low-level experience in replay buffer."""
            self.low_buffer.append((state, subgoal, action, reward, next_state, done))

            # Update when buffer is full
            if len(self.low_buffer) >= self.batch_size:
                self.replay_low()

        def replay_meta(self) -> None:
            """Update meta Q-function using experience replay."""
            if len(self.meta_buffer) < self.batch_size:
                return

            # Sample batch from meta buffer
            batch = np.random.choice(len(self.meta_buffer), self.batch_size, replace=False)

            for idx in batch:
                state, subgoal, reward, next_state, done = self.meta_buffer[idx]
                self.update_meta_q(state, subgoal, reward, next_state, done)

            # Clear buffer
            self.meta_buffer.clear()

        def replay_low(self) -> None:
            """Update low-level Q-function using experience replay."""
            if len(self.low_buffer) < self.batch_size:
                return

            # Sample batch from low buffer
            batch = np.random.choice(len(self.low_buffer), self.batch_size, replace=False)

            for idx in batch:
                state, subgoal, action, reward, next_state, done = self.low_buffer[idx]
                self.update_low_level_q(state, subgoal, action, reward, next_state, done)

            # Clear buffer
            self.low_buffer.clear()
    ```

=== "HQL with Function Approximation"
    ```python
    class HQLWithFunctionApproximation(HierarchicalQLearningAgent):
        """
        Hierarchical Q-Learning agent with function approximation for large state spaces.
        """

        def __init__(self, state_size: int, subgoal_size: int, action_size: int, **kwargs):
            super().__init__(state_size, subgoal_size, action_size, **kwargs)

            # Function approximators (simplified linear models)
            self.meta_weights = np.random.randn(state_size, subgoal_size) * 0.01
            self.low_weights = np.random.randn(state_size + subgoal_size, action_size) * 0.01

        def get_meta_q_value(self, state: int, subgoal: int) -> float:
            """Get Q-value using function approximation."""
            return self.meta_weights[state, subgoal]

        def get_low_q_value(self, state: int, subgoal: int, action: int) -> float:
            """Get Q-value using function approximation."""
            features = np.concatenate([np.eye(self.state_size)[state], np.eye(self.subgoal_size)[subgoal]])
            return np.dot(features, self.low_weights[:, action])

        def update_meta_weights(self, state: int, subgoal: int, target: float) -> None:
            """Update meta Q-function weights."""
            current_q = self.get_meta_q_value(state, subgoal)
            error = target - current_q
            self.meta_weights[state, subgoal] += self.alpha * error

        def update_low_weights(self, state: int, subgoal: int, action: int, target: float) -> None:
            """Update low-level Q-function weights."""
            current_q = self.get_low_q_value(state, subgoal, action)
            error = target - current_q
            features = np.concatenate([np.eye(self.state_size)[state], np.eye(self.subgoal_size)[subgoal]])
            self.low_weights[:, action] += self.alpha * error * features
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/hierarchical_rl/hierarchical_q_learning.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/hierarchical_rl/hierarchical_q_learning.py)
    - **Tests**: [`tests/unit/hierarchical_rl/test_hierarchical_q_learning.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/hierarchical_rl/test_hierarchical_q_learning.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Basic HQL** | O(|S| × |G| × |A|) per update | O(|S| × |G| × |A|) | Tabular representation |
    **HQL + Replay** | O(batch_size × |S| × |G| × |A|) | O(|S| × |G| × |A| + buffer_size) | Better sample efficiency |
    **HQL + Function Approx** | O(feature_size × |A|) per update | O(feature_size × |A|) | Scales to large state spaces |

!!! warning "Performance Considerations"
    - **Tabular representation** becomes infeasible for large state spaces
    - **Experience replay** improves sample efficiency but requires more memory
    - **Function approximation** scales better but may lose convergence guarantees
    - **Subgoal horizon** affects exploration and learning efficiency

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Robotics & Control"
        - **Robot Manipulation**: Complex manipulation tasks with subgoals
        - **Autonomous Navigation**: Multi-level path planning and execution
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
        - **Hierarchical Learning**: Understanding temporal abstraction in RL
        - **Subgoal Decomposition**: Learning to break complex tasks into subproblems
        - **Multi-Level Q-Functions**: Understanding coordination between Q-functions
        - **Transfer Learning**: Learning reusable low-level skills

!!! success "Educational Value"
    - **Hierarchical Learning**: Perfect example of temporal abstraction in RL
    - **Task Decomposition**: Shows how to break complex problems into manageable parts
    - **Multi-Level Coordination**: Demonstrates learning at different time scales
    - **Transfer Learning**: Illustrates how skills can be reused across tasks

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Papers"
        1. **Dietterich, T. G.** (2000). Hierarchical reinforcement learning with the MAXQ value function decomposition. *Journal of Artificial Intelligence Research*, 13.
        2. **Kaelbling, L. P., et al.** (1998). Hierarchical reinforcement learning with the MAXQ value function decomposition. *Journal of Artificial Intelligence Research*, 13.

    !!! grid-item "Hierarchical RL Textbooks"
        3. **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction*. MIT Press.
        4. **Kaelbling, L. P., et al.** (1998). Hierarchical reinforcement learning with the MAXQ value function decomposition. *Journal of Artificial Intelligence Research*, 13.

    !!! grid-item "Online Resources"
        5. [Hierarchical Reinforcement Learning - Wikipedia](https://en.wikipedia.org/wiki/Hierarchical_reinforcement_learning)
        6. [MAXQ Implementation Guide](https://github.com/andrew-j-levy/Hierarchical-Actor-Critic-HAC-)
        7. [Hierarchical RL Tutorial](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

    !!! grid-item "Implementation & Practice"
        8. [NumPy Documentation](https://numpy.org/doc/)
        9. [OpenAI Gym](https://www.gymlibrary.dev/) - RL environments
        10. [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - RL implementations

!!! tip "Interactive Learning"
    Try implementing Hierarchical Q-Learning yourself! Start with simple environments that have natural hierarchical structure, like navigation tasks with waypoints. Implement the basic two-level Q-functions first, then add experience replay for better sample efficiency. Experiment with different subgoal horizons and achievement detection methods to see their impact on learning. Compare with flat Q-learning methods to see the benefits of hierarchical decomposition. This will give you deep insight into the power of temporal abstraction in reinforcement learning.

## Navigation

{{ nav_grid(current_algorithm="hierarchical-q-learning", current_family="hierarchical-rl", max_related=5) }}
