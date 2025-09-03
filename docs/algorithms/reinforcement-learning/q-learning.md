---
tags: [reinforcement-learning, algorithms, q-learning, temporal-difference, value-based, markov-decision-process]
title: "Q-Learning"
family: "reinforcement-learning"
complexity: "O(|S| × |A|)"
---

# Q-Learning

!!! info "Algorithm Family"
    **Family:** [Reinforcement Learning](../../families/reinforcement-learning.md)

!!! abstract "Overview"
    Q-Learning is a model-free, off-policy reinforcement learning algorithm that learns the optimal action-value function (Q-function) for a Markov Decision Process (MDP). It's one of the most fundamental and widely-used algorithms in reinforcement learning, capable of learning optimal policies without requiring a model of the environment.

    The algorithm works by iteratively updating Q-values based on the Bellman equation, using temporal difference learning to estimate the expected future rewards. Q-Learning is particularly powerful because it can learn optimal policies even when taking random actions during training, making it suitable for exploration-heavy scenarios.

## Mathematical Formulation

!!! math "Q-Learning Update Rule"
    The core Q-Learning update rule follows the Bellman optimality equation:
    
    $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$
    
    Where:
    - $Q(s, a)$ is the Q-value for state $s$ and action $a$
    - $\alpha$ is the learning rate (controls update magnitude)
    - $r_t$ is the immediate reward at time $t$
    - $\gamma$ is the discount factor (future reward importance)
    - $\max_{a'} Q(s_{t+1}, a')$ is the maximum Q-value for the next state

!!! success "Key Properties"
    - **Model-Free**: Doesn't require knowledge of environment dynamics
    - **Off-Policy**: Can learn optimal policy while following different behavior policy
    - **Temporal Difference**: Updates estimates based on difference between predicted and actual values
    - **Convergence**: Guaranteed to converge to optimal Q-values under certain conditions

## Implementation Approaches

=== "Standard Q-Learning (Recommended)"
    ```python
    import numpy as np
    from typing import Dict, Tuple, Any
    
    class QLearningAgent:
        """
        Q-Learning agent implementation.
        
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
        
        def get_action(self, state: int) -> int:
            """Choose action using epsilon-greedy policy."""
            if np.random.random() < self.epsilon:
                # Exploration: random action
                return np.random.randint(self.action_size)
            else:
                # Exploitation: best action according to Q-table
                return np.argmax(self.q_table[state])
        
        def update(self, state: int, action: int, reward: float, 
                  next_state: int, done: bool) -> None:
            """Update Q-value using Q-Learning update rule."""
            current_q = self.q_table[state, action]
            
            if done:
                # Terminal state: no future rewards
                max_next_q = 0
            else:
                # Non-terminal state: maximum Q-value of next state
                max_next_q = np.max(self.q_table[next_state])
            
            # Q-Learning update rule
            new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
            self.q_table[state, action] = new_q
        
        def get_policy(self) -> np.ndarray:
            """Extract greedy policy from Q-table."""
            return np.argmax(self.q_table, axis=1)
    ```

=== "Experience Replay Q-Learning (Advanced)"
    ```python
    import random
    from collections import deque
    
    class QLearningWithReplay:
        """
        Q-Learning with experience replay for better sample efficiency.
        """
        
        def __init__(self, state_size: int, action_size: int, 
                     learning_rate: float = 0.1, discount_factor: float = 0.95,
                     epsilon: float = 0.1, replay_buffer_size: int = 10000,
                     batch_size: int = 32):
            self.state_size = state_size
            self.action_size = action_size
            self.alpha = learning_rate
            self.gamma = discount_factor
            self.epsilon = epsilon
            self.batch_size = batch_size
            
            # Initialize Q-table and replay buffer
            self.q_table = np.zeros((state_size, action_size))
            self.replay_buffer = deque(maxlen=replay_buffer_size)
        
        def store_experience(self, state: int, action: int, reward: float,
                           next_state: int, done: bool) -> None:
            """Store experience in replay buffer."""
            self.replay_buffer.append((state, action, reward, next_state, done))
        
        def replay(self) -> None:
            """Update Q-values using batch of experiences from replay buffer."""
            if len(self.replay_buffer) < self.batch_size:
                return
            
            # Sample random batch of experiences
            batch = random.sample(self.replay_buffer, self.batch_size)
            
            for state, action, reward, next_state, done in batch:
                current_q = self.q_table[state, action]
                
                if done:
                    max_next_q = 0
                else:
                    max_next_q = np.max(self.q_table[next_state])
                
                # Q-Learning update
                new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
                self.q_table[state, action] = new_q
    ```

=== "Function Approximation Q-Learning"
    ```python
    from sklearn.linear_model import SGDRegressor
    
    class QLearningFunctionApproximation:
        """
        Q-Learning with function approximation for continuous/large state spaces.
        """
        
        def __init__(self, action_size: int, feature_size: int,
                     learning_rate: float = 0.01, discount_factor: float = 0.95,
                     epsilon: float = 0.1):
            self.action_size = action_size
            self.feature_size = feature_size
            self.alpha = learning_rate
            self.gamma = discount_factor
            self.epsilon = epsilon
            
            # Separate Q-function approximator for each action
            self.q_approximators = [
                SGDRegressor(learning_rate='constant', eta0=learning_rate)
                for _ in range(action_size)
            ]
            
            # Initialize with dummy data
            for approximator in self.q_approximators:
                approximator.partial_fit([[0] * feature_size], [0])
        
        def get_q_value(self, state_features: np.ndarray, action: int) -> float:
            """Get Q-value for state-action pair using function approximation."""
            return self.q_approximators[action].predict([state_features])[0]
        
        def update(self, state_features: np.ndarray, action: int, reward: float,
                  next_state_features: np.ndarray, done: bool) -> None:
            """Update Q-function approximator."""
            current_q = self.get_q_value(state_features, action)
            
            if done:
                target_q = reward
            else:
                # Find maximum Q-value for next state
                next_q_values = [self.get_q_value(next_state_features, a) 
                               for a in range(self.action_size)]
                target_q = reward + self.gamma * max(next_q_values)
            
            # Update the approximator for this action
            self.q_approximators[action].partial_fit([state_features], [target_q])
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/reinforcement_learning/q_learning.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/reinforcement_learning/q_learning.py)
    - **Tests**: [`tests/unit/reinforcement_learning/test_q_learning.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/reinforcement_learning/test_q_learning.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Tabular Q-Learning** | O(|S| × |A|) per update | O(|S| × |A|) | Optimal for discrete state spaces |
    **Experience Replay** | O(batch_size) per update | O(|S| × |A| + buffer_size) | Better sample efficiency |
    **Function Approximation** | O(feature_size) per update | O(feature_size × |A|) | Scales to continuous states |

!!! warning "Performance Considerations"
    - **Tabular approach** becomes infeasible for large state spaces
    - **Experience replay** improves sample efficiency but requires more memory
    - **Function approximation** scales better but may lose convergence guarantees
    - **Exploration strategy** significantly affects learning performance

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Game AI"
        - **Board Games**: Chess, Go, and strategy game AI
        - **Video Games**: NPC behavior and pathfinding
        - **Puzzle Games**: Optimal solution finding
        - **Simulation Games**: Resource management AI

    !!! grid-item "Robotics & Control"
        - **Autonomous Navigation**: Path planning and obstacle avoidance
        - **Robot Manipulation**: Optimal control policies
        - **Swarm Robotics**: Multi-agent coordination
        - **Industrial Automation**: Process optimization

    !!! grid-item "Real-World Applications"
        - **Recommendation Systems**: User preference learning
        - **Trading Algorithms**: Market strategy optimization
        - **Resource Management**: Optimal allocation policies
        - **Healthcare**: Treatment optimization and scheduling

    !!! grid-item "Educational Value"
        - **Reinforcement Learning**: Understanding value-based methods
        - **Temporal Difference**: Learning from experience
        - **Exploration vs Exploitation**: Balancing learning and performance
        - **Markov Decision Processes**: Understanding sequential decision making

!!! success "Educational Value"
    - **Reinforcement Learning**: Perfect introduction to value-based methods
    - **Temporal Difference Learning**: Shows how to learn from experience
    - **Exploration Strategies**: Demonstrates importance of exploration vs exploitation
    - **Convergence Properties**: Illustrates theoretical guarantees in practice

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Textbooks"
        1. **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction*. MIT Press. ISBN 978-0-262-03924-7.
        2. **Szepesvári, C.** (2010). *Algorithms for Reinforcement Learning*. Morgan & Claypool. ISBN 978-1-60845-492-1.

    !!! grid-item "Q-Learning Specific"
        3. **Watkins, C. J. C. H.** (1989). *Learning from Delayed Rewards*. PhD Thesis, University of Cambridge.
        4. **Watkins, C. J. C. H., & Dayan, P.** (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292.

    !!! grid-item "Online Resources"
        5. [Q-Learning - Wikipedia](https://en.wikipedia.org/wiki/Q-learning)
        6. [Reinforcement Learning Tutorial](https://www.gymlibrary.dev/)
        7. [Q-Learning Implementation Guide](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

    !!! grid-item "Implementation & Practice"
        8. [Python Official Documentation](https://docs.python.org/3/)
        9. [NumPy Documentation](https://numpy.org/doc/)
        10. [OpenAI Gym](https://www.gymlibrary.dev/) - RL environments for testing

!!! tip "Interactive Learning"
    Try implementing Q-Learning yourself! Start with simple environments like a grid world or the classic "Cliff Walking" problem. Implement both the tabular version and experience replay variant to see how replay improves learning efficiency. Experiment with different exploration strategies (epsilon-greedy, softmax) and observe their impact on convergence. This will give you deep insight into the fundamentals of reinforcement learning.
