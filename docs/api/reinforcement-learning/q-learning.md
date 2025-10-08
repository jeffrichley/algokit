# Q-Learning API Reference

## Overview

Q-Learning is a model-free, off-policy reinforcement learning algorithm that learns the optimal action-value function. It's one of the most fundamental and widely-used RL algorithms, particularly effective for discrete state and action spaces.

**Key Features:**
- Off-policy learning (learns optimal policy while following exploratory policy)
- Guaranteed convergence under proper conditions
- Simple and interpretable
- Works well with tabular representations

For algorithmic details and theory, see the [Reinforcement Learning Overview](../../algorithms/reinforcement-learning/overview.md).

---

## Configuration

::: algokit.algorithms.reinforcement_learning.q_learning.QLearningConfig
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      show_source: false
      show_bases: true

---

## Agent

::: algokit.algorithms.reinforcement_learning.q_learning.QLearningAgent
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      show_source: false
      show_bases: true

---

## Quick Start Example

```python
from algokit.algorithms.reinforcement_learning.q_learning import (
    QLearningAgent,
    QLearningConfig
)
import gymnasium as gym

# Create environment
env = gym.make('FrozenLake-v1')

# Configure agent (method 1: using config object)
config = QLearningConfig(
    state_size=16,
    action_size=4,
    learning_rate=0.1,
    gamma=0.99,
    epsilon=0.1
)
agent = QLearningAgent(config=config)

# Or use kwargs (method 2: backward compatible)
agent = QLearningAgent(
    state_size=16,
    action_size=4,
    learning_rate=0.1
)

# Training loop
for episode in range(1000):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if episode % 100 == 0:
        print(f"Episode {episode}: Reward = {total_reward}")

# Get statistics
stats = agent.get_statistics()
print(f"Average reward: {stats['avg_reward']}")
```

---

## See Also

### Related Algorithms
- [SARSA](sarsa.md) - On-policy alternative to Q-Learning
- [DQN](dqn.md) - Deep learning extension of Q-Learning
- [Actor-Critic](actor-critic.md) - Hybrid approach

### Documentation
- [Reinforcement Learning Overview](../../algorithms/reinforcement-learning/overview.md)
- [Q-Learning Examples](../../examples/reinforcement-learning.md)
- [RL Best Practices](../../guides/rl-best-practices.md)
