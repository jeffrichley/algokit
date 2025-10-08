# PPO (Proximal Policy Optimization) API Reference

## Overview

PPO is a state-of-the-art policy gradient method that achieves a good balance between sample efficiency, simplicity, and ease of tuning. It's widely used in production systems and robotics applications.

**Key Features:**
- Clipped surrogate objective for stable updates
- On-policy learning with mini-batch updates
- Robust to hyperparameter choices
- Effective for both continuous and discrete action spaces
- Industry standard for real-world applications

For algorithmic details and theory, see the [Reinforcement Learning Overview](../../algorithms/reinforcement-learning/overview.md).

---

## Configuration

::: algokit.algorithms.reinforcement_learning.ppo.PPOConfig
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      show_source: false
      show_bases: true

---

## Agent

::: algokit.algorithms.reinforcement_learning.ppo.PPOAgent
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      show_source: false
      show_bases: true

---

## Quick Start Example

```python
from algokit.algorithms.reinforcement_learning.ppo import (
    PPOAgent,
    PPOConfig
)
import gymnasium as gym

# Create environment
env = gym.make('CartPole-v1')

# Configure agent
config = PPOConfig(
    state_size=4,
    action_size=2,
    learning_rate=0.0003,
    gamma=0.99,
    epsilon_clip=0.2,
    value_coef=0.5,
    entropy_coef=0.01
)
agent = PPOAgent(config=config)

# Training loop
for episode in range(1000):
    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store transition
        agent.store_transition(state, action, reward, done)
        state = next_state
        episode_reward += reward

    # Update policy at end of episode
    agent.update()

    if episode % 50 == 0:
        stats = agent.get_statistics()
        print(f"Episode {episode}: Reward = {episode_reward:.2f}, "
              f"Policy Loss = {stats['policy_loss']:.4f}")

# Save trained model
agent.save('ppo_cartpole.pt')
```

---

## Advanced Usage

### Custom Network Architecture

```python
import torch.nn as nn

# Define custom actor network
class CustomActor(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

# Use with PPO
agent = PPOAgent(
    state_size=4,
    action_size=2,
    actor_network=CustomActor  # Custom architecture
)
```

### Hyperparameter Tuning

```python
# Conservative (stable but slower learning)
conservative_config = PPOConfig(
    state_size=4,
    action_size=2,
    learning_rate=0.0001,
    epsilon_clip=0.1,  # Small clipping
    batch_size=64,
    epochs=10
)

# Aggressive (faster but less stable)
aggressive_config = PPOConfig(
    state_size=4,
    action_size=2,
    learning_rate=0.001,
    epsilon_clip=0.3,  # Larger clipping
    batch_size=32,
    epochs=5
)
```

---

## See Also

### Related Algorithms
- [Policy Gradient](policy-gradient.md) - Foundational policy optimization
- [Actor-Critic](actor-critic.md) - Combines value and policy learning
- [Q-Learning](q-learning.md) - Value-based alternative

### Documentation
- [Reinforcement Learning Overview](../../algorithms/reinforcement-learning/overview.md)
- [PPO Paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning Up - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
