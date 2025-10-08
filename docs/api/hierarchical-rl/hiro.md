# HIRO (Data-Efficient Hierarchical RL) API Reference

## Overview

HIRO is a state-of-the-art hierarchical reinforcement learning algorithm that uses goal-conditioned policies with off-policy correction for exceptional sample efficiency. It's particularly effective for long-horizon continuous control tasks.

**Key Features:**
- Higher-level policy proposes goals (as state deltas)
- Lower-level goal-conditioned policy learns primitive actions
- Off-policy correction through goal relabeling
- Hindsight experience replay for data efficiency
- TD3-style target policy smoothing for stability
- Normalized intrinsic rewards

For algorithmic details and theory, see the [Hierarchical RL Overview](../../algorithms/hierarchical-rl/overview.md).

---

## Configuration

::: algokit.algorithms.hierarchical_rl.hiro.HIROConfig
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      show_source: false
      show_bases: true

---

## Agent

::: algokit.algorithms.hierarchical_rl.hiro.HIROAgent
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      show_source: false
      show_bases: true

---

## Policy Networks

### Higher-Level Policy

::: algokit.algorithms.hierarchical_rl.hiro.HigherLevelPolicy
    options:
      show_root_heading: true
      heading_level: 4
      members_order: source
      show_source: false

### Lower-Level Policy

::: algokit.algorithms.hierarchical_rl.hiro.LowerLevelPolicy
    options:
      show_root_heading: true
      heading_level: 4
      members_order: source
      show_source: false

---

## Quick Start Example

```python
from algokit.algorithms.hierarchical_rl.hiro import (
    HIROAgent,
    HIROConfig
)
import gymnasium as gym

# Create environment
env = gym.make('FetchReach-v2')  # Goal-conditioned environment

# Configure agent
config = HIROConfig(
    state_size=10,
    action_size=4,
    goal_size=3,  # Goals are state deltas
    goal_horizon=10,  # Steps between high-level decisions
    learning_rate=0.0003,
    gamma=0.99
)
agent = HIROAgent(config=config)

# Training loop
for episode in range(1000):
    metrics = agent.train_episode(env, max_steps=200, epsilon=0.1)

    if episode % 100 == 0:
        print(f"Episode {episode}:")
        print(f"  Reward: {metrics['reward']:.2f}")
        print(f"  Steps: {metrics['steps']}")
        print(f"  Lower Critic Loss: {metrics['avg_lower_critic_loss']:.4f}")
        print(f"  Higher Critic Loss: {metrics['avg_higher_critic_loss']:.4f}")

# Get detailed statistics
stats = agent.get_statistics()
print(f"\nTraining Statistics:")
print(f"  Total Episodes: {stats['total_episodes']}")
print(f"  Avg Reward: {stats['avg_reward']:.2f}")
print(f"  Intrinsic/Extrinsic Ratio: {stats['intrinsic_extrinsic_ratio']:.2f}")
```

---

## Advanced Usage

### Understanding Goal-Conditioned Policies

HIRO uses goals as **state deltas** (relative displacements), not absolute states:

```python
# High-level policy proposes goal delta
goal_delta = agent.select_goal(current_state)  # e.g., [+2.0, -1.0, +0.5]

# Low-level policy tries to achieve this delta
action = agent.select_action(current_state, goal_delta)

# After action, check achievement
achieved_delta = next_state - current_state
intrinsic_reward = agent.goal_distance(achieved_delta, goal_delta)
```

### Off-Policy Correction (Goal Relabeling)

HIRO relabels goals using hindsight to improve sample efficiency:

```python
# Original goal proposed by manager
original_goal = [+2.0, -1.0, +0.5]

# What was actually achieved
start_state = [0.0, 0.0, 0.0]
achieved_state = [1.5, -0.8, +0.3]
achieved_delta = achieved_state - start_state  # [1.5, -0.8, +0.3]

# Relabel: what if manager had proposed what we achieved?
relabeled_goal = agent.relabel_goal(start_state, trajectory, horizon=10)
# relabeled_goal = achieved_delta

# Store with both original and relabeled goals
# This doubles effective sample efficiency!
```

### Intrinsic Reward Computation

```python
# Goals are deltas, so compare achieved delta to goal delta
achieved_delta = next_state - goal_state
intrinsic = agent.goal_distance(achieved_delta, current_goal)

# Normalized by running statistics
# Scaled by state dimensionality
# This prevents reward scale issues
```

---

## Hyperparameter Guidelines

### Goal Horizon
- **Short (5-10)**: More frequent subgoal updates, easier to achieve
- **Long (20-50)**: More temporal abstraction, harder credit assignment
- **Typical**: 10 for most tasks

### Learning Rate
- **Conservative**: 0.0001 (more stable)
- **Standard**: 0.0003 (good default)
- **Aggressive**: 0.001 (faster learning, less stable)

### Intrinsic Scale
- **Low (0.1-0.5)**: When extrinsic rewards are large
- **Medium (1.0)**: Balanced (default)
- **High (2.0-5.0)**: For sparse extrinsic rewards

---

## See Also

### Related Algorithms
- [Options Framework](options-framework.md) - Discrete temporal abstraction
- [Feudal RL](feudal-rl.md) - Manager-worker architecture
- [DQN](../reinforcement-learning/dqn.md) - Deep Q-learning foundation

### Documentation
- [Hierarchical RL Overview](../../algorithms/hierarchical-rl/overview.md)
- [HIRO Coverage Report](../../../HIRO_COVERAGE_IMPROVEMENT.md)
- [HIRO Paper (Nachum et al., 2018)](https://arxiv.org/abs/1805.08296)

### Research
- Original HIRO paper: "Data-Efficient Hierarchical Reinforcement Learning" (NeurIPS 2018)
- Extensions and improvements in goal-conditioned RL
- Applications in robotics and manipulation
