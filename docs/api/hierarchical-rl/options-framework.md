# Options Framework API Reference

## Overview

The Options Framework provides formal temporal abstraction for hierarchical reinforcement learning. An option is a temporally extended action consisting of an initiation set, policy, and termination condition, enabling reusable skills and accelerated learning.

**Key Features:**
- Formal mathematical framework with convergence guarantees
- Temporal abstraction through options (closed-loop policies)
- Semi-Markov decision process framework
- Flexible option discovery methods
- Supports transfer learning across tasks
- Intra-option learning for efficiency

For algorithmic details and theory, see the [Hierarchical RL Overview](../../algorithms/hierarchical-rl/overview.md).

---

## Configuration

::: algokit.algorithms.hierarchical_rl.options_framework.OptionsConfig
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      show_source: false
      show_bases: true

---

## Framework

::: algokit.algorithms.hierarchical_rl.options_framework.OptionsFramework
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      show_source: false
      show_bases: true

---

## Option Class

::: algokit.algorithms.hierarchical_rl.options_framework.Option
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      show_source: false
      show_bases: true

---

## Quick Start Example

```python
from algokit.algorithms.hierarchical_rl.options_framework import (
    OptionsFramework,
    OptionsConfig,
    Option
)
import gymnasium as gym

# Create environment
env = gym.make('FrozenLake-v1')

# Configure framework
config = OptionsConfig(
    state_size=16,
    action_size=4,
    num_options=4,  # Number of options to discover
    learning_rate=0.1,
    gamma=0.99,
    epsilon=0.1
)
framework = OptionsFramework(config=config)

# Training loop
for episode in range(1000):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Select option
        option = framework.select_option(state)

        # Execute option until termination
        while not option.should_terminate(state) and not done:
            action = option.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Learn from transition
            framework.learn(state, option, action, reward, next_state, done)
            state = next_state
            total_reward += reward

    if episode % 100 == 0:
        print(f"Episode {episode}: Reward = {total_reward}")

# Get learned options
options = framework.get_options()
print(f"Discovered {len(options)} options")
```

---

## Advanced Usage

### Custom Option Definition

```python
# Define a custom option manually
class NavigateRightOption(Option):
    """Option that navigates to the right until hitting a wall."""

    def __init__(self):
        super().__init__(name="navigate_right")

    def can_initiate(self, state):
        # Can start anywhere
        return True

    def select_action(self, state):
        # Always move right
        return 2  # Right action

    def should_terminate(self, state):
        # Terminate at right wall
        return is_at_right_wall(state)

# Add custom option to framework
framework.add_option(NavigateRightOption())
```

### Option Discovery

```python
# Automatic option discovery through clustering
from algokit.algorithms.hierarchical_rl.options_framework import (
    discover_options_clustering
)

# Collect trajectories
trajectories = []
for _ in range(100):
    trajectory = collect_trajectory(env, random_policy)
    trajectories.append(trajectory)

# Discover options via state clustering
discovered_options = discover_options_clustering(
    trajectories,
    num_options=4,
    method='kmeans'
)

# Use discovered options
for option in discovered_options:
    framework.add_option(option)
```

### Transfer Learning

```python
# Train on source task
source_env = gym.make('FrozenLake-v1')
framework.train(source_env, episodes=1000)

# Transfer options to target task
target_env = gym.make('FrozenLake8x8-v1')  # Larger version

# Options are reusable!
framework.train(target_env, episodes=500)  # Learns faster with pre-trained options
```

---

## See Also

### Related Algorithms
- [Feudal RL](feudal-rl.md) - Manager-worker hierarchy
- [HIRO](hiro.md) - Goal-conditioned hierarchical RL
- [SARSA](../reinforcement-learning/sarsa.md) - On-policy foundation

### Documentation
- [Hierarchical RL Overview](../../algorithms/hierarchical-rl/overview.md)
- [Options Paper (Sutton et al., 1999)](http://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf)
- [Option Discovery Methods](../../guides/option-discovery.md)

### Research
- "Between MDPs and Semi-MDPs" (Sutton, Precup, Singh, 1999)
- "Learning Options in Reinforcement Learning" (various modern approaches)
- Applications in transfer learning and skill composition
