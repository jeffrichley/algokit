# Hierarchical Reinforcement Learning Algorithms

## Overview

Hierarchical Reinforcement Learning (HRL) extends traditional reinforcement learning by decomposing complex tasks into simpler subtasks or options. This hierarchical structure enables agents to learn more efficiently by reusing learned skills and operating at multiple levels of abstraction, from high-level strategic planning to low-level control execution.

**Key Characteristics:**
- **Temporal Abstraction**: Actions that operate over different time scales
- **Skill Reuse**: Learned behaviors that can be applied to new tasks
- **Multi-level Decision Making**: Hierarchical structure from strategy to control
- **Option Policies**: Temporally extended actions over multiple time steps

**Common Applications:**
- Robotics and autonomous systems
- Game playing and strategy games
- Multi-agent coordination
- Complex control systems
- Natural language processing
- Computer vision and perception

## Key Concepts

- **Options**: Temporally extended actions that can be executed over multiple time steps
- **Hierarchy**: Multiple levels of decision-making from high-level strategy to low-level control
- **Skill Reuse**: Learned behaviors that can be applied to new tasks
- **Temporal Abstraction**: Actions that operate over different time scales
- **Subgoal Decomposition**: Breaking complex tasks into manageable subtasks
- **Policy Hierarchies**: Layered policies operating at different abstraction levels

## Comparison Table

| Algorithm | Complexity | Strengths | Weaknesses | Applications |
|-----------|------------|-----------|------------|--------------|
| **Hierarchical Actor-Critic** | O(n²) time, O(n) space | Stable learning, hierarchical structure | Complex architecture, training overhead | Robotics, autonomous navigation, game playing |
| **Hierarchical Policy Gradient** | O(n²) time, O(n) space | Direct policy optimization, hierarchical actions | High variance, sample inefficiency | Continuous control, robotics manipulation |
| **Hierarchical Q-Learning** | O(n²) time, O(n²) space | Extends Q-learning, option-based learning | Large state-action space, exploration challenges | Discrete environments, strategy games |
| **Hierarchical Task Networks** | O(n³) time, O(n²) space | Structured task decomposition, planning | Complex planning, computational overhead | Automated planning, robotics assembly |
| **Feudal Networks** | O(n²) time, O(n) space | End-to-end learning, neural architecture | Training complexity, interpretability | Deep RL, complex environments |
| **Option-Critic** | O(n²) time, O(n) space | End-to-end option learning, policy gradients | Convergence issues, hyperparameter sensitivity | Skill learning, temporal abstraction |

## Algorithms in This Family

- [**Hierarchical Actor-Critic**](../algorithms/hierarchical-rl/hierarchical-actor-critic.md) - Hierarchical policy gradient with actor-critic framework and temporal abstraction
- [**Hierarchical Policy Gradient**](../algorithms/hierarchical-rl/hierarchical-policy-gradient.md) - Direct policy optimization with temporal abstraction and subgoal decomposition
- [**Hierarchical Q-Learning**](../algorithms/hierarchical-rl/hierarchical-q-learning.md) - Value-based learning with hierarchical Q-functions and subgoal structure
- [**Hierarchical Task Networks**](../algorithms/hierarchical-rl/hierarchical-task-networks.md) - Task decomposition and hierarchical planning with modular learning
- [**Feudal Networks**](../algorithms/hierarchical-rl/feudal-networks.md) - Manager-worker architecture with goal-based learning and temporal abstraction
- [**Option-Critic**](../algorithms/hierarchical-rl/option-critic.md) - Learning options end-to-end with automatic discovery and termination functions

## Implementation Status

- **Complete**: 6/6 algorithms (100%)
- **In Progress**: 0/6 algorithms (0%)
- **Planned**: 0/6 algorithms (0%)

## Hierarchical Structures

### **Temporal Hierarchy**
- **High-level**: Strategic decisions and goal setting
- **Mid-level**: Skill selection and coordination
- **Low-level**: Primitive actions and control

### **Spatial Hierarchy**
- **Global**: Environment-wide planning
- **Regional**: Local area navigation
- **Local**: Immediate obstacle avoidance

### **Functional Hierarchy**
- **Planning**: Long-term strategy
- **Navigation**: Path finding and movement
- **Control**: Actuator commands

## Learning Approaches

### **Predefined Hierarchies**
- Human-designed task decomposition
- Fixed skill libraries
- Structured learning objectives

### **Learned Hierarchies**
- Automatic task decomposition
- Dynamic skill discovery
- Adaptive abstraction levels

### **Hybrid Approaches**
- Combine predefined and learned components
- Incremental hierarchy construction
- Skill refinement over time

## Related Algorithm Families

- **Reinforcement Learning**: Foundation for HRL algorithms
- **Multi-Agent Systems**: Coordination and cooperation at multiple levels
- **Planning Algorithms**: Task decomposition and strategic planning
- **Neural Networks**: Deep learning approaches for hierarchical policies
- **Control Systems**: Multi-level control and decision making
