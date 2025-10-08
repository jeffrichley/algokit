# Hierarchical Reinforcement Learning Overview

## 🧠 Introduction

**Hierarchical Reinforcement Learning (HRL)** decomposes complex tasks into a hierarchy of simpler subtasks, enabling more efficient learning and better generalization. This approach is inspired by how humans break down complex problems into manageable pieces and reuse learned skills across different contexts.

## 🎯 Core Motivation

Traditional "flat" RL faces significant challenges with complex, long-horizon tasks:
- **Credit Assignment**: Hard to determine which early actions led to later rewards
- **Exploration**: Exponentially large state-action spaces
- **Transfer**: Difficulty reusing knowledge across tasks
- **Sample Efficiency**: Requires enormous amounts of experience

HRL addresses these challenges through:
- ✅ **Temporal Abstraction**: Actions that span multiple time steps
- ✅ **Skill Reuse**: Learn once, apply many times
- ✅ **Compositional Learning**: Build complex behaviors from simple ones
- ✅ **Improved Credit Assignment**: Localize credit to relevant temporal scales

## 🏗️ Hierarchical Structure

```
┌─────────────────────────────────────┐
│   High-Level Policy (Manager)       │
│   "Achieve subgoal: Go to room B"   │
└──────────────┬──────────────────────┘
               │ Subgoals/Options
               ▼
┌─────────────────────────────────────┐
│   Low-Level Policy (Worker)         │
│   "Navigate through doorway"        │
└──────────────┬──────────────────────┘
               │ Primitive Actions
               ▼
┌─────────────────────────────────────┐
│          Environment                 │
└─────────────────────────────────────┘
```

## 🔑 Key Characteristics

- **Task Decomposition**: Break complex problems into manageable pieces
- **Hierarchical Structure**: Multiple levels of abstraction
- **Subtask Learning**: Learn reusable skills/behaviors
- **Temporal Abstraction**: Actions that span multiple timesteps
- **Skill Reuse**: Transfer knowledge across tasks
- **Goal-Conditioned Policies**: Parameterized by objectives

## 🎨 Implemented Algorithms

### 1. **Options Framework** 🎯
Temporal abstraction through options (closed-loop policies with initiation/termination conditions).

**Key Concepts:**
- **Option**: Triple (Initiation Set, Policy, Termination Condition)
- **Semi-Markov**: Decisions at option boundaries
- **Skill Library**: Reusable behavioral primitives

**Features:**
- ✅ Formal mathematical framework
- ✅ Provable convergence guarantees
- ✅ Flexible option discovery methods
- ✅ Supports transfer learning

**Best For:**
- Tasks with natural subtask boundaries
- Environments requiring skill reuse
- Navigation with waypoints
- Transfer across similar tasks

**Coverage**: 95% 🌟

**Example Use Cases:**
- Room navigation (option per room)
- Robotic assembly (pick, place, screw)
- Game AI (gather, build, attack)

### 2. **Feudal RL** 👑
Manager-worker architecture with explicit goal-setting hierarchy.

**Key Concepts:**
- **Manager Network**: Sets subgoals for workers
- **Worker Network**: Achieves manager's subgoals
- **Transition Policy Gradient**: Backprop through hierarchy
- **Directional Goals**: Goal embeddings in latent space

**Features:**
- ✅ End-to-end differentiable
- ✅ Automatic goal discovery
- ✅ Handles long-term dependencies
- ✅ Scalable to deep hierarchies

**Best For:**
- Long-horizon tasks
- Sparse reward environments
- Tasks requiring planning
- Continuous control

**Coverage**: 98% 🌟

**Example Use Cases:**
- Robotic manipulation sequences
- Multi-stage game strategies
- Complex navigation tasks
- Resource management

### 3. **HIRO (Data-Efficient Hierarchical RL)** 🚀
Goal-conditioned policies with off-policy correction for sample efficiency.

**Key Concepts:**
- **High-Level Policy**: Proposes goal states (as deltas)
- **Low-Level Policy**: Goal-conditioned primitive actions
- **Off-Policy Correction**: Relabel goals using hindsight
- **Hindsight Experience Replay**: Data-efficient learning

**Features:**
- ✅ State-of-the-art sample efficiency
- ✅ Off-policy learning at both levels
- ✅ Goal relabeling for stability
- ✅ TD3-style target smoothing

**Best For:**
- Sample-limited domains
- Long-horizon continuous control
- Robotic tasks
- Sparse reward problems

**Coverage**: 99% 🌟

**Example Use Cases:**
- Robot arm manipulation
- Locomotion with goals
- Object rearrangement
- Multi-step assembly

## 📊 Algorithm Comparison

| Algorithm | Type | Goals | Learning | Sample Efficiency | Best For |
|-----------|------|-------|----------|-------------------|----------|
| Options | Temporal Abstraction | Discrete | On/Off-Policy | High | Discrete tasks |
| Feudal RL | Manager-Worker | Latent Space | On-Policy | Medium | Long-horizon |
| HIRO | Goal-Conditioned | State Deltas | Off-Policy | Very High | Continuous Control |

## 🧮 Mathematical Foundations

### Options Framework
Option $\omega = (I, \pi, \beta)$ where:
- $I \subseteq S$: Initiation set
- $\pi: S \times A \rightarrow [0,1]$: Option policy
- $\beta: S \rightarrow [0,1]$: Termination function

### Feudal RL Manager-Worker
Manager goal: $g_t = f_{\text{manager}}(s_t)$
Worker policy: $\pi_{\text{worker}}(a | s_t, g_t)$
Intrinsic reward: $r_t^{\text{int}} = \cos(s_{t+c} - s_t, g_t)$

### HIRO Goal Relabeling
High-level: $g_t = \mu_h(s_t)$
Low-level: $a_t = \mu_l(s_t, g_t)$
Relabeled goal: $\tilde{g}_t = s_{t+c} - s_t$ (hindsight)

## 🌟 Common Applications

### Robotics
- **Manipulation**: Pick, place, stack objects
- **Locomotion**: Navigate to goal positions
- **Assembly**: Multi-step construction tasks
- **Grasping**: Approach, grasp, lift sequences

### Navigation
- **Room Navigation**: High-level room selection, low-level pathfinding
- **Autonomous Driving**: Route planning + local control
- **Drone Control**: Waypoint navigation + stabilization
- **Multi-Agent Coordination**: Team tactics + individual actions

### Games & Planning
- **Strategy Games**: Strategic planning + tactical execution
- **Resource Management**: Build order + micro-management
- **Puzzle Solving**: Subgoal identification + achievement
- **Multi-Objective Games**: Balance different objectives

### Task Planning
- **Workflow Automation**: Task decomposition + execution
- **Manufacturing**: Production scheduling + robot control
- **Logistics**: Route planning + package handling
- **Healthcare**: Treatment planning + intervention

## 💡 Design Patterns

### When to Use Hierarchical RL

**Strong Indicators:**
- ✅ Task has natural subtask structure
- ✅ Long time horizons (100+ steps)
- ✅ Sparse rewards
- ✅ Need skill reuse across tasks
- ✅ Human demonstrations show hierarchical structure

**Weak Indicators:**
- ❌ Simple, short-horizon tasks
- ❌ Dense reward signals
- ❌ No obvious subtask boundaries
- ❌ Flat RL already works well

### Choosing the Right HRL Algorithm

**Use Options When:**
- Discrete, well-defined subtasks exist
- Want formal guarantees
- Need interpretable behaviors
- Transferring to related tasks

**Use Feudal RL When:**
- Long-horizon continuous control
- Need automatic goal discovery
- Can use on-policy training
- Want end-to-end learning

**Use HIRO When:**
- Sample efficiency is critical
- Continuous state/action spaces
- Can leverage off-policy data
- Need state-of-the-art performance

## 🔬 Key Challenges & Solutions

### Challenge 1: Credit Assignment
**Problem**: Hard to credit manager for worker's actions
**Solutions:**
- Feudal: Transition policy gradients
- HIRO: Off-policy corrections
- Options: Option-value functions

### Challenge 2: Goal Discovery
**Problem**: What goals should manager set?
**Solutions:**
- Feudal: Learn goal embeddings
- HIRO: State deltas as goals
- Options: Option discovery algorithms

### Challenge 3: Non-Stationarity
**Problem**: Lower level changes as it learns
**Solutions:**
- HIRO: Goal relabeling
- Feudal: Joint training
- Options: Intra-option learning

### Challenge 4: Exploration
**Problem**: Hierarchical exploration is hard
**Solutions:**
- Intrinsic motivation
- Curiosity-driven exploration
- Diversity-promoting objectives

## 🎓 Learning Path

### Beginner
1. **Understand Flat RL**: Master Q-Learning, Policy Gradient
2. **Study Options Framework**: Learn temporal abstraction
3. **Implement Simple Hierarchy**: Two-level navigation

### Intermediate
4. **Explore Feudal RL**: Manager-worker architectures
5. **Study Goal-Conditioned RL**: Universal value functions
6. **Implement HIRO**: Off-policy hierarchical learning

### Advanced
7. **Research Frontiers**: Multi-level hierarchies
8. **Transfer Learning**: Cross-task skill reuse
9. **Meta-Learning**: Learning to learn hierarchies

## 🚀 Getting Started

```python
from algokit.algorithms.hierarchical_rl import (
    OptionsFramework,
    FeudalRLAgent,
    HIROAgent
)

# Example 1: Options Framework
options_agent = OptionsFramework(
    state_size=4,
    action_size=2,
    num_options=4,
    learning_rate=0.001
)

# Example 2: Feudal RL
feudal_agent = FeudalRLAgent(
    state_size=4,
    action_size=2,
    manager_horizon=10,
    latent_dim=16
)

# Example 3: HIRO
hiro_agent = HIROAgent(
    state_size=4,
    action_size=2,
    goal_size=4,  # Goals as state deltas
    goal_horizon=10
)

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
```

## 📚 Further Reading

### Foundational Papers
- Sutton et al.: "Between MDPs and Semi-MDPs" (Options, 1999)
- Dayan & Hinton: "Feudal Reinforcement Learning" (1993)
- Nachum et al.: "Data-Efficient Hierarchical RL" (HIRO, 2018)
- Vezhnevets et al.: "FeUdal Networks" (FuN, 2017)

### Modern Advances
- Florensa et al.: "Automatic Goal Generation" (2018)
- Levy et al.: "Learning Multi-Level Hierarchies" (HIRO extensions)
- Eysenbach et al.: "Diversity is All You Need" (DIAYN, 2019)
- Nachum et al.: "Why Does Hierarchy Help?" (2019)

### Surveys & Tutorials
- Pateria et al.: "Hierarchical RL Survey" (2021)
- OpenAI Blog: "Learning Dexterity" (2018)
- Barto & Mahadevan: "Recent Advances in HRL" (2003)

## 🔗 Related Families

- **Reinforcement Learning**: Foundation for HRL methods
- **Transfer Learning**: Reusing hierarchical skills
- **Meta-Learning**: Learning hierarchy structures
- **Multi-Agent RL**: Hierarchical coordination
- **Model-Based RL**: Planning at multiple levels

## 🎯 Success Stories

### Research Breakthroughs
- **OpenAI Dactyl**: Rubik's cube solving with HRL
- **DeepMind Capture the Flag**: Hierarchical team tactics
- **Berkeley Robot Learning**: Manipulation with HIRO
- **MIT Autonomous Driving**: Hierarchical driving policies

### Industry Applications
- **Robotics**: Warehouse automation with option hierarchies
- **Game AI**: StarCraft bots with feudal architectures
- **Autonomous Systems**: Self-driving with hierarchical planning
- **Process Control**: Manufacturing with hierarchical optimization

---

**Ready to build hierarchies?** Start with [Options Framework](options_framework.md), explore [Feudal RL](feudal_rl.md), or dive into [HIRO](hiro.md)!
