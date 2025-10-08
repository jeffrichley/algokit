# Reinforcement Learning Overview

## 🤖 Introduction

**Reinforcement Learning (RL)** is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward. Unlike supervised learning, the agent learns through trial and error, receiving feedback in the form of rewards or penalties without being explicitly told which action to take.

## 🎯 Core Concepts

### The RL Framework

```
┌─────────┐                    ┌──────────────┐
│  Agent  │◄───── State ────────┤              │
│         │                     │ Environment  │
│         ├────── Action ───────►              │
└─────────┘◄───── Reward ───────┘              │
                                └──────────────┘
```

### Key Components

- **Agent**: The learner/decision maker
- **Environment**: The world the agent interacts with
- **State (S)**: Current situation of the environment
- **Action (A)**: Choices available to the agent
- **Reward (R)**: Feedback signal indicating desirability of outcome
- **Policy (π)**: Agent's strategy for choosing actions
- **Value Function (V)**: Expected long-term return from a state
- **Q-Function (Q)**: Expected return from taking an action in a state

## 🔑 Key Characteristics

- **Agent-Environment Interaction**: Learning through experience
- **Reward Maximization**: Goal is to maximize cumulative reward
- **Policy Learning**: Discovering optimal behavior strategy
- **Value Function Approximation**: Estimating long-term value
- **Exploration vs Exploitation**: Balancing trying new actions vs known good actions
- **Markov Decision Processes**: Mathematical framework for sequential decisions

## 🎨 Implemented Algorithms

### 1. **Q-Learning** 📊
Model-free, off-policy algorithm that learns optimal action-value function.
- **Type**: Value-based, Tabular
- **Learning**: Off-policy (learns optimal policy while following exploratory policy)
- **Convergence**: Guaranteed with proper conditions
- **Best For**: Discrete state/action spaces
- **Coverage**: 97%

### 2. **SARSA** 🎲
On-policy TD control algorithm that updates Q-values based on actual actions taken.
- **Type**: Value-based, Tabular
- **Learning**: On-policy (learns policy being followed)
- **Convergence**: More conservative than Q-learning
- **Best For**: Safety-critical applications
- **Coverage**: 96%

### 3. **Actor-Critic** 🎭
Hybrid approach combining policy-based and value-based methods.
- **Type**: Hybrid, Policy Gradient
- **Components**: Actor (policy) + Critic (value function)
- **Learning**: On-policy or off-policy variants
- **Best For**: Continuous action spaces
- **Coverage**: 90%

### 4. **Deep Q-Network (DQN)** 🧠
Deep learning extension of Q-learning using neural networks.
- **Type**: Value-based, Deep RL
- **Innovation**: Experience replay + target networks
- **Learning**: Off-policy with function approximation
- **Best For**: High-dimensional state spaces (images, complex observations)
- **Coverage**: 94%

### 5. **Policy Gradient** 🎯
Directly optimizes policy without value function.
- **Type**: Policy-based
- **Method**: REINFORCE algorithm
- **Learning**: Monte Carlo, on-policy
- **Best For**: Continuous action spaces, stochastic policies
- **Coverage**: 100%

### 6. **PPO (Proximal Policy Optimization)** 🚀
State-of-the-art policy gradient method with stability improvements.
- **Type**: Policy-based, Advanced
- **Innovation**: Clipped surrogate objective for stable updates
- **Learning**: On-policy, mini-batch updates
- **Best For**: Production systems, robotics
- **Coverage**: 91%

## 🧮 Mathematical Foundations

### Bellman Equation
The fundamental recursive relationship in RL:

$$V(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a)V(s') \right]$$

### Q-Learning Update Rule
$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

### Policy Gradient Theorem
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} [\nabla_\theta \log \pi_\theta(a|s) Q^{\pi}(s,a)]$$

## 📊 Algorithm Comparison

| Algorithm | Type | Space | Best For | Sample Efficiency | Stability |
|-----------|------|-------|----------|-------------------|-----------|
| Q-Learning | Value | Discrete | Tabular | High | High |
| SARSA | Value | Discrete | Safety | High | Very High |
| Actor-Critic | Hybrid | Any | Balanced | Medium | Medium |
| DQN | Deep Value | Large/Cont | Images | Medium | Medium |
| Policy Gradient | Policy | Any | Continuous | Low | Low |
| PPO | Policy | Any | Production | Medium | High |

## 🌟 Common Applications

### Game Playing
- Chess, Go, Shogi (AlphaGo, AlphaZero)
- Video games (Atari, StarCraft)
- Board games and puzzles
- Multi-player game AI

### Robotics & Control
- Robot manipulation and grasping
- Locomotion and walking
- Autonomous navigation
- Drone control
- Warehouse automation

### Finance & Trading
- Portfolio optimization
- Algorithmic trading
- Risk management
- Market making
- Option pricing

### Healthcare & Medicine
- Treatment optimization
- Drug discovery
- Personalized medicine
- Resource allocation
- Diagnostic assistance

### Autonomous Systems
- Self-driving cars
- Traffic control
- Energy management
- Network optimization
- Smart grids

## 💡 Choosing the Right Algorithm

### Tabular Methods (Q-Learning, SARSA)
**Use When:**
- ✅ Small, discrete state/action spaces
- ✅ Need guaranteed convergence
- ✅ Want interpretable results
- ✅ Have sufficient compute for full exploration

**Avoid When:**
- ❌ Large or continuous state spaces
- ❌ High-dimensional observations
- ❌ Limited memory available

### Deep RL (DQN, PPO)
**Use When:**
- ✅ High-dimensional observations (images, sensors)
- ✅ Complex state spaces
- ✅ Need function approximation
- ✅ Have GPU compute available

**Avoid When:**
- ❌ Small, simple problems (overkill)
- ❌ Need theoretical guarantees
- ❌ Limited data/compute
- ❌ Require full interpretability

### Policy Gradient (REINFORCE, PPO)
**Use When:**
- ✅ Continuous action spaces
- ✅ Stochastic policies needed
- ✅ Direct policy optimization preferred
- ✅ Have sufficient samples

**Avoid When:**
- ❌ High variance is problematic
- ❌ Need sample efficiency
- ❌ Value-based methods suffice

## 🔬 Exploration Strategies

### ε-Greedy
- Simple: With probability ε, take random action
- Effective for discrete spaces
- Used in Q-Learning, DQN

### Softmax / Boltzmann
- Temperature-based probabilistic selection
- Smoother exploration than ε-greedy
- Common in policy gradient methods

### Upper Confidence Bound (UCB)
- Optimistic initialization
- Balance exploration/exploitation mathematically
- Used in bandits and tree search

### Curiosity & Intrinsic Motivation
- Reward agent for exploring novel states
- Effective in sparse reward environments
- Advanced technique for complex problems

## 🎓 Learning Path

### Beginner
1. **Q-Learning**: Start here - understand value functions
2. **SARSA**: Learn on-policy vs off-policy
3. **Grid World**: Practice with simple environments

### Intermediate
4. **Actor-Critic**: Combine value and policy
5. **DQN**: Introduction to deep RL
6. **Policy Gradient**: Direct policy optimization

### Advanced
7. **PPO**: State-of-the-art stable learning
8. **Hierarchical RL**: Complex task decomposition
9. **Multi-Agent RL**: Coordination and competition

## 📈 Training Best Practices

### Hyperparameter Tuning
```python
# Critical hyperparameters
learning_rate = 0.001   # Step size for updates
gamma = 0.99            # Discount factor
epsilon = 0.1           # Exploration rate
batch_size = 32         # Mini-batch size
```

### Reward Engineering
- Shape rewards to guide learning
- Avoid sparse rewards when possible
- Normalize rewards for stability
- Use reward clipping if needed

### Network Architecture
- Start simple, add complexity as needed
- Use appropriate activations (ReLU common)
- Normalize inputs to neural networks
- Consider dueling architectures for DQN

### Debugging RL
- Monitor learning curves (reward, loss)
- Visualize policy behavior
- Check Q-value estimates
- Validate exploration is happening
- Use simpler environments first

## 🔗 Related Families

- **Hierarchical RL**: Decompose complex tasks (Options, HIRO, Feudal)
- **Deep Learning**: Neural network function approximation
- **Control Systems**: Classical control theory connections
- **Optimization**: Policy optimization techniques
- **Planning**: Model-based RL and planning algorithms

## 🚀 Getting Started

```python
from algokit.algorithms.reinforcement_learning import (
    QLearningAgent,
    SARSAAgent,
    ActorCriticAgent,
    DQNAgent,
    PolicyGradientAgent,
    PPOAgent
)

# Example: Q-Learning
agent = QLearningAgent(
    state_size=16,
    action_size=4,
    learning_rate=0.1,
    gamma=0.99,
    epsilon=0.1
)

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    print(f"Episode {episode}: Reward = {total_reward}")
```

## 📚 Further Reading

### Classic Papers
- Sutton & Barto: "Reinforcement Learning: An Introduction" (2018)
- Mnih et al.: "Human-level control through deep RL" (DQN, 2015)
- Schulman et al.: "Proximal Policy Optimization" (PPO, 2017)
- Silver et al.: "Mastering the game of Go" (AlphaGo, 2016)

### Online Resources
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [DeepMind x UCL RL Course](https://www.deepmind.com/learning-resources/reinforcement-learning-lecture-series-2021)
- [Sutton & Barto Book (Free)](http://incompleteideas.net/book/the-book.html)
- [CleanRL Implementation Reference](https://github.com/vwxyzjn/cleanrl)

### Benchmarks
- OpenAI Gym / Gymnasium
- MuJoCo Physics Engine
- Atari 2600 Games
- DeepMind Control Suite
- ProcGen (Generalization)

## 🎯 Practice Environments

### Beginner
- CartPole: Balance pole on cart
- FrozenLake: Navigate icy grid
- Taxi: Pick up and drop off passengers
- Mountain Car: Drive car up hill

### Intermediate
- Lunar Lander: Land spacecraft safely
- Atari Games: Learn from pixels
- MuJoCo Humanoid: Bipedal walking
- Multi-armed Bandit: Resource allocation

### Advanced
- StarCraft II: Real-time strategy
- CARLA: Autonomous driving
- Unity ML-Agents: 3D environments
- Multi-Agent Particle: Coordination

---

**Ready to start learning?** Begin with [Q-Learning](q_learning.md) or explore [Deep RL](dqn.md) for complex problems!
