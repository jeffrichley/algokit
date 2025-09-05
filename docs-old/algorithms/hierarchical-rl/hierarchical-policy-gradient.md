---
algorithm_key: "hierarchical-policy-gradient"
tags: [hierarchical-rl, algorithms, hierarchical-policy-gradient, temporal-abstraction, subgoals, policy-optimization]
title: "Hierarchical Policy Gradient"
family: "hierarchical-rl"
---

# Hierarchical Policy Gradient

{{ algorithm_card("hierarchical-policy-gradient") }}

!!! abstract "Overview"
    Hierarchical Policy Gradient extends the traditional policy gradient framework to handle temporal abstraction and hierarchical task decomposition. The algorithm learns policies at multiple levels: a high-level meta-policy that selects subgoals or options, and low-level policies that execute primitive actions to achieve these subgoals.

    This hierarchical approach enables the agent to solve complex, long-horizon tasks by breaking them down into manageable subproblems. The meta-policy learns to sequence subgoals effectively, while the low-level policies learn to achieve specific subgoals efficiently. Hierarchical Policy Gradient is particularly powerful in domains where tasks have natural hierarchical structure, such as robotics manipulation, navigation, and game playing.

## Mathematical Formulation

!!! math "Hierarchical Policy Gradient Theorem"
    The hierarchical policy gradient can be derived from the policy gradient theorem:
    
    $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(\tau) R(\tau) \right]$$
    
    For hierarchical policies, this decomposes into:
    
    $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_h} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_{meta}(g_t|s_t) R(\tau) + \nabla_\theta \log \pi_{low}(a_t|s_t, g_t) R(\tau) \right]$$
    
    Where:
    - $\pi_h$ is the hierarchical policy
    - $\pi_{meta}$ is the meta-policy that selects subgoals $g_t$
    - $\pi_{low}$ is the low-level policy that executes actions $a_t$ given subgoal $g_t$
    - $R(\tau)$ is the return of trajectory $\tau$
    
    The hierarchical advantage function is:
    
    $$A_h(s_t, g_t, a_t) = Q_h(s_t, g_t, a_t) - V_h(s_t)$$
    
    Where $Q_h$ and $V_h$ are the hierarchical Q-function and value function respectively.

!!! success "Key Properties"
    - **Temporal Abstraction**: High-level policies operate over longer time horizons
    - **Subgoal Decomposition**: Complex tasks broken into manageable subproblems
    - **Hierarchical Learning**: Policies at different levels learn simultaneously
    - **Transfer Learning**: Low-level policies can be reused across different tasks
    - **Sample Efficiency**: Better exploration and learning in complex environments

## Implementation Approaches

=== "Basic Hierarchical Policy Gradient (Recommended)"
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import numpy as np
    
    class MetaPolicyNetwork(nn.Module):
        """Meta-policy network that selects subgoals."""
        
        def __init__(self, state_size: int, subgoal_size: int, hidden_size: int = 128):
            super(MetaPolicyNetwork, self).__init__()
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, subgoal_size)
        
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            subgoal_probs = F.softmax(self.fc3(x), dim=-1)
            return subgoal_probs
    
    class LowLevelPolicyNetwork(nn.Module):
        """Low-level policy network that executes actions given subgoals."""
        
        def __init__(self, state_size: int, subgoal_size: int, action_size: int, hidden_size: int = 128):
            super(LowLevelPolicyNetwork, self).__init__()
            self.fc1 = nn.Linear(state_size + subgoal_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, action_size)
        
        def forward(self, state: torch.Tensor, subgoal: torch.Tensor) -> torch.Tensor:
            # Concatenate state and subgoal
            combined = torch.cat([state, subgoal], dim=-1)
            x = F.relu(self.fc1(combined))
            x = F.relu(self.fc2(x))
            action_probs = F.softmax(self.fc3(x), dim=-1)
            return action_probs
    
    class HierarchicalPolicyGradientAgent:
        """
        Hierarchical Policy Gradient agent implementation.
        
        Args:
            state_size: Dimension of state space
            subgoal_size: Dimension of subgoal space
            action_size: Number of possible actions
            meta_lr: Learning rate for meta-policy (default: 0.001)
            low_lr: Learning rate for low-level policy (default: 0.001)
            discount_factor: Discount factor gamma (default: 0.99)
            subgoal_horizon: Maximum steps to achieve subgoal (default: 50)
        """
        
        def __init__(self, state_size: int, subgoal_size: int, action_size: int,
                     meta_lr: float = 0.001, low_lr: float = 0.001,
                     discount_factor: float = 0.99, subgoal_horizon: int = 50):
            
            self.state_size = state_size
            self.subgoal_size = subgoal_size
            self.action_size = action_size
            self.gamma = discount_factor
            self.subgoal_horizon = subgoal_horizon
            
            # Networks
            self.meta_policy = MetaPolicyNetwork(state_size, subgoal_size)
            self.low_policy = LowLevelPolicyNetwork(state_size, subgoal_size, action_size)
            
            # Optimizers
            self.meta_optimizer = optim.Adam(self.meta_policy.parameters(), lr=meta_lr)
            self.low_optimizer = optim.Adam(self.low_policy.parameters(), lr=low_lr)
            
            # Experience buffers
            self.meta_buffer = []
            self.low_buffer = []
            
            # Current subgoal tracking
            self.current_subgoal = None
            self.subgoal_steps = 0
        
        def get_subgoal(self, state: np.ndarray) -> tuple[int, float]:
            """Select subgoal using meta-policy."""
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            subgoal_probs = self.meta_policy(state_tensor)
            
            # Sample subgoal
            dist = torch.distributions.Categorical(subgoal_probs)
            subgoal = dist.sample()
            log_prob = dist.log_prob(subgoal)
            
            return subgoal.item(), log_prob.item()
        
        def get_action(self, state: np.ndarray, subgoal: int) -> tuple[int, float]:
            """Get action using low-level policy given subgoal."""
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            subgoal_tensor = torch.FloatTensor([subgoal]).unsqueeze(0)
            
            action_probs = self.low_policy(state_tensor, subgoal_tensor)
            
            # Sample action
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item()
        
        def step(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
            """Process one step and potentially update subgoal."""
            # Store low-level transition
            if self.current_subgoal is not None:
                self.low_buffer.append((state, self.current_subgoal, action, reward, next_state, done))
            
            # Check if subgoal should be updated
            self.subgoal_steps += 1
            if (self.subgoal_steps >= self.subgoal_horizon or 
                self.is_subgoal_achieved(state, next_state) or done):
                
                # Store meta-level transition
                if len(self.low_buffer) > 0:
                    total_reward = sum(r for _, _, _, r, _, _ in self.low_buffer)
                    self.meta_buffer.append((state, self.current_subgoal, total_reward, next_state, done))
                
                # Select new subgoal
                if not done:
                    new_subgoal, _ = self.get_subgoal(next_state)
                    self.current_subgoal = new_subgoal
                    self.subgoal_steps = 0
                
                # Clear low-level buffer
                self.low_buffer.clear()
        
        def is_subgoal_achieved(self, state: np.ndarray, next_state: np.ndarray) -> bool:
            """Check if current subgoal has been achieved."""
            # Simple distance-based subgoal achievement
            # In practice, this would be domain-specific
            return np.linalg.norm(next_state - state) < 0.1
        
        def update_policies(self) -> None:
            """Update all policies using stored experience."""
            self.update_low_level_policy()
            self.update_meta_policy()
        
        def update_low_level_policy(self) -> None:
            """Update low-level policy using Monte Carlo policy gradient."""
            if len(self.low_buffer) < 10:
                return
            
            # Calculate discounted returns for low-level transitions
            states, subgoals, actions, rewards, next_states, dones = zip(*self.low_buffer)
            
            # Convert to tensors
            states = torch.FloatTensor(states)
            subgoals = torch.LongTensor(subgoals)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            dones = torch.BoolTensor(dones)
            
            # Calculate discounted returns
            returns = []
            discounted_return = 0
            for reward, done in zip(reversed(rewards), reversed(dones)):
                if done:
                    discounted_return = 0
                discounted_return = reward + self.gamma * discounted_return
                returns.insert(0, discounted_return)
            
            returns = torch.FloatTensor(returns)
            
            # Normalize returns for stability
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Get current action probabilities
            action_probs = self.low_policy(states, subgoals)
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            
            # Policy gradient loss
            policy_loss = -(log_probs * returns).mean()
            
            # Update low-level policy
            self.low_optimizer.zero_grad()
            policy_loss.backward()
            self.low_optimizer.step()
        
        def update_meta_policy(self) -> None:
            """Update meta-policy using Monte Carlo policy gradient."""
            if len(self.meta_buffer) < 10:
                return
            
            # Calculate discounted returns for meta-level transitions
            states, subgoals, rewards, next_states, dones = zip(*self.meta_buffer)
            
            # Convert to tensors
            states = torch.FloatTensor(states)
            subgoals = torch.LongTensor(subgoals)
            rewards = torch.FloatTensor(rewards)
            dones = torch.BoolTensor(dones)
            
            # Calculate discounted returns
            returns = []
            discounted_return = 0
            for reward, done in zip(reversed(rewards), reversed(dones)):
                if done:
                    discounted_return = 0
                discounted_return = reward + self.gamma * discounted_return
                returns.insert(0, discounted_return)
            
            returns = torch.FloatTensor(returns)
            
            # Normalize returns for stability
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Get current subgoal probabilities
            subgoal_probs = self.meta_policy(states)
            dist = torch.distributions.Categorical(subgoal_probs)
            log_probs = dist.log_prob(subgoals)
            
            # Policy gradient loss
            policy_loss = -(log_probs * returns).mean()
            
            # Update meta-policy
            self.meta_optimizer.zero_grad()
            policy_loss.backward()
            self.meta_optimizer.step()
    ```

=== "HPC with Baseline (Advanced)"
    ```python
    class HPCWithBaseline(HierarchicalPolicyGradientAgent):
        """
        Hierarchical Policy Gradient agent with baseline for variance reduction.
        """
        
        def __init__(self, state_size: int, subgoal_size: int, action_size: int, **kwargs):
            super().__init__(state_size, subgoal_size, action_size, **kwargs)
            
            # Baseline networks for variance reduction
            self.meta_baseline = nn.Sequential(
                nn.Linear(state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            
            self.low_baseline = nn.Sequential(
                nn.Linear(state_size + subgoal_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            
            # Baseline optimizers
            self.meta_baseline_optimizer = optim.Adam(self.meta_baseline.parameters(), lr=0.001)
            self.low_baseline_optimizer = optim.Adam(self.low_baseline.parameters(), lr=0.001)
        
        def update_low_level_policy(self) -> None:
            """Update low-level policy with baseline correction."""
            if len(self.low_buffer) < 10:
                return
            
            states, subgoals, actions, rewards, next_states, dones = zip(*self.low_buffer)
            
            states = torch.FloatTensor(states)
            subgoals = torch.LongTensor(subgoals)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            dones = torch.BoolTensor(dones)
            
            # Calculate discounted returns
            returns = []
            discounted_return = 0
            for reward, done in zip(reversed(rewards), reversed(dones)):
                if done:
                    discounted_return = 0
                discounted_return = reward + self.gamma * discounted_return
                returns.insert(0, discounted_return)
            
            returns = torch.FloatTensor(returns)
            
            # Get baseline values
            combined_inputs = torch.cat([states, F.one_hot(subgoals, self.subgoal_size).float()], dim=-1)
            baselines = self.low_baseline(combined_inputs).squeeze()
            
            # Calculate advantages
            advantages = returns - baselines.detach()
            
            # Get current action probabilities
            action_probs = self.low_policy(states, subgoals)
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            
            # Policy gradient loss with baseline
            policy_loss = -(log_probs * advantages).mean()
            
            # Baseline loss
            baseline_loss = F.mse_loss(baselines, returns)
            
            # Update networks
            self.low_optimizer.zero_grad()
            policy_loss.backward()
            self.low_optimizer.step()
            
            self.low_baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.low_baseline_optimizer.step()
    ```

=== "HPC with Entropy Regularization"
    ```python
    class HPCWithEntropy(HierarchicalPolicyGradientAgent):
        """
        Hierarchical Policy Gradient agent with entropy regularization.
        """
        
        def __init__(self, state_size: int, subgoal_size: int, action_size: int, 
                     entropy_coef: float = 0.01, **kwargs):
            super().__init__(state_size, subgoal_size, action_size, **kwargs)
            self.entropy_coef = entropy_coef
        
        def update_low_level_policy(self) -> None:
            """Update low-level policy with entropy regularization."""
            if len(self.low_buffer) < 10:
                return
            
            states, subgoals, actions, rewards, next_states, dones = zip(*self.low_buffer)
            
            states = torch.FloatTensor(states)
            subgoals = torch.LongTensor(subgoals)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            dones = torch.BoolTensor(dones)
            
            # Calculate discounted returns
            returns = []
            discounted_return = 0
            for reward, done in zip(reversed(rewards), reversed(dones)):
                if done:
                    discounted_return = 0
                discounted_return = reward + self.gamma * discounted_return
                returns.insert(0, discounted_return)
            
            returns = torch.FloatTensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Get current action probabilities
            action_probs = self.low_policy(states, subgoals)
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Policy gradient loss with entropy regularization
            policy_loss = -(log_probs * returns).mean() - self.entropy_coef * entropy
            
            # Update low-level policy
            self.low_optimizer.zero_grad()
            policy_loss.backward()
            self.low_optimizer.step()
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/hierarchical_rl/hierarchical_policy_gradient.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/hierarchical_rl/hierarchical_policy_gradient.py)
    - **Tests**: [`tests/unit/hierarchical_rl/test_hierarchical_policy_gradient.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/hierarchical_rl/test_hierarchical_policy_gradient.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Basic HPC** | O(episode_length × (policy_params + meta_policy_params)) | O(episode_length × (state_size + subgoal_size)) | Monte Carlo updates |
    **HPC + Baseline** | O(episode_length × (policy_params + baseline_params)) | O(episode_length × (state_size + subgoal_size)) | Reduced variance |
    **HPC + Entropy** | O(episode_length × (policy_params + meta_policy_params)) | O(episode_length × (state_size + subgoal_size)) | Better exploration |

!!! warning "Performance Considerations"
    - **Monte Carlo updates** require complete episodes
    - **Two-level learning** requires careful coordination between policies
    - **Subgoal horizon** affects exploration and learning efficiency
    - **Baseline networks** add computational overhead but reduce variance

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
        - **Multi-Level Policies**: Understanding coordination between policy levels
        - **Transfer Learning**: Learning reusable low-level skills

!!! success "Educational Value"
    - **Hierarchical Learning**: Perfect example of temporal abstraction in RL
    - **Task Decomposition**: Shows how to break complex problems into manageable parts
    - **Multi-Level Coordination**: Demonstrates learning at different time scales
    - **Transfer Learning**: Illustrates how skills can be reused across tasks

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Papers"
        1. **Levy, A., et al.** (2019). Hierarchical actor-critic. *Advances in Neural Information Processing Systems*, 32.
        2. **Nachum, O., et al.** (2018). Data-efficient hierarchical reinforcement learning. *Advances in Neural Information Processing Systems*, 31.

    !!! grid-item "Hierarchical RL Textbooks"
        3. **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction*. MIT Press.
        4. **Kaelbling, L. P., et al.** (1998). Hierarchical reinforcement learning with the MAXQ value function decomposition. *Journal of Artificial Intelligence Research*, 13.

    !!! grid-item "Online Resources"
        5. [Hierarchical Reinforcement Learning - Wikipedia](https://en.wikipedia.org/wiki/Hierarchical_reinforcement_learning)
        6. [HAC Implementation Guide](https://github.com/andrew-j-levy/Hierarchical-Actor-Critic-HAC-)
        7. [Hierarchical RL Tutorial](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

    !!! grid-item "Implementation & Practice"
        8. [PyTorch Documentation](https://pytorch.org/docs/)
        9. [OpenAI Gym](https://www.gymlibrary.dev/) - RL environments
        10. [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - RL implementations

!!! tip "Interactive Learning"
    Try implementing Hierarchical Policy Gradient yourself! Start with simple environments that have natural hierarchical structure, like navigation tasks with waypoints. Implement the basic two-level hierarchy first, then add baseline networks for variance reduction. Experiment with different subgoal horizons and entropy regularization to see their impact on exploration and learning. Compare with flat policy gradient methods to see the benefits of hierarchical decomposition. This will give you deep insight into the power of temporal abstraction in reinforcement learning.

## Navigation

{{ nav_grid(current_algorithm="hierarchical-policy-gradient", current_family="hierarchical-rl", max_related=5) }}
