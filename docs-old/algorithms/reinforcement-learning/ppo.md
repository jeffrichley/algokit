---
algorithm_key: "ppo"
tags: [reinforcement-learning, algorithms, ppo, proximal-policy-optimization, trust-region, clipping, actor-critic]
title: "Proximal Policy Optimization (PPO)"
family: "reinforcement-learning"
---

# Proximal Policy Optimization (PPO)

{{ algorithm_card("ppo") }}

!!! abstract "Overview"
    Proximal Policy Optimization (PPO) is a state-of-the-art policy gradient algorithm that addresses the key challenges of policy optimization: sample efficiency, training stability, and ease of implementation. PPO introduces a novel objective function that penalizes large policy updates, preventing the performance collapses that often occur with traditional policy gradient methods.

    The algorithm achieves this through a "clipped surrogate" objective that constrains policy updates to be close to the previous policy, while maintaining the benefits of policy gradient methods. PPO has become one of the most popular and successful reinforcement learning algorithms, achieving state-of-the-art performance across a wide range of domains including robotics, game playing, and autonomous systems.

## Mathematical Formulation

!!! math "PPO Clipped Surrogate Objective"
    The PPO algorithm optimizes the following clipped surrogate objective:
    
    $$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$
    
    Where:
    - $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio
    - $A_t$ is the advantage estimate at time $t$
    - $\epsilon$ is the clipping parameter (typically 0.2)
    - $\text{clip}(x, a, b) = \max(a, \min(x, b))$ clamps $x$ to $[a, b]$
    
    The clipping ensures that $r_t(\theta)$ stays within $[1-\epsilon, 1+\epsilon]$, preventing large policy updates.

!!! success "Key Properties"
    - **Clipped Objective**: Prevents large policy updates that could cause performance collapse
    - **Sample Efficiency**: Can reuse data for multiple epochs
    - **Training Stability**: More stable than traditional policy gradient methods
    - **Easy Implementation**: Simpler than TRPO while maintaining performance
    - **Wide Applicability**: Works well across many domains and problem types

## Implementation Approaches

=== "PPO with Clipped Objective (Recommended)"
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import numpy as np
    from collections import deque
    
    class PPONetwork(nn.Module):
        """PPO network with separate policy and value heads."""
        
        def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
            super(PPONetwork, self).__init__()
            
            # Shared feature layers
            self.feature_layer = nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )
            
            # Policy head
            self.policy_head = nn.Linear(hidden_size, action_size)
            
            # Value head
            self.value_head = nn.Linear(hidden_size, 1)
        
        def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            features = self.feature_layer(state)
            action_probs = F.softmax(self.policy_head(features), dim=-1)
            value = self.value_head(features)
            return action_probs, value
        
        def get_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Get action, log probability, and value for given state."""
            action_probs, value = self.forward(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob, value
    
    class PPOAgent:
        """
        PPO agent implementation.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer (default: 0.0003)
            discount_factor: Discount factor gamma (default: 0.99)
            gae_lambda: GAE lambda parameter (default: 0.95)
            clip_epsilon: PPO clipping parameter (default: 0.2)
            value_coef: Value function loss coefficient (default: 0.5)
            entropy_coef: Entropy regularization coefficient (default: 0.01)
            max_grad_norm: Maximum gradient norm for clipping (default: 0.5)
            epochs_per_update: Number of epochs per update (default: 4)
        """
        
        def __init__(self, state_size: int, action_size: int,
                     learning_rate: float = 0.0003, discount_factor: float = 0.99,
                     gae_lambda: float = 0.95, clip_epsilon: float = 0.2,
                     value_coef: float = 0.5, entropy_coef: float = 0.01,
                     max_grad_norm: float = 0.5, epochs_per_update: int = 4):
            
            self.state_size = state_size
            self.action_size = action_size
            self.gamma = discount_factor
            self.gae_lambda = gae_lambda
            self.clip_epsilon = clip_epsilon
            self.value_coef = value_coef
            self.entropy_coef = entropy_coef
            self.max_grad_norm = max_grad_norm
            self.epochs_per_update = epochs_per_update
            
            # Networks
            self.policy = PPONetwork(state_size, action_size)
            self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
            
            # Experience buffer
            self.buffer = []
        
        def get_action(self, state: np.ndarray) -> tuple[int, float, float]:
            """Sample action from policy and return action, log_prob, and value."""
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value = self.policy.get_action(state_tensor)
            return action.item(), log_prob.item(), value.item()
        
        def store_transition(self, state: np.ndarray, action: int, reward: float,
                           next_state: np.ndarray, done: bool, log_prob: float, value: float) -> None:
            """Store transition in buffer."""
            self.buffer.append((state, action, reward, next_state, done, log_prob, value))
        
        def compute_gae(self, rewards: list, values: list, dones: list) -> list:
            """Compute Generalized Advantage Estimation (GAE)."""
            advantages = []
            gae = 0
            
            for i in reversed(range(len(rewards))):
                if i == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[i + 1]
                
                delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
                advantages.insert(0, gae)
            
            return advantages
        
        def update_policy(self) -> None:
            """Update policy using PPO algorithm."""
            if len(self.buffer) == 0:
                return
            
            # Extract data from buffer
            states, actions, rewards, next_states, dones, old_log_probs, values = zip(*self.buffer)
            
            # Convert to tensors
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            old_log_probs = torch.stack(old_log_probs)
            values = torch.FloatTensor(values)
            rewards = torch.FloatTensor(rewards)
            dones = torch.BoolTensor(dones)
            
            # Compute GAE advantages
            advantages = self.compute_gae(rewards, values, dones)
            advantages = torch.FloatTensor(advantages)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Compute returns
            returns = advantages + values
            
            # Multiple epochs of updates
            for _ in range(self.epochs_per_update):
                # Get current policy outputs
                action_probs, current_values = self.policy(states)
                dist = torch.distributions.Categorical(action_probs)
                current_log_probs = dist.log_prob(actions)
                
                # Compute probability ratio
                ratio = torch.exp(current_log_probs - old_log_probs)
                
                # PPO clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value function loss
                value_loss = F.mse_loss(current_values.squeeze(), returns)
                
                # Entropy regularization
                entropy = dist.entropy().mean()
                
                # Total loss
                total_loss = (policy_loss + 
                            self.value_coef * value_loss - 
                            self.entropy_coef * entropy)
                
                # Update policy
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            # Clear buffer
            self.buffer.clear()
    ```

=== "PPO with Multiple Environments"
    ```python
    class PPOAgentMultiEnv(PPOAgent):
        """
        PPO agent that can handle multiple environments simultaneously.
        """
        
        def __init__(self, state_size: int, action_size: int, num_envs: int = 4, **kwargs):
            super().__init__(state_size, action_size, **kwargs)
            self.num_envs = num_envs
            self.env_buffers = [[] for _ in range(num_envs)]
        
        def store_transition(self, env_id: int, state: np.ndarray, action: int, reward: float,
                           next_state: np.ndarray, done: bool, log_prob: float, value: float) -> None:
            """Store transition for specific environment."""
            self.env_buffers[env_id].append((state, action, reward, next_state, done, log_prob, value))
            
            # If episode ends, move to main buffer
            if done:
                self.buffer.extend(self.env_buffers[env_id])
                self.env_buffers[env_id].clear()
        
        def get_actions(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Get actions for multiple environments simultaneously."""
            states_tensor = torch.FloatTensor(states)
            actions, log_probs, values = self.policy.get_action(states_tensor)
            return actions.numpy(), log_probs.numpy(), values.numpy()
    ```

=== "PPO with Continuous Actions"
    ```python
    class ContinuousPPONetwork(nn.Module):
        """PPO network for continuous action spaces."""
        
        def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
            super(ContinuousPPONetwork, self).__init__()
            
            # Shared feature layers
            self.feature_layer = nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )
            
            # Policy head (mean and log_std)
            self.policy_mean = nn.Linear(hidden_size, action_size)
            self.policy_log_std = nn.Linear(hidden_size, action_size)
            
            # Value head
            self.value_head = nn.Linear(hidden_size, 1)
        
        def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            features = self.feature_layer(state)
            
            # Policy outputs
            mean = torch.tanh(self.policy_mean(features))
            log_std = self.policy_log_std(features)
            
            # Value output
            value = self.value_head(features)
            
            return mean, log_std, value
        
        def get_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Get action, log probability, and value for continuous actions."""
            mean, log_std, value = self.forward(state)
            std = torch.exp(log_std)
            
            # Create normal distribution
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            return action, log_prob, value
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/reinforcement_learning/ppo.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/reinforcement_learning/ppo.py)
    - **Tests**: [`tests/unit/reinforcement_learning/test_ppo.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/reinforcement_learning/test_ppo.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **PPO (Discrete)** | O(batch_size × epochs × network_params) | O(batch_size × state_size) | Clipped objective, multiple epochs |
    **PPO Multi-Env** | O(total_batch_size × epochs × network_params) | O(total_batch_size × state_size) | Parallel environment sampling |
    **PPO Continuous** | O(batch_size × epochs × network_params) | O(batch_size × state_size) | Continuous action spaces |

!!! warning "Performance Considerations"
    - **Multiple epochs** increase computational cost per update
    - **GAE computation** adds overhead but improves advantage estimation
    - **Clipping parameter** affects training stability and convergence
    - **Hyperparameter tuning** is crucial for optimal performance

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Robotics & Control"
        - **Robot Manipulation**: Complex grasping and manipulation tasks
        - **Autonomous Vehicles**: Navigation and control in dynamic environments
        - **Industrial Automation**: Process optimization and control
        - **Swarm Robotics**: Multi-agent coordination and control

    !!! grid-item "Game AI & Entertainment"
        - **Video Games**: Character behavior and strategy learning
        - **Simulation Games**: Resource management and decision making
        - **Board Games**: Complex strategy optimization
        - **Virtual Environments**: NPC behavior and interaction

    !!! grid-item "Real-World Applications"
        - **Finance**: Portfolio optimization and trading strategies
        - **Healthcare**: Treatment scheduling and optimization
        - **Energy Management**: Consumption optimization and demand response
        - **Network Control**: Traffic routing and congestion management

    !!! grid-item "Educational Value"
        - **Modern RL**: Understanding state-of-the-art policy optimization
        - **Trust Region Methods**: Learning importance of constrained updates
        - **Advantage Estimation**: Understanding GAE and its benefits
        - **Training Stability**: Learning techniques for stable policy learning

!!! success "Educational Value"
    - **Modern Algorithms**: Perfect example of state-of-the-art policy optimization
    - **Trust Region Methods**: Shows importance of constraining policy updates
    - **Advantage Estimation**: Demonstrates benefits of GAE for stable learning
    - **Implementation Best Practices**: Illustrates modern RL implementation patterns

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Papers"
        1. **Schulman, J., et al.** (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
        2. **Schulman, J., et al.** (2015). Trust region policy optimization. *ICML*, 37.

    !!! grid-item "PPO and Trust Region Textbooks"
        3. **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction*. MIT Press.
        4. **François-Lavet, V., et al.** (2018). *Deep Reinforcement Learning: Fundamentals, Research and Applications*. Springer.

    !!! grid-item "Online Resources"
        5. [PPO Algorithm - Wikipedia](https://en.wikipedia.org/wiki/Proximal_policy_optimization)
        6. [OpenAI PPO Implementation](https://github.com/openai/baselines)
        7. [PPO Tutorial](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

    !!! grid-item "Implementation & Practice"
        8. [PyTorch Documentation](https://pytorch.org/docs/)
        9. [OpenAI Gym](https://www.gymlibrary.dev/) - RL environments
        10. [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - PPO implementation

!!! tip "Interactive Learning"
    Try implementing PPO yourself! Start with discrete action spaces to understand the clipped objective, then move to continuous control problems. Implement GAE to see how it improves advantage estimation. Experiment with different clipping parameters and number of epochs to understand their impact on training stability. Compare with other policy gradient methods to see why PPO is often more stable and sample efficient. This will give you deep insight into modern reinforcement learning best practices.

## Navigation

{{ nav_grid(current_algorithm="ppo", current_family="reinforcement-learning", max_related=5) }}
