---
tags: [reinforcement-learning, algorithms, policy-gradient, monte-carlo, variance-reduction, baseline-methods]
title: "Policy Gradient Methods"
family: "reinforcement-learning"
complexity: "O(episode_length × policy_params)"
---

# Policy Gradient Methods

!!! info "Algorithm Family"
    **Family:** [Reinforcement Learning](../../families/reinforcement-learning.md)

!!! abstract "Overview"
    Policy Gradient methods are a fundamental class of reinforcement learning algorithms that directly optimize the policy by following the gradient of expected return with respect to policy parameters. Unlike value-based methods that learn value functions and derive policies, policy gradient methods directly parameterize and optimize the policy function.

    These methods have several advantages: they can handle continuous action spaces naturally, work well with stochastic policies, and can incorporate domain knowledge through policy parameterization. However, they also face challenges like high variance in gradient estimates and potential convergence to local optima. Policy gradient methods form the foundation for many modern algorithms including REINFORCE, TRPO, and PPO.

## Mathematical Formulation

!!! math "Policy Gradient Theorem"
    The policy gradient theorem states that:
    
    $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(\tau) R(\tau) \right]$$
    
    Where:
    - $J(\theta)$ is the expected return objective
    - $\pi_\theta(\tau)$ is the probability of trajectory $\tau$ under policy $\pi_\theta$
    - $R(\tau)$ is the return of trajectory $\tau$
    
    For a single timestep, this becomes:
    
    $$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) Q^\pi(s, a) \right]$$
    
    Where $\rho^\pi$ is the state distribution under policy $\pi$.

!!! success "Key Properties"
    - **Direct Policy Optimization**: Optimizes policy parameters directly
    - **Continuous Actions**: Naturally handles continuous action spaces
    - **Stochastic Policies**: Works well with probabilistic action selection
    - **High Variance**: Gradient estimates can have high variance
    - **Local Optima**: May converge to suboptimal local solutions

## Implementation Approaches

=== "REINFORCE (Monte Carlo Policy Gradient)"
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import numpy as np
    
    class PolicyNetwork(nn.Module):
        """Policy network that outputs action probabilities."""
        
        def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
            super(PolicyNetwork, self).__init__()
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, action_size)
        
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            action_probs = F.softmax(self.fc3(x), dim=-1)
            return action_probs
    
    class REINFORCEAgent:
        """
        REINFORCE agent implementation (Monte Carlo Policy Gradient).
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            learning_rate: Learning rate for policy network (default: 0.001)
            discount_factor: Discount factor gamma (default: 0.99)
        """
        
        def __init__(self, state_size: int, action_size: int,
                     learning_rate: float = 0.001, discount_factor: float = 0.99):
            
            self.state_size = state_size
            self.action_size = action_size
            self.gamma = discount_factor
            
            # Policy network
            self.policy = PolicyNetwork(state_size, action_size)
            self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
            
            # Episode storage
            self.episode_states = []
            self.episode_actions = []
            self.episode_rewards = []
            self.episode_log_probs = []
        
        def get_action(self, state: np.ndarray) -> tuple[int, float]:
            """Sample action from policy and return action with log probability."""
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = self.policy(state_tensor)
            
            # Sample action from distribution
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            return action.item(), log_prob.item()
        
        def store_transition(self, state: np.ndarray, action: int, reward: float, log_prob: float) -> None:
            """Store transition for current episode."""
            self.episode_states.append(state)
            self.episode_actions.append(action)
            self.episode_rewards.append(reward)
            self.episode_log_probs.append(log_prob)
        
        def update_policy(self) -> None:
            """Update policy using Monte Carlo policy gradient."""
            if len(self.episode_rewards) == 0:
                return
            
            # Calculate discounted returns
            returns = []
            discounted_return = 0
            for reward in reversed(self.episode_rewards):
                discounted_return = reward + self.gamma * discounted_return
                returns.insert(0, discounted_return)
            
            # Normalize returns for stability
            returns = torch.FloatTensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Convert episode data to tensors
            states = torch.FloatTensor(self.episode_states)
            actions = torch.LongTensor(self.episode_actions)
            log_probs = torch.stack(self.episode_log_probs)
            
            # Calculate policy gradient loss
            policy_loss = -(log_probs * returns).mean()
            
            # Update policy
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            
            # Clear episode data
            self.episode_states.clear()
            self.episode_actions.clear()
            self.episode_rewards.clear()
            self.episode_log_probs.clear()
    ```

=== "REINFORCE with Baseline"
    ```python
    class BaselineNetwork(nn.Module):
        """Baseline network for variance reduction."""
        
        def __init__(self, state_size: int, hidden_size: int = 128):
            super(BaselineNetwork, self).__init__()
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, 1)
        
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            baseline = self.fc3(x)
            return baseline
    
    class REINFORCEBaselineAgent(REINFORCEAgent):
        """
        REINFORCE agent with baseline for variance reduction.
        """
        
        def __init__(self, state_size: int, action_size: int,
                     learning_rate: float = 0.001, baseline_lr: float = 0.001,
                     discount_factor: float = 0.99):
            super().__init__(state_size, action_size, learning_rate, discount_factor)
            
            # Baseline network
            self.baseline = BaselineNetwork(state_size)
            self.baseline_optimizer = optim.Adam(self.baseline.parameters(), lr=baseline_lr)
        
        def update_policy(self) -> None:
            """Update policy and baseline using baseline-corrected policy gradient."""
            if len(self.episode_rewards) == 0:
                return
            
            # Calculate discounted returns
            returns = []
            discounted_return = 0
            for reward in reversed(self.episode_rewards):
                discounted_return = reward + self.gamma * discounted_return
                returns.insert(0, discounted_return)
            
            # Convert episode data to tensors
            states = torch.FloatTensor(self.episode_states)
            actions = torch.LongTensor(self.episode_actions)
            log_probs = torch.stack(self.episode_log_probs)
            returns = torch.FloatTensor(returns)
            
            # Get baseline values
            baselines = self.baseline(states).squeeze()
            
            # Calculate advantage (return - baseline)
            advantages = returns - baselines.detach()
            
            # Policy gradient loss
            policy_loss = -(log_probs * advantages).mean()
            
            # Baseline loss (MSE regression)
            baseline_loss = F.mse_loss(baselines, returns)
            
            # Update networks
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()
            
            # Clear episode data
            self.episode_states.clear()
            self.episode_actions.clear()
            self.episode_rewards.clear()
            self.episode_log_probs.clear()
    ```

=== "Continuous Policy Gradient"
    ```python
    class ContinuousPolicyNetwork(nn.Module):
        """Policy network for continuous action spaces."""
        
        def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
            super(ContinuousPolicyNetwork, self).__init__()
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, action_size)
            self.fc4 = nn.Linear(hidden_size, action_size)  # For log_std
        
        def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            mean = torch.tanh(self.fc3(x))  # Bound actions to [-1, 1]
            log_std = self.fc4(x)
            return mean, log_std
    
    class ContinuousPolicyGradientAgent(REINFORCEAgent):
        """
        Policy gradient agent for continuous action spaces.
        """
        
        def __init__(self, state_size: int, action_size: int,
                     learning_rate: float = 0.001, discount_factor: float = 0.99):
            super().__init__(state_size, action_size, learning_rate, discount_factor)
            # Replace policy with continuous version
            self.policy = ContinuousPolicyNetwork(state_size, action_size)
            self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        def get_action(self, state: np.ndarray) -> tuple[np.ndarray, float]:
            """Sample continuous action from policy."""
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mean, log_std = self.policy(state_tensor)
            
            # Create normal distribution
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            
            # Sample action
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            return action.squeeze().numpy(), log_prob.item()
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/reinforcement_learning/policy_gradient.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/reinforcement_learning/policy_gradient.py)
    - **Tests**: [`tests/unit/reinforcement_learning/test_policy_gradient.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/reinforcement_learning/test_policy_gradient.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **REINFORCE** | O(episode_length × policy_params) | O(episode_length × state_size) | Monte Carlo, high variance |
    **REINFORCE + Baseline** | O(episode_length × (policy_params + baseline_params)) | O(episode_length × state_size) | Reduced variance, two networks |
    **Continuous PG** | O(episode_length × policy_params) | O(episode_length × state_size) | Handles continuous actions |

!!! warning "Performance Considerations"
    - **High variance** in gradient estimates affects training stability
    - **Monte Carlo updates** require complete episodes
    - **Baseline networks** add computational overhead but reduce variance
    - **Hyperparameter tuning** is crucial for convergence

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Continuous Control"
        - **Robotics**: Joint control and manipulation
        - **Autonomous Vehicles**: Steering and acceleration control
        - **Industrial Automation**: Process control and optimization
        - **Game AI**: Continuous movement and aiming

    !!! grid-item "Game AI & Entertainment"
        - **Video Games**: Character behavior and decision making
        - **Simulation Games**: Resource management and strategy
        - **Board Games**: Complex strategy optimization
        - **Virtual Environments**: NPC behavior and interaction

    !!! grid-item "Real-World Applications"
        - **Finance**: Portfolio optimization and trading strategies
        - **Healthcare**: Treatment scheduling and optimization
        - **Energy Management**: Consumption optimization
        - **Network Control**: Traffic routing and optimization

    !!! grid-item "Educational Value"
        - **Policy Optimization**: Understanding direct policy learning
        - **Variance Reduction**: Learning importance of baselines
        - **Continuous Actions**: Handling infinite action spaces
        - **Monte Carlo Methods**: Learning from complete episodes

!!! success "Educational Value"
    - **Direct Policy Learning**: Perfect example of policy optimization
    - **Variance Reduction**: Shows importance of baseline methods
    - **Continuous Control**: Demonstrates handling of continuous action spaces
    - **Monte Carlo Learning**: Illustrates learning from complete trajectories

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Papers"
        1. **Williams, R. J.** (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3-4), 229-256.
        2. **Sutton, R. S., et al.** (2000). Policy gradient methods for reinforcement learning with function approximation. *Advances in Neural Information Processing Systems*, 12.

    !!! grid-item "Policy Gradient Textbooks"
        3. **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction*. MIT Press.
        4. **Szepesvári, C.** (2010). *Algorithms for Reinforcement Learning*. Morgan & Claypool.

    !!! grid-item "Online Resources"
        5. [Policy Gradient Methods - Wikipedia](https://en.wikipedia.org/wiki/Policy_gradient_methods)
        6. [REINFORCE Algorithm Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
        7. [OpenAI Baselines](https://github.com/openai/baselines) - Policy gradient implementations

    !!! grid-item "Implementation & Practice"
        8. [PyTorch Documentation](https://pytorch.org/docs/)
        9. [OpenAI Gym](https://www.gymlibrary.dev/) - RL environments
        10. [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - RL implementations

!!! tip "Interactive Learning"
    Try implementing Policy Gradient methods yourself! Start with REINFORCE on simple environments like CartPole to understand the basics. Add a baseline network to see how it reduces variance and improves training stability. Experiment with continuous action spaces to understand how to handle infinite action sets. Compare with value-based methods to see the trade-offs between direct policy optimization and value function learning. This will give you deep insight into the fundamentals of policy-based reinforcement learning.
