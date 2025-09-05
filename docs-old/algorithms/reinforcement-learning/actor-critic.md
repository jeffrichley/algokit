---
algorithm_key: "actor-critic"
tags: [reinforcement-learning, algorithms, actor-critic, policy-gradient, value-based, advantage-estimation]
title: "Actor-Critic Methods"
family: "reinforcement-learning"
---

# Actor-Critic Methods

{{ algorithm_card("actor-critic") }}

!!! abstract "Overview"
    Actor-Critic methods represent a powerful hybrid approach that combines the benefits of both policy gradient methods (actor) and value function methods (critic). The actor learns a parameterized policy that maps states to actions, while the critic learns a value function that estimates the expected return from each state.

    This architecture provides several advantages: it reduces variance compared to pure policy gradient methods, enables continuous action spaces, and often achieves better sample efficiency. Actor-Critic methods have become fundamental building blocks for many modern reinforcement learning algorithms, including A3C, A2C, and SAC.

## Mathematical Formulation

!!! math "Actor-Critic Policy Gradient"
    The actor-critic policy gradient is given by:
    
    $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) A^\pi(s_t, a_t) \right]$$
    
    Where the advantage function $A^\pi(s_t, a_t)$ is estimated by the critic:
    
    $$A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t) \approx r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)$$
    
    The actor network updates the policy parameters $\theta$ using this gradient, while the critic network updates the value function parameters $\phi$ to minimize:
    
    $$L(\phi) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \left( r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t) \right)^2 \right]$$

!!! success "Key Properties"
    - **Hybrid Approach**: Combines policy gradient and value function methods
    - **Reduced Variance**: Value function baseline reduces policy gradient variance
    - **Continuous Actions**: Naturally handles continuous action spaces
    - **Sample Efficiency**: Often more sample efficient than pure policy gradients

## Implementation Approaches

=== "Basic Actor-Critic (Recommended)"
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import numpy as np
    
    class ActorNetwork(nn.Module):
        """Actor network that outputs action probabilities."""
        
        def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
            super(ActorNetwork, self).__init__()
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, action_size)
        
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            action_probs = F.softmax(self.fc3(x), dim=-1)
            return action_probs
    
    class CriticNetwork(nn.Module):
        """Critic network that estimates state values."""
        
        def __init__(self, state_size: int, hidden_size: int = 128):
            super(CriticNetwork, self).__init__()
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, 1)
        
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            value = self.fc3(x)
            return value
    
    class ActorCriticAgent:
        """
        Basic Actor-Critic agent implementation.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            actor_lr: Learning rate for actor network (default: 0.001)
            critic_lr: Learning rate for critic network (default: 0.001)
            discount_factor: Discount factor gamma (default: 0.99)
        """
        
        def __init__(self, state_size: int, action_size: int,
                     actor_lr: float = 0.001, critic_lr: float = 0.001,
                     discount_factor: float = 0.99):
            
            self.state_size = state_size
            self.action_size = action_size
            self.gamma = discount_factor
            
            # Networks
            self.actor = ActorNetwork(state_size, action_size)
            self.critic = CriticNetwork(state_size)
            
            # Optimizers
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        def get_action(self, state: np.ndarray) -> tuple[int, float]:
            """Sample action from policy and return action with log probability."""
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = self.actor(state_tensor)
            
            # Sample action from distribution
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            return action.item(), log_prob.item()
        
        def update(self, states: list, actions: list, rewards: list, 
                  log_probs: list, next_states: list, dones: list) -> None:
            """Update actor and critic networks."""
            # Convert to tensors
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            log_probs = torch.stack(log_probs)
            next_states = torch.FloatTensor(next_states)
            dones = torch.BoolTensor(dones)
            
            # Calculate advantages
            current_values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            
            # TD(0) advantage estimation
            advantages = rewards + self.gamma * next_values * ~dones - current_values
            
            # Actor loss (policy gradient)
            actor_loss = -(log_probs * advantages.detach()).mean()
            
            # Critic loss (value function regression)
            critic_loss = F.mse_loss(current_values, rewards + self.gamma * next_values * ~dones)
            
            # Update networks
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
    ```

=== "Advantage Actor-Critic (A2C)"
    ```python
    class A2CAgent(ActorCriticAgent):
        """
        Advantage Actor-Critic (A2C) agent with n-step returns.
        """
        
        def __init__(self, state_size: int, action_size: int,
                     actor_lr: float = 0.001, critic_lr: float = 0.001,
                     discount_factor: float = 0.99, n_steps: int = 5):
            super().__init__(state_size, action_size, actor_lr, critic_lr, discount_factor)
            self.n_steps = n_steps
            self.buffer = []
        
        def store_transition(self, state: np.ndarray, action: int, reward: float,
                           next_state: np.ndarray, done: bool, log_prob: float) -> None:
            """Store transition in buffer."""
            self.buffer.append((state, action, reward, next_state, done, log_prob))
            
            # Update when buffer is full or episode ends
            if len(self.buffer) >= self.n_steps or done:
                self.update_from_buffer()
                self.buffer.clear()
        
        def update_from_buffer(self) -> None:
            """Update networks using n-step returns from buffer."""
            if len(self.buffer) == 0:
                return
            
            # Calculate n-step returns
            states, actions, rewards, next_states, dones, log_probs = zip(*self.buffer)
            
            # Bootstrap final value if episode didn't end
            if not dones[-1]:
                final_state = torch.FloatTensor(next_states[-1]).unsqueeze(0)
                final_value = self.critic(final_state).item()
            else:
                final_value = 0
            
            # Calculate n-step returns
            returns = []
            current_return = final_value
            for reward, done in zip(reversed(rewards), reversed(dones)):
                if done:
                    current_return = 0
                current_return = reward + self.gamma * current_return
                returns.insert(0, current_return)
            
            # Convert to tensors
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            returns = torch.FloatTensor(returns)
            log_probs = torch.stack(log_probs)
            
            # Calculate advantages
            current_values = self.critic(states).squeeze()
            advantages = returns - current_values
            
            # Actor loss
            actor_loss = -(log_probs * advantages.detach()).mean()
            
            # Critic loss
            critic_loss = F.mse_loss(current_values, returns)
            
            # Update networks
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
    ```

=== "Continuous Actor-Critic"
    ```python
    class ContinuousActorNetwork(nn.Module):
        """Actor network for continuous action spaces."""
        
        def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
            super(ContinuousActorNetwork, self).__init__()
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
    
    class ContinuousActorCriticAgent(ActorCriticAgent):
        """
        Actor-Critic agent for continuous action spaces.
        """
        
        def __init__(self, state_size: int, action_size: int,
                     actor_lr: float = 0.001, critic_lr: float = 0.001,
                     discount_factor: float = 0.99):
            super().__init__(state_size, action_size, actor_lr, critic_lr, discount_factor)
            # Replace actor with continuous version
            self.actor = ContinuousActorNetwork(state_size, action_size)
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        def get_action(self, state: np.ndarray) -> tuple[np.ndarray, float]:
            """Sample continuous action from policy."""
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mean, log_std = self.actor(state_tensor)
            
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

    - **Main Implementation**: [`src/algokit/reinforcement_learning/actor_critic.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/reinforcement_learning/actor_critic.py)
    - **Tests**: [`tests/unit/reinforcement_learning/test_actor_critic.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/reinforcement_learning/test_actor_critic.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Basic Actor-Critic** | O(batch_size × network_params) | O(network_params) | Baseline hybrid approach |
    **A2C** | O(n_steps × network_params) | O(n_steps × state_size) | N-step returns, better stability |
    **Continuous AC** | O(batch_size × network_params) | O(network_params) | Handles continuous actions |

!!! warning "Performance Considerations"
    - **Two networks** require more computational resources
    - **Advantage estimation** affects training stability
    - **Hyperparameter tuning** is crucial for both networks
    - **Value function accuracy** directly impacts policy learning

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Continuous Control"
        - **Robotics**: Joint control and manipulation
        - **Autonomous Vehicles**: Steering and acceleration control
        - **Industrial Automation**: Process control and optimization
        - **Game AI**: Continuous movement and aiming

    !!! grid-item "Game AI & Entertainment"
        - **Video Games**: Character movement and decision making
        - **Simulation Games**: Resource management and strategy
        - **Board Games**: Complex strategy optimization
        - **Virtual Environments**: NPC behavior and interaction

    !!! grid-item "Real-World Applications"
        - **Finance**: Portfolio optimization and trading
        - **Healthcare**: Treatment scheduling and optimization
        - **Energy Management**: Consumption optimization
        - **Network Control**: Traffic routing and optimization

    !!! grid-item "Educational Value"
        - **Hybrid Methods**: Understanding policy and value function combination
        - **Advantage Estimation**: Learning importance of baseline reduction
        - **Continuous Actions**: Handling infinite action spaces
        - **Training Stability**: Balancing actor and critic learning

!!! success "Educational Value"
    - **Hybrid Learning**: Perfect example of combining different RL approaches
    - **Advantage Functions**: Shows importance of reducing variance in policy gradients
    - **Continuous Control**: Demonstrates handling of continuous action spaces
    - **Architecture Design**: Illustrates benefits of specialized networks

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Papers"
        1. **Sutton, R. S., et al.** (2000). Policy gradient methods for reinforcement learning with function approximation. *Advances in Neural Information Processing Systems*, 12.
        2. **Mnih, V., et al.** (2016). Asynchronous methods for deep reinforcement learning. *ICML*, 48.

    !!! grid-item "Actor-Critic Textbooks"
        3. **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction*. MIT Press.
        4. **Szepesvári, C.** (2010). *Algorithms for Reinforcement Learning*. Morgan & Claypool.

    !!! grid-item "Online Resources"
        5. [Actor-Critic Methods - Wikipedia](https://en.wikipedia.org/wiki/Actor%E2%80%93critic_methods)
        6. [PyTorch Actor-Critic Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
        7. [OpenAI Baselines](https://github.com/openai/baselines) - A2C implementation

    !!! grid-item "Implementation & Practice"
        8. [PyTorch Documentation](https://pytorch.org/docs/)
        9. [OpenAI Gym](https://www.gymlibrary.dev/) - RL environments
        10. [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - RL implementations

!!! tip "Interactive Learning"
    Try implementing Actor-Critic methods yourself! Start with discrete action spaces to understand the basics, then move to continuous control problems like CartPole or LunarLander. Implement different advantage estimation methods (TD(0), n-step returns) to see their impact on training stability. Experiment with continuous action spaces to understand how to handle infinite action sets. This will give you deep insight into the power of hybrid reinforcement learning approaches.

## Navigation

{{ nav_grid(current_algorithm="actor-critic", current_family="reinforcement-learning", max_related=5) }}
