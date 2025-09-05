---
algorithm_key: "feudal-networks"
tags: [hierarchical-rl, algorithms, feudal-networks, temporal-abstraction, subgoals, manager-worker]
title: "Feudal Networks (FuN)"
family: "hierarchical-rl"
---

# Feudal Networks (FuN)

{{ algorithm_card("feudal-networks") }}

!!! abstract "Overview"
    Feudal Networks (FuN) is a hierarchical reinforcement learning algorithm that implements a manager-worker architecture for temporal abstraction. The algorithm consists of two neural networks: a manager that operates at a high level and sets abstract goals, and a worker that operates at a low level and executes actions to achieve these goals.

    This hierarchical approach enables the agent to solve complex, long-horizon tasks by breaking them down into manageable subproblems. The manager learns to set useful goals, while the worker learns to achieve specific goals efficiently. Feudal Networks are particularly powerful in domains where tasks have natural hierarchical structure, such as robotics manipulation, navigation, and game playing.

## Mathematical Formulation

!!! math "Feudal Network Architecture"
    The Feudal Network consists of a manager policy $\pi_m$ and a worker policy $\pi_w$:
    
    $$\pi_m(g_t|s_t) = \text{softmax}(f_m(s_t))$$
    
    $$\pi_w(a_t|s_t, g_t) = \text{softmax}(f_w(s_t, g_t))$$
    
    Where:
    - $\pi_m$ is the manager policy that selects goals $g_t$
    - $\pi_w$ is the worker policy that executes actions $a_t$ given goal $g_t$
    - $f_m$ and $f_w$ are neural network functions
    - $s_t$ is the current state
    
    The hierarchical value function is:
    
    $$V_h(s_t) = \mathbb{E}_{g_t \sim \pi_m} \left[ V_w(s_t, g_t) \right]$$
    
    Where $V_w(s_t, g_t)$ is the worker's value function.

!!! success "Key Properties"
    - **Manager-Worker Architecture**: Clear separation of high-level planning and low-level execution
    - **Temporal Abstraction**: Manager operates over longer time horizons
    - **Goal-Based Learning**: Worker learns to achieve abstract goals
    - **Hierarchical Learning**: Both networks learn simultaneously
    - **Transfer Learning**: Worker skills can be reused across different goals

## Implementation Approaches

=== "Basic Feudal Networks (Recommended)"
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import numpy as np
    
    class ManagerNetwork(nn.Module):
        """Manager network that selects abstract goals."""
        
        def __init__(self, state_size: int, goal_size: int, hidden_size: int = 128):
            super(ManagerNetwork, self).__init__()
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, goal_size)
        
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            goal_probs = F.softmax(self.fc3(x), dim=-1)
            return goal_probs
    
    class WorkerNetwork(nn.Module):
        """Worker network that executes actions given goals."""
        
        def __init__(self, state_size: int, goal_size: int, action_size: int, hidden_size: int = 128):
            super(WorkerNetwork, self).__init__()
            self.fc1 = nn.Linear(state_size + goal_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, action_size)
        
        def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
            # Concatenate state and goal
            combined = torch.cat([state, goal], dim=-1)
            x = F.relu(self.fc1(combined))
            x = F.relu(self.fc2(x))
            action_probs = F.softmax(self.fc3(x), dim=-1)
            return action_probs
    
    class FeudalNetworkAgent:
        """
        Feudal Networks agent implementation.
        
        Args:
            state_size: Dimension of state space
            goal_size: Dimension of goal space
            action_size: Number of possible actions
            manager_lr: Learning rate for manager network (default: 0.001)
            worker_lr: Learning rate for worker network (default: 0.001)
            discount_factor: Discount factor gamma (default: 0.99)
            goal_horizon: Maximum steps to achieve goal (default: 50)
        """
        
        def __init__(self, state_size: int, goal_size: int, action_size: int,
                     manager_lr: float = 0.001, worker_lr: float = 0.001,
                     discount_factor: float = 0.99, goal_horizon: int = 50):
            
            self.state_size = state_size
            self.goal_size = goal_size
            self.action_size = action_size
            self.gamma = discount_factor
            self.goal_horizon = goal_horizon
            
            # Networks
            self.manager = ManagerNetwork(state_size, goal_size)
            self.worker = WorkerNetwork(state_size, goal_size, action_size)
            
            # Optimizers
            self.manager_optimizer = optim.Adam(self.manager.parameters(), lr=manager_lr)
            self.worker_optimizer = optim.Adam(self.worker.parameters(), lr=worker_lr)
            
            # Experience buffers
            self.manager_buffer = []
            self.worker_buffer = []
            
            # Current goal tracking
            self.current_goal = None
            self.goal_steps = 0
        
        def get_goal(self, state: np.ndarray) -> tuple[int, float]:
            """Select goal using manager network."""
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            goal_probs = self.manager(state_tensor)
            
            # Sample goal
            dist = torch.distributions.Categorical(goal_probs)
            goal = dist.sample()
            log_prob = dist.log_prob(goal)
            
            return goal.item(), log_prob.item()
        
        def get_action(self, state: np.ndarray, goal: int) -> tuple[int, float]:
            """Get action using worker network given goal."""
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            goal_tensor = torch.FloatTensor([goal]).unsqueeze(0)
            
            action_probs = self.worker(state_tensor, goal_tensor)
            
            # Sample action
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item()
        
        def step(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
            """Process one step and potentially update goal."""
            # Store worker transition
            if self.current_goal is not None:
                self.worker_buffer.append((state, self.current_goal, action, reward, next_state, done))
            
            # Check if goal should be updated
            self.goal_steps += 1
            if (self.goal_steps >= self.goal_horizon or 
                self.is_goal_achieved(state, next_state) or done):
                
                # Store manager transition
                if len(self.worker_buffer) > 0:
                    total_reward = sum(r for _, _, _, r, _, _ in self.worker_buffer)
                    self.manager_buffer.append((state, self.current_goal, total_reward, next_state, done))
                
                # Select new goal
                if not done:
                    new_goal, _ = self.get_goal(next_state)
                    self.current_goal = new_goal
                    self.goal_steps = 0
                
                # Clear worker buffer
                self.worker_buffer.clear()
        
        def is_goal_achieved(self, state: np.ndarray, next_state: np.ndarray) -> bool:
            """Check if current goal has been achieved."""
            # Simple distance-based goal achievement
            # In practice, this would be domain-specific
            return np.linalg.norm(next_state - state) < 0.1
        
        def update_networks(self) -> None:
            """Update both manager and worker networks."""
            self.update_worker_network()
            self.update_manager_network()
        
        def update_worker_network(self) -> None:
            """Update worker network using stored experience."""
            if len(self.worker_buffer) < 10:
                return
            
            # Sample batch from worker buffer
            batch = np.random.choice(len(self.worker_buffer), min(32, len(self.worker_buffer)), replace=False)
            
            states, goals, actions, rewards, next_states, dones = zip(*[self.worker_buffer[i] for i in batch])
            
            # Convert to tensors
            states = torch.FloatTensor(states)
            goals = torch.LongTensor(goals)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.BoolTensor(dones)
            
            # Compute advantages
            current_values = self.get_worker_value(states, goals)
            next_values = self.get_worker_value(next_states, goals)
            
            advantages = rewards + self.gamma * next_values * ~dones - current_values
            
            # Get current action probabilities
            action_probs = self.worker(states, F.one_hot(goals, self.goal_size).float())
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            
            # Policy gradient loss
            policy_loss = -(log_probs * advantages.detach()).mean()
            
            # Update worker network
            self.worker_optimizer.zero_grad()
            policy_loss.backward()
            self.worker_optimizer.step()
        
        def update_manager_network(self) -> None:
            """Update manager network using stored experience."""
            if len(self.manager_buffer) < 10:
                return
            
            # Sample batch from manager buffer
            batch = np.random.choice(len(self.manager_buffer), min(32, len(self.manager_buffer)), replace=False)
            
            states, goals, rewards, next_states, dones = zip(*[self.manager_buffer[i] for i in batch])
            
            # Convert to tensors
            states = torch.FloatTensor(states)
            goals = torch.LongTensor(goals)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.BoolTensor(dones)
            
            # Compute advantages for manager
            current_values = self.get_manager_value(states)
            next_values = self.get_manager_value(next_states)
            
            advantages = rewards + self.gamma * next_values * ~dones - current_values
            
            # Get current goal probabilities
            goal_probs = self.manager(states)
            dist = torch.distributions.Categorical(goal_probs)
            log_probs = dist.log_prob(goals)
            
            # Policy gradient loss
            policy_loss = -(log_probs * advantages.detach()).mean()
            
            # Update manager network
            self.manager_optimizer.zero_grad()
            policy_loss.backward()
            self.manager_optimizer.step()
        
        def get_worker_value(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
            """Get worker value estimates."""
            # Simplified value estimation
            return torch.zeros(len(states))
        
        def get_manager_value(self, states: torch.Tensor) -> torch.Tensor:
            """Get manager value estimates."""
            # Simplified value estimation
            return torch.zeros(len(states))
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/hierarchical_rl/feudal_networks.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/hierarchical_rl/feudal_networks.py)
    - **Tests**: [`tests/unit/hierarchical_rl/test_feudal_networks.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/hierarchical_rl/test_feudal_networks.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Basic FuN** | O(batch_size × (manager_params + worker_params)) | O(batch_size × (state_size + goal_size)) | Two-network architecture |
    **FuN + Experience Replay** | O(batch_size × (manager_params + worker_params)) | O(batch_size × (state_size + goal_size) + buffer_size) | Better sample efficiency |
    **FuN + Value Networks** | O(batch_size × (manager_params + worker_params + value_params)) | O(batch_size × (state_size + goal_size)) | Value function estimation |

!!! warning "Performance Considerations"
    - **Two networks** require careful coordination
    - **Goal horizon** affects exploration and learning efficiency
    - **Goal achievement detection** is crucial for proper hierarchy
    - **Manager updates** depend on worker performance

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Robotics & Control"
        - **Robot Manipulation**: Complex manipulation tasks with goals
        - **Autonomous Navigation**: Multi-level navigation planning
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
        - **Manager-Worker Architecture**: Understanding hierarchical control
        - **Goal-Based Learning**: Learning to set and achieve abstract goals
        - **Temporal Abstraction**: Understanding different time scales
        - **Transfer Learning**: Learning reusable worker skills

!!! success "Educational Value"
    - **Hierarchical Control**: Perfect example of manager-worker architecture
    - **Goal Setting**: Shows how to set abstract goals for workers
    - **Temporal Abstraction**: Demonstrates learning at different time scales
    - **Skill Reuse**: Illustrates how worker skills can be reused

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Papers"
        1. **Vezhnevets, A., et al.** (2017). FeUdal Networks for Hierarchical Reinforcement Learning. *ICML*, 70.

    !!! grid-item "Hierarchical RL Textbooks"
        2. **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction*. MIT Press.
        3. **Kaelbling, L. P., et al.** (1998). Hierarchical reinforcement learning with the MAXQ value function decomposition. *Journal of Artificial Intelligence Research*, 13.

    !!! grid-item "Online Resources"
        4. [Feudal Networks - Wikipedia](https://en.wikipedia.org/wiki/Feudal_networks)
        5. [FuN Implementation Guide](https://github.com/andrew-j-levy/Hierarchical-Actor-Critic-HAC-)
        6. [Hierarchical RL Tutorial](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

    !!! grid-item "Implementation & Practice"
        7. [PyTorch Documentation](https://pytorch.org/docs/)
        8. [OpenAI Gym](https://www.gymlibrary.dev/) - RL environments
        9. [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - RL implementations

!!! tip "Interactive Learning"
    Try implementing Feudal Networks yourself! Start with simple environments that have natural hierarchical structure, like navigation tasks with waypoints. Implement the basic manager-worker architecture first, then add experience replay for better sample efficiency. Experiment with different goal horizons and achievement detection methods to see their impact on learning. Compare with flat approaches to see the benefits of hierarchical decomposition. This will give you deep insight into the power of manager-worker architectures in reinforcement learning.

## Navigation

{{ nav_grid(current_algorithm="feudal-networks", current_family="hierarchical-rl", max_related=5) }}
