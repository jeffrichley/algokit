---
tags: [hierarchical-rl, algorithms, hierarchical-task-networks, task-decomposition, planning, temporal-abstraction]
title: "Hierarchical Task Networks"
family: "hierarchical-rl"
complexity: "O(task_depth × action_space)"
---

# Hierarchical Task Networks

!!! info "Algorithm Family"
    **Family:** [Hierarchical Reinforcement Learning](../../families/hierarchical-rl.md)

!!! abstract "Overview"
    Hierarchical Task Networks (HTNs) represent a powerful approach to reinforcement learning that decomposes complex tasks into hierarchical structures of subtasks. The algorithm learns to plan and execute tasks at multiple levels of abstraction, where high-level tasks are broken down into simpler subtasks that can be learned and executed independently.

    This hierarchical approach enables agents to solve complex, long-horizon problems by leveraging task decomposition and temporal abstraction. HTNs are particularly effective in domains where tasks have natural hierarchical structure, such as robotics manipulation, autonomous navigation, and complex game playing scenarios.

## Mathematical Formulation

!!! math "Task Decomposition Framework"
    A hierarchical task network can be represented as:
    
    $$T = \{T_1, T_2, \ldots, T_n\}$$
    
    Where each task $T_i$ can be decomposed into subtasks:
    
    $$T_i = \{t_{i1}, t_{i2}, \ldots, t_{im}\}$$
    
    The value function for a composite task is:
    
    $$V(T_i) = \max_{\pi} \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{H} \gamma^t r_t \right]$$
    
    Where $\pi$ is the policy for executing the task decomposition, and $H$ is the horizon.

!!! success "Key Properties"
    - **Task Decomposition**: Complex tasks broken into manageable subtasks
    - **Temporal Abstraction**: Different levels operate at different time scales
    - **Modular Learning**: Subtasks can be learned independently
    - **Reusability**: Learned subtasks can be applied to different composite tasks
    - **Planning**: Hierarchical planning for task execution

## Implementation Approaches

=== "Basic Hierarchical Task Networks (Recommended)"
    ```python
    import numpy as np
    from typing import Dict, List, Tuple, Any
    from dataclasses import dataclass
    
    @dataclass
    class Task:
        """Represents a task in the hierarchy."""
        name: str
        subtasks: List['Task']
        policy: Any = None
        value: float = 0.0
        
        def is_primitive(self) -> bool:
            """Check if this is a primitive task (no subtasks)."""
            return len(self.subtasks) == 0
    
    class HierarchicalTaskNetworkAgent:
        """
        Hierarchical Task Networks agent implementation.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            learning_rate: Learning rate for policies (default: 0.001)
            discount_factor: Discount factor gamma (default: 0.99)
        """
        
        def __init__(self, state_size: int, action_size: int,
                     learning_rate: float = 0.001, discount_factor: float = 0.99):
            
            self.state_size = state_size
            self.action_size = action_size
            self.alpha = learning_rate
            self.gamma = discount_factor
            
            # Task hierarchy
            self.task_hierarchy = self.build_task_hierarchy()
            
            # Policies for each task
            self.task_policies = {}
            
            # Experience buffers for each task
            self.task_buffers = {}
        
        def build_task_hierarchy(self) -> Task:
            """Build the hierarchical task structure."""
            # Example: Navigation task with subgoals
            navigate = Task("navigate", [
                Task("move_to_waypoint", []),
                Task("avoid_obstacles", []),
                Task("reach_destination", [])
            ])
            
            return navigate
        
        def get_action(self, state: np.ndarray, current_task: Task) -> int:
            """Get action for current task."""
            if current_task.is_primitive():
                # Execute primitive action
                return self.execute_primitive_task(state, current_task)
            else:
                # Select next subtask
                return self.select_subtask(state, current_task)
        
        def execute_primitive_task(self, state: np.ndarray, task: Task) -> int:
            """Execute a primitive task."""
            if task.name not in self.task_policies:
                # Initialize random policy for new task
                self.task_policies[task.name] = np.random.rand(self.state_size, self.action_size)
                self.task_buffers[task.name] = []
            
            # Epsilon-greedy action selection
            if np.random.random() < 0.1:
                return np.random.randint(self.action_size)
            else:
                return np.argmax(self.task_policies[task.name][state])
        
        def select_subtask(self, state: np.ndarray, task: Task) -> int:
            """Select next subtask to execute."""
            # Simple heuristic: select subtask with highest value
            subtask_values = [self.get_task_value(subtask) for subtask in task.subtasks]
            return np.argmax(subtask_values)
        
        def get_task_value(self, task: Task) -> float:
            """Get the value of a task."""
            if task.is_primitive():
                return task.value
            else:
                # Recursively compute value from subtasks
                subtask_values = [self.get_task_value(subtask) for subtask in task.subtasks]
                return max(subtask_values) if subtask_values else 0.0
        
        def update_task_policy(self, task_name: str, state: int, action: int, 
                             reward: float, next_state: int) -> None:
            """Update policy for a specific task."""
            if task_name not in self.task_policies:
                return
            
            current_q = self.task_policies[task_name][state, action]
            
            # Q-Learning update
            max_next_q = np.max(self.task_policies[task_name][next_state])
            new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
            self.task_policies[task_name][state, action] = new_q
        
        def store_experience(self, task_name: str, state: int, action: int, 
                           reward: float, next_state: int) -> None:
            """Store experience for a specific task."""
            if task_name not in self.task_buffers:
                self.task_buffers[task_name] = []
            
            self.task_buffers[task_name].append((state, action, reward, next_state))
            
            # Update policy when buffer is full
            if len(self.task_buffers[task_name]) >= 10:
                self.update_from_buffer(task_name)
        
        def update_from_buffer(self, task_name: str) -> None:
            """Update task policy from experience buffer."""
            buffer = self.task_buffers[task_name]
            
            for state, action, reward, next_state in buffer:
                self.update_task_policy(task_name, state, action, reward, next_state)
            
            # Clear buffer
            self.task_buffers[task_name].clear()
    ```

=== "HTN with Dynamic Task Decomposition (Advanced)"
    ```python
    class DynamicHTNAgent(HierarchicalTaskNetworkAgent):
        """
        HTN agent with dynamic task decomposition.
        """
        
        def __init__(self, state_size: int, action_size: int, **kwargs):
            super().__init__(state_size, action_size, **kwargs)
            
            # Task decomposition network
            self.decomposition_network = np.random.randn(state_size, 10) * 0.01
        
        def decompose_task_dynamically(self, state: np.ndarray, task: Task) -> List[Task]:
            """Dynamically decompose tasks based on current state."""
            if task.is_primitive():
                return [task]
            
            # Use decomposition network to determine subtask priorities
            state_features = state / np.linalg.norm(state)
            decomposition_scores = np.dot(state_features, self.decomposition_network)
            
            # Sort subtasks by decomposition scores
            subtask_scores = list(zip(task.subtasks, decomposition_scores[:len(task.subtasks)]))
            subtask_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [subtask for subtask, _ in subtask_scores]
    ```

=== "HTN with Transfer Learning"
    ```python
    class TransferHTNAgent(HierarchicalTaskNetworkAgent):
        """
        HTN agent with transfer learning capabilities.
        """
        
        def __init__(self, state_size: int, action_size: int, **kwargs):
            super().__init__(state_size, action_size, **kwargs)
            
            # Transfer learning mappings
            self.transfer_mappings = {}
        
        def transfer_task_policy(self, source_task: str, target_task: str) -> None:
            """Transfer policy from source task to target task."""
            if source_task in self.task_policies and target_task not in self.task_policies:
                # Copy policy with some noise for exploration
                source_policy = self.task_policies[source_task]
                noise = np.random.normal(0, 0.1, source_policy.shape)
                self.task_policies[target_task] = source_policy + noise
                self.task_buffers[target_task] = []
        
        def adapt_transferred_policy(self, task_name: str, adaptation_rate: float = 0.1) -> None:
            """Adapt transferred policy to new task."""
            if task_name in self.task_policies:
                # Add adaptation noise
                noise = np.random.normal(0, adaptation_rate, self.task_policies[task_name].shape)
                self.task_policies[task_name] += noise
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/hierarchical_rl/hierarchical_task_networks.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/hierarchical_rl/hierarchical_task_networks.py)
    - **Tests**: [`tests/unit/hierarchical_rl/test_hierarchical_task_networks.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/hierarchical_rl/test_hierarchical_task_networks.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Basic HTN** | O(task_depth × action_space) | O(num_tasks × state_size × action_size) | Static task hierarchy |
    **Dynamic HTN** | O(task_depth × action_space + decomposition_cost) | O(num_tasks × state_size × action_size) | Dynamic decomposition |
    **Transfer HTN** | O(task_depth × action_space) | O(num_tasks × state_size × action_size) | Policy transfer capabilities |

!!! warning "Performance Considerations"
    - **Task hierarchy design** is crucial for performance
    - **Dynamic decomposition** adds computational overhead
    - **Policy transfer** requires careful adaptation
    - **Task coordination** affects overall performance

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Robotics & Control"
        - **Robot Manipulation**: Complex manipulation task decomposition
        - **Autonomous Navigation**: Multi-level navigation planning
        - **Industrial Automation**: Process optimization with subtasks
        - **Swarm Robotics**: Coordinated multi-agent task execution

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
        - **Task Decomposition**: Understanding how to break complex problems
        - **Hierarchical Planning**: Learning multi-level planning strategies
        - **Modular Learning**: Understanding independent subtask learning
        - **Transfer Learning**: Learning to reuse knowledge across tasks

!!! success "Educational Value"
    - **Task Decomposition**: Perfect example of breaking complex problems into parts
    - **Hierarchical Planning**: Shows how to plan at multiple levels
    - **Modular Learning**: Demonstrates independent learning of components
    - **Transfer Learning**: Illustrates knowledge reuse across domains

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Papers"
        1. **Erol, K., et al.** (1994). HTN planning: Complexity and expressivity. *AAAI*, 94.
        2. **Nau, D., et al.** (2003). SHOP2: An HTN planning system. *Journal of Artificial Intelligence Research*, 20.

    !!! grid-item "Hierarchical RL Textbooks"
        3. **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction*. MIT Press.
        4. **Kaelbling, L. P., et al.** (1998). Hierarchical reinforcement learning with the MAXQ value function decomposition. *Journal of Artificial Intelligence Research*, 13.

    !!! grid-item "Online Resources"
        5. [Hierarchical Task Networks - Wikipedia](https://en.wikipedia.org/wiki/Hierarchical_task_network)
        6. [HTN Planning Tutorial](https://www.cs.umd.edu/~nau/planning/)
        7. [Hierarchical RL Tutorial](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

    !!! grid-item "Implementation & Practice"
        8. [NumPy Documentation](https://numpy.org/doc/)
        9. [OpenAI Gym](https://www.gymlibrary.dev/) - RL environments
        10. [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - RL implementations

!!! tip "Interactive Learning"
    Try implementing Hierarchical Task Networks yourself! Start with simple task hierarchies like navigation or manipulation tasks. Implement the basic task decomposition first, then add dynamic decomposition capabilities. Experiment with policy transfer between similar tasks to see how knowledge can be reused. Compare with flat approaches to see the benefits of hierarchical task decomposition. This will give you deep insight into the power of structured task planning in reinforcement learning.
