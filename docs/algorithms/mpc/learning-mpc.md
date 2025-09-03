---
tags: [mpc, algorithms, learning-mpc, adaptive-control, machine-learning, neural-networks, reinforcement-learning]
title: "Learning MPC"
family: "mpc"
complexity: "O(N³) for horizon length N, additional complexity for learning algorithms"
---

# Learning MPC

!!! info "Algorithm Family"
    **Family:** [MPC Algorithms](../../families/mpc.md)

!!! abstract "Overview"
    Learning MPC combines the predictive control framework with machine learning techniques to create adaptive controllers that can improve their performance over time. Unlike traditional MPC that relies on fixed, pre-specified models, Learning MPC continuously updates its system model, cost function, or control policy based on observed data and performance feedback.

    This approach is essential in applications where system dynamics are complex, time-varying, or poorly understood, such as autonomous vehicles, robotics, and adaptive manufacturing systems. Learning MPC can handle model uncertainties, adapt to changing operating conditions, and improve performance through experience, making it particularly valuable in dynamic and uncertain environments.

## Mathematical Formulation

!!! math "Learning MPC Framework"
    **1. Adaptive System Model:**
    
    The learning system model is described by:
    
    $$x(k+1) = f_\theta(x(k), u(k), d(k))$$
    $$y(k) = h_\theta(x(k), u(k))$$
    
    Where:
    - $x(k) \in \mathbb{R}^{n_x}$ is the state vector
    - $u(k) \in \mathbb{R}^{n_u}$ is the control input vector
    - $d(k) \in \mathbb{R}^{n_d}$ is the disturbance vector
    - $y(k) \in \mathbb{R}^{n_y}$ is the output vector
    - $\theta$ represents learnable parameters that are updated online
    
    **2. Learning MPC Optimization Problem:**
    
    At each time step $k$, solve:
    
    $$\min_{U_k} J_\theta(x(k), U_k) = \sum_{i=0}^{N_p-1} L_\theta(x(k+i|k), u(k+i)) + V_\theta(x(k+N_p|k))$$
    
    Subject to:
    - $x(k+i+1|k) = f_\theta(x(k+i|k), u(k+i), d(k+i))$
    - $y(k+i|k) = h_\theta(x(k+i|k), u(k+i))$
    - $g(x(k+i|k), u(k+i)) \leq 0$ (constraints)
    - $u_{min} \leq u(k+i) \leq u_{max}$
    - $x_{min} \leq x(k+i|k) \leq x_{max}$
    
    Where:
    - $U_k = [u(k), u(k+1), ..., u(k+N_c-1)]$ is the control sequence
    - $L_\theta(\cdot)$ is the learnable stage cost function
    - $V_\theta(\cdot)$ is the learnable terminal cost function
    
    **3. Online Learning Update:**
    
    The parameters $\theta$ are updated based on observed data:
    
    $$\theta(k+1) = \theta(k) + \alpha \nabla_\theta J_\theta(x(k), U_k^*)$$
    
    Where:
    - $\alpha$ is the learning rate
    - $\nabla_\theta J_\theta$ is the gradient with respect to parameters
    - $U_k^*$ is the optimal control sequence found at time $k$

!!! success "Key Properties"
    - **Adaptive Models**: Continuously updates system models based on data
    - **Performance Learning**: Improves control performance through experience
    - **Uncertainty Handling**: Adapts to unknown or changing system dynamics
    - **Data-Driven**: Leverages observed data for model improvement
    - **Real-Time Adaptation**: Updates parameters during operation

## Implementation Approaches

=== "Basic Learning MPC Controller (Recommended)"
    ```python
    import numpy as np
    from scipy.optimize import minimize
    from typing import Callable, Optional, Tuple, Dict, List
    import torch
    import torch.nn as nn
    
    class LearningMPCController:
        """
        Basic Learning MPC Controller implementation.
        
        Args:
            prediction_horizon: Number of prediction steps
            control_horizon: Number of control steps
            state_dim: Dimension of state vector
            input_dim: Dimension of input vector
            output_dim: Dimension of output vector
            learning_rate: Learning rate for parameter updates
            Q: State weight matrix
            R: Input weight matrix
        """
        
        def __init__(self, prediction_horizon: int, control_horizon: int,
                     state_dim: int, input_dim: int, output_dim: int,
                     learning_rate: float = 0.01,
                     Q: np.ndarray = None, R: np.ndarray = None):
            
            self.Np = prediction_horizon
            self.Nc = min(control_horizon, prediction_horizon)
            self.nx = state_dim
            self.nu = input_dim
            self.ny = output_dim
            self.learning_rate = learning_rate
            
            # Weighting matrices
            self.Q = Q if Q is not None else np.eye(state_dim)
            self.R = R if R is not None else np.eye(input_dim)
            
            # Learnable system model (neural network)
            self.system_model = self._create_system_model()
            
            # Learnable cost function (neural network)
            self.cost_function = self._create_cost_function()
            
            # Data buffer for learning
            self.data_buffer = []
            self.buffer_size = 1000
            
            # Constraints
            self.u_min = -np.inf * np.ones(input_dim)
            self.u_max = np.inf * np.ones(input_dim)
            self.x_min = -np.inf * np.ones(state_dim)
            self.x_max = np.inf * np.ones(state_dim)
            
            # History
            self.control_history = []
            self.state_history = []
            self.cost_history = []
            self.learning_history = []
        
        def _create_system_model(self) -> nn.Module:
            """
            Create a neural network for the system model.
            """
            model = nn.Sequential(
                nn.Linear(self.nx + self.nu, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, self.nx)
            )
            return model
        
        def _create_cost_function(self) -> nn.Module:
            """
            Create a neural network for the cost function.
            """
            model = nn.Sequential(
                nn.Linear(self.nx + self.nu, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            return model
        
        def set_constraints(self, u_min: np.ndarray = None, u_max: np.ndarray = None,
                          x_min: np.ndarray = None, x_max: np.ndarray = None) -> None:
            """
            Set input and state constraints.
            """
            if u_min is not None:
                self.u_min = np.array(u_min)
            if u_max is not None:
                self.u_max = np.array(u_max)
            if x_min is not None:
                self.x_min = np.array(x_min)
            if x_max is not None:
                self.x_max = np.array(x_max)
        
        def predict_next_state(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
            """
            Predict next state using the learned system model.
            """
            # Convert to torch tensors
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            control_tensor = torch.FloatTensor(control).unsqueeze(0)
            
            # Concatenate state and control
            input_tensor = torch.cat([state_tensor, control_tensor], dim=1)
            
            # Predict next state
            with torch.no_grad():
                next_state = self.system_model(input_tensor)
            
            return next_state.squeeze(0).numpy()
        
        def compute_cost(self, state: np.ndarray, control: np.ndarray) -> float:
            """
            Compute cost using the learned cost function.
            """
            # Convert to torch tensors
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            control_tensor = torch.FloatTensor(control).unsqueeze(0)
            
            # Concatenate state and control
            input_tensor = torch.cat([state_tensor, control_tensor], dim=1)
            
            # Compute cost
            with torch.no_grad():
                cost = self.cost_function(input_tensor)
            
            return cost.squeeze(0).item()
        
        def compute_control(self, current_state: np.ndarray, 
                          reference_trajectory: np.ndarray = None,
                          current_disturbance: np.ndarray = None) -> np.ndarray:
            """
            Compute optimal control input using Learning MPC.
            """
            # Initial guess for control sequence
            u0 = np.zeros(self.Nc * self.nu)
            
            # Bounds for optimization
            bounds = []
            for i in range(self.Nc):
                for j in range(self.nu):
                    bounds.append((self.u_min[j], self.u_max[j]))
            
            # Solve optimization problem
            result = minimize(
                fun=lambda u: self._learning_objective_function(u, current_state, 
                                                             reference_trajectory, current_disturbance),
                x0=u0,
                bounds=bounds,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            if not result.success:
                print(f"Learning MPC optimization failed: {result.message}")
                if self.control_history:
                    return self.control_history[-1]
                return np.zeros(self.nu)
            
            # Extract first control input
            optimal_control = result.x[:self.nu]
            
            # Store history
            self.control_history.append(optimal_control)
            self.state_history.append(current_state)
            self.cost_history.append(result.fun)
            
            return optimal_control
        
        def _learning_objective_function(self, u: np.ndarray, current_state: np.ndarray,
                                       reference: np.ndarray, disturbance: np.ndarray = None) -> float:
            """
            Compute Learning MPC objective function.
            """
            # Reshape control sequence
            U = u.reshape(self.Nc, self.nu)
            
            # Initialize cost
            total_cost = 0.0
            x = current_state.copy()
            
            # Prediction loop
            for i in range(self.Np):
                # Get control input
                if i < self.Nc:
                    u_i = U[i]
                else:
                    u_i = U[-1]
                
                # Predict next state using learned model
                x_next = self.predict_next_state(x, u_i)
                
                # Compute cost using learned cost function
                stage_cost = self.compute_cost(x, u_i)
                total_cost += stage_cost
                
                # Add reference tracking cost if provided
                if reference is not None and i < len(reference):
                    ref_i = reference[i]
                    tracking_error = x - ref_i
                    tracking_cost = tracking_error.T @ self.Q @ tracking_error
                    total_cost += tracking_cost
                
                # Add control penalty
                control_cost = u_i.T @ self.R @ u_i
                total_cost += control_cost
                
                # State constraints penalty
                if np.any(x < self.x_min) or np.any(x > self.x_max):
                    total_cost += 1e6
                
                # Update state
                x = x_next
            
            return total_cost
        
        def update_models(self, state_sequence: List[np.ndarray], 
                         control_sequence: List[np.ndarray],
                         next_state_sequence: List[np.ndarray],
                         cost_sequence: List[float]) -> None:
            """
            Update the learned models using observed data.
            """
            # Add data to buffer
            for i in range(len(state_sequence)):
                data_point = {
                    'state': state_sequence[i],
                    'control': control_sequence[i],
                    'next_state': next_state_sequence[i],
                    'cost': cost_sequence[i]
                }
                self.data_buffer.append(data_point)
            
            # Maintain buffer size
            if len(self.data_buffer) > self.buffer_size:
                self.data_buffer = self.data_buffer[-self.buffer_size:]
            
            # Update system model
            self._update_system_model()
            
            # Update cost function
            self._update_cost_function()
        
        def _update_system_model(self) -> None:
            """
            Update the system model using observed state transitions.
            """
            if len(self.data_buffer) < 10:  # Need minimum data
                return
            
            # Prepare training data
            states = []
            controls = []
            next_states = []
            
            for data_point in self.data_buffer:
                states.append(data_point['state'])
                controls.append(data_point['control'])
                next_states.append(data_point['next_state'])
            
            # Convert to torch tensors
            states_tensor = torch.FloatTensor(states)
            controls_tensor = torch.FloatTensor(controls)
            next_states_tensor = torch.FloatTensor(next_states)
            
            # Input to the model
            inputs = torch.cat([states_tensor, controls_tensor], dim=1)
            
            # Training
            optimizer = torch.optim.Adam(self.system_model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            
            for epoch in range(10):  # Few epochs for online learning
                optimizer.zero_grad()
                predictions = self.system_model(inputs)
                loss = criterion(predictions, next_states_tensor)
                loss.backward()
                optimizer.step()
        
        def _update_cost_function(self) -> None:
            """
            Update the cost function using observed costs.
            """
            if len(self.data_buffer) < 10:  # Need minimum data
                return
            
            # Prepare training data
            states = []
            controls = []
            costs = []
            
            for data_point in self.data_buffer:
                states.append(data_point['state'])
                controls.append(data_point['control'])
                costs.append(data_point['cost'])
            
            # Convert to torch tensors
            states_tensor = torch.FloatTensor(states)
            controls_tensor = torch.FloatTensor(controls)
            costs_tensor = torch.FloatTensor(costs).unsqueeze(1)
            
            # Input to the model
            inputs = torch.cat([states_tensor, controls_tensor], dim=1)
            
            # Training
            optimizer = torch.optim.Adam(self.cost_function.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            
            for epoch in range(10):  # Few epochs for online learning
                optimizer.zero_grad()
                predictions = self.cost_function(inputs)
                loss = criterion(predictions, costs_tensor)
                loss.backward()
                optimizer.step()
        
        def get_control_history(self) -> np.ndarray:
            """Get control input history."""
            return np.array(self.control_history) if self.control_history else np.array([])
        
        def get_state_history(self) -> np.ndarray:
            """Get state history."""
            return np.array(self.state_history) if self.state_history else np.array([])
        
        def get_cost_history(self) -> np.ndarray:
            """Get cost history."""
            return np.array(self.cost_history) if self.cost_history else np.array([])
        
        def get_learning_history(self) -> list:
            """Get learning update history."""
            return self.learning_history
        
        def reset(self) -> None:
            """Reset controller state."""
            self.control_history.clear()
            self.state_history.clear()
            self.cost_history.clear()
            self.learning_history.clear()
    ```

=== "Reinforcement Learning MPC (Advanced)"
    ```python
    class RLMPCController(LearningMPCController):
        """
        Learning MPC using reinforcement learning techniques.
        """
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            
            # RL parameters
            self.exploration_rate = 0.1
            self.discount_factor = 0.99
            self.experience_buffer = []
            
            # Policy network
            self.policy_network = self._create_policy_network()
        
        def _create_policy_network(self) -> nn.Module:
            """
            Create a policy network for action selection.
            """
            model = nn.Sequential(
                nn.Linear(self.nx, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, self.nu),
                nn.Tanh()  # Output in [-1, 1]
            )
            return model
        
        def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
            """
            Select action using the policy network.
            """
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action = self.policy_network(state_tensor)
            
            action = action.squeeze(0).numpy()
            
            # Add exploration noise during training
            if training and np.random.random() < self.exploration_rate:
                noise = np.random.normal(0, 0.1, action.shape)
                action = np.clip(action + noise, -1, 1)
            
            # Scale action to control bounds
            action = self._scale_action(action)
            
            return action
        
        def _scale_action(self, action: np.ndarray) -> np.ndarray:
            """
            Scale action from [-1, 1] to control bounds.
            """
            action_range = self.u_max - self.u_min
            action_center = (self.u_max + self.u_min) / 2
            return action_center + action * action_range / 2
        
        def update_policy(self, states: List[np.ndarray], actions: List[np.ndarray],
                         rewards: List[float], next_states: List[np.ndarray]) -> None:
            """
            Update the policy network using RL algorithms.
            """
            # This is a simplified policy update
            # In practice, you would implement specific RL algorithms like DDPG, SAC, etc.
            
            if len(states) < 10:
                return
            
            # Convert to tensors
            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.FloatTensor(actions)
            rewards_tensor = torch.FloatTensor(rewards)
            next_states_tensor = torch.FloatTensor(next_states)
            
            # Simple policy gradient update
            optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
            
            # Compute policy loss (simplified)
            current_actions = self.policy_network(states_tensor)
            action_loss = torch.mean((current_actions - actions_tensor) ** 2)
            
            # Add reward-based loss
            reward_loss = -torch.mean(rewards_tensor)
            
            total_loss = action_loss + 0.1 * reward_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    ```

=== "Adaptive Learning MPC"
    ```python
    class AdaptiveLearningMPC(LearningMPCController):
        """
        Learning MPC with adaptive learning rates and model complexity.
        """
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            
            # Adaptive parameters
            self.adaptive_learning_rate = True
            self.model_complexity_adaptation = True
            self.performance_threshold = 0.1
            
            # Performance tracking
            self.performance_history = []
            self.learning_rate_history = []
        
        def _update_system_model(self) -> None:
            """
            Update system model with adaptive learning rate.
            """
            if len(self.data_buffer) < 10:
                return
            
            # Compute current performance
            current_performance = self._compute_performance()
            self.performance_history.append(current_performance)
            
            # Adapt learning rate based on performance
            if self.adaptive_learning_rate and len(self.performance_history) > 1:
                performance_change = current_performance - self.performance_history[-2]
                
                if performance_change < -self.performance_threshold:
                    # Performance degraded, reduce learning rate
                    self.learning_rate *= 0.9
                elif performance_change > self.performance_threshold:
                    # Performance improved, increase learning rate
                    self.learning_rate *= 1.1
                
                # Clamp learning rate
                self.learning_rate = np.clip(self.learning_rate, 1e-6, 0.1)
            
            self.learning_rate_history.append(self.learning_rate)
            
            # Call parent update method
            super()._update_system_model()
        
        def _compute_performance(self) -> float:
            """
            Compute current performance metric.
            """
            if len(self.data_buffer) < 10:
                return float('inf')
            
            # Simple performance metric: average prediction error
            total_error = 0.0
            count = 0
            
            for data_point in self.data_buffer[-10:]:  # Last 10 data points
                predicted_next_state = self.predict_next_state(
                    data_point['state'], data_point['control']
                )
                actual_next_state = data_point['next_state']
                error = np.linalg.norm(predicted_next_state - actual_next_state)
                total_error += error
                count += 1
            
            return total_error / count if count > 0 else float('inf')
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/mpc/learning_mpc.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/mpc/learning_mpc.py)
    - **Tests**: [`tests/unit/mpc/test_learning_mpc.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/mpc/test_learning_mpc.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Basic Learning MPC** | O(N³) per time step | O(N² + M) | M neural network parameters |
    **RL MPC** | O(N³) per time step | O(N² + M + E) | E experience buffer size |
    **Adaptive Learning MPC** | O(N³) per time step | O(N² + M + P) | P performance history |

!!! warning "Performance Considerations"
    - **Neural network training** adds computational overhead
    - **Learning rate tuning** affects convergence speed
    - **Data buffer size** impacts memory usage and learning quality
    - **Model complexity** balances expressiveness and training time

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Autonomous Vehicles"
        - **Self-Driving Cars**: Adapt to changing road conditions
        - **UAV Control**: Learn flight dynamics and wind patterns
        - **Marine Vehicles**: Adapt to changing sea conditions
        - **Spacecraft Control**: Learn orbital dynamics

    !!! grid-item "Robotics & Automation"
        - **Robot Manipulators**: Learn contact dynamics and friction
        - **Mobile Robots**: Adapt to changing environments
        - **Human-Robot Interaction**: Learn human behavior patterns
        - **Industrial Automation**: Adapt to process variations

    !!! grid-item "Manufacturing & Process Control"
        - **Chemical Processes**: Learn reaction kinetics
        - **Manufacturing Systems**: Adapt to tool wear and variations
        - **Quality Control**: Learn defect patterns
        - **Supply Chain**: Adapt to demand variations

    !!! grid-item "Energy & Smart Grids"
        - **Power Generation**: Learn demand patterns
        - **Renewable Energy**: Adapt to weather variations
        - **Energy Storage**: Learn degradation patterns
        - **Grid Management**: Adapt to load variations

!!! success "Educational Value"
    - **Machine Learning Integration**: Understanding how to combine ML with control
    - **Adaptive Systems**: Learning how systems can improve over time
    - **Data-Driven Control**: Understanding the role of data in control design
    - **Online Learning**: Learning how to update models during operation

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Textbooks"
        1. **Rawlings, J. B., et al.** (2017). *Model Predictive Control: Theory, Computation, and Design*. Nob Hill.
        2. **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction*. MIT Press.

    !!! grid-item "Historical & Cultural"
        3. **Kocijan, J., et al.** (2005). Predictive control of a gas-liquid separation plant based on a Gaussian process model. *Computers & Chemical Engineering*, 29(4).
        4. **Hewing, L., et al.** (2020). Learning-based model predictive control: Toward safe learning in control. *Annual Review of Control, Robotics, and Autonomous Systems*.

    !!! grid-item "Online Resources"
        5. [Learning MPC - Wikipedia](https://en.wikipedia.org/wiki/Model_predictive_control)
        6. [Neural Network Control](https://pytorch.org/tutorials/)
        7. [Reinforcement Learning](https://www.youtube.com/watch?v=example)

    !!! grid-item "Implementation & Practice"
        8. [Python Learning MPC](https://pypi.org/project/learning-mpc/)
        9. [PyTorch](https://pytorch.org/)
        10. [OpenAI Gym](https://gym.openai.com/)

!!! tip "Interactive Learning"
    Try implementing Learning MPC yourself! Start with a simple system and implement a basic neural network model, then add the learning update loop. Experiment with different neural network architectures and see how they affect prediction accuracy. Try implementing reinforcement learning techniques and see how they improve control performance over time. Add adaptive learning rates and see how they affect convergence. This will give you deep insight into how to design controllers that can learn and adapt to changing conditions while maintaining stability and performance.
