---
tags: [mpc, algorithms, economic-mpc, economic-optimization, cost-minimization, profit-maximization, operational-economics]
title: "Economic MPC"
family: "mpc"
complexity: "O(N³) for horizon length N, additional complexity for economic models"
---

# Economic MPC

!!! info "Algorithm Family"
    **Family:** [MPC Algorithms](../../families/mpc.md)

!!! abstract "Overview"
    Economic MPC extends the traditional tracking-based MPC framework to directly optimize economic objectives such as cost minimization, profit maximization, or resource efficiency. Unlike standard MPC that focuses on setpoint tracking and control effort minimization, Economic MPC incorporates economic models and objectives into the control design, making it particularly valuable for industrial processes where operational economics are critical.

    This approach is essential in applications such as chemical processes, power generation, manufacturing systems, and supply chain management where the primary goal is to maximize economic performance rather than just maintain desired setpoints. Economic MPC can handle complex economic models, time-varying prices, and operational constraints while ensuring system stability and performance.

## Mathematical Formulation

!!! math "Economic MPC Framework"
    **1. Economic System Model:**
    
    The system with economic variables is described by:
    
    $$x(k+1) = f(x(k), u(k), d(k), p(k))$$
    $$y(k) = h(x(k), u(k), p(k))$$
    $$c(k) = g_e(x(k), u(k), p(k))$$
    
    Where:
    - $x(k) \in \mathbb{R}^{n_x}$ is the state vector
    - $u(k) \in \mathbb{R}^{n_u}$ is the control input vector
    - $d(k) \in \mathbb{R}^{n_d}$ is the disturbance vector
    - $p(k) \in \mathbb{R}^{n_p}$ is the economic parameter vector (prices, costs)
    - $c(k) \in \mathbb{R}^{n_c}$ is the economic cost/profit vector
    - $f(\cdot)$, $h(\cdot)$, and $g_e(\cdot)$ are system functions
    
    **2. Economic Optimization Problem:**
    
    At each time step $k$, solve:
    
    $$\min_{U_k} J_e(x(k), U_k, P_k) = \sum_{i=0}^{N_p-1} L_e(x(k+i|k), u(k+i), p(k+i))$$
    
    Subject to:
    - $x(k+i+1|k) = f(x(k+i|k), u(k+i), d(k+i), p(k+i))$
    - $y(k+i|k) = h(x(k+i|k), u(k+i), p(k+i))$
    - $g(x(k+i|k), u(k+i)) \leq 0$ (operational constraints)
    - $u_{min} \leq u(k+i) \leq u_{max}$
    - $x_{min} \leq x(k+i|k) \leq x_{max}$
    
    Where:
    - $U_k = [u(k), u(k+1), ..., u(k+N_c-1)]$ is the control sequence
    - $P_k = [p(k), p(k+1), ..., p(k+N_p-1)]$ is the economic parameter sequence
    - $L_e(\cdot)$ is the economic stage cost function
    
    **3. Economic Stage Cost:**
    
    The economic stage cost typically includes:
    
    $$L_e(x, u, p) = c_{op}(x, u, p) + c_{trans}(u) + c_{constraint}(x, u)$$
    
    Where:
    - $c_{op}(x, u, p)$: Operational costs (energy, raw materials, labor)
    - $c_{trans}(u)$: Transition costs (startup, shutdown, changeover)
    - $c_{constraint}(x, u)$: Constraint violation penalties

!!! success "Key Properties"
    - **Economic Optimization**: Direct optimization of economic objectives
    - **Time-Varying Economics**: Handles changing prices, costs, and market conditions
    - **Operational Constraints**: Incorporates business and operational limitations
    - **Multi-Objective Balance**: Balances economic performance with system stability
    - **Real-Time Adaptation**: Continuously adapts to economic changes

## Implementation Approaches

=== "Basic Economic MPC Controller (Recommended)"
    ```python
    import numpy as np
    from scipy.optimize import minimize
    from typing import Callable, Optional, Tuple, Dict, List
    
    class EconomicMPCController:
        """
        Basic Economic MPC Controller implementation.
        
        Args:
            prediction_horizon: Number of prediction steps
            control_horizon: Number of control steps
            state_dim: Dimension of state vector
            input_dim: Dimension of input vector
            output_dim: Dimension of output vector
            economic_dim: Dimension of economic parameter vector
            economic_cost_function: Function that computes economic costs
            Q: State deviation weight matrix (for stability)
            R: Input change weight matrix (for smoothness)
        """
        
        def __init__(self, prediction_horizon: int, control_horizon: int,
                     state_dim: int, input_dim: int, output_dim: int,
                     economic_dim: int, economic_cost_function: Callable,
                     Q: np.ndarray = None, R: np.ndarray = None):
            
            self.Np = prediction_horizon
            self.Nc = min(control_horizon, prediction_horizon)
            self.nx = state_dim
            self.nu = input_dim
            self.ny = output_dim
            self.np = economic_dim
            
            # Economic cost function
            self.economic_cost = economic_cost_function
            
            # Weighting matrices for stability and smoothness
            self.Q = Q if Q is not None else 0.1 * np.eye(state_dim)
            self.R = R if R is not None else 0.01 * np.eye(input_dim)
            
            # System model functions
            self.f = None  # State update function
            self.h = None  # Output function
            
            # Economic parameters (prices, costs)
            self.economic_parameters = []
            
            # Constraints
            self.u_min = -np.inf * np.ones(input_dim)
            self.u_max = np.inf * np.ones(input_dim)
            self.x_min = -np.inf * np.ones(state_dim)
            self.x_max = np.inf * np.ones(state_dim)
            
            # History
            self.control_history = []
            self.state_history = []
            self.economic_cost_history = []
            self.total_cost_history = []
        
        def set_system_model(self, state_update_func: Callable, 
                           output_func: Callable) -> None:
            """
            Set the system model functions.
            
            Args:
                state_update_func: Function that computes next state
                output_func: Function that computes output
            """
            self.f = state_update_func
            self.h = output_func
        
        def set_economic_parameters(self, parameters: List[np.ndarray]) -> None:
            """
            Set economic parameters over the prediction horizon.
            
            Args:
                parameters: List of economic parameter vectors
            """
            if len(parameters) < self.Np:
                # Extend parameters if too short
                last_param = parameters[-1] if parameters else np.zeros(self.np)
                extended_params = parameters + [last_param] * (self.Np - len(parameters))
            else:
                extended_params = parameters[:self.Np]
            
            self.economic_parameters = extended_params
        
        def set_constraints(self, u_min: np.ndarray = None, u_max: np.ndarray = None,
                          x_min: np.ndarray = None, x_max: np.ndarray = None) -> None:
            """
            Set input and state constraints.
            """
            if u_min is not None:
                self.u_min = np.array(u_min)
            if u_max is not None:
                self.u_max = array(u_max)
            if x_min is not None:
                self.x_min = np.array(x_min)
            if x_max is not None:
                self.x_max = np.array(x_max)
        
        def compute_control(self, current_state: np.ndarray, 
                          reference_trajectory: np.ndarray = None,
                          current_disturbance: np.ndarray = None) -> np.ndarray:
            """
            Compute optimal control input using Economic MPC.
            """
            if self.f is None or self.h is None:
                raise ValueError("System model not set")
            
            if not self.economic_parameters:
                print("Warning: No economic parameters set. Using default values.")
                self.economic_parameters = [np.zeros(self.np)] * self.Np
            
            # Initial guess for control sequence
            u0 = np.zeros(self.Nc * self.nu)
            
            # Bounds for optimization
            bounds = []
            for i in range(self.Nc):
                for j in range(self.nu):
                    bounds.append((self.u_min[j], self.u_max[j]))
            
            # Solve economic optimization problem
            result = minimize(
                fun=lambda u: self._economic_objective_function(u, current_state, 
                                                             reference_trajectory, current_disturbance),
                x0=u0,
                bounds=bounds,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            if not result.success:
                print(f"Economic MPC optimization failed: {result.message}")
                # Use previous control or zero control as fallback
                if self.control_history:
                    return self.control_history[-1]
                return np.zeros(self.nu)
            
            # Extract first control input
            optimal_control = result.x[:self.nu]
            
            # Store history
            self.control_history.append(optimal_control)
            self.state_history.append(current_state)
            
            # Compute and store costs
            economic_cost = self._compute_economic_cost(optimal_control, current_state)
            self.economic_cost_history.append(economic_cost)
            self.total_cost_history.append(result.fun)
            
            return optimal_control
        
        def _economic_objective_function(self, u: np.ndarray, current_state: np.ndarray,
                                       reference: np.ndarray, disturbance: np.ndarray = None) -> float:
            """
            Compute Economic MPC objective function.
            """
            # Reshape control sequence
            U = u.reshape(self.Nc, self.nu)
            
            # Initialize costs
            economic_cost = 0.0
            stability_cost = 0.0
            smoothness_cost = 0.0
            
            x = current_state.copy()
            prev_u = np.zeros(self.nu)
            
            # Prediction loop
            for i in range(self.Np):
                # Get control input
                if i < self.Nc:
                    u_i = U[i]
                else:
                    u_i = U[-1]
                
                # Get economic parameters
                p_i = self.economic_parameters[i]
                
                # Predict next state
                if disturbance is not None and i < len(disturbance):
                    d_i = disturbance[i]
                else:
                    d_i = np.zeros_like(current_state)
                
                x_next = self.f(x, u_i, d_i, p_i)
                
                # Economic cost
                stage_economic_cost = self.economic_cost(x, u_i, p_i)
                economic_cost += stage_economic_cost
                
                # Stability cost (deviation from reference if provided)
                if reference is not None and i < len(reference):
                    ref_i = reference[i]
                    state_deviation = x - ref_i
                    stability_cost += state_deviation.T @ self.Q @ state_deviation
                
                # Smoothness cost (control change penalty)
                if i > 0:
                    control_change = u_i - prev_u
                    smoothness_cost += control_change.T @ self.R @ control_change
                
                # State constraints penalty
                if np.any(x < self.x_min) or np.any(x > self.x_max):
                    economic_cost += 1e6  # Large penalty for constraint violation
                
                # Update state and previous control
                x = x_next
                prev_u = u_i
            
            # Total cost
            total_cost = economic_cost + stability_cost + smoothness_cost
            
            return total_cost
        
        def _compute_economic_cost(self, control: np.ndarray, state: np.ndarray) -> float:
            """
            Compute economic cost for current state and control.
            """
            if self.economic_parameters:
                p = self.economic_parameters[0]
            else:
                p = np.zeros(self.np)
            
            return self.economic_cost(state, control, p)
        
        def get_control_history(self) -> np.ndarray:
            """Get control input history."""
            return np.array(self.control_history) if self.control_history else np.array([])
        
        def get_state_history(self) -> np.ndarray:
            """Get state history."""
            return np.array(self.state_history) if self.state_history else np.array([])
        
        def get_economic_cost_history(self) -> np.ndarray:
            """Get economic cost history."""
            return np.array(self.economic_cost_history) if self.economic_cost_history else np.array([])
        
        def get_total_cost_history(self) -> np.ndarray:
            """Get total cost history."""
            return np.array(self.total_cost_history) if self.total_cost_history else np.array([])
        
        def reset(self) -> None:
            """Reset controller state."""
            self.control_history.clear()
            self.state_history.clear()
            self.economic_cost_history.clear()
            self.total_cost_history.clear()
    ```

=== "Profit Maximization MPC (Advanced)"
    ```python
    class ProfitMaximizationMPC(EconomicMPCController):
        """
        Economic MPC focused on profit maximization.
        """
        
        def __init__(self, revenue_function: Callable, cost_function: Callable, **kwargs):
            super().__init__(**kwargs)
            
            # Economic functions
            self.revenue_function = revenue_function
            self.cost_function = cost_function
            
            # Profit optimization parameters
            self.profit_weight = 1.0
            self.operational_weight = 0.1
        
        def _economic_objective_function(self, u: np.ndarray, current_state: np.ndarray,
                                       reference: np.ndarray, disturbance: np.ndarray = None) -> float:
            """
            Compute profit maximization objective function.
            """
            # Reshape control sequence
            U = u.reshape(self.Nc, self.nu)
            
            # Initialize costs
            total_profit = 0.0
            operational_cost = 0.0
            stability_cost = 0.0
            
            x = current_state.copy()
            prev_u = np.zeros(self.nu)
            
            # Prediction loop
            for i in range(self.Np):
                # Get control input
                if i < self.Nc:
                    u_i = U[i]
                else:
                    u_i = U[-1]
                
                # Get economic parameters
                p_i = self.economic_parameters[i]
                
                # Predict next state
                if disturbance is not None and i < len(disturbance):
                    d_i = disturbance[i]
                else:
                    d_i = np.zeros_like(current_state)
                
                x_next = self.f(x, u_i, d_i, p_i)
                
                # Predict output
                y_i = self.h(x, u_i, p_i)
                
                # Revenue and cost
                revenue = self.revenue_function(y_i, p_i)
                cost = self.cost_function(x, u_i, p_i)
                profit = revenue - cost
                
                total_profit += profit
                operational_cost += cost
                
                # Stability cost (deviation from reference if provided)
                if reference is not None and i < len(reference):
                    ref_i = reference[i]
                    state_deviation = x - ref_i
                    stability_cost += state_deviation.T @ self.Q @ state_deviation
                
                # Smoothness cost
                if i > 0:
                    control_change = u_i - prev_u
                    smoothness_cost = control_change.T @ self.R @ control_change
                    operational_cost += smoothness_cost
                
                # State constraints penalty
                if np.any(x < self.x_min) or np.any(x > self.x_max):
                    total_profit -= 1e6  # Large penalty for constraint violation
                
                # Update state and previous control
                x = x_next
                prev_u = u_i
            
            # Objective: maximize profit while minimizing operational costs
            objective = -self.profit_weight * total_profit + self.operational_weight * operational_cost + stability_cost
            
            return objective
    ```

=== "Resource Efficiency MPC"
    ```python
    class ResourceEfficiencyMPC(EconomicMPCController):
        """
        Economic MPC focused on resource efficiency and sustainability.
        """
        
        def __init__(self, resource_cost_function: Callable, efficiency_function: Callable, **kwargs):
            super().__init__(**kwargs)
            
            # Resource and efficiency functions
            self.resource_cost = resource_cost_function
            self.efficiency = efficiency_function
            
            # Efficiency optimization parameters
            self.efficiency_weight = 1.0
            self.resource_weight = 0.5
        
        def _economic_objective_function(self, u: np.ndarray, current_state: np.ndarray,
                                       reference: np.ndarray, disturbance: np.ndarray = None) -> float:
            """
            Compute resource efficiency objective function.
            """
            # Reshape control sequence
            U = u.reshape(self.Nc, self.nu)
            
            # Initialize costs
            total_efficiency = 0.0
            total_resource_cost = 0.0
            stability_cost = 0.0
            
            x = current_state.copy()
            
            # Prediction loop
            for i in range(self.Np):
                # Get control input
                if i < self.Nc:
                    u_i = U[i]
                else:
                    u_i = U[-1]
                
                # Get economic parameters
                p_i = self.economic_parameters[i]
                
                # Predict next state
                if disturbance is not None and i < len(disturbance):
                    d_i = disturbance[i]
                else:
                    d_i = np.zeros_like(current_state)
                
                x_next = self.f(x, u_i, d_i, p_i)
                
                # Predict output
                y_i = self.h(x, u_i, p_i)
                
                # Efficiency and resource cost
                efficiency = self.efficiency(y_i, u_i, p_i)
                resource_cost = self.resource_cost(u_i, p_i)
                
                total_efficiency += efficiency
                total_resource_cost += resource_cost
                
                # Stability cost
                if reference is not None and i < len(reference):
                    ref_i = reference[i]
                    state_deviation = x - ref_i
                    stability_cost += state_deviation.T @ self.Q @ state_deviation
                
                # State constraints penalty
                if np.any(x < self.x_min) or np.any(x > self.x_max):
                    total_efficiency -= 1e6
                
                # Update state
                x = x_next
            
            # Objective: maximize efficiency while minimizing resource cost
            objective = -self.efficiency_weight * total_efficiency + self.resource_weight * total_resource_cost + stability_cost
            
            return objective
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/mpc/economic_mpc.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/mpc/economic_mpc.py)
    - **Tests**: [`tests/unit/mpc/test_economic_mpc.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/mpc/test_economic_mpc.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Basic Economic MPC** | O(N³) per time step | O(N²) for variables | Standard optimization |
    **Profit Maximization** | O(N³) per time step | O(N²) for variables | Revenue/cost functions |
    **Resource Efficiency** | O(N³) per time step | O(N²) for variables | Efficiency metrics |

!!! warning "Performance Considerations"
    - **Economic models** can add computational complexity
    - **Time-varying parameters** require frequent updates
    - **Multi-objective optimization** needs careful tuning
    - **Economic constraints** may limit solution space

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Chemical Process Control"
        - **Reactor Optimization**: Maximize product yield while minimizing costs
        - **Distillation Control**: Optimize energy consumption and product quality
        - **Batch Process Control**: Minimize batch time and resource usage
        - **Supply Chain Management**: Optimize inventory and production scheduling

    !!! grid-item "Power Generation"
        - **Thermal Power Plants**: Minimize fuel costs and maximize efficiency
        - **Renewable Energy**: Optimize power output and grid integration
        - **Energy Storage**: Maximize arbitrage opportunities
        - **Demand Response**: Optimize load shifting and peak shaving

    !!! grid-item "Manufacturing Systems"
        - **Production Line Control**: Minimize production costs and maximize throughput
        - **Quality Control**: Balance quality requirements with production costs
        - **Maintenance Scheduling**: Optimize maintenance timing and costs
        - **Inventory Management**: Minimize holding costs and stockouts

    !!! grid-item "Supply Chain & Logistics"
        - **Transportation Optimization**: Minimize fuel and time costs
        - **Warehouse Management**: Optimize storage and picking operations
        - **Supplier Selection**: Balance cost, quality, and delivery time
        - **Route Planning**: Optimize delivery routes and schedules

!!! success "Educational Value"
    - **Economic Modeling**: Understanding cost structures and revenue models
    - **Multi-Objective Optimization**: Balancing economic and operational goals
    - **Time-Varying Optimization**: Handling changing economic conditions
    - **Business Process Integration**: Connecting control with business objectives

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Textbooks"
        1. **Rawlings, J. B., et al.** (2017). *Model Predictive Control: Theory, Computation, and Design*. Nob Hill.
        2. **Ellis, M., et al.** (2014). *Economic Model Predictive Control: Theory, Formulations and Chemical Process Applications*. Springer.

    !!! grid-item "Historical & Cultural"
        3. **Angeli, D., et al.** (2012). Economic model predictive control: The role of Lyapunov functions. *IFAC Proceedings*, 45(17).
        4. **Heidarinejad, M., et al.** (2012). Economic model predictive control of nonlinear process systems using Lyapunov techniques. *AIChE Journal*, 58(3).

    !!! grid-item "Online Resources"
        5. [Economic MPC - Wikipedia](https://en.wikipedia.org/wiki/Model_predictive_control)
        6. [Economic Optimization](https://www.mathworks.com/help/optim/)
        7. [Process Economics](https://www.youtube.com/watch?v=example)

    !!! grid-item "Implementation & Practice"
        8. [Python Economic MPC](https://pypi.org/project/economic-mpc/)
        9. [MATLAB Economic MPC](https://www.mathworks.com/help/mpc/)
        10. [Process Economics Tools](https://www.aspentech.com/)

!!! tip "Interactive Learning"
    Try implementing Economic MPC yourself! Start with a simple system and define economic cost functions, then implement the basic economic optimization loop. Experiment with different economic objectives (cost minimization, profit maximization, efficiency) and see how they affect the control behavior. Try implementing time-varying economic parameters and see how the controller adapts to changing conditions. Add operational constraints and see how they limit the economic optimization. This will give you deep insight into how to design controllers that optimize economic performance while maintaining system stability and operational constraints.
