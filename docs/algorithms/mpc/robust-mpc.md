---
tags: [mpc, algorithms, robust-mpc, uncertainty-handling, min-max-optimization, tube-mpc]
title: "Robust MPC"
family: "mpc"
complexity: "O(N³) to O(N⁴) for horizon length N, depending on uncertainty model"
---

# Robust MPC

!!! info "Algorithm Family"
    **Family:** [MPC Algorithms](../../families/mpc.md)

!!! abstract "Overview"
    Robust MPC extends the predictive control framework to handle systems with uncertainties, disturbances, and model errors while maintaining performance and constraint satisfaction. Unlike standard MPC that assumes perfect knowledge of the system, Robust MPC explicitly accounts for bounded uncertainties and provides guaranteed performance under worst-case conditions.

    This approach is essential in applications where system parameters are uncertain, external disturbances are significant, or when safety and reliability are critical. Robust MPC methods include tube-based approaches, min-max optimization, and scenario-based methods that ensure robust performance across a range of operating conditions.

## Mathematical Formulation

!!! math "Robust MPC Framework"
    **1. Uncertain System Model:**
    
    The uncertain system is described by:
    
    $$x(k+1) = f(x(k), u(k), d(k), \theta)$$
    $$y(k) = h(x(k), u(k), \theta)$$
    
    Where:
    - $x(k) \in \mathbb{R}^{n_x}$ is the state vector
    - $u(k) \in \mathbb{R}^{n_u}$ is the control input vector
    - $d(k) \in \mathbb{R}^{n_d}$ is the disturbance vector
    - $\theta \in \Theta$ is the uncertain parameter vector
    - $\Theta$ is the uncertainty set
    
    **2. Robust Optimization Problem:**
    
    At each time step $k$, solve:
    
    $$\min_{U_k} \max_{\theta \in \Theta, d \in \mathcal{D}} J(x(k), U_k, \theta, d)$$
    
    Subject to:
    - $x(k+i+1|k) = f(x(k+i|k), u(k+i), d(k+i), \theta)$
    - $y(k+i|k) = h(x(k+i|k), u(k+i), \theta)$
    - $g(x(k+i|k), u(k+i), \theta) \leq 0$ for all $\theta \in \Theta$
    - $u_{min} \leq u(k+i) \leq u_{max}$
    - $x_{min} \leq x(k+i|k) \leq x_{max}$ for all $\theta \in \Theta$
    
    Where:
    - $U_k = [u(k), u(k+1), ..., u(k+N_c-1)]$ is the control sequence
    - $\mathcal{D}$ is the disturbance set
    - The objective function $J$ is maximized over uncertainties and minimized over controls
    
    **3. Tube MPC Approach:**
    
    For linear systems with additive uncertainties:
    
    $$x(k+1) = Ax(k) + Bu(k) + w(k)$$
    
    Where $w(k) \in \mathcal{W}$ (bounded uncertainty set).
    
    The tube MPC ensures:
    
    $$x(k) \in \mathcal{X} \ominus \mathcal{R}$$
    
    Where $\mathcal{R}$ is the robust positively invariant set.

!!! success "Key Properties"
    - **Uncertainty Handling**: Explicitly accounts for parameter and disturbance uncertainties
    - **Robust Performance**: Guarantees performance under worst-case conditions
    - **Constraint Satisfaction**: Ensures constraints are met for all admissible uncertainties
    - **Safety Guarantees**: Provides robust stability and safety margins
    - **Conservative Design**: May sacrifice nominal performance for robustness

## Implementation Approaches

=== "Basic Robust MPC Controller (Recommended)"
    ```python
    import numpy as np
    from scipy.optimize import minimize
    from typing import Callable, Optional, Tuple, List
    
    class RobustMPCController:
        """
        Basic Robust MPC Controller implementation.
        
        Args:
            prediction_horizon: Number of prediction steps
            control_horizon: Number of control steps
            state_dim: Dimension of state vector
            input_dim: Dimension of input vector
            output_dim: Dimension of output vector
            uncertainty_set: List of uncertainty scenarios
            Q: Output tracking weight matrix
            R: Input penalty weight matrix
            Qf: Terminal state weight matrix
        """
        
        def __init__(self, prediction_horizon: int, control_horizon: int,
                     state_dim: int, input_dim: int, output_dim: int,
                     uncertainty_set: List[dict] = None,
                     Q: np.ndarray = None, R: np.ndarray = None, 
                     Qf: np.ndarray = None):
            
            self.Np = prediction_horizon
            self.Nc = min(control_horizon, prediction_horizon)
            self.nx = state_dim
            self.nu = input_dim
            self.ny = output_dim
            
            # Uncertainty set
            self.uncertainty_set = uncertainty_set or []
            
            # Weighting matrices
            self.Q = Q if Q is not None else np.eye(output_dim)
            self.R = R if R is not None else np.eye(input_dim)
            self.Qf = Qf if Qf is not None else np.eye(state_dim)
            
            # System model functions
            self.f = None  # State update function
            self.h = None  # Output function
            
            # Constraints
            self.u_min = -np.inf * np.ones(input_dim)
            self.u_max = np.inf * np.ones(input_dim)
            self.x_min = -np.inf * np.ones(state_dim)
            self.x_max = np.inf * np.ones(state_dim)
            
            # History
            self.control_history = []
            self.state_history = []
            self.cost_history = []
            self.worst_case_scenarios = []
        
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
        
        def add_uncertainty_scenario(self, scenario: dict) -> None:
            """
            Add an uncertainty scenario.
            
            Args:
                scenario: Dictionary with parameter and disturbance values
            """
            self.uncertainty_set.append(scenario)
        
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
        
        def compute_control(self, current_state: np.ndarray, 
                          reference_trajectory: np.ndarray) -> np.ndarray:
            """
            Compute robust control input using Robust MPC.
            """
            if self.f is None or self.h is None:
                raise ValueError("System model not set")
            
            if not self.uncertainty_set:
                print("Warning: No uncertainty scenarios defined. Using nominal MPC.")
                return self._compute_nominal_control(current_state, reference_trajectory)
            
            # Initial guess for control sequence
            u0 = np.zeros(self.Nc * self.nu)
            
            # Bounds for optimization
            bounds = []
            for i in range(self.Nc):
                for j in range(self.nu):
                    bounds.append((self.u_min[j], self.u_max[j]))
            
            # Solve robust optimization problem
            result = minimize(
                fun=lambda u: self._robust_objective_function(u, current_state, reference_trajectory),
                x0=u0,
                bounds=bounds,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            if not result.success:
                print(f"Robust MPC optimization failed: {result.message}")
                # Use previous control or zero control as fallback
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
        
        def _robust_objective_function(self, u: np.ndarray, current_state: np.ndarray,
                                     reference: np.ndarray) -> float:
            """
            Compute robust objective function (worst-case over uncertainties).
            """
            # Reshape control sequence
            U = u.reshape(self.Nc, self.nu)
            
            # Evaluate objective for all uncertainty scenarios
            scenario_costs = []
            
            for scenario in self.uncertainty_set:
                cost = self._evaluate_scenario_cost(U, current_state, reference, scenario)
                scenario_costs.append(cost)
            
            # Return worst-case cost
            worst_case_cost = max(scenario_costs)
            
            # Store worst-case scenario
            worst_scenario_idx = np.argmax(scenario_costs)
            self.worst_case_scenarios.append(self.uncertainty_set[worst_scenario_idx])
            
            return worst_case_cost
        
        def _evaluate_scenario_cost(self, U: np.ndarray, current_state: np.ndarray,
                                  reference: np.ndarray, scenario: dict) -> float:
            """
            Evaluate cost for a specific uncertainty scenario.
            """
            # Initialize cost
            cost = 0.0
            x = current_state.copy()
            
            # Extract scenario parameters
            params = scenario.get('parameters', {})
            disturbances = scenario.get('disturbances', [])
            
            # Prediction loop
            for i in range(self.Np):
                # Get control input
                if i < self.Nc:
                    u_i = U[i]
                else:
                    u_i = U[-1]
                
                # Get disturbance for this step
                if i < len(disturbances):
                    d_i = disturbances[i]
                else:
                    d_i = np.zeros_like(current_state)
                
                # Predict next state with uncertainty
                x_next = self.f(x, u_i, d_i, **params)
                
                # Predict output with uncertainty
                y_i = self.h(x, u_i, **params)
                
                # Tracking cost
                if i < len(reference):
                    ref_i = reference[i]
                else:
                    ref_i = reference[-1] if reference else np.zeros_like(y_i)
                
                tracking_error = y_i - ref_i
                cost += tracking_error.T @ self.Q @ tracking_error
                
                # Control cost
                if i < self.Nc:
                    cost += u_i.T @ self.R @ u_i
                
                # State constraints penalty
                if np.any(x < self.x_min) or np.any(x > self.x_max):
                    cost += 1e6
                
                # Update state
                x = x_next
            
            # Terminal cost
            terminal_error = x - (reference[-1] if reference else np.zeros_like(x))
            cost += terminal_error.T @ self.Qf @ terminal_error
            
            return cost
        
        def _compute_nominal_control(self, current_state: np.ndarray, 
                                   reference_trajectory: np.ndarray) -> np.ndarray:
            """
            Compute control using nominal MPC (no uncertainties).
            """
            # Simple nominal MPC implementation
            u0 = np.zeros(self.Nc * self.nu)
            
            bounds = []
            for i in range(self.Nc):
                for j in range(self.nu):
                    bounds.append((self.u_min[j], self.u_max[j]))
            
            result = minimize(
                fun=lambda u: self._nominal_objective_function(u, current_state, reference_trajectory),
                x0=u0,
                bounds=bounds,
                method='SLSQP',
                options={'maxiter': 100}
            )
            
            if result.success:
                optimal_control = result.x[:self.nu]
            else:
                optimal_control = np.zeros(self.nu)
            
            return optimal_control
        
        def _nominal_objective_function(self, u: np.ndarray, current_state: np.ndarray,
                                      reference: np.ndarray) -> float:
            """
            Compute nominal objective function (no uncertainties).
            """
            # Reshape control sequence
            U = u.reshape(self.Nc, self.nu)
            
            # Initialize cost
            cost = 0.0
            x = current_state.copy()
            
            # Prediction loop
            for i in range(self.Np):
                # Get control input
                if i < self.Nc:
                    u_i = U[i]
                else:
                    u_i = U[-1]
                
                # Predict next state (nominal)
                x_next = self.f(x, u_i, np.zeros_like(current_state), {})
                
                # Predict output (nominal)
                y_i = self.h(x, u_i, {})
                
                # Tracking cost
                if i < len(reference):
                    ref_i = reference[i]
                else:
                    ref_i = reference[-1] if reference else np.zeros_like(y_i)
                
                tracking_error = y_i - ref_i
                cost += tracking_error.T @ self.Q @ tracking_error
                
                # Control cost
                if i < self.Nc:
                    cost += u_i.T @ self.R @ u_i
                
                # State constraints penalty
                if np.any(x < self.x_min) or np.any(x > self.x_max):
                    cost += 1e6
                
                # Update state
                x = x_next
            
            # Terminal cost
            terminal_error = x - (reference[-1] if reference else np.zeros_like(x))
            cost += terminal_error.T @ self.Qf @ terminal_error
            
            return cost
        
        def get_control_history(self) -> np.ndarray:
            """Get control input history."""
            return np.array(self.control_history) if self.control_history else np.array([])
        
        def get_state_history(self) -> np.ndarray:
            """Get state history."""
            return np.array(self.state_history) if self.state_history else np.array([])
        
        def get_cost_history(self) -> np.ndarray:
            """Get cost history."""
            return np.array(self.cost_history) if self.cost_history else np.array([])
        
        def get_worst_case_scenarios(self) -> list:
            """Get worst-case scenarios history."""
            return self.worst_case_scenarios
        
        def reset(self) -> None:
            """Reset controller state."""
            self.control_history.clear()
            self.state_history.clear()
            self.cost_history.clear()
            self.worst_case_scenarios.clear()
    ```

=== "Tube MPC Controller (Advanced)"
    ```python
    class TubeMPCController(RobustMPCController):
        """
        Tube MPC Controller for linear systems with additive uncertainties.
        """
        
        def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray,
                     uncertainty_set: np.ndarray, **kwargs):
            super().__init__(**kwargs)
            
            # Linear system matrices
            self.A = A
            self.B = B
            self.C = C
            
            # Uncertainty set (assumed to be polytopic)
            self.uncertainty_set = uncertainty_set
            
            # Compute robust positively invariant set
            self.robust_set = self._compute_robust_set()
            
            # Set linear system model
            self.set_system_model(self._linear_state_update, self._linear_output)
        
        def _linear_state_update(self, x: np.ndarray, u: np.ndarray, 
                               d: np.ndarray = None, **kwargs) -> np.ndarray:
            """Linear state update function."""
            x_next = self.A @ x + self.B @ u
            if d is not None:
                x_next += d
            return x_next
        
        def _linear_output(self, x: np.ndarray, u: np.ndarray, **kwargs) -> np.ndarray:
            """Linear output function."""
            return self.C @ x
        
        def _compute_robust_set(self) -> np.ndarray:
            """
            Compute robust positively invariant set.
            This is a simplified implementation.
            """
            # For simplicity, use a scaled version of the uncertainty set
            # In practice, you would use more sophisticated methods
            return self.uncertainty_set * 2.0
        
        def _robust_objective_function(self, u: np.ndarray, current_state: np.ndarray,
                                     reference: np.ndarray) -> float:
            """
            Compute robust objective function for tube MPC.
            """
            # Reshape control sequence
            U = u.reshape(self.Nc, self.nu)
            
            # Initialize cost
            cost = 0.0
            x = current_state.copy()
            
            # Prediction loop with tube constraints
            for i in range(self.Np):
                # Get control input
                if i < self.Nc:
                    u_i = U[i]
                else:
                    u_i = U[-1]
                
                # Predict nominal state
                x_nominal = self.A @ x + self.B @ u_i
                
                # Add tube constraint: x must be in X ⊖ R
                if not self._is_in_tube(x_nominal):
                    cost += 1e6  # Large penalty for tube violation
                
                # Predict output
                y_i = self.C @ x_nominal
                
                # Tracking cost
                if i < len(reference):
                    ref_i = reference[i]
                else:
                    ref_i = reference[-1] if reference else np.zeros_like(y_i)
                
                tracking_error = y_i - ref_i
                cost += tracking_error.T @ self.Q @ tracking_error
                
                # Control cost
                if i < self.Nc:
                    cost += u_i.T @ self.R @ u_i
                
                # Update state
                x = x_nominal
            
            # Terminal cost
            terminal_error = x - (reference[-1] if reference else np.zeros_like(x))
            cost += terminal_error.T @ self.Qf @ terminal_error
            
            return cost
        
        def _is_in_tube(self, state: np.ndarray) -> bool:
            """
            Check if state is within the tube (X ⊖ R).
            """
            # Simplified tube membership check
            # In practice, you would use proper set operations
            return np.all(np.abs(state) <= self.robust_set)
    ```

=== "Min-Max MPC Controller"
    ```python
    class MinMaxMPCController(RobustMPCController):
        """
        Min-Max MPC Controller using nested optimization.
        """
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            
            # Min-max parameters
            self.max_uncertainty_iterations = 10
            self.uncertainty_tolerance = 1e-4
        
        def compute_control_minmax(self, current_state: np.ndarray, 
                                  reference_trajectory: np.ndarray) -> np.ndarray:
            """
            Compute control using min-max optimization.
            """
            # Initial guess for control sequence
            u0 = np.zeros(self.Nc * self.nu)
            
            # Bounds for optimization
            bounds = []
            for i in range(self.Nc):
                for j in range(self.nu):
                    bounds.append((self.u_min[j], self.u_max[j]))
            
            # Solve min-max problem
            result = minimize(
                fun=lambda u: self._minmax_objective_function(u, current_state, reference_trajectory),
                x0=u0,
                bounds=bounds,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            if not result.success:
                print(f"Min-Max MPC optimization failed: {result.message}")
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
        
        def _minmax_objective_function(self, u: np.ndarray, current_state: np.ndarray,
                                     reference: np.ndarray) -> float:
            """
            Compute min-max objective function.
            """
            # For each control sequence, find the worst-case uncertainty
            worst_case_cost = -np.inf
            
            for scenario in self.uncertainty_set:
                cost = self._evaluate_scenario_cost(u.reshape(self.Nc, self.nu), 
                                                 current_state, reference, scenario)
                worst_case_cost = max(worst_case_cost, cost)
            
            return worst_case_cost
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/mpc/robust_mpc.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/mpc/robust_mpc.py)
    - **Tests**: [`tests/unit/mpc/test_robust_mpc.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/mpc/test_robust_mpc.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Basic Robust MPC** | O(N³ × S) | O(N² × S) | S uncertainty scenarios |
    **Tube MPC** | O(N³) | O(N²) | Linear system with polytopic uncertainty |
    **Min-Max MPC** | O(N³ × S) | O(N² × S) | Nested optimization |

!!! warning "Performance Considerations"
    - **Uncertainty scenarios** significantly increase computational cost
    - **Worst-case optimization** can be conservative
    - **Tube constraints** add complexity to optimization
    - **Scenario generation** affects robustness guarantees

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Chemical Process Control"
        - **Reactor Control**: Parameter uncertainties in kinetics
        - **Distillation**: Feed composition variations
        - **Crystallization**: Temperature and concentration uncertainties
        - **Polymerization**: Catalyst activity variations

    !!! grid-item "Aerospace & Defense"
        - **Flight Control**: Aerodynamic parameter uncertainties
        - **Missile Guidance**: Target motion uncertainties
        - **Satellite Control**: Orbital perturbation uncertainties
        - **Spacecraft Docking**: Relative position uncertainties

    !!! grid-item "Robotics & Automation"
        - **Robot Manipulators**: Payload and friction uncertainties
        - **Mobile Robots**: Terrain and sensor uncertainties
        - **Aerial Vehicles**: Wind and aerodynamic uncertainties
        - **Underwater Vehicles**: Current and buoyancy uncertainties

    !!! grid-item "Power Systems"
        - **Power Plants**: Load and fuel quality uncertainties
        - **Renewable Energy**: Wind and solar forecast uncertainties
        - **Smart Grids**: Demand and generation uncertainties
        - **Battery Management**: Aging and temperature uncertainties

!!! success "Educational Value"
    - **Uncertainty Modeling**: Understanding how to represent system uncertainties
    - **Robust Optimization**: Learning worst-case optimization techniques
    - **Safety Margins**: Understanding robust constraint satisfaction
    - **Performance vs. Robustness**: Balancing nominal and worst-case performance

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Textbooks"
        1. **Rawlings, J. B., et al.** (2017). *Model Predictive Control: Theory, Computation, and Design*. Nob Hill.
        2. **Bemporad, A., & Morari, M.** (1999). Robust model predictive control: A survey. *Robustness in Identification and Control*.

    !!! grid-item "Historical & Cultural"
        3. **Mayne, D. Q., et al.** (2005). Robust model predictive control of constrained linear systems with bounded disturbances. *Automatica*, 41(2).
        4. **Langson, W., et al.** (2004). Robust model predictive control using tubes. *Automatica*, 40(1).

    !!! grid-item "Online Resources"
        5. [Robust MPC - Wikipedia](https://en.wikipedia.org/wiki/Model_predictive_control)
        6. [Tube MPC Methods](https://www.mathworks.com/help/mpc/)
        7. [Robust Control Methods](https://www.youtube.com/watch?v=example)

    !!! grid-item "Implementation & Practice"
        8. [Python Robust MPC](https://pypi.org/project/robust-mpc/)
        9. [MATLAB Robust MPC](https://www.mathworks.com/help/mpc/)
        10. [CasADi Robust Optimization](https://web.casadi.org/)

!!! tip "Interactive Learning"
    Try implementing Robust MPC yourself! Start with a simple system and add parameter uncertainties, then implement scenario-based robust optimization. Experiment with different uncertainty sets and see how they affect the control behavior. Try implementing tube MPC for linear systems and compare the performance with scenario-based approaches. Add disturbance uncertainties and see how they affect the robust constraint satisfaction. This will give you deep insight into how to design controllers that are robust to real-world uncertainties while maintaining performance.
