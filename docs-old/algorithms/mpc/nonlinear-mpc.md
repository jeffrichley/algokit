---
algorithm_key: "nonlinear-mpc"
tags: [mpc, algorithms, nonlinear-mpc, sequential-quadratic-programming, nonlinear-optimization, real-time-optimization]
title: "Nonlinear MPC"
family: "mpc"
---

# Nonlinear MPC

{{ algorithm_card("nonlinear-mpc") }}

!!! abstract "Overview"
    Nonlinear MPC extends the predictive control framework to handle systems with nonlinear dynamics, constraints, and objectives. Unlike Linear MPC, which can be formulated as a Quadratic Programming problem, Nonlinear MPC requires solving a Nonlinear Programming (NLP) problem at each time step. This approach provides the ability to control complex nonlinear systems while maintaining the predictive and constraint-handling capabilities of MPC.

    Nonlinear MPC is essential in applications where system dynamics are inherently nonlinear, such as chemical processes, robotics, aerospace systems, and many other domains. While computationally more demanding than linear approaches, it offers superior performance for nonlinear systems and can handle complex constraints and objectives.

## Mathematical Formulation

!!! math "Nonlinear MPC Framework"
    **1. Nonlinear System Model:**
    
    The discrete-time nonlinear system is described by:
    
    $$x(k+1) = f(x(k), u(k), d(k))$$
    $$y(k) = h(x(k), u(k))$$
    
    Where:
    - $x(k) \in \mathbb{R}^{n_x}$ is the state vector
    - $u(k) \in \mathbb{R}^{n_u}$ is the control input vector
    - $d(k) \in \mathbb{R}^{n_d}$ is the disturbance vector
    - $y(k) \in \mathbb{R}^{n_y}$ is the output vector
    - $f(\cdot)$ and $h(\cdot)$ are nonlinear functions
    
    **2. Nonlinear Optimization Problem:**
    
    At each time step $k$, solve:
    
    $$\min_{U_k} J(x(k), U_k) = \sum_{i=0}^{N_p-1} \|y(k+i|k) - r(k+i)\|_Q^2 + \sum_{i=0}^{N_c-1} \|u(k+i)\|_R^2$$
    
    Subject to:
    - $x(k+i+1|k) = f(x(k+i|k), u(k+i), d(k+i))$
    - $y(k+i|k) = h(x(k+i|k), u(k+i))$
    - $g(x(k+i|k), u(k+i)) \leq 0$ (inequality constraints)
    - $h_c(x(k+i|k), u(k+i)) = 0$ (equality constraints)
    - $u_{min} \leq u(k+i) \leq u_{max}$
    - $x_{min} \leq x(k+i|k) \leq x_{max}$
    
    Where:
    - $U_k = [u(k), u(k+1), ..., u(k+N_c-1)]$ is the control sequence
    - $N_p$ is the prediction horizon
    - $N_c$ is the control horizon
    - $r(k)$ is the reference trajectory
    - $Q$ and $R$ are weighting matrices
    - $g(\cdot)$ and $h_c(\cdot)$ are constraint functions
    
    **3. Sequential Quadratic Programming (SQP):**
    
    The NLP is solved using SQP, which iteratively solves QP subproblems:
    
    $$\min_{\Delta U_k} \frac{1}{2} \Delta U_k^T H_k \Delta U_k + g_k^T \Delta U_k$$
    
    Subject to linearized constraints around the current iterate.

!!! success "Key Properties"
    - **Nonlinear Dynamics**: Handles complex system behaviors
    - **General Constraints**: Supports nonlinear equality and inequality constraints
    - **Global Optimization**: Can find globally optimal solutions
    - **Real-time Capability**: Fast convergence with modern solvers
    - **Constraint Satisfaction**: Ensures all constraints are met

## Implementation Approaches

=== "Basic Nonlinear MPC Controller (Recommended)"
    ```python
    import numpy as np
    from scipy.optimize import minimize
    from typing import Callable, Optional, Tuple, Dict
    
    class NonlinearMPCController:
        """
        Basic Nonlinear MPC Controller implementation.
        
        Args:
            prediction_horizon: Number of prediction steps
            control_horizon: Number of control steps
            state_dim: Dimension of state vector
            input_dim: Dimension of input vector
            output_dim: Dimension of output vector
            Q: Output tracking weight matrix
            R: Input penalty weight matrix
            Qf: Terminal state weight matrix
        """
        
        def __init__(self, prediction_horizon: int, control_horizon: int,
                     state_dim: int, input_dim: int, output_dim: int,
                     Q: np.ndarray = None, R: np.ndarray = None, 
                     Qf: np.ndarray = None):
            
            self.Np = prediction_horizon
            self.Nc = min(control_horizon, prediction_horizon)
            self.nx = state_dim
            self.nu = input_dim
            self.ny = output_dim
            
            # Weighting matrices
            self.Q = Q if Q is not None else np.eye(output_dim)
            self.R = R if R is not None else np.eye(input_dim)
            self.Qf = Qf if Qf is not None else np.eye(state_dim)
            
            # System model functions
            self.f = None  # State update function
            self.h = None  # Output function
            
            # Constraint functions
            self.g = None  # Inequality constraints
            self.h_c = None  # Equality constraints
            
            # Constraints
            self.u_min = -np.inf * np.ones(input_dim)
            self.u_max = np.inf * np.ones(input_dim)
            self.x_min = -np.inf * np.ones(state_dim)
            self.x_max = np.inf * np.ones(state_dim)
            
            # History
            self.control_history = []
            self.state_history = []
            self.cost_history = []
            self.optimization_info = []
        
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
        
        def set_constraint_functions(self, inequality_constraints: Callable = None,
                                   equality_constraints: Callable = None) -> None:
            """
            Set constraint functions.
            
            Args:
                inequality_constraints: Function g(x,u) <= 0
                equality_constraints: Function h_c(x,u) = 0
            """
            self.g = inequality_constraints
            self.h_c = equality_constraints
        
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
                          reference_trajectory: np.ndarray,
                          current_disturbance: np.ndarray = None) -> np.ndarray:
            """
            Compute optimal control input using Nonlinear MPC.
            """
            if self.f is None or self.h is None:
                raise ValueError("System model not set")
            
            # Initial guess for control sequence
            u0 = np.zeros(self.Nc * self.nu)
            
            # Bounds for optimization
            bounds = []
            for i in range(self.Nc):
                for j in range(self.nu):
                    bounds.append((self.u_min[j], self.u_max[j]))
            
            # Constraints
            constraints = []
            
            # Add inequality constraints if defined
            if self.g is not None:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda u: -self._evaluate_inequality_constraints(u, current_state, current_disturbance)
                })
            
            # Add equality constraints if defined
            if self.h_c is not None:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda u: self._evaluate_equality_constraints(u, current_state, current_disturbance)
                })
            
            # Solve optimization problem
            result = minimize(
                fun=lambda u: self._objective_function(u, current_state, 
                                                    reference_trajectory, current_disturbance),
                x0=u0,
                bounds=bounds,
                constraints=constraints,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            if not result.success:
                print(f"Nonlinear MPC optimization failed: {result.message}")
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
            self.optimization_info.append({
                'success': result.success,
                'iterations': result.nit,
                'message': result.message
            })
            
            return optimal_control
        
        def _objective_function(self, u: np.ndarray, current_state: np.ndarray,
                              reference: np.ndarray, disturbance: np.ndarray = None) -> float:
            """
            Compute Nonlinear MPC objective function value.
            """
            # Reshape control sequence
            U = u.reshape(self.Nc, self.nu)
            
            # Initialize cost
            cost = 0.0
            x = current_state.copy()
            
            # Prediction loop
            for i in range(self.Np):
                # Get control input (use last control if beyond control horizon)
                if i < self.Nc:
                    u_i = U[i]
                else:
                    u_i = U[-1]
                
                # Predict next state
                if disturbance is not None and i < len(disturbance):
                    d_i = disturbance[i]
                else:
                    d_i = np.zeros_like(current_state)
                
                x_next = self.f(x, u_i, d_i)
                
                # Predict output
                y_i = self.h(x, u_i)
                
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
                    cost += 1e6  # Large penalty for constraint violation
                
                # Update state
                x = x_next
            
            # Terminal cost
            terminal_error = x - (reference[-1] if reference else np.zeros_like(x))
            cost += terminal_error.T @ self.Qf @ terminal_error
            
            return cost
        
        def _evaluate_inequality_constraints(self, u: np.ndarray, current_state: np.ndarray,
                                          disturbance: np.ndarray = None) -> np.ndarray:
            """
            Evaluate inequality constraints g(x,u) <= 0.
            """
            if self.g is None:
                return np.array([])
            
            # Reshape control sequence
            U = u.reshape(self.Nc, self.nu)
            
            constraints = []
            x = current_state.copy()
            
            # Evaluate constraints over prediction horizon
            for i in range(self.Np):
                if i < self.Nc:
                    u_i = U[i]
                else:
                    u_i = U[-1]
                
                # Evaluate constraints at current state and input
                g_val = self.g(x, u_i)
                if np.isscalar(g_val):
                    g_val = np.array([g_val])
                constraints.extend(g_val)
                
                # Update state for next iteration
                if disturbance is not None and i < len(disturbance):
                    d_i = disturbance[i]
                else:
                    d_i = np.zeros_like(current_state)
                x = self.f(x, u_i, d_i)
            
            return np.array(constraints)
        
        def _evaluate_equality_constraints(self, u: np.ndarray, current_state: np.ndarray,
                                        disturbance: np.ndarray = None) -> np.ndarray:
            """
            Evaluate equality constraints h_c(x,u) = 0.
            """
            if self.h_c is None:
                return np.array([])
            
            # Reshape control sequence
            U = u.reshape(self.Nc, self.nu)
            
            constraints = []
            x = current_state.copy()
            
            # Evaluate constraints over prediction horizon
            for i in range(self.Np):
                if i < self.Nc:
                    u_i = U[i]
                else:
                    u_i = U[-1]
                
                # Evaluate constraints at current state and input
                h_val = self.h_c(x, u_i)
                if np.isscalar(h_val):
                    h_val = np.array([h_val])
                constraints.extend(h_val)
                
                # Update state for next iteration
                if disturbance is not None and i < len(disturbance):
                    d_i = disturbance[i]
                else:
                    d_i = np.zeros_like(current_state)
                x = self.f(x, u_i, d_i)
            
            return np.array(constraints)
        
        def get_control_history(self) -> np.ndarray:
            """Get control input history."""
            return np.array(self.control_history) if self.control_history else np.array([])
        
        def get_state_history(self) -> np.ndarray:
            """Get state history."""
            return np.array(self.state_history) if self.state_history else np.array([])
        
        def get_cost_history(self) -> np.ndarray:
            """Get cost history."""
            return np.array(self.cost_history) if self.cost_history else np.array([])
        
        def get_optimization_info(self) -> list:
            """Get optimization information history."""
            return self.optimization_info
        
        def reset(self) -> None:
            """Reset controller state."""
            self.control_history.clear()
            self.state_history.clear()
            self.cost_history.clear()
            self.optimization_info.clear()
    ```

=== "SQP-Based Nonlinear MPC (Advanced)"
    ```python
    class SQPNonlinearMPC(NonlinearMPCController):
        """
        Nonlinear MPC using Sequential Quadratic Programming.
        """
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            
            # SQP parameters
            self.max_sqp_iterations = 10
            self.sqp_tolerance = 1e-6
            self.merit_function_weight = 1.0
        
        def compute_control_sqp(self, current_state: np.ndarray, 
                               reference_trajectory: np.ndarray,
                               current_disturbance: np.ndarray = None) -> np.ndarray:
            """
            Compute control using SQP method.
            """
            # Initial guess
            u_k = np.zeros(self.Nc * self.nu)
            
            # SQP iterations
            for iteration in range(self.max_sqp_iterations):
                # Linearize constraints around current iterate
                A_eq, b_eq, A_ineq, b_ineq = self._linearize_constraints(u_k, current_state, current_disturbance)
                
                # Compute gradient and Hessian
                grad = self._compute_gradient(u_k, current_state, reference_trajectory, current_disturbance)
                H = self._compute_hessian(u_k, current_state, current_disturbance)
                
                # Solve QP subproblem
                try:
                    delta_u = self._solve_qp_subproblem(H, grad, A_eq, b_eq, A_ineq, b_ineq)
                except:
                    print(f"SQP QP subproblem failed at iteration {iteration}")
                    break
                
                # Update control
                u_k_new = u_k + delta_u
                
                # Check convergence
                if np.linalg.norm(delta_u) < self.sqp_tolerance:
                    break
                
                u_k = u_k_new
            
            # Extract first control input
            optimal_control = u_k[:self.nu]
            
            # Store history
            self.control_history.append(optimal_control)
            self.state_history.append(current_state)
            
            return optimal_control
        
        def _linearize_constraints(self, u: np.ndarray, current_state: np.ndarray,
                                 disturbance: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """
            Linearize constraints around current iterate.
            """
            # This is a simplified linearization
            # In practice, you would compute Jacobians
            
            # For now, return empty matrices
            return np.zeros((0, self.Nc * self.nu)), np.zeros(0), np.zeros((0, self.Nc * self.nu)), np.zeros(0)
        
        def _compute_gradient(self, u: np.ndarray, current_state: np.ndarray,
                            reference: np.ndarray, disturbance: np.ndarray = None) -> np.ndarray:
            """
            Compute gradient of objective function.
            """
            # Finite difference approximation
            epsilon = 1e-6
            grad = np.zeros_like(u)
            
            for i in range(len(u)):
                u_plus = u.copy()
                u_plus[i] += epsilon
                u_minus = u.copy()
                u_minus[i] -= epsilon
                
                f_plus = self._objective_function(u_plus, current_state, reference, disturbance)
                f_minus = self._objective_function(u_minus, current_state, reference, disturbance)
                
                grad[i] = (f_plus - f_minus) / (2 * epsilon)
            
            return grad
        
        def _compute_hessian(self, u: np.ndarray, current_state: np.ndarray,
                            disturbance: np.ndarray = None) -> np.ndarray:
            """
            Compute Hessian of objective function.
            """
            # Simplified Hessian (identity matrix)
            # In practice, you would compute the actual Hessian
            return np.eye(len(u))
        
        def _solve_qp_subproblem(self, H: np.ndarray, grad: np.ndarray,
                                A_eq: np.ndarray, b_eq: np.ndarray,
                                A_ineq: np.ndarray, b_ineq: np.ndarray) -> np.ndarray:
            """
            Solve QP subproblem.
            """
            # Simple QP solver using scipy
            from scipy.optimize import minimize
            
            def qp_objective(delta_u):
                return 0.5 * delta_u.T @ H @ delta_u + grad.T @ delta_u
            
            # Bounds for delta_u (simplified)
            bounds = [(-1.0, 1.0)] * len(grad)
            
            # Constraints
            constraints = []
            if len(A_eq) > 0:
                constraints.append({'type': 'eq', 'fun': lambda x: A_eq @ x - b_eq})
            if len(A_ineq) > 0:
                constraints.append({'type': 'ineq', 'fun': lambda x: b_ineq - A_ineq @ x})
            
            result = minimize(qp_objective, np.zeros_like(grad), bounds=bounds, constraints=constraints)
            
            if result.success:
                return result.x
            else:
                raise ValueError("QP subproblem failed to converge")
    ```

=== "Real-Time Iteration MPC"
    ```python
    class RealTimeIterationMPC(NonlinearMPCController):
        """
        Nonlinear MPC with real-time iteration scheme.
        """
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            
            # Real-time iteration parameters
            self.max_iterations_per_step = 1
            self.warm_start = True
            self.previous_solution = None
        
        def compute_control_rti(self, current_state: np.ndarray, 
                               reference_trajectory: np.ndarray,
                               current_disturbance: np.ndarray = None) -> np.ndarray:
            """
            Compute control using real-time iteration scheme.
            """
            # Use warm start if available
            if self.warm_start and self.previous_solution is not None:
                u0 = self._shift_solution(self.previous_solution)
            else:
                u0 = np.zeros(self.Nc * self.nu)
            
            # Perform limited iterations
            u_k = u0
            for iteration in range(self.max_iterations_per_step):
                # Single SQP iteration
                grad = self._compute_gradient(u_k, current_state, reference_trajectory, current_disturbance)
                H = self._compute_hessian(u_k, current_state, current_disturbance)
                
                # Simple gradient step
                step_size = 0.1
                u_k = u_k - step_size * np.linalg.solve(H, grad)
                
                # Apply bounds
                u_k = np.clip(u_k.reshape(self.Nc, self.nu), self.u_min, self.u_max).flatten()
            
            # Store solution for next warm start
            self.previous_solution = u_k.copy()
            
            # Extract first control input
            optimal_control = u_k[:self.nu]
            
            # Store history
            self.control_history.append(optimal_control)
            self.state_history.append(current_state)
            
            return optimal_control
        
        def _shift_solution(self, u: np.ndarray) -> np.ndarray:
            """
            Shift previous solution for warm start.
            """
            u_reshaped = u.reshape(self.Nc, self.nu)
            
            # Shift control sequence
            u_shifted = np.roll(u_reshaped, -1, axis=0)
            u_shifted[-1] = u_shifted[-2]  # Repeat last control
            
            return u_shifted.flatten()
        
        def _compute_gradient(self, u: np.ndarray, current_state: np.ndarray,
                            reference: np.ndarray, disturbance: np.ndarray = None) -> np.ndarray:
            """
            Compute gradient using finite differences.
            """
            epsilon = 1e-6
            grad = np.zeros_like(u)
            
            for i in range(len(u)):
                u_plus = u.copy()
                u_plus[i] += epsilon
                u_minus = u.copy()
                u_minus[i] -= epsilon
                
                f_plus = self._objective_function(u_plus, current_state, reference, disturbance)
                f_minus = self._objective_function(u_minus, current_state, reference, disturbance)
                
                grad[i] = (f_plus - f_minus) / (2 * epsilon)
            
            return grad
        
        def _compute_hessian(self, u: np.ndarray, current_state: np.ndarray,
                            disturbance: np.ndarray = None) -> np.ndarray:
            """
            Compute Hessian approximation.
            """
            # Use BFGS approximation or identity matrix
            return np.eye(len(u))
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/mpc/nonlinear_mpc.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/mpc/nonlinear_mpc.py)
    - **Tests**: [`tests/unit/mpc/test_nonlinear_mpc.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/mpc/test_nonlinear_mpc.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Basic Nonlinear MPC** | O(N³) to O(N⁴) | O(N²) for variables | Standard NLP solution |
    **SQP-Based MPC** | O(N³) per iteration | O(N²) for matrices | Multiple QP subproblems |
    **Real-Time Iteration** | O(N³) per iteration | O(N²) for matrices | Limited iterations per step |

!!! warning "Performance Considerations"
    - **Nonlinear optimization** is computationally expensive
    - **SQP iterations** can be time-consuming
    - **Real-time constraints** require careful iteration limits
    - **Warm start** significantly improves convergence

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Chemical Process Control"
        - **Reactor Control**: Temperature and concentration control
        - **Distillation**: Column control with phase changes
        - **Crystallization**: Particle size and shape control
        - **Polymerization**: Molecular weight distribution control

    !!! grid-item "Robotics & Automation"
        - **Robot Manipulators**: Joint and end-effector control
        - **Mobile Robots**: Navigation in complex environments
        - **Aerial Vehicles**: Flight control and path planning
        - **Underwater Vehicles**: Depth and trajectory control

    !!! grid-item "Aerospace & Defense"
        - **Aircraft Control**: Flight dynamics and trajectory
        - **Missile Guidance**: Nonlinear guidance laws
        - **Satellite Control**: Orbit and attitude control
        - **Spacecraft Docking**: Precise position control

    !!! grid-item "Energy Systems"
        - **Power Plants**: Boiler and turbine control
        - **Renewable Energy**: Wind turbine and solar control
        - **Smart Grids**: Load balancing and demand response
        - **Battery Management**: Charging and thermal control

!!! success "Educational Value"
    - **Nonlinear Systems**: Understanding complex system dynamics
    - **Optimization**: Learning nonlinear programming techniques
    - **Real-Time Control**: Managing computational constraints
    - **Constraint Handling**: Dealing with complex constraints

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Textbooks"
        1. **Rawlings, J. B., et al.** (2017). *Model Predictive Control: Theory, Computation, and Design*. Nob Hill.
        2. **Diehl, M., et al.** (2009). *Real-Time Optimization for Large Scale Nonlinear Processes*. Springer.

    !!! grid-item "Historical & Cultural"
        3. **Biegler, L. T.** (2010). Nonlinear programming: Concepts, algorithms, and applications to chemical processes. *SIAM*.
        4. **Nocedal, J., & Wright, S. J.** (2006). *Numerical Optimization*. Springer.

    !!! grid-item "Online Resources"
        5. [Nonlinear MPC - Wikipedia](https://en.wikipedia.org/wiki/Model_predictive_control)
        6. [SQP Methods](https://www.mathworks.com/help/optim/ug/constrained-nonlinear-optimization.html)
        7. [Real-Time Optimization](https://www.youtube.com/watch?v=example)

    !!! grid-item "Implementation & Practice"
        8. [Python NLP Solvers](https://pypi.org/project/ipopt/)
        9. [MATLAB fmincon](https://www.mathworks.com/help/optim/ug/fmincon.html)
        10. [CasADi Optimization](https://web.casadi.org/)

!!! tip "Interactive Learning"
    Try implementing Nonlinear MPC yourself! Start with a simple nonlinear system like a pendulum or inverted pendulum, then implement the basic prediction and optimization loop. Experiment with different optimization methods (SQP, interior point) and see how they affect convergence. Try implementing real-time iteration schemes with limited iterations per step, and compare performance with full optimization. Add nonlinear constraints and see how they affect the optimization problem. This will give you deep insight into how to handle nonlinear systems in predictive control and how to balance performance with computational requirements.

## Navigation

{{ nav_grid(current_algorithm="nonlinear-mpc", current_family="mpc", max_related=5) }}
