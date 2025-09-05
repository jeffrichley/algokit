---
algorithm_key: "model-predictive-control"
tags: [mpc, algorithms, model-predictive-control, receding-horizon, optimization, predictive-control]
title: "Model Predictive Control"
family: "mpc"
---

# Model Predictive Control

{{ algorithm_card("model-predictive-control") }}

!!! abstract "Overview"
    Model Predictive Control (MPC) is an advanced control strategy that uses a mathematical model of the system to predict future behavior and optimize control actions over a finite prediction horizon. The controller solves an optimization problem at each time step to determine the optimal control sequence, applies the first control action, and then repeats the process in a receding horizon fashion.

    This approach is particularly valuable in process control, automotive systems, robotics, and other applications where future behavior prediction and constraint handling are critical. MPC provides excellent performance, handles multiple inputs and outputs naturally, and can incorporate various constraints and objectives in a unified framework.

## Mathematical Formulation

!!! math "MPC Framework"
    **1. Prediction Model:**

    The system is described by the discrete-time model:

    $$x(k+1) = f(x(k), u(k), d(k))$$
    $$y(k) = h(x(k), u(k))$$

    Where:
    - $x(k)$ is the state vector at time $k$
    - $u(k)$ is the control input vector
    - $d(k)$ is the disturbance vector
    - $y(k)$ is the output vector
    - $f(\cdot)$ and $h(\cdot)$ are system functions

    **2. Optimization Problem:**

    At each time step $k$, solve:

    $$\min_{U_k} J(x(k), U_k) = \sum_{i=0}^{N_p-1} \|y(k+i|k) - r(k+i)\|_Q^2 + \sum_{i=0}^{N_c-1} \|u(k+i)\|_R^2$$

    Subject to:
    - $x(k+i+1|k) = f(x(k+i|k), u(k+i), d(k+i))$
    - $y(k+i|k) = h(x(k+i|k), u(k+i))$
    - $u_{min} \leq u(k+i) \leq u_{max}$
    - $x_{min} \leq x(k+i|k) \leq x_{max}$
    - $u(k+i) = u(k+N_c-1)$ for $i \geq N_c$

    Where:
    - $U_k = [u(k), u(k+1), ..., u(k+N_c-1)]$ is the control sequence
    - $N_p$ is the prediction horizon
    - $N_c$ is the control horizon
    - $r(k)$ is the reference trajectory
    - $Q$ and $R$ are weighting matrices

    **3. Receding Horizon Implementation:**

    Apply only the first control action:
    $$u(k) = u^*(k)$$

    Then shift the horizon and repeat the optimization.

!!! success "Key Properties"
    - **Predictive Capability**: Uses system model to predict future behavior
    - **Constraint Handling**: Naturally incorporates input, state, and output constraints
    - **Multi-Objective Optimization**: Balances tracking performance and control effort
    - **Receding Horizon**: Continuously updates control strategy
    - **Model-Based**: Requires accurate system model for good performance

## Implementation Approaches

=== "Basic MPC Controller (Recommended)"
    ```python
    import numpy as np
    from scipy.optimize import minimize
    from typing import Callable, Tuple, Optional, Dict

    class ModelPredictiveController:
        """
        Basic Model Predictive Controller implementation.

        Args:
            prediction_horizon: Number of prediction steps
            control_horizon: Number of control steps (≤ prediction_horizon)
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

            # Constraints
            self.u_min = -np.inf * np.ones(input_dim)
            self.u_max = np.inf * np.ones(input_dim)
            self.x_min = -np.inf * np.ones(state_dim)
            self.x_max = np.inf * np.ones(state_dim)

            # System model functions
            self.f = None  # State update function
            self.h = None  # Output function

            # History
            self.control_history = []
            self.state_history = []
            self.output_history = []
            self.cost_history = []

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

        def set_constraints(self, u_min: np.ndarray = None, u_max: np.ndarray = None,
                          x_min: np.ndarray = None, x_max: np.ndarray = None) -> None:
            """
            Set input and state constraints.

            Args:
                u_min: Minimum input bounds
                u_max: Maximum input bounds
                x_min: Minimum state bounds
                x_max: Maximum state bounds
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
            Compute optimal control input using MPC.

            Args:
                current_state: Current system state
                reference_trajectory: Reference trajectory over prediction horizon
                current_disturbance: Current disturbance (optional)

            Returns:
                Optimal control input
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

            # Solve optimization problem
            result = minimize(
                fun=lambda u: self._objective_function(u, current_state,
                                                    reference_trajectory, current_disturbance),
                x0=u0,
                bounds=bounds,
                method='SLSQP',
                options={'maxiter': 100}
            )

            if not result.success:
                print(f"MPC optimization failed: {result.message}")
                # Use previous control or zero control as fallback
                if self.control_history:
                    return self.control_history[-1]
                return np.zeros(self.nu)

            # Extract first control input
            optimal_control = result.x[:self.nu]

            # Store history
            self.control_history.append(optimal_control)
            self.state_history.append(current_state)

            # Predict output
            predicted_output = self.h(current_state, optimal_control)
            self.output_history.append(predicted_output)
            self.cost_history.append(result.fun)

            return optimal_control

        def _objective_function(self, u: np.ndarray, current_state: np.ndarray,
                              reference: np.ndarray, disturbance: np.ndarray = None) -> float:
            """
            Compute MPC objective function value.

            Args:
                u: Control sequence (flattened)
                current_state: Current system state
                reference: Reference trajectory
                disturbance: Disturbance sequence

            Returns:
                Objective function value
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

        def get_control_history(self) -> np.ndarray:
            """Get control input history."""
            return np.array(self.control_history) if self.control_history else np.array([])

        def get_state_history(self) -> np.ndarray:
            """Get state history."""
            return np.array(self.state_history) if self.state_history else np.array([])

        def get_output_history(self) -> np.ndarray:
            """Get output history."""
            return np.array(self.output_history) if self.output_history else np.array([])

        def get_cost_history(self) -> np.ndarray:
            """Get cost history."""
            return np.array(self.cost_history) if self.cost_history else np.array([])

        def reset(self) -> None:
            """Reset controller state."""
            self.control_history.clear()
            self.state_history.clear()
            self.output_history.clear()
            self.cost_history.clear()
    ```

=== "Linear MPC Controller (Advanced)"
    ```python
    class LinearMPCController(ModelPredictiveController):
        """
        Linear MPC Controller for linear systems.
        """

        def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
                     **kwargs):
            super().__init__(**kwargs)

            # Linear system matrices
            self.A = A
            self.B = B
            self.C = C
            self.D = D

            # Set linear system model
            self.set_system_model(self._linear_state_update, self._linear_output)

        def _linear_state_update(self, x: np.ndarray, u: np.ndarray,
                               d: np.ndarray = None) -> np.ndarray:
            """Linear state update function."""
            x_next = self.A @ x + self.B @ u
            if d is not None:
                x_next += d
            return x_next

        def _linear_output(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
            """Linear output function."""
            return self.C @ x + self.D @ u

        def _objective_function(self, u: np.ndarray, current_state: np.ndarray,
                              reference: np.ndarray, disturbance: np.ndarray = None) -> float:
            """
            Compute linear MPC objective function.
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

                # Predict next state
                if disturbance is not None and i < len(disturbance):
                    d_i = disturbance[i]
                else:
                    d_i = np.zeros_like(current_state)

                x_next = self._linear_state_update(x, u_i, d_i)

                # Predict output
                y_i = self._linear_output(x, u_i)

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
    ```

=== "MPC with Warm Start (Advanced)"
    ```python
    class WarmStartMPCController(ModelPredictiveController):
        """
        MPC Controller with warm start for better convergence.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            # Warm start variables
            self.previous_solution = None
            self.previous_state = None

        def compute_control(self, current_state: np.ndarray,
                          reference_trajectory: np.ndarray,
                          current_disturbance: np.ndarray = None) -> np.ndarray:
            """
            Compute control with warm start.
            """
            # Generate warm start
            warm_start = self._generate_warm_start(current_state)

            # Bounds for optimization
            bounds = []
            for i in range(self.Nc):
                for j in range(self.nu):
                    bounds.append((self.u_min[j], self.u_max[j]))

            # Solve optimization with warm start
            result = minimize(
                fun=lambda u: self._objective_function(u, current_state,
                                                    reference_trajectory, current_disturbance),
                x0=warm_start,
                bounds=bounds,
                method='SLSQP',
                options={'maxiter': 100}
            )

            if not result.success:
                print(f"MPC optimization failed: {result.message}")
                if self.control_history:
                    return self.control_history[-1]
                return np.zeros(self.nu)

            # Store solution for next warm start
            self.previous_solution = result.x
            self.previous_state = current_state

            # Extract first control input
            optimal_control = result.x[:self.nu]

            # Store history
            self.control_history.append(optimal_control)
            self.state_history.append(current_state)

            # Predict output
            predicted_output = self.h(current_state, optimal_control)
            self.output_history.append(predicted_output)
            self.cost_history.append(result.fun)

            return optimal_control

        def _generate_warm_start(self, current_state: np.ndarray) -> np.ndarray:
            """
            Generate warm start for optimization.
            """
            if (self.previous_solution is not None and
                self.previous_state is not None):

                # Shift previous solution
                warm_start = np.roll(self.previous_solution, -self.nu)
                warm_start[-self.nu:] = self.previous_solution[-self.nu:]  # Repeat last control

                # Adjust for state change
                state_change = current_state - self.previous_state
                if np.linalg.norm(state_change) < 0.1:  # Small state change
                    return warm_start
                else:
                    # Larger state change, use zero initial guess
                    return np.zeros(self.Nc * self.nu)

            # No previous solution, use zero initial guess
            return np.zeros(self.Nc * self.nu)
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/mpc/model_predictive_control.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/mpc/model_predictive_control.py)
    - **Tests**: [`tests/unit/mpc/test_model_predictive_control.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/mpc/test_model_predictive_control.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Basic MPC** | O(N³) per time step | O(N²) for variables | Standard optimization |
    **Linear MPC** | O(N³) per time step | O(N²) for variables | Linear system matrices |
    **Warm Start MPC** | O(N³) per time step | O(N²) for variables | Better convergence |

!!! warning "Performance Considerations"
    - **Prediction horizon** affects computational complexity and performance
    - **Control horizon** balances performance and computational cost
    - **Weighting matrices** significantly impact control behavior
    - **Constraint handling** adds complexity to optimization problem

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Process Control"
        - **Chemical Processes**: Temperature, pressure, and flow control
        - **Oil Refining**: Distillation column control
        - **Power Generation**: Boiler and turbine control
        - **Water Treatment**: pH and chemical dosing control

    !!! grid-item "Automotive & Transportation"
        - **Engine Control**: Fuel injection and timing optimization
        - **Vehicle Control**: Trajectory tracking and obstacle avoidance
        - **Traffic Control**: Signal timing optimization
        - **Autonomous Driving**: Path planning and vehicle control

    !!! grid-item "Robotics & Automation"
        - **Robot Manipulators**: Trajectory tracking and force control
        - **Mobile Robots**: Navigation and obstacle avoidance
        - **Industrial Automation**: Production line optimization
        - **Aerial Vehicles**: Flight control and path planning

    !!! grid-item "Energy Systems"
        - **Smart Grids**: Load balancing and demand response
        - **Renewable Energy**: Wind turbine and solar panel control
        - **Battery Management**: Charging optimization and thermal control
        - **Building Control**: HVAC and lighting optimization

!!! success "Educational Value"
    - **Predictive Control**: Understanding future behavior prediction
    - **Optimization**: Learning constrained optimization techniques
    - **Model-Based Control**: Using system models for control design
    - **Constraint Handling**: Managing system limitations and requirements

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Textbooks"
        1. **Rawlings, J. B., et al.** (2017). *Model Predictive Control: Theory, Computation, and Design*. Nob Hill.
        2. **Maciejowski, J. M.** (2002). *Predictive Control with Constraints*. Prentice Hall.

    !!! grid-item "Historical & Cultural"
        3. **Richalet, J., et al.** (1978). Model predictive heuristic control: Applications to industrial processes. *Automatica*, 14(5).
        4. **Cutler, C. R., & Ramaker, B. L.** (1980). Dynamic matrix control—A computer control algorithm. *AIChE Journal*, 26(4).

    !!! grid-item "Online Resources"
        5. [Model Predictive Control - Wikipedia](https://en.wikipedia.org/wiki/Model_predictive_control)
        6. [MPC Tutorial](https://www.mathworks.com/help/mpc/)
        7. [MPC Applications](https://www.youtube.com/watch?v=example)

    !!! grid-item "Implementation & Practice"
        8. [Python MPC Library](https://pypi.org/project/mpc/)
        9. [MATLAB MPC Toolbox](https://www.mathworks.com/help/mpc/)
        10. [CasADi Optimization](https://web.casadi.org/)

!!! tip "Interactive Learning"
    Try implementing MPC yourself! Start with a simple first-order system, then implement the basic prediction and optimization loop. Add constraints to see how they affect the control behavior, and experiment with different prediction and control horizons to understand their impact on performance and computational cost. Try implementing warm start techniques to improve convergence, and add disturbance handling for robustness. This will give you deep insight into how predictive control works and how to design effective MPC controllers for various applications.

## Navigation

{{ nav_grid(current_algorithm="model-predictive-control", current_family="mpc", max_related=5) }}
