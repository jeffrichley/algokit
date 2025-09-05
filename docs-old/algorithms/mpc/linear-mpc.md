---
algorithm_key: "linear-mpc"
tags: [mpc, algorithms, linear-mpc, quadratic-programming, state-space, linear-systems]
title: "Linear MPC"
family: "mpc"
---

# Linear MPC

{{ algorithm_card("linear-mpc") }}

!!! abstract "Overview"
    Linear MPC is a specialized form of Model Predictive Control that applies to linear time-invariant (LTI) systems. By leveraging the linearity of the system, Linear MPC can be formulated as a Quadratic Programming (QP) problem, which can be solved efficiently using specialized optimization algorithms. This approach provides excellent performance for linear systems while maintaining the predictive and constraint-handling capabilities of MPC.

    Linear MPC is widely used in industrial process control, automotive applications, and aerospace systems where the system dynamics can be well-approximated by linear models. The QP formulation enables real-time implementation and provides theoretical guarantees for stability and performance.

## Mathematical Formulation

!!! math "Linear MPC Framework"
    **1. Linear System Model:**

    The discrete-time linear system is described by:

    $$x(k+1) = Ax(k) + Bu(k) + Ed(k)$$
    $$y(k) = Cx(k) + Du(k)$$

    Where:
    - $x(k) \in \mathbb{R}^{n_x}$ is the state vector
    - $u(k) \in \mathbb{R}^{n_u}$ is the control input vector
    - $d(k) \in \mathbb{R}^{n_d}$ is the disturbance vector
    - $y(k) \in \mathbb{R}^{n_y}$ is the output vector
    - $A, B, C, D, E$ are constant matrices of appropriate dimensions

    **2. Prediction Equations:**

    The state prediction over horizon $N_p$ is:

    $$X_k = \Phi x(k) + \Gamma U_k + \Psi D_k$$

    Where:
    - $X_k = [x(k+1|k), x(k+2|k), ..., x(k+N_p|k)]^T$
    - $U_k = [u(k), u(k+1), ..., u(k+N_c-1)]^T$
    - $D_k = [d(k), d(k+1), ..., d(k+N_p-1)]^T$
    - $\Phi, \Gamma, \Psi$ are prediction matrices

    **3. QP Formulation:**

    The optimization problem becomes:

    $$\min_{U_k} \frac{1}{2} U_k^T H U_k + f^T U_k$$

    Subject to:
    - $G U_k \leq w$
    - $A_{eq} U_k = b_{eq}$

    Where:
    - $H = \Gamma^T \bar{Q} \Gamma + \bar{R}$ is the Hessian matrix
    - $f = \Gamma^T \bar{Q} (\Phi x(k) + \Psi D_k - R_k)$ is the gradient
    - $\bar{Q} = \text{diag}(Q, Q, ..., Q, Q_f)$ and $\bar{R} = \text{diag}(R, R, ..., R)$
    - $R_k$ is the reference trajectory

    **4. Constraint Matrices:**

    Input constraints: $u_{min} \leq u(k+i) \leq u_{max}$
    State constraints: $x_{min} \leq x(k+i|k) \leq x_{max}$
    Output constraints: $y_{min} \leq y(k+i|k) \leq y_{max}$

!!! success "Key Properties"
    - **QP Formulation**: Convex optimization problem with efficient solvers
    - **Linear Predictions**: Exact state predictions using matrix operations
    - **Real-time Capability**: Fast solution using specialized QP algorithms
    - **Stability Guarantees**: Theoretical stability under certain conditions
    - **Constraint Handling**: Natural incorporation of linear constraints

## Implementation Approaches

=== "Basic Linear MPC Controller (Recommended)"
    ```python
    import numpy as np
    from scipy.optimize import minimize
    from typing import Optional, Tuple

    class LinearMPCController:
        """
        Linear MPC Controller implementation.

        Args:
            A: State matrix
            B: Input matrix
            C: Output matrix
            D: Direct feedthrough matrix
            prediction_horizon: Number of prediction steps
            control_horizon: Number of control steps
            Q: Output tracking weight matrix
            R: Input penalty weight matrix
            Qf: Terminal state weight matrix
        """

        def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
                     prediction_horizon: int, control_horizon: int,
                     Q: np.ndarray = None, R: np.ndarray = None, Qf: np.ndarray = None):

            self.A = A
            self.B = B
            self.C = C
            self.D = D

            self.Np = prediction_horizon
            self.Nc = min(control_horizon, prediction_horizon)

            # Dimensions
            self.nx = A.shape[0]
            self.nu = B.shape[1]
            self.ny = C.shape[0]

            # Weighting matrices
            self.Q = Q if Q is not None else np.eye(self.ny)
            self.R = R if R is not None else np.eye(self.nu)
            self.Qf = Qf if Qf is not None else np.eye(self.nx)

            # Constraints
            self.u_min = -np.inf * np.ones(self.nu)
            self.u_max = np.inf * np.ones(self.nu)
            self.x_min = -np.inf * np.ones(self.nx)
            self.x_max = np.inf * np.ones(self.nx)
            self.y_min = -np.inf * np.ones(self.ny)
            self.y_max = np.inf * np.ones(self.ny)

            # Pre-compute prediction matrices
            self._compute_prediction_matrices()

            # History
            self.control_history = []
            self.state_history = []
            self.cost_history = []

        def _compute_prediction_matrices(self) -> None:
            """Pre-compute prediction matrices for efficiency."""
            # State prediction matrix
            self.Phi = np.zeros((self.Np * self.nx, self.nx))
            for i in range(self.Np):
                self.Phi[i*self.nx:(i+1)*self.nx, :] = np.linalg.matrix_power(self.A, i+1)

            # Input prediction matrix
            self.Gamma = np.zeros((self.Np * self.nx, self.Nc * self.nu))
            for i in range(self.Np):
                for j in range(min(i+1, self.Nc)):
                    if i-j >= 0:
                        self.Gamma[i*self.nx:(i+1)*self.nx, j*self.nu:(j+1)*self.nu] = \
                            np.linalg.matrix_power(self.A, i-j) @ self.B

            # Output prediction matrix
            self.Phi_y = self.Phi @ self.C.T
            self.Gamma_y = self.Gamma @ self.C.T

        def set_constraints(self, u_min: np.ndarray = None, u_max: np.ndarray = None,
                          x_min: np.ndarray = None, x_max: np.ndarray = None,
                          y_min: np.ndarray = None, y_max: np.ndarray = None) -> None:
            """
            Set input, state, and output constraints.
            """
            if u_min is not None:
                self.u_min = np.array(u_min)
            if u_max is not None:
                self.u_max = np.array(u_max)
            if x_min is not None:
                self.x_min = np.array(x_min)
            if x_max is not None:
                self.x_max = np.array(x_max)
            if y_min is not None:
                self.y_min = np.array(y_min)
            if y_max is not None:
                self.y_max = np.array(y_max)

        def compute_control(self, current_state: np.ndarray,
                          reference_trajectory: np.ndarray,
                          current_disturbance: np.ndarray = None) -> np.ndarray:
            """
            Compute optimal control input using Linear MPC.
            """
            # Prepare reference trajectory
            if len(reference_trajectory) < self.Np:
                # Extend reference if too short
                ref_extended = np.tile(reference_trajectory[-1], (self.Np, 1))
                ref_extended[:len(reference_trajectory)] = reference_trajectory
            else:
                ref_extended = reference_trajectory[:self.Np]

            # Prepare disturbance
            if current_disturbance is not None:
                d_k = np.tile(current_disturbance, (self.Np, 1))
            else:
                d_k = np.zeros((self.Np, self.nx))

            # Build QP matrices
            H, f = self._build_qp_matrices(current_state, ref_extended, d_k)
            G, w = self._build_constraint_matrices()

            # Solve QP problem
            try:
                result = minimize(
                    fun=lambda u: 0.5 * u.T @ H @ u + f.T @ u,
                    x0=np.zeros(self.Nc * self.nu),
                    constraints={'type': 'ineq', 'fun': lambda u: w - G @ u},
                    method='SLSQP',
                    options={'maxiter': 100}
                )

                if result.success:
                    optimal_control = result.x[:self.nu]
                    cost = result.fun
                else:
                    print(f"QP optimization failed: {result.message}")
                    optimal_control = np.zeros(self.nu)
                    cost = float('inf')

            except Exception as e:
                print(f"Error in QP solution: {e}")
                optimal_control = np.zeros(self.nu)
                cost = float('inf')

            # Store history
            self.control_history.append(optimal_control)
            self.state_history.append(current_state)
            self.cost_history.append(cost)

            return optimal_control

        def _build_qp_matrices(self, current_state: np.ndarray,
                              reference: np.ndarray, disturbance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Build QP matrices H and f.
            """
            # Build block diagonal matrices
            Q_block = np.kron(np.eye(self.Np), self.Q)
            R_block = np.kron(np.eye(self.Nc), self.R)

            # Add terminal cost
            Q_block[-self.nx:, -self.nx:] = self.Qf

            # Hessian matrix
            H = self.Gamma_y.T @ Q_block @ self.Gamma_y + R_block

            # Gradient vector
            predicted_output = self.Phi_y @ current_state
            tracking_error = predicted_output - reference.flatten()
            f = self.Gamma_y.T @ Q_block @ tracking_error

            return H, f

        def _build_constraint_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
            """
            Build constraint matrices G and w.
            """
            constraints = []

            # Input constraints
            for i in range(self.Nc):
                for j in range(self.nu):
                    # u_min <= u[i,j]
                    row = np.zeros(self.Nc * self.nu)
                    row[i * self.nu + j] = -1
                    constraints.append((row, -self.u_min[j]))

                    # u[i,j] <= u_max
                    row = np.zeros(self.Nc * self.nu)
                    row[i * self.nu + j] = 1
                    constraints.append((row, self.u_max[j]))

            # State constraints (simplified)
            for i in range(self.Np):
                for j in range(self.nx):
                    # x_min <= x[i,j]
                    row = self.Gamma[i*self.nx+j, :]
                    constraints.append((row, self.x_min[j]))

                    # x[i,j] <= x_max
                    row = -self.Gamma[i*self.nx+j, :]
                    constraints.append((row, -self.x_max[j]))

            # Build constraint matrix and vector
            if constraints:
                G = np.vstack([c[0] for c in constraints])
                w = np.array([c[1] for c in constraints])
            else:
                G = np.zeros((0, self.Nc * self.nu))
                w = np.array([])

            return G, w

        def get_prediction_matrices(self) -> dict:
            """Get prediction matrices for analysis."""
            return {
                'Phi': self.Phi,
                'Gamma': self.Gamma,
                'Phi_y': self.Phi_y,
                'Gamma_y': self.Gamma_y
            }

        def get_control_history(self) -> np.ndarray:
            """Get control input history."""
            return np.array(self.control_history) if self.control_history else np.array([])

        def get_state_history(self) -> np.ndarray:
            """Get state history."""
            return np.array(self.state_history) if self.state_history else np.array([])

        def get_cost_history(self) -> np.ndarray:
            """Get cost history."""
            return np.array(self.cost_history) if self.cost_history else np.array([])

        def reset(self) -> None:
            """Reset controller state."""
            self.control_history.clear()
            self.state_history.clear()
            self.cost_history.clear()
    ```

=== "Fast Linear MPC with Explicit Solution (Advanced)"
    ```python
    class FastLinearMPCController(LinearMPCController):
        """
        Fast Linear MPC using explicit solution for unconstrained case.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            # Pre-compute explicit solution for unconstrained case
            self._compute_explicit_solution()

        def _compute_explicit_solution(self) -> None:
            """Pre-compute explicit solution matrices."""
            # For unconstrained case, the solution is: U = -H^(-1) * f
            # We can pre-compute H^(-1) for efficiency
            Q_block = np.kron(np.eye(self.Np), self.Q)
            R_block = np.kron(np.eye(self.Nc), self.R)
            Q_block[-self.nx:, -self.nx:] = self.Qf

            H = self.Gamma_y.T @ Q_block @ self.Gamma_y + R_block
            self.H_inv = np.linalg.inv(H)

            # Pre-compute constant part of solution
            self.K_const = -self.H_inv @ self.Gamma_y.T @ Q_block

        def compute_control_unconstrained(self, current_state: np.ndarray,
                                        reference_trajectory: np.ndarray) -> np.ndarray:
            """
            Compute control using explicit solution (unconstrained case).
            """
            # Prepare reference trajectory
            if len(reference_trajectory) < self.Np:
                ref_extended = np.tile(reference_trajectory[-1], (self.Np, 1))
                ref_extended[:len(reference_trajectory)] = reference_trajectory
            else:
                ref_extended = reference_trajectory[:self.Np]

            # Compute control using explicit solution
            predicted_output = self.Phi_y @ current_state
            tracking_error = predicted_output - ref_extended.flatten()

            U = self.K_const @ tracking_error
            optimal_control = U[:self.nu]

            # Store history
            self.control_history.append(optimal_control)
            self.state_history.append(current_state)

            return optimal_control
    ```

=== "Linear MPC with Soft Constraints"
    ```python
    class SoftConstraintLinearMPC(LinearMPCController):
        """
        Linear MPC with soft constraints using slack variables.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            # Soft constraint parameters
            self.soft_constraint_weight = 1e6
            self.slack_variables = True

        def _build_qp_matrices(self, current_state: np.ndarray,
                              reference: np.ndarray, disturbance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Build QP matrices with slack variables for soft constraints.
            """
            if not self.slack_variables:
                return super()._build_qp_matrices(current_state, reference, disturbance)

            # Add slack variables for soft constraints
            n_slack = self.Np * self.nx + self.Np * self.ny  # State and output constraints

            # Extended Hessian with slack variables
            H_base, f_base = super()._build_qp_matrices(current_state, reference, disturbance)

            # Extended Hessian: [H_base, 0; 0, W_slack]
            H_extended = np.block([
                [H_base, np.zeros((H_base.shape[0], n_slack))],
                [np.zeros((n_slack, H_base.shape[1])), self.soft_constraint_weight * np.eye(n_slack)]
            ])

            # Extended gradient: [f_base; 0]
            f_extended = np.concatenate([f_base, np.zeros(n_slack)])

            return H_extended, f_extended

        def _build_constraint_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
            """
            Build constraint matrices with slack variables.
            """
            if not self.slack_variables:
                return super()._build_constraint_matrices()

            # Build soft constraints using slack variables
            n_slack = self.Np * self.nx + self.Np * self.ny
            n_vars = self.Nc * self.nu + n_slack

            constraints = []

            # Input constraints (hard constraints)
            for i in range(self.Nc):
                for j in range(self.nu):
                    # u_min <= u[i,j]
                    row = np.zeros(n_vars)
                    row[i * self.nu + j] = -1
                    constraints.append((row, -self.u_min[j]))

                    # u[i,j] <= u_max
                    row = np.zeros(n_vars)
                    row[i * self.nu + j] = 1
                    constraints.append((row, self.u_max[j]))

            # State constraints (soft constraints with slack)
            for i in range(self.Np):
                for j in range(self.nx):
                    slack_idx = i * self.nx + j
                    # x_min <= x[i,j] + slack
                    row = np.zeros(n_vars)
                    row[self.Nc * self.nu + slack_idx] = 1  # Slack variable
                    row[:self.Nc * self.nu] = self.Gamma[i*self.nx+j, :]  # State constraint
                    constraints.append((row, self.x_min[j]))

                    # x[i,j] - slack <= x_max
                    row = np.zeros(n_vars)
                    row[self.Nc * self.nu + slack_idx] = -1  # Slack variable
                    row[:self.Nc * self.nu] = -self.Gamma[i*self.nx+j, :]  # State constraint
                    constraints.append((row, -self.x_max[j]))

            # Build constraint matrix and vector
            if constraints:
                G = np.vstack([c[0] for c in constraints])
                w = np.array([c[1] for c in constraints])
            else:
                G = np.zeros((0, n_vars))
                w = np.array([])

            return G, w
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/mpc/linear_mpc.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/mpc/linear_mpc.py)
    - **Tests**: [`tests/unit/mpc/test_linear_mpc.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/mpc/test_linear_mpc.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Basic Linear MPC** | O(N³) per time step | O(N²) for matrices | Standard QP solution |
    **Fast Linear MPC** | O(N²) per time step | O(N²) for matrices | Pre-computed solution |
    **Soft Constraint MPC** | O(N³) per time step | O(N²) for matrices | Additional slack variables |

!!! warning "Performance Considerations"
    - **Prediction matrices** can be pre-computed for efficiency
    - **QP solvers** significantly impact real-time performance
    - **Constraint handling** adds computational complexity
    - **Horizon length** affects both performance and computation time

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Industrial Process Control"
        - **Chemical Plants**: Temperature and pressure control
        - **Oil Refineries**: Distillation and separation control
        - **Power Plants**: Boiler and turbine control
        - **Water Treatment**: pH and chemical dosing control

    !!! grid-item "Automotive Systems"
        - **Engine Control**: Fuel injection and timing
        - **Vehicle Dynamics**: Trajectory tracking and stability
        - **Brake Systems**: ABS and traction control
        - **Steering Control**: Lane keeping and parking

    !!! grid-item "Aerospace & Defense"
        - **Flight Control**: Aircraft attitude and trajectory
        - **Missile Guidance**: Target tracking and guidance
        - **Satellite Control**: Orbit and attitude maintenance
        - **UAV Systems**: Autonomous navigation and control

    !!! grid-item "Robotics & Automation"
        - **Robot Manipulators**: Joint and end-effector control
        - **Mobile Robots**: Path following and obstacle avoidance
        - **Industrial Automation**: Production line control
        - **Exoskeletons**: Human-robot interaction control

!!! success "Educational Value"
    - **Linear Systems**: Understanding state-space representations
    - **Optimization**: Learning quadratic programming techniques
    - **Predictive Control**: Mastering prediction-based control design
    - **Constraint Handling**: Managing system limitations effectively

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Textbooks"
        1. **Rawlings, J. B., et al.** (2017). *Model Predictive Control: Theory, Computation, and Design*. Nob Hill.
        2. **Maciejowski, J. M.** (2002). *Predictive Control with Constraints*. Prentice Hall.

    !!! grid-item "Historical & Cultural"
        3. **Camacho, E. F., & Bordons, C.** (2007). *Model Predictive Control*. Springer.
        4. **Mayne, D. Q., et al.** (2000). Constrained model predictive control: Stability and optimality. *Automatica*, 36(6).

    !!! grid-item "Online Resources"
        5. [Linear MPC - Wikipedia](https://en.wikipedia.org/wiki/Model_predictive_control)
        6. [QP Solvers](https://www.mathworks.com/help/optim/ug/quadratic-programming.html)
        7. [MPC Applications](https://www.youtube.com/watch?v=example)

    !!! grid-item "Implementation & Practice"
        8. [Python QP Solvers](https://pypi.org/project/qpsolvers/)
        9. [MATLAB QP](https://www.mathworks.com/help/optim/ug/quadprog.html)
        10. [OSQP Solver](https://osqp.org/)

!!! tip "Interactive Learning"
    Try implementing Linear MPC yourself! Start with a simple second-order system, then implement the prediction matrices and QP formulation. Experiment with different prediction and control horizons to see their effects on performance and computational cost. Try implementing soft constraints using slack variables, and compare the performance with hard constraints. Implement the explicit solution for unconstrained cases to see the performance improvement. This will give you deep insight into how linearity enables efficient MPC implementation and how to design effective controllers for linear systems.

## Navigation

{{ nav_grid(current_algorithm="linear-mpc", current_family="mpc", max_related=5) }}
