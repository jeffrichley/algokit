---
algorithm_key: "distributed-mpc"
tags: [mpc, algorithms, distributed-mpc, multi-agent-control, coordination, consensus, distributed-optimization]
title: "Distributed MPC"
family: "mpc"
---

# Distributed MPC

{{ algorithm_card("distributed-mpc") }}

!!! abstract "Overview"
    Distributed MPC extends the predictive control framework to handle large-scale systems composed of multiple interconnected subsystems. Instead of solving a centralized optimization problem, Distributed MPC decomposes the system into smaller, manageable subsystems and coordinates their control actions through communication and consensus mechanisms.

    This approach is essential in applications such as smart grids, transportation networks, large-scale manufacturing systems, and multi-robot coordination where centralized control becomes computationally intractable or communication constraints limit centralized solutions. Distributed MPC provides scalability, robustness, and fault tolerance while maintaining performance through coordinated control actions.

## Mathematical Formulation

!!! math "Distributed MPC Framework"
    **1. Multi-Subsystem System Model:**

    The system is decomposed into $M$ subsystems:

    $$x_i(k+1) = f_i(x_i(k), u_i(k), \sum_{j \in \mathcal{N}_i} H_{ij} x_j(k), d_i(k))$$
    $$y_i(k) = h_i(x_i(k), u_i(k))$$

    Where:
    - $x_i(k) \in \mathbb{R}^{n_{x_i}}$ is the state vector of subsystem $i$
    - $u_i(k) \in \mathbb{R}^{n_{u_i}}$ is the control input vector of subsystem $i$
    - $\mathcal{N}_i$ is the set of neighbors of subsystem $i$
    - $H_{ij}$ is the coupling matrix between subsystems $i$ and $j$
    - $d_i(k)$ is the local disturbance vector

    **2. Distributed Optimization Problem:**

    Each subsystem $i$ solves:

    $$\min_{U_i} J_i(x_i(k), U_i, \hat{U}_{-i}) = \sum_{t=0}^{N_p-1} L_i(x_i(k+t|k), u_i(k+t), \hat{x}_{-i}(k+t|k))$$

    Subject to:
    - $x_i(k+t+1|k) = f_i(x_i(k+t|k), u_i(k+t), \sum_{j \in \mathcal{N}_i} H_{ij} \hat{x}_j(k+t|k), d_i(k+t))$
    - $g_i(x_i(k+t|k), u_i(k+t)) \leq 0$ (local constraints)
    - $u_{i,min} \leq u_i(k+t) \leq u_{i,max}$
    - $x_{i,min} \leq x_i(k+t|k) \leq x_{i,max}$

    Where:
    - $U_i = [u_i(k), u_i(k+1), ..., u_i(k+N_c-1)]$ is the local control sequence
    - $\hat{U}_{-i}$ represents estimated control sequences from other subsystems
    - $\hat{x}_{-i}$ represents estimated states from neighboring subsystems

    **3. Coordination Mechanisms:**

    **Consensus-based coordination:**
    $$\hat{U}_i^{(l+1)} = \hat{U}_i^{(l)} + \alpha \sum_{j \in \mathcal{N}_i} (\hat{U}_j^{(l)} - \hat{U}_i^{(l)})$$

    **Gauss-Seidel coordination:**
    $$\hat{U}_i^{(l+1)} = \arg\min_{U_i} J_i(x_i(k), U_i, \hat{U}_{-i}^{(l+1)})$$

!!! success "Key Properties"
    - **Scalability**: Handles large-scale systems through decomposition
    - **Parallel Computation**: Subsystems can solve problems concurrently
    - **Local Information**: Each subsystem uses primarily local information
    - **Coordination**: Maintains system-wide performance through coordination
    - **Fault Tolerance**: System continues operating if some subsystems fail

## Implementation Approaches

=== "Basic Distributed MPC Controller (Recommended)"
    ```python
    import numpy as np
    from scipy.optimize import minimize
    from typing import Callable, List, Dict, Tuple, Optional
    import threading
    import time

    class DistributedMPCController:
        """
        Basic Distributed MPC Controller implementation.

        Args:
            subsystem_id: Unique identifier for this subsystem
            prediction_horizon: Number of prediction steps
            control_horizon: Number of control steps
            state_dim: Dimension of local state vector
            input_dim: Dimension of local input vector
            output_dim: Dimension of local output vector
            Q: Local state weight matrix
            R: Local input weight matrix
            Qf: Local terminal state weight matrix
        """

        def __init__(self, subsystem_id: int, prediction_horizon: int, control_horizon: int,
                     state_dim: int, input_dim: int, output_dim: int,
                     Q: np.ndarray = None, R: np.ndarray = None, Qf: np.ndarray = None):

            self.subsystem_id = subsystem_id
            self.Np = prediction_horizon
            self.Nc = min(control_horizon, prediction_horizon)
            self.nx = state_dim
            self.nu = input_dim
            self.ny = output_dim

            # Weighting matrices
            self.Q = Q if Q is not None else np.eye(state_dim)
            self.R = R if R is not None else np.eye(input_dim)
            self.Qf = Qf if Qf is not None else np.eye(state_dim)

            # System model functions
            self.f = None  # Local state update function
            self.h = None  # Local output function

            # Neighbor information
            self.neighbors = []  # List of neighbor subsystem IDs
            self.coupling_matrices = {}  # H_ij matrices for each neighbor

            # Coordination parameters
            self.coordination_iterations = 5
            self.consensus_parameter = 0.1

            # Estimated neighbor states and controls
            self.estimated_neighbor_states = {}
            self.estimated_neighbor_controls = {}

            # Local constraints
            self.u_min = -np.inf * np.ones(input_dim)
            self.u_max = np.inf * np.ones(input_dim)
            self.x_min = -np.inf * np.ones(state_dim)
            self.x_max = np.inf * np.ones(state_dim)

            # History
            self.control_history = []
            self.state_history = []
            self.cost_history = []
            self.coordination_history = []

        def set_system_model(self, state_update_func: Callable,
                           output_func: Callable) -> None:
            """
            Set the local system model functions.
            """
            self.f = state_update_func
            self.h = output_func

        def add_neighbor(self, neighbor_id: int, coupling_matrix: np.ndarray) -> None:
            """
            Add a neighboring subsystem.

            Args:
                neighbor_id: ID of the neighboring subsystem
                coupling_matrix: Coupling matrix H_ij
            """
            self.neighbors.append(neighbor_id)
            self.coupling_matrices[neighbor_id] = coupling_matrix

            # Initialize estimated states and controls
            self.estimated_neighbor_states[neighbor_id] = np.zeros(self.nx)
            self.estimated_neighbor_controls[neighbor_id] = np.zeros(self.nu)

        def set_constraints(self, u_min: np.ndarray = None, u_max: np.ndarray = None,
                          x_min: np.ndarray = None, x_max: np.ndarray = None) -> None:
            """
            Set local input and state constraints.
            """
            if u_min is not None:
                self.u_min = np.array(u_min)
            if u_max is not None:
                self.u_max = np.array(u_max)
            if x_min is not None:
                self.x_min = np.array(x_min)
            if x_max is not None:
                self.x_max = np.array(x_max)

        def update_neighbor_estimates(self, neighbor_id: int,
                                    estimated_state: np.ndarray,
                                    estimated_control: np.ndarray) -> None:
            """
            Update estimates from a neighboring subsystem.
            """
            if neighbor_id in self.neighbors:
                self.estimated_neighbor_states[neighbor_id] = estimated_state
                self.estimated_neighbor_controls[neighbor_id] = estimated_control

        def compute_control(self, current_state: np.ndarray,
                          reference_trajectory: np.ndarray = None,
                          current_disturbance: np.ndarray = None) -> np.ndarray:
            """
            Compute optimal control input using Distributed MPC.
            """
            if self.f is None or self.h is None:
                raise ValueError("System model not set")

            # Initial guess for local control sequence
            u0 = np.zeros(self.Nc * self.nu)

            # Bounds for optimization
            bounds = []
            for i in range(self.Nc):
                for j in range(self.nu):
                    bounds.append((self.u_min[j], self.u_max[j]))

            # Coordination loop
            local_control = u0
            coordination_costs = []

            for coord_iter in range(self.coordination_iterations):
                # Solve local optimization problem
                result = minimize(
                    fun=lambda u: self._local_objective_function(u, current_state,
                                                               reference_trajectory, current_disturbance),
                    x0=local_control,
                    bounds=bounds,
                    method='SLSQP',
                    options={'maxiter': 100, 'ftol': 1e-6}
                )

                if result.success:
                    local_control = result.x
                    coordination_costs.append(result.fun)
                else:
                    print(f"Local optimization failed at iteration {coord_iter}: {result.message}")
                    break

                # Update neighbor estimates (simplified coordination)
                self._update_coordination()

            # Extract first control input
            optimal_control = local_control[:self.nu]

            # Store history
            self.control_history.append(optimal_control)
            self.state_history.append(current_state)
            self.cost_history.append(coordination_costs[-1] if coordination_costs else float('inf'))
            self.coordination_history.append(len(coordination_costs))

            return optimal_control

        def _local_objective_function(self, u: np.ndarray, current_state: np.ndarray,
                                     reference: np.ndarray, disturbance: np.ndarray = None) -> float:
            """
            Compute local objective function for Distributed MPC.
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

                # Compute coupling effect from neighbors
                coupling_effect = np.zeros_like(current_state)
                for neighbor_id in self.neighbors:
                    if neighbor_id in self.estimated_neighbor_states:
                        neighbor_state = self.estimated_neighbor_states[neighbor_id]
                        coupling_matrix = self.coupling_matrices[neighbor_id]
                        coupling_effect += coupling_matrix @ neighbor_state

                # Predict next state with coupling
                if disturbance is not None and i < len(disturbance):
                    d_i = disturbance[i]
                else:
                    d_i = np.zeros_like(current_state)

                x_next = self.f(x, u_i, coupling_effect, d_i)

                # Predict output
                y_i = self.h(x, u_i)

                # Local tracking cost
                if reference is not None and i < len(reference):
                    ref_i = reference[i]
                else:
                    ref_i = np.zeros_like(y_i)

                tracking_error = y_i - ref_i
                cost += tracking_error.T @ self.Q @ tracking_error

                # Local control cost
                if i < self.Nc:
                    cost += u_i.T @ self.R @ u_i

                # Local state constraints penalty
                if np.any(x < self.x_min) or np.any(x > self.x_max):
                    cost += 1e6

                # Update state
                x = x_next

            # Terminal cost
            if reference is not None:
                terminal_error = x - reference[-1]
            else:
                terminal_error = x
            cost += terminal_error.T @ self.Qf @ terminal_error

            return cost

        def _update_coordination(self) -> None:
            """
            Update coordination with neighbors (simplified implementation).
            """
            # In a real implementation, this would involve communication
            # with neighboring subsystems to exchange state and control estimates

            # For now, we'll use a simple consensus update
            for neighbor_id in self.neighbors:
                if neighbor_id in self.estimated_neighbor_states:
                    # Simple consensus update
                    current_estimate = self.estimated_neighbor_states[neighbor_id]
                    # This would normally come from the neighbor
                    # For simulation, we'll add some noise
                    noise = np.random.normal(0, 0.01, current_estimate.shape)
                    self.estimated_neighbor_states[neighbor_id] = current_estimate + noise

        def get_control_history(self) -> np.ndarray:
            """Get local control input history."""
            return np.array(self.control_history) if self.control_history else np.array([])

        def get_state_history(self) -> np.ndarray:
            """Get local state history."""
            return np.array(self.state_history) if self.state_history else np.array([])

        def get_cost_history(self) -> np.ndarray:
            """Get local cost history."""
            return np.array(self.cost_history) if self.cost_history else np.array([])

        def get_coordination_history(self) -> np.ndarray:
            """Get coordination iteration history."""
            return np.array(self.coordination_history) if self.coordination_history else np.array([])

        def reset(self) -> None:
            """Reset controller state."""
            self.control_history.clear()
            self.state_history.clear()
            self.cost_history.clear()
            self.coordination_history.clear()
    ```

=== "Consensus-Based Distributed MPC (Advanced)"
    ```python
    class ConsensusDistributedMPC(DistributedMPCController):
        """
        Distributed MPC using consensus-based coordination.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            # Consensus parameters
            self.consensus_weight = 0.5
            self.max_consensus_iterations = 10

        def _update_coordination(self) -> None:
            """
            Update coordination using consensus algorithm.
            """
            # Consensus update for neighbor estimates
            for neighbor_id in self.neighbors:
                if neighbor_id in self.estimated_neighbor_states:
                    current_estimate = self.estimated_neighbor_states[neighbor_id]

                    # Consensus update rule
                    # In practice, this would be the actual neighbor's estimate
                    # For simulation, we'll use a consensus-like update
                    consensus_update = self.consensus_parameter * (np.random.normal(0, 0.1, current_estimate.shape))
                    self.estimated_neighbor_states[neighbor_id] = current_estimate + consensus_update
    ```

=== "Parallel Distributed MPC System"
    ```python
    class ParallelDistributedMPCSystem:
        """
        System coordinator for parallel Distributed MPC.
        """

        def __init__(self, num_subsystems: int):
            self.num_subsystems = num_subsystems
            self.subsystems = {}
            self.coordination_threads = {}
            self.running = False

        def add_subsystem(self, subsystem_id: int, controller: DistributedMPCController) -> None:
            """
            Add a subsystem to the distributed system.
            """
            self.subsystems[subsystem_id] = controller

        def start_coordination(self) -> None:
            """
            Start the coordination process.
            """
            self.running = True

            # Start coordination threads for each subsystem
            for subsystem_id, controller in self.subsystems.items():
                thread = threading.Thread(target=self._coordination_loop,
                                       args=(subsystem_id, controller))
                thread.daemon = True
                thread.start()
                self.coordination_threads[subsystem_id] = thread

        def stop_coordination(self) -> None:
            """
            Stop the coordination process.
            """
            self.running = False

            # Wait for all threads to finish
            for thread in self.coordination_threads.values():
                thread.join()

        def _coordination_loop(self, subsystem_id: int, controller: DistributedMPCController) -> None:
            """
            Coordination loop for a single subsystem.
            """
            while self.running:
                # Simulate coordination updates
                time.sleep(0.1)  # Coordination frequency

                # In practice, this would involve actual communication
                # between subsystems to exchange estimates

                # For simulation, we'll just update the coordination
                controller._update_coordination()

        def compute_all_controls(self, states: Dict[int, np.ndarray],
                               references: Dict[int, np.ndarray] = None) -> Dict[int, np.ndarray]:
            """
            Compute controls for all subsystems in parallel.
            """
            controls = {}

            # Create threads for parallel computation
            threads = {}
            results = {}

            for subsystem_id, controller in self.subsystems.items():
                state = states.get(subsystem_id, np.zeros(controller.nx))
                reference = references.get(subsystem_id, None) if references else None

                thread = threading.Thread(target=self._compute_subsystem_control,
                                       args=(subsystem_id, controller, state, reference, results))
                thread.start()
                threads[subsystem_id] = thread

            # Wait for all threads to complete
            for thread in threads.values():
                thread.join()

            # Collect results
            for subsystem_id in self.subsystems:
                controls[subsystem_id] = results[subsystem_id]

            return controls

        def _compute_subsystem_control(self, subsystem_id: int, controller: DistributedMPCController,
                                     state: np.ndarray, reference: np.ndarray, results: dict) -> None:
            """
            Compute control for a single subsystem.
            """
            control = controller.compute_control(state, reference)
            results[subsystem_id] = control
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/mpc/distributed_mpc.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/mpc/distributed_mpc.py)
    - **Tests**: [`tests/unit/mpc/test_distributed_mpc.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/mpc/test_distributed_mpc.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Basic Distributed MPC** | O(N³ × M) | O(N² × M) | M subsystems, sequential coordination |
    **Consensus-Based** | O(N³ × M × C) | O(N² × M) | C consensus iterations |
    **Parallel Distributed** | O(N³) per subsystem | O(N² × M) | Parallel computation, communication overhead |

!!! warning "Performance Considerations"
    - **Communication overhead** can limit scalability
    - **Coordination iterations** affect convergence time
    - **Coupling strength** impacts coordination requirements
    - **Network topology** affects coordination efficiency

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Smart Grids & Power Systems"
        - **Microgrid Control**: Coordinate multiple energy sources
        - **Load Management**: Balance demand across distribution networks
        - **Renewable Integration**: Coordinate solar and wind generation
        - **Energy Storage**: Optimize battery and thermal storage

    !!! grid-item "Transportation Networks"
        - **Traffic Signal Control**: Coordinate intersections
        - **Autonomous Vehicle Coordination**: Multi-vehicle path planning
        - **Public Transportation**: Bus and train scheduling
        - **Logistics Networks**: Supply chain coordination

    !!! grid-item "Manufacturing Systems"
        - **Production Line Control**: Coordinate multiple workstations
        - **Supply Chain Management**: Inventory and production coordination
        - **Quality Control**: Multi-stage inspection coordination
        - **Maintenance Scheduling**: Equipment maintenance coordination

    !!! grid-item "Multi-Robot Systems"
        - **Swarm Robotics**: Coordinated movement and task allocation
        - **Industrial Automation**: Multi-robot assembly coordination
        - **Search and Rescue**: Coordinated exploration and mapping
        - **Agricultural Robotics**: Field monitoring and intervention

!!! success "Educational Value"
    - **System Decomposition**: Understanding how to break down large systems
    - **Coordination Mechanisms**: Learning consensus and coordination algorithms
    - **Parallel Computation**: Understanding concurrent optimization
    - **Network Effects**: Learning how coupling affects coordination

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Textbooks"
        1. **Rawlings, J. B., et al.** (2017). *Model Predictive Control: Theory, Computation, and Design*. Nob Hill.
        2. **Scattolini, R.** (2009). Architectures for distributed and hierarchical Model Predictive Control. *Journal of Process Control*.

    !!! grid-item "Historical & Cultural"
        3. **Stewart, B. T., et al.** (2010). Cooperative distributed model predictive control. *Systems & Control Letters*, 59(8).
        4. **Maestre, J. M., et al.** (2014). Distributed model predictive control: A tutorial review and future research directions. *Computers & Chemical Engineering*.

    !!! grid-item "Online Resources"
        5. [Distributed MPC - Wikipedia](https://en.wikipedia.org/wiki/Model_predictive_control)
        6. [Multi-Agent Systems](https://www.mathworks.com/help/mpc/)
        7. [Distributed Control](https://www.youtube.com/watch?v=example)

    !!! grid-item "Implementation & Practice"
        8. [Python Distributed MPC](https://pypi.org/project/distributed-mpc/)
        9. [MATLAB Distributed MPC](https://www.mathworks.com/help/mpc/)
        10. [Multi-Agent Simulation](https://www.anylogic.com/)

!!! tip "Interactive Learning"
    Try implementing Distributed MPC yourself! Start with a simple two-subsystem system, then implement the basic coordination loop. Experiment with different coordination mechanisms (consensus, Gauss-Seidel) and see how they affect convergence. Try implementing parallel computation for multiple subsystems and compare performance with sequential coordination. Add coupling between subsystems and see how it affects the coordination requirements. This will give you deep insight into how to design scalable control systems that can handle large-scale applications through decomposition and coordination.

## Navigation

{{ nav_grid(current_algorithm="distributed-mpc", current_family="mpc", max_related=5) }}
