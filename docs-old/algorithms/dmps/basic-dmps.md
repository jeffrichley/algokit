---
algorithm_key: "basic-dmps"
tags: [dmps, algorithms, dynamic-movement-primitives, motor-control, robotics, trajectory-learning]
title: "Basic Dynamic Movement Primitives (DMPs)"
family: "dmps"
---

# Basic Dynamic Movement Primitives (DMPs)

{{ algorithm_card("basic-dmps") }}

!!! abstract "Overview"
    Dynamic Movement Primitives (DMPs) are a powerful framework for learning and reproducing complex motor behaviors in robotics. They provide a way to represent movements as dynamical systems that can be learned from demonstrations and then adapted to new situations while maintaining the essential characteristics of the original movement.

    The basic DMP formulation uses a second-order dynamical system with a forcing function that captures the shape of the movement. This allows for temporal and spatial scaling, goal adaptation, and smooth trajectory generation. DMPs are particularly effective for robotic applications where precise, smooth, and adaptable movements are required.

## Mathematical Formulation

!!! math "Basic DMP Dynamics"
    A basic DMP is defined by the following second-order dynamical system:
    
    $$\tau \dot{v} = \alpha_v (\beta_v (g - y) - v) + f(s)$$
    
    $$\tau \dot{y} = v$$
    
    Where:
    - $y$ is the position, $v$ is the velocity
    - $g$ is the goal position
    - $\tau$ is the temporal scaling factor
    - $\alpha_v$ and $\beta_v$ are damping parameters
    - $f(s)$ is the forcing function that captures the movement shape
    - $s$ is the phase variable that monotonically decreases from 1 to 0
    
    The phase variable follows:
    
    $$\tau \dot{s} = -\alpha_s s$$
    
    The forcing function is represented as:
    
    $$f(s) = \frac{\sum_{i=1}^N w_i \psi_i(s) s}{\sum_{i=1}^N \psi_i(s)}$$
    
    Where $\psi_i(s) = \exp(-h_i (s - c_i)^2)$ are Gaussian basis functions.

!!! success "Key Properties"
    - **Temporal Scaling**: Movements can be sped up or slowed down
    - **Spatial Scaling**: Movements can be scaled in amplitude
    - **Goal Adaptation**: Target positions can be changed
    - **Smooth Trajectories**: Guaranteed smooth and stable movements

## Implementation Approaches

=== "Basic DMP Implementation (Recommended)"
    ```python
    import numpy as np
    from scipy.integrate import odeint
    from typing import Tuple, List, Optional, Callable
    import matplotlib.pyplot as plt
    
    class BasicDMP:
        """
        Basic Dynamic Movement Primitives implementation.
        
        Args:
            n_basis: Number of basis functions
            alpha_v: Damping parameter for velocity
            beta_v: Damping parameter for position
            alpha_s: Phase variable decay rate
            dt: Time step for integration
        """
        
        def __init__(self, n_basis: int = 50, alpha_v: float = 25.0, beta_v: float = 6.25,
                     alpha_s: float = 1.0, dt: float = 0.01):
            self.n_basis = n_basis
            self.alpha_v = alpha_v
            self.beta_v = beta_v
            self.alpha_s = alpha_s
            self.dt = dt
            
            # Initialize basis functions
            self._initialize_basis_functions()
            
            # Weights for forcing function
            self.weights = np.zeros(n_basis)
            
            # Movement parameters
            self.y0 = 0.0  # Start position
            self.g = 1.0   # Goal position
            self.tau = 1.0 # Temporal scaling
            
            # Learned flag
            self.is_learned = False
        
        def _initialize_basis_functions(self):
            """Initialize Gaussian basis functions."""
            # Centers distributed uniformly in phase space
            self.centers = np.linspace(0, 1, self.n_basis)
            
            # Widths (heights) of basis functions
            self.widths = np.ones(self.n_basis) * 0.5
        
        def learn_from_demonstration(self, trajectory: np.ndarray, time: np.ndarray) -> 'BasicDMP':
            """
            Learn DMP parameters from a demonstrated trajectory.
            
            Args:
                trajectory: Demonstrated trajectory (n_samples,)
                time: Time vector (n_samples,)
                
            Returns:
                self
            """
            trajectory = np.asarray(trajectory)
            time = np.asarray(time)
            
            if len(trajectory) != len(time):
                raise ValueError("Trajectory and time must have the same length")
            
            # Set movement parameters
            self.y0 = trajectory[0]
            self.g = trajectory[-1]
            self.tau = time[-1] - time[0]
            
            # Compute velocities and accelerations
            dt = time[1] - time[0]
            velocity = np.gradient(trajectory, dt)
            acceleration = np.gradient(velocity, dt)
            
            # Compute phase variable
            phase = self._compute_phase(time)
            
            # Compute forcing function target
            forcing_target = self._compute_forcing_target(trajectory, velocity, acceleration, phase)
            
            # Learn weights using least squares
            self._learn_weights(phase, forcing_target)
            
            self.is_learned = True
            return self
        
        def _compute_phase(self, time: np.ndarray) -> np.ndarray:
            """Compute phase variable from time."""
            # Phase variable: s(t) = exp(-alpha_s * t / tau)
            phase = np.exp(-self.alpha_s * time / self.tau)
            return phase
        
        def _compute_forcing_target(self, trajectory: np.ndarray, velocity: np.ndarray, 
                                  acceleration: np.ndarray, phase: np.ndarray) -> np.ndarray:
            """Compute target forcing function from demonstration."""
            # From the DMP equation: f(s) = tau * a - alpha_v * (beta_v * (g - y) - v)
            forcing_target = (self.tau * acceleration - 
                            self.alpha_v * (self.beta_v * (self.g - trajectory) - velocity))
            return forcing_target
        
        def _learn_weights(self, phase: np.ndarray, forcing_target: np.ndarray):
            """Learn weights using least squares regression."""
            # Compute basis function activations
            basis_activations = self._compute_basis_activations(phase)
            
            # Normalize by sum of activations
            normalized_activations = basis_activations / (np.sum(basis_activations, axis=1, keepdims=True) + 1e-10)
            
            # Multiply by phase variable
            weighted_activations = normalized_activations * phase.reshape(-1, 1)
            
            # Solve least squares problem: Aw = f
            # where A is the weighted activations and f is the forcing target
            self.weights = np.linalg.lstsq(weighted_activations, forcing_target, rcond=None)[0]
        
        def _compute_basis_activations(self, phase: np.ndarray) -> np.ndarray:
            """Compute activations of all basis functions."""
            activations = np.zeros((len(phase), self.n_basis))
            
            for i in range(self.n_basis):
                # Gaussian activation
                activations[:, i] = np.exp(-self.widths[i] * (phase - self.centers[i])**2)
            
            return activations
        
        def generate_trajectory(self, duration: Optional[float] = None, 
                              goal: Optional[float] = None, 
                              start: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Generate trajectory using the learned DMP.
            
            Args:
                duration: Duration of the movement (if None, uses learned tau)
                goal: Goal position (if None, uses learned goal)
                start: Start position (if None, uses learned start)
                
            Returns:
                Tuple of (time, position, velocity)
            """
            if not self.is_learned:
                raise ValueError("DMP must be learned before generating trajectories")
            
            # Set parameters
            if duration is not None:
                self.tau = duration
            if goal is not None:
                self.g = goal
            if start is not None:
                self.y0 = start
            
            # Time vector
            time = np.arange(0, self.tau, self.dt)
            
            # Initial conditions
            y0 = self.y0
            v0 = 0.0
            s0 = 1.0
            
            # Integrate the DMP system
            initial_conditions = [y0, v0, s0]
            solution = odeint(self._dmp_dynamics, initial_conditions, time)
            
            position = solution[:, 0]
            velocity = solution[:, 1]
            phase = solution[:, 2]
            
            return time, position, velocity
        
        def _dmp_dynamics(self, state: List[float], t: float) -> List[float]:
            """DMP dynamics for integration."""
            y, v, s = state
            
            # Compute forcing function
            f = self._compute_forcing_function(s)
            
            # DMP equations
            dy_dt = v / self.tau
            dv_dt = (self.alpha_v * (self.beta_v * (self.g - y) - v) + f) / self.tau
            ds_dt = -self.alpha_s * s / self.tau
            
            return [dy_dt, dv_dt, ds_dt]
        
        def _compute_forcing_function(self, s: float) -> float:
            """Compute forcing function at given phase."""
            # Compute basis function activations
            activations = np.exp(-self.widths * (s - self.centers)**2)
            
            # Normalize
            normalized_activations = activations / (np.sum(activations) + 1e-10)
            
            # Weighted sum
            f = np.sum(self.weights * normalized_activations) * s
            
            return f
        
        def set_goal(self, goal: float):
            """Set new goal position."""
            self.g = goal
        
        def set_start(self, start: float):
            """Set new start position."""
            self.y0 = start
        
        def set_temporal_scaling(self, tau: float):
            """Set temporal scaling factor."""
            self.tau = tau
        
        def get_parameters(self) -> dict:
            """Get DMP parameters."""
            return {
                'weights': self.weights.copy(),
                'y0': self.y0,
                'g': self.g,
                'tau': self.tau,
                'alpha_v': self.alpha_v,
                'beta_v': self.beta_v,
                'alpha_s': self.alpha_s,
                'n_basis': self.n_basis
            }
        
        def set_parameters(self, params: dict):
            """Set DMP parameters."""
            self.weights = params['weights']
            self.y0 = params['y0']
            self.g = params['g']
            self.tau = params['tau']
            self.alpha_v = params['alpha_v']
            self.beta_v = params['beta_v']
            self.alpha_s = params['alpha_s']
            self.n_basis = params['n_basis']
            self.is_learned = True
    ```

=== "Multi-dimensional DMP"
    ```python
    class MultiDimensionalDMP:
        """
        Multi-dimensional DMP for joint space or Cartesian space movements.
        """
        
        def __init__(self, n_dims: int, n_basis: int = 50, alpha_v: float = 25.0, 
                     beta_v: float = 6.25, alpha_s: float = 1.0, dt: float = 0.01):
            self.n_dims = n_dims
            self.n_basis = n_basis
            self.alpha_v = alpha_v
            self.beta_v = beta_v
            self.alpha_s = alpha_s
            self.dt = dt
            
            # Initialize DMPs for each dimension
            self.dmps = [BasicDMP(n_basis, alpha_v, beta_v, alpha_s, dt) 
                        for _ in range(n_dims)]
            
            # Movement parameters
            self.y0 = np.zeros(n_dims)
            self.g = np.ones(n_dims)
            self.tau = 1.0
        
        def learn_from_demonstration(self, trajectory: np.ndarray, time: np.ndarray) -> 'MultiDimensionalDMP':
            """
            Learn multi-dimensional DMP from demonstration.
            
            Args:
                trajectory: Demonstrated trajectory (n_samples, n_dims)
                time: Time vector (n_samples,)
                
            Returns:
                self
            """
            trajectory = np.asarray(trajectory)
            time = np.asarray(time)
            
            if trajectory.shape[1] != self.n_dims:
                raise ValueError(f"Trajectory must have {self.n_dims} dimensions")
            
            # Set movement parameters
            self.y0 = trajectory[0]
            self.g = trajectory[-1]
            self.tau = time[-1] - time[0]
            
            # Learn each dimension independently
            for i in range(self.n_dims):
                self.dmps[i].learn_from_demonstration(trajectory[:, i], time)
            
            return self
        
        def generate_trajectory(self, duration: Optional[float] = None, 
                              goal: Optional[np.ndarray] = None, 
                              start: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Generate multi-dimensional trajectory.
            
            Args:
                duration: Duration of the movement
                goal: Goal position (n_dims,)
                start: Start position (n_dims,)
                
            Returns:
                Tuple of (time, position, velocity)
            """
            # Set parameters for all DMPs
            if duration is not None:
                self.tau = duration
            if goal is not None:
                self.g = goal
            if start is not None:
                self.y0 = start
            
            # Generate trajectory for each dimension
            trajectories = []
            velocities = []
            
            for i in range(self.n_dims):
                time, pos, vel = self.dmps[i].generate_trajectory(duration, goal[i], start[i])
                trajectories.append(pos)
                velocities.append(vel)
            
            # Stack trajectories
            position = np.column_stack(trajectories)
            velocity = np.column_stack(velocities)
            
            return time, position, velocity
        
        def set_goal(self, goal: np.ndarray):
            """Set new goal position."""
            self.g = goal
            for i in range(self.n_dims):
                self.dmps[i].set_goal(goal[i])
        
        def set_start(self, start: np.ndarray):
            """Set new start position."""
            self.y0 = start
            for i in range(self.n_dims):
                self.dmps[i].set_start(start[i])
        
        def set_temporal_scaling(self, tau: float):
            """Set temporal scaling factor."""
            self.tau = tau
            for i in range(self.n_dims):
                self.dmps[i].set_temporal_scaling(tau)
    ```

=== "DMP with Obstacle Avoidance"
    ```python
    class DMPWithObstacleAvoidance(BasicDMP):
        """
        DMP with obstacle avoidance capabilities.
        """
        
        def __init__(self, n_basis: int = 50, alpha_v: float = 25.0, beta_v: float = 6.25,
                     alpha_s: float = 1.0, dt: float = 0.01, gamma: float = 1000.0):
            super().__init__(n_basis, alpha_v, beta_v, alpha_s, dt)
            self.gamma = gamma  # Obstacle avoidance strength
            self.obstacles = []  # List of obstacle positions
        
        def add_obstacle(self, position: float, radius: float = 0.1):
            """Add an obstacle to avoid."""
            self.obstacles.append({'position': position, 'radius': radius})
        
        def remove_obstacle(self, index: int):
            """Remove obstacle by index."""
            if 0 <= index < len(self.obstacles):
                del self.obstacles[index]
        
        def clear_obstacles(self):
            """Clear all obstacles."""
            self.obstacles = []
        
        def _dmp_dynamics(self, state: List[float], t: float) -> List[float]:
            """DMP dynamics with obstacle avoidance."""
            y, v, s = state
            
            # Compute forcing function
            f = self._compute_forcing_function(s)
            
            # Compute obstacle avoidance force
            f_obs = self._compute_obstacle_avoidance_force(y, v)
            
            # DMP equations with obstacle avoidance
            dy_dt = v / self.tau
            dv_dt = (self.alpha_v * (self.beta_v * (self.g - y) - v) + f + f_obs) / self.tau
            ds_dt = -self.alpha_s * s / self.tau
            
            return [dy_dt, dv_dt, ds_dt]
        
        def _compute_obstacle_avoidance_force(self, y: float, v: float) -> float:
            """Compute obstacle avoidance force."""
            f_obs = 0.0
            
            for obstacle in self.obstacles:
                obs_pos = obstacle['position']
                obs_radius = obstacle['radius']
                
                # Distance to obstacle
                dist = abs(y - obs_pos)
                
                if dist < obs_radius:
                    # Repulsive force
                    direction = np.sign(y - obs_pos)
                    strength = self.gamma * (obs_radius - dist) / obs_radius
                    f_obs += direction * strength
            
            return f_obs
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/dmps/basic_dmps.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/dmps/basic_dmps.py)
    - **Tests**: [`tests/unit/dmps/test_basic_dmps.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/dmps/test_basic_dmps.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Basic DMP** | $O(N \cdot T)$ | $O(N)$ | N basis functions, T time steps |
    **Multi-dimensional DMP** | $O(D \cdot N \cdot T)$ | $O(D \cdot N)$ | D dimensions |
    **DMP with Obstacles** | $O(N \cdot T \cdot O)$ | $O(N + O)$ | O obstacles |

!!! warning "Performance Considerations"
    - **Basis function computation** can be expensive for large N
    - **Integration time** depends on movement duration
    - **Obstacle avoidance** adds computational overhead
    - **Memory usage** grows with number of basis functions

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Robotic Manipulation"
        - **Pick and Place**: Object manipulation tasks
        - **Assembly**: Precise part placement
        - **Welding**: Smooth welding trajectories
        - **Painting**: Artistic and industrial painting

    !!! grid-item "Human-Robot Interaction"
        - **Imitation Learning**: Learning from human demonstrations
        - **Collaborative Tasks**: Human-robot cooperation
        - **Assistive Robotics**: Helping with daily tasks
        - **Rehabilitation**: Physical therapy assistance

    !!! grid-item "Mobile Robotics"
        - **Navigation**: Smooth path following
        - **Formation Control**: Multi-robot coordination
        - **Search and Rescue**: Adaptive exploration
        - **Autonomous Vehicles**: Smooth driving behaviors

    !!! grid-item "Educational Value"
        - **Motor Control**: Understanding movement generation
        - **Learning from Demonstration**: Imitation learning concepts
        - **Dynamical Systems**: Understanding system dynamics
        - **Robotics**: Practical robotics applications

!!! success "Educational Value"
    - **Motor Control**: Perfect example of movement generation in robotics
    - **Learning from Demonstration**: Shows how to learn from examples
    - **Dynamical Systems**: Demonstrates stable system design
    - **Adaptation**: Illustrates how to adapt learned behaviors

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Papers"
        1. **Ijspeert, A. J., Nakanishi, J., & Schaal, S.** (2002). Movement imitation with nonlinear dynamical systems in humanoid robots. *ICRA*, 2, 1398-1403.
        2. **Schaal, S.** (2006). Dynamic movement primitives-a framework for motor control in humans and humanoid robotics. *Adaptive motion of animals and machines*, 261-280.

    !!! grid-item "Robotics Textbooks"
        3. **Siciliano, B., & Khatib, O.** (2016). *Springer Handbook of Robotics*. Springer.
        4. **Siciliano, B., Sciavicco, L., Villani, L., & Oriolo, G.** (2010). *Robotics: Modelling, Planning and Control*. Springer.

    !!! grid-item "Online Resources"
        5. [Dynamic Movement Primitives - Wikipedia](https://en.wikipedia.org/wiki/Dynamic_movement_primitives)
        6. [DMP Tutorial](https://www.cs.cmu.edu/~cga/dynopt/readings/Schaal06-DMP.pdf)
        7. [DMP Implementation](https://github.com/studywolf/pydmps)

    !!! grid-item "Implementation & Practice"
        8. [PyDMPs Library](https://github.com/studywolf/pydmps) - Python DMP implementation
        9. [DMPy Library](https://github.com/studywolf/dmpy) - Another Python DMP library
        10. [DMP Tutorial](https://www.cs.cmu.edu/~cga/dynopt/readings/Schaal06-DMP.pdf)

!!! tip "Interactive Learning"
    Try implementing Basic DMPs yourself! Start with simple 1D movements to understand how the algorithm works. Experiment with different numbers of basis functions to see how they affect the learned trajectory. Try implementing multi-dimensional DMPs to understand how to handle complex movements. Compare DMPs with other trajectory generation methods to see the benefits of the dynamical systems approach. This will give you deep insight into motor control and learning from demonstration.

## Navigation

{{ nav_grid(current_algorithm="basic-dmps", current_family="dmps", max_related=5) }}
