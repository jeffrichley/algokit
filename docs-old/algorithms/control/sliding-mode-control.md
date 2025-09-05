---
algorithm_key: "sliding-mode-control"
tags: [control, algorithms, sliding-mode-control, robust-control, variable-structure, chattering-reduction]
title: "Sliding Mode Control"
family: "control"
---

# Sliding Mode Control

{{ algorithm_card("sliding-mode-control") }}

!!! abstract "Overview"
    Sliding Mode Control (SMC) is a robust control strategy that forces the system state to reach and remain on a predefined sliding surface, regardless of parameter uncertainties and external disturbances. The controller switches between different control structures based on the system's position relative to this surface, creating a discontinuous control law that provides excellent robustness properties.

    This approach is particularly valuable in applications where system parameters are uncertain, external disturbances are present, or when high-precision control is required. Sliding mode controllers are widely used in aerospace systems, robotics, power electronics, and other domains where robustness is critical.

## Mathematical Formulation

!!! math "Sliding Mode Control Framework"
    **1. Sliding Surface Definition:**

    The sliding surface $s(x,t)$ is defined as:

    $$s(x,t) = \left( \frac{d}{dt} + \lambda \right)^{n-1} e(t)$$

    Where:
    - $e(t) = x_d(t) - x(t)$ is the tracking error
    - $x_d(t)$ is the desired trajectory
    - $x(t)$ is the current state
    - $\lambda > 0$ is a design parameter
    - $n$ is the system order

    **2. Sliding Mode Condition:**

    The sliding mode occurs when $s(x,t) = 0$, which implies:

    $$\dot{s}(x,t) = 0$$

    **3. Control Law:**

    The control input $u(t)$ consists of two components:

    $$u(t) = u_{eq}(t) + u_{sw}(t)$$

    Where:
    - $u_{eq}(t)$ is the equivalent control (continuous part)
    - $u_{sw}(t)$ is the switching control (discontinuous part)

    **4. Equivalent Control:**

    $$u_{eq}(t) = -B^{-1}(x) \left[ f(x) + \lambda \dot{e}(t) \right]$$

    **5. Switching Control:**

    $$u_{sw}(t) = -K \text{sign}(s(x,t))$$

    Where $K > 0$ is the switching gain and $\text{sign}(\cdot)$ is the signum function.

!!! success "Key Properties"
    - **Robustness**: Insensitive to parameter uncertainties and disturbances
    - **Finite-Time Convergence**: System reaches sliding surface in finite time
    - **Invariance**: Once on sliding surface, system becomes invariant to disturbances
    - **Chattering**: High-frequency switching can cause unwanted oscillations
    - **Design Flexibility**: Can handle nonlinear and time-varying systems

## Implementation Approaches

=== "Basic Sliding Mode Controller (Recommended)"
    ```python
    import numpy as np
    from typing import Tuple, Callable, Optional

    class SlidingModeController:
        """
        Basic Sliding Mode Controller implementation.

        Args:
            lambda_param: Sliding surface parameter (default: 1.0)
            switching_gain: Switching control gain (default: 10.0)
            boundary_layer: Boundary layer thickness for chattering reduction (default: 0.1)
            sampling_time: Controller sampling time (default: 0.01)
        """

        def __init__(self, lambda_param: float = 1.0, switching_gain: float = 10.0,
                     boundary_layer: float = 0.1, sampling_time: float = 0.01):

            self.lambda_param = lambda_param
            self.K = switching_gain
            self.boundary_layer = boundary_layer
            self.dt = sampling_time

            # State tracking
            self.prev_error = 0.0
            self.prev_error_derivative = 0.0

            # History for analysis
            self.sliding_surface_history = []
            self.control_history = []
            self.error_history = []

        def compute_sliding_surface(self, error: float, error_derivative: float) -> float:
            """
            Compute the sliding surface value.

            Args:
                error: Current tracking error
                error_derivative: Derivative of tracking error

            Returns:
                Sliding surface value
            """
            # For second-order system: s = e_dot + lambda * e
            sliding_surface = error_derivative + self.lambda_param * error
            return sliding_surface

        def compute_equivalent_control(self, desired_acceleration: float,
                                     system_dynamics: Callable, current_state: np.ndarray) -> float:
            """
            Compute the equivalent control component.

            Args:
                desired_acceleration: Desired second derivative
                system_dynamics: Function describing system dynamics
                current_state: Current system state

            Returns:
                Equivalent control value
            """
            # Simplified equivalent control calculation
            # In practice, this would use the actual system model
            u_eq = desired_acceleration - system_dynamics(current_state)
            return u_eq

        def compute_switching_control(self, sliding_surface: float) -> float:
            """
            Compute the switching control component with boundary layer.

            Args:
                sliding_surface: Current sliding surface value

            Returns:
                Switching control value
            """
            # Boundary layer approach to reduce chattering
            if abs(sliding_surface) <= self.boundary_layer:
                # Inside boundary layer: use saturation function
                u_sw = -self.K * (sliding_surface / self.boundary_layer)
            else:
                # Outside boundary layer: use signum function
                u_sw = -self.K * np.sign(sliding_surface)

            return u_sw

        def compute_control(self, setpoint: float, current_output: float,
                          current_derivative: float, system_dynamics: Callable,
                          current_state: np.ndarray) -> float:
            """
            Compute the total control input.

            Args:
                setpoint: Desired output value
                current_output: Current system output
                current_derivative: Current output derivative
                system_dynamics: System dynamics function
                current_state: Current system state

            Returns:
                Total control input
            """
            # Calculate tracking error and its derivative
            error = setpoint - current_output
            error_derivative = -current_derivative  # Assuming setpoint is constant

            # Compute sliding surface
            sliding_surface = self.compute_sliding_surface(error, error_derivative)

            # Compute equivalent control
            desired_acceleration = -self.lambda_param * error_derivative
            u_eq = self.compute_equivalent_control(desired_acceleration, system_dynamics, current_state)

            # Compute switching control
            u_sw = self.compute_switching_control(sliding_surface)

            # Total control input
            control_input = u_eq + u_sw

            # Store history
            self.sliding_surface_history.append(sliding_surface)
            self.control_history.append(control_input)
            self.error_history.append(error)

            # Update previous values
            self.prev_error = error
            self.prev_error_derivative = error_derivative

            return control_input

        def get_sliding_surface(self) -> float:
            """Get current sliding surface value."""
            if self.sliding_surface_history:
                return self.sliding_surface_history[-1]
            return 0.0

        def get_control_components(self) -> dict:
            """Get control components for analysis."""
            if not self.control_history:
                return {}

            return {
                'sliding_surface': self.get_sliding_surface(),
                'total_control': self.control_history[-1],
                'error': self.error_history[-1] if self.error_history else 0.0
            }

        def reset(self) -> None:
            """Reset the controller state."""
            self.prev_error = 0.0
            self.prev_error_derivative = 0.0
            self.sliding_surface_history.clear()
            self.control_history.clear()
            self.error_history.clear()
    ```

=== "Higher-Order Sliding Mode Controller (Advanced)"
    ```python
    class HigherOrderSlidingModeController(SlidingModeController):
        """
        Higher-Order Sliding Mode Controller for systems with relative degree > 1.
        """

        def __init__(self, relative_degree: int = 2, **kwargs):
            super().__init__(**kwargs)
            self.relative_degree = relative_degree

            # Higher-order sliding surface parameters
            self.alpha = 1.0
            self.beta = 1.0

            # State history for higher-order derivatives
            self.error_history = []
            self.error_derivatives = []

        def compute_higher_order_surface(self, error: float, error_derivatives: list) -> float:
            """
            Compute higher-order sliding surface.

            Args:
                error: Current tracking error
                error_derivatives: List of error derivatives

            Returns:
                Higher-order sliding surface value
            """
            if len(error_derivatives) < self.relative_degree - 1:
                return error

            # For relative degree 2: s = e_ddot + alpha * e_dot + beta * e
            if self.relative_degree == 2:
                e_dot = error_derivatives[0]
                return error_derivatives[1] + self.alpha * e_dot + self.beta * error

            # For relative degree 3: s = e_ddot_dot + alpha * e_ddot + beta * e_dot + gamma * e
            elif self.relative_degree == 3:
                e_dot = error_derivatives[0]
                e_ddot = error_derivatives[1]
                return error_derivatives[2] + self.alpha * e_ddot + self.beta * e_dot + self.beta * error

            # Default to basic sliding surface
            return super().compute_sliding_surface(error, error_derivatives[0] if error_derivatives else 0.0)

        def compute_super_twisting_control(self, sliding_surface: float,
                                        sliding_surface_derivative: float) -> float:
            """
            Implement super-twisting algorithm for chattering reduction.

            Args:
                sliding_surface: Current sliding surface value
                sliding_surface_derivative: Derivative of sliding surface

            Returns:
                Super-twisting control value
            """
            # Super-twisting parameters
            k1 = 1.5
            k2 = 1.1

            # Super-twisting control law
            u_st = -k1 * np.sign(sliding_surface) * np.sqrt(abs(sliding_surface)) - k2 * np.sign(sliding_surface_derivative)

            return u_st
    ```

=== "Adaptive Sliding Mode Controller"
    ```python
    class AdaptiveSlidingModeController(SlidingModeController):
        """
        Sliding Mode Controller with adaptive switching gain.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            # Adaptive parameters
            self.adaptive_gain = self.K
            self.adaptation_rate = 0.1
            self.min_gain = 1.0
            self.max_gain = 100.0

            # Performance tracking
            self.surface_norm_history = []

        def adapt_switching_gain(self, sliding_surface: float,
                               surface_derivative: float) -> None:
            """
            Adaptively adjust the switching gain.

            Args:
                sliding_surface: Current sliding surface value
                surface_derivative: Derivative of sliding surface
            """
            # Surface norm for adaptation
            surface_norm = np.sqrt(sliding_surface**2 + surface_derivative**2)
            self.surface_norm_history.append(surface_norm)

            # Adaptive law: increase gain when surface norm is large
            if len(self.surface_norm_history) >= 10:
                recent_norm = np.mean(self.surface_norm_history[-10:])

                # Adaptation rule
                if recent_norm > 0.1:  # Threshold for adaptation
                    self.adaptive_gain += self.adaptation_rate * recent_norm
                else:
                    self.adaptive_gain -= self.adaptation_rate * 0.01  # Gradual decrease

                # Bound the adaptive gain
                self.adaptive_gain = np.clip(self.adaptive_gain, self.min_gain, self.max_gain)

        def compute_switching_control(self, sliding_surface: float) -> float:
            """Compute switching control with adaptive gain."""
            # Use adaptive gain instead of fixed gain
            K_adaptive = self.adaptive_gain

            # Boundary layer approach
            if abs(sliding_surface) <= self.boundary_layer:
                u_sw = -K_adaptive * (sliding_surface / self.boundary_layer)
            else:
                u_sw = -K_adaptive * np.sign(sliding_surface)

            return u_sw

        def compute_control(self, setpoint: float, current_output: float,
                          current_derivative: float, system_dynamics: Callable,
                          current_state: np.ndarray) -> float:
            """Compute control with adaptive gain adjustment."""
            # Calculate error and sliding surface
            error = setpoint - current_output
            error_derivative = -current_derivative

            sliding_surface = self.compute_sliding_surface(error, error_derivative)

            # Adapt switching gain
            self.adapt_switching_gain(sliding_surface, error_derivative)

            # Compute control components
            desired_acceleration = -self.lambda_param * error_derivative
            u_eq = self.compute_equivalent_control(desired_acceleration, system_dynamics, current_state)
            u_sw = self.compute_switching_control(sliding_surface)

            # Total control
            control_input = u_eq + u_sw

            # Store history
            self.sliding_surface_history.append(sliding_surface)
            self.control_history.append(control_input)
            self.error_history.append(error)

            return control_input
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/control/sliding_mode_control.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/control/sliding_mode_control.py)
    - **Tests**: [`tests/unit/control/test_sliding_mode_control.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/control/test_sliding_mode_control.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Basic SMC** | O(n) per time step | O(n) for state history | Simple sliding surface computation |
    **Higher-Order SMC** | O(n²) per time step | O(n²) for derivatives | Higher-order surface computation |
    **Adaptive SMC** | O(n) per time step | O(n) for adaptation history | Additional gain adaptation |

!!! warning "Performance Considerations"
    - **Chattering reduction** is crucial for practical implementation
    - **Switching gain selection** affects robustness and performance
    - **Boundary layer thickness** trades off chattering vs. tracking accuracy
    - **Sampling rate** must be high enough to handle switching

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Aerospace & Defense"
        - **Missile Guidance**: Robust tracking under aerodynamic uncertainties
        - **Satellite Control**: Attitude control with parameter variations
        - **Aircraft Control**: Flight control with wind disturbances
        - **Spacecraft Docking**: Precise position control in space

    !!! grid-item "Robotics & Automation"
        - **Robot Manipulators**: Trajectory tracking with payload changes
        - **Mobile Robots**: Path following on uncertain terrain
        - **Industrial Automation**: Process control with parameter drift
        - **Exoskeletons**: Human-robot interaction with varying loads

    !!! grid-item "Power Electronics"
        - **Motor Drives**: Speed control with load variations
        - **Power Converters**: Voltage regulation with load changes
        - **Inverters**: Current control with grid disturbances
        - **Battery Management**: Charging control with aging effects

    !!! grid-item "Automotive & Transportation"
        - **Engine Control**: Robust control under varying conditions
        - **Brake Systems**: ABS control with road surface changes
        - **Steering Control**: Lane keeping with road conditions
        - **Electric Vehicles**: Motor control with battery variations

!!! success "Educational Value"
    - **Robust Control Theory**: Understanding disturbance rejection
    - **Nonlinear Control**: Handling system nonlinearities
    - **Variable Structure Systems**: Learning about switching control
    - **Chattering Analysis**: Understanding control signal quality

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Textbooks"
        1. **Utkin, V. I.** (1992). *Sliding Modes in Control and Optimization*. Springer.
        2. **Edwards, C., & Spurgeon, S.** (1998). *Sliding Mode Control: Theory and Applications*. CRC Press.

    !!! grid-item "Historical & Cultural"
        3. **Utkin, V. I.** (1977). Variable structure systems with sliding modes. *IEEE Transactions on Automatic Control*, 22(2).
        4. **Emelyanov, S. V.** (1967). *Variable Structure Control Systems*. Nauka.

    !!! grid-item "Online Resources"
        5. [Sliding Mode Control - Wikipedia](https://en.wikipedia.org/wiki/Sliding_mode_control)
        6. [SMC Tutorial](https://www.mathworks.com/help/slcontrol/ug/sliding-mode-control.html)
        7. [Chattering Reduction Methods](https://www.researchgate.net/publication/2245766)

    !!! grid-item "Implementation & Practice"
        8. [Python Control Library](https://python-control.readthedocs.io/)
        9. [MATLAB Sliding Mode Control](https://www.mathworks.com/help/slcontrol/)
        10. [Simulink SMC Examples](https://www.mathworks.com/help/slcontrol/examples/)

!!! tip "Interactive Learning"
    Try implementing sliding mode control yourself! Start with a simple second-order system like a mass-spring-damper, then implement the basic sliding surface and switching control. Add boundary layer methods to reduce chattering, and experiment with different switching gains to see their effects on robustness. Implement higher-order sliding mode control for systems with relative degree > 1, and try adaptive switching gains for automatic tuning. This will give you deep insight into the power of variable structure control and how to design robust controllers for uncertain systems.

## Navigation

{{ nav_grid(current_algorithm="sliding-mode-control", current_family="control", max_related=5) }}
