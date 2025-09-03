---
tags: [control, algorithms, pid-control, feedback-control, proportional-integral-derivative, linear-control]
title: "PID Control"
family: "control"
complexity: "O(1) per time step"
---

# PID Control

!!! info "Algorithm Family"
    **Family:** [Control Algorithms](../../families/control.md)

!!! abstract "Overview"
    PID Control (Proportional-Integral-Derivative Control) is a fundamental feedback control algorithm that combines three control actions to achieve desired system behavior. The algorithm continuously calculates an error value as the difference between a desired setpoint and a measured process variable, then applies a correction based on proportional, integral, and derivative terms.

    This approach is widely used in industrial control systems, robotics, automotive applications, and many other domains where precise control is required. PID controllers are valued for their simplicity, effectiveness, and ability to handle a wide range of control problems with minimal tuning.

## Mathematical Formulation

!!! math "PID Control Law"
    The PID control law is given by:
    
    $$u(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{d}{dt} e(t)$$
    
    Where:
    - $u(t)$ is the control output at time $t$
    - $e(t) = r(t) - y(t)$ is the error (setpoint - process variable)
    - $K_p$ is the proportional gain
    - $K_i$ is the integral gain
    - $K_d$ is the derivative gain
    
    In discrete-time form with sampling period $T_s$:
    
    $$u_k = K_p e_k + K_i T_s \sum_{i=0}^k e_i + K_d \frac{e_k - e_{k-1}}{T_s}$$
    
    Where $k$ represents the discrete time step.

!!! success "Key Properties"
    - **Proportional Term**: Provides immediate response proportional to current error
    - **Integral Term**: Eliminates steady-state error by accumulating past errors
    - **Derivative Term**: Improves transient response and reduces overshoot
    - **Linear Control**: Simple linear combination of error terms
    - **Robust Performance**: Works well across many different systems

## Implementation Approaches

=== "Basic PID Controller (Recommended)"
    ```python
    import numpy as np
    from typing import Tuple, Optional
    
    class PIDController:
        """
        Basic PID Controller implementation.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            setpoint: Desired target value
            sample_time: Sampling time in seconds (default: 0.01)
            output_limits: Tuple of (min, max) output limits (default: None)
            anti_windup: Enable anti-windup for integral term (default: True)
        """
        
        def __init__(self, kp: float, ki: float, kd: float, setpoint: float = 0.0,
                     sample_time: float = 0.01, output_limits: Optional[Tuple[float, float]] = None,
                     anti_windup: bool = True):
            
            # PID gains
            self.kp = kp
            self.ki = ki
            self.kd = kd
            
            # Control parameters
            self.setpoint = setpoint
            self.sample_time = sample_time
            self.output_limits = output_limits
            self.anti_windup = anti_windup
            
            # State variables
            self.prev_error = 0.0
            self.integral = 0.0
            self.output = 0.0
            
            # Performance tracking
            self.error_history = []
            self.output_history = []
        
        def set_setpoint(self, setpoint: float) -> None:
            """Update the setpoint value."""
            self.setpoint = setpoint
        
        def compute(self, process_variable: float) -> float:
            """
            Compute the control output.
            
            Args:
                process_variable: Current measured value
                
            Returns:
                Control output value
            """
            # Calculate error
            error = self.setpoint - process_variable
            
            # Proportional term
            p_term = self.kp * error
            
            # Integral term
            self.integral += error * self.sample_time
            i_term = self.ki * self.integral
            
            # Derivative term
            derivative = (error - self.prev_error) / self.sample_time
            d_term = self.kd * derivative
            
            # Combine terms
            self.output = p_term + i_term + d_term
            
            # Apply output limits
            if self.output_limits is not None:
                min_output, max_output = self.output_limits
                self.output = np.clip(self.output, min_output, max_output)
                
                # Anti-windup: prevent integral from accumulating when output is limited
                if self.anti_windup:
                    if self.output == min_output or self.output == max_output:
                        self.integral = self.integral - error * self.sample_time
            
            # Update state for next iteration
            self.prev_error = error
            
            # Store history for analysis
            self.error_history.append(error)
            self.output_history.append(self.output)
            
            return self.output
        
        def reset(self) -> None:
            """Reset the controller state."""
            self.prev_error = 0.0
            self.integral = 0.0
            self.output = 0.0
            self.error_history.clear()
            self.output_history.clear()
        
        def get_performance_metrics(self) -> dict:
            """Calculate performance metrics."""
            if not self.error_history:
                return {}
            
            errors = np.array(self.error_history)
            
            metrics = {
                'steady_state_error': np.mean(errors[-10:]) if len(errors) >= 10 else np.mean(errors),
                'max_overshoot': np.max(errors) if np.max(errors) > 0 else 0,
                'settling_time': self._calculate_settling_time(errors),
                'rise_time': self._calculate_rise_time(errors),
                'integral_absolute_error': np.sum(np.abs(errors)) * self.sample_time,
                'integral_squared_error': np.sum(errors**2) * self.sample_time
            }
            
            return metrics
        
        def _calculate_settling_time(self, errors: np.ndarray, tolerance: float = 0.05) -> float:
            """Calculate settling time (time to reach within tolerance of setpoint)."""
            if len(errors) < 2:
                return 0.0
            
            # Find when error stays within tolerance
            within_tolerance = np.abs(errors) <= tolerance
            if not np.any(within_tolerance):
                return float('inf')
            
            # Find first time error enters tolerance and stays there
            for i in range(len(errors)):
                if np.all(within_tolerance[i:]):
                    return i * self.sample_time
            
            return float('inf')
        
        def _calculate_rise_time(self, errors: np.ndarray, threshold: float = 0.9) -> float:
            """Calculate rise time (time to reach threshold of final value)."""
            if len(errors) < 2:
                return 0.0
            
            # Find when error reaches threshold of setpoint
            threshold_value = threshold * self.setpoint
            for i, error in enumerate(errors):
                if abs(error) <= threshold_value:
                    return i * self.sample_time
            
            return float('inf')
    ```

=== "PID with Auto-tuning (Advanced)"
    ```python
    class AutoTuningPID(PIDController):
        """
        PID Controller with automatic tuning capabilities.
        """
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.tuning_mode = 'manual'
            self.relay_amplitude = 1.0
            self.critical_gain = 0.0
            self.critical_period = 0.0
        
        def relay_feedback_tuning(self, process_variable: float, relay_amplitude: float = 1.0) -> dict:
            """
            Perform relay feedback tuning to find critical gain and period.
            
            Args:
                process_variable: Current measured value
                relay_amplitude: Amplitude of relay signal
                
            Returns:
                Dictionary with tuning parameters
            """
            self.tuning_mode = 'relay_tuning'
            self.relay_amplitude = relay_amplitude
            
            # Simple relay feedback implementation
            # In practice, this would be more sophisticated
            
            # Simulate relay feedback response
            error = self.setpoint - process_variable
            
            if error > 0:
                output = relay_amplitude
            else:
                output = -relay_amplitude
            
            # Estimate critical parameters (simplified)
            self.critical_gain = 4 * relay_amplitude / (np.pi * abs(error)) if abs(error) > 0.01 else 1.0
            self.critical_period = 2 * np.pi / (2 * np.pi / 10)  # Simplified estimation
            
            # Apply Ziegler-Nichols tuning rules
            tuning_params = self._ziegler_nichols_tuning()
            
            self.tuning_mode = 'manual'
            return tuning_params
        
        def _ziegler_nichols_tuning(self) -> dict:
            """Apply Ziegler-Nichols tuning rules."""
            if self.critical_gain <= 0 or self.critical_period <= 0:
                return {}
            
            # Ziegler-Nichols tuning rules
            kp = 0.6 * self.critical_gain
            ki = 1.2 * self.critical_gain / self.critical_period
            kd = 0.075 * self.critical_gain * self.critical_period
            
            # Update controller gains
            self.kp = kp
            self.ki = ki
            self.kd = kd
            
            return {
                'kp': kp,
                'ki': ki,
                'kd': kd,
                'critical_gain': self.critical_gain,
                'critical_period': self.critical_period
            }
    ```

=== "PID with Filtering and Anti-windup"
    ```python
    class FilteredPID(PIDController):
        """
        PID Controller with filtering and advanced anti-windup.
        """
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            
            # Filtering parameters
            self.derivative_filter_alpha = 0.1  # Low-pass filter for derivative
            self.filtered_derivative = 0.0
            
            # Anti-windup parameters
            self.anti_windup_gain = 0.1
            self.saturation_error = 0.0
        
        def compute(self, process_variable: float) -> float:
            """Compute control output with filtering and anti-windup."""
            # Calculate error
            error = self.setpoint - process_variable
            
            # Proportional term
            p_term = self.kp * error
            
            # Integral term with anti-windup
            if self.output_limits is not None:
                min_output, max_output = self.output_limits
                if self.output == min_output or self.output == max_output:
                    self.saturation_error = self.output - (p_term + self.ki * self.integral + self.kd * self.filtered_derivative)
                else:
                    self.saturation_error = 0.0
            
            # Update integral with anti-windup correction
            self.integral += error * self.sample_time - self.anti_windup_gain * self.saturation_error
            i_term = self.ki * self.integral
            
            # Derivative term with filtering
            derivative = (error - self.prev_error) / self.sample_time
            self.filtered_derivative = (self.derivative_filter_alpha * derivative + 
                                      (1 - self.derivative_filter_alpha) * self.filtered_derivative)
            d_term = self.kd * self.filtered_derivative
            
            # Combine terms
            self.output = p_term + i_term + d_term
            
            # Apply output limits
            if self.output_limits is not None:
                min_output, max_output = self.output_limits
                self.output = np.clip(self.output, min_output, max_output)
            
            # Update state
            self.prev_error = error
            
            # Store history
            self.error_history.append(error)
            self.output_history.append(self.output)
            
            return self.output
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/control/pid_control.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/control/pid_control.py)
    - **Tests**: [`tests/unit/control/test_pid_control.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/control/test_pid_control.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Basic PID** | O(1) per time step | O(n) for history | Simple arithmetic operations |
    **Auto-tuning PID** | O(1) per time step | O(n) for history | Additional tuning calculations |
    **Filtered PID** | O(1) per time step | O(n) for history | Filter operations add minimal overhead |

!!! warning "Performance Considerations"
    - **Sampling rate** affects control performance and computational load
    - **Anti-windup** prevents integral term from saturating
    - **Filtering** reduces noise but adds phase lag
    - **Tuning parameters** significantly impact system response

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Industrial Control"
        - **Temperature Control**: Furnaces, reactors, HVAC systems
        - **Pressure Control**: Compressors, boilers, pipelines
        - **Flow Control**: Pumps, valves, chemical processes
        - **Level Control**: Tanks, reservoirs, storage systems

    !!! grid-item "Robotics & Automation"
        - **Position Control**: Robot arms, CNC machines, 3D printers
        - **Velocity Control**: Motors, drives, conveyor systems
        - **Force Control**: Grippers, assembly robots, haptic devices
        - **Trajectory Tracking**: Autonomous vehicles, drones, mobile robots

    !!! grid-item "Automotive & Aerospace"
        - **Engine Control**: Fuel injection, ignition timing, turbo control
        - **Flight Control**: Aircraft stability, missile guidance, drone navigation
        - **Brake Control**: ABS systems, traction control, stability control
        - **Climate Control**: Cabin temperature, humidity, air quality

    !!! grid-item "Consumer Electronics"
        - **Audio Systems**: Volume control, equalization, noise cancellation
        - **Display Control**: Brightness, contrast, color temperature
        - **Power Management**: Battery charging, voltage regulation, thermal management
        - **Sensor Fusion**: IMU calibration, GPS filtering, camera stabilization

!!! success "Educational Value"
    - **Control Theory Fundamentals**: Perfect introduction to feedback control
    - **System Dynamics**: Understanding how systems respond to inputs
    - **Tuning Methods**: Learning to optimize control parameters
    - **Real-world Applications**: Seeing control theory in practice

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Textbooks"
        1. **Åström, K. J., & Hägglund, T.** (2006). *Advanced PID Control*. ISA.
        2. **Ogata, K.** (2010). *Modern Control Engineering*. Pearson.

    !!! grid-item "Historical & Cultural"
        3. **Ziegler, J. G., & Nichols, N. B.** (1942). Optimum settings for automatic controllers. *Transactions of the ASME*, 64(8).
        4. **Callender, A., et al.** (1936). Time lag in a control system. *Philosophical Transactions of the Royal Society*, 235(756).

    !!! grid-item "Online Resources"
        5. [PID Controller - Wikipedia](https://en.wikipedia.org/wiki/PID_controller)
        6. [PID Tuning Guide](https://www.controleng.com/articles/pid-tuning-guide/)
        7. [Control Tutorials](https://ctms.engin.umich.edu/CTMS/index.php?example=Introduction&section=ControlPID)

    !!! grid-item "Implementation & Practice"
        8. [Python Control Library](https://python-control.readthedocs.io/)
        9. [MATLAB Control Toolbox](https://www.mathworks.com/help/control/)
        10. [Arduino PID Library](https://playground.arduino.cc/Code/PIDLibrary/)

!!! tip "Interactive Learning"
    Try implementing PID control yourself! Start with a simple first-order system simulation, then experiment with different gain values to see their effects on system response. Implement anti-windup to handle output saturation, and add filtering to reduce noise sensitivity. Try the Ziegler-Nichols tuning method to automatically find good parameters. This will give you deep insight into the fundamental principles of feedback control and how to tune controllers for optimal performance.
