---
algorithm_key: "adaptive-control"
tags: [control, algorithms, adaptive-control, parameter-estimation, self-tuning, model-reference]
title: "Adaptive Control"
family: "control"
---

{{ algorithm_card(config.extra.algorithm_key) }}

!!! abstract "Overview"
    Adaptive Control is a control strategy that automatically adjusts controller parameters based on real-time system identification and performance evaluation. Unlike fixed-parameter controllers like PID, adaptive controllers continuously learn and adapt to changing system dynamics, making them ideal for systems with unknown or time-varying parameters.

    This approach is particularly valuable in aerospace applications, robotics, process control, and other domains where system characteristics may change over time or are initially unknown. Adaptive controllers can handle parameter uncertainties, environmental changes, and system degradation while maintaining desired performance.

## Mathematical Formulation

!!! math "Adaptive Control Framework"
    The adaptive control system consists of two main components:
    
    **1. Parameter Estimation (System Identification):**
    
    $$\hat{\theta}(t) = \hat{\theta}(t-1) + \gamma(t) \phi(t) e(t)$$
    
    Where:
    - $\hat{\theta}(t)$ is the estimated parameter vector at time $t$
    - $\gamma(t)$ is the adaptation gain (learning rate)
    - $\phi(t)$ is the regressor vector
    - $e(t)$ is the prediction error
    
    **2. Control Law:**
    
    $$u(t) = \hat{\theta}^T(t) \phi(t) + K_p e(t) + K_i \int e(\tau) d\tau$$
    
    Where:
    - $u(t)$ is the control input
    - $K_p, K_i$ are fixed control gains
    - $e(t)$ is the tracking error
    
    **3. Lyapunov Stability Criterion:**
    
    $$V(t) = \frac{1}{2} e^2(t) + \frac{1}{2} \tilde{\theta}^T(t) \Gamma^{-1} \tilde{\theta}(t)$$
    
    Where $\tilde{\theta}(t) = \theta - \hat{\theta}(t)$ is the parameter estimation error.

!!! success "Key Properties"
    - **Online Learning**: Continuously updates parameters during operation
    - **Robustness**: Handles parameter uncertainties and system changes
    - **Self-tuning**: Automatically adjusts to maintain performance
    - **Convergence**: Parameters converge to true values under certain conditions
    - **Stability**: Lyapunov-based stability guarantees

## Implementation Approaches

=== "Basic Adaptive Controller (Recommended)"
    ```python
    import numpy as np
    from typing import List, Tuple, Optional
    from scipy.linalg import solve
    
    class AdaptiveController:
        """
        Basic Adaptive Controller implementation.
        
        Args:
            n_params: Number of parameters to estimate
            learning_rate: Adaptation gain (default: 0.1)
            forgetting_factor: Forgetting factor for RLS (default: 0.99)
            control_gains: Fixed control gains (Kp, Ki) (default: (1.0, 0.1))
        """
        
        def __init__(self, n_params: int, learning_rate: float = 0.1,
                     forgetting_factor: float = 0.99, control_gains: Tuple[float, float] = (1.0, 0.1)):
            
            self.n_params = n_params
            self.gamma = learning_rate
            self.lambda_ = forgetting_factor
            
            # Parameter estimation
            self.theta_hat = np.zeros(n_params)
            self.P = np.eye(n_params) * 1000  # Initial covariance
            
            # Control parameters
            self.kp, self.ki = control_gains
            self.integral_error = 0.0
            
            # History for analysis
            self.parameter_history = []
            self.control_history = []
            self.error_history = []
        
        def update_parameters(self, regressor: np.ndarray, output: float, 
                            predicted_output: float) -> None:
            """
            Update parameter estimates using recursive least squares.
            
            Args:
                regressor: Feature vector
                output: Actual system output
                predicted_output: Predicted output using current parameters
            """
            # Prediction error
            error = output - predicted_output
            
            # Update covariance matrix
            P_phi = self.P @ regressor
            denominator = self.lambda_ + regressor.T @ P_phi
            
            if denominator > 1e-10:  # Avoid division by zero
                # Kalman gain
                K = P_phi / denominator
                
                # Update parameter estimates
                self.theta_hat += K * error
                
                # Update covariance matrix
                self.P = (self.P - np.outer(K, P_phi)) / self.lambda_
            
            # Store history
            self.parameter_history.append(self.theta_hat.copy())
            self.error_history.append(error)
        
        def compute_control(self, setpoint: float, current_output: float, 
                          regressor: np.ndarray) -> float:
            """
            Compute control input using adaptive control law.
            
            Args:
                setpoint: Desired output value
                current_output: Current system output
                regressor: Feature vector for parameter estimation
                
            Returns:
                Control input value
            """
            # Tracking error
            error = setpoint - current_output
            
            # Update integral error
            self.integral_error += error
            
            # Adaptive control component
            adaptive_term = self.theta_hat.T @ regressor
            
            # Fixed control component
            fixed_term = self.kp * error + self.ki * self.integral_error
            
            # Total control input
            control_input = adaptive_term + fixed_term
            
            # Store control history
            self.control_history.append(control_input)
            
            return control_input
        
        def get_parameter_estimates(self) -> np.ndarray:
            """Get current parameter estimates."""
            return self.theta_hat.copy()
        
        def get_estimation_quality(self) -> dict:
            """Get quality metrics for parameter estimation."""
            if len(self.error_history) < 10:
                return {}
            
            recent_errors = self.error_history[-10:]
            
            metrics = {
                'mean_error': np.mean(recent_errors),
                'error_variance': np.var(recent_errors),
                'parameter_variance': np.trace(self.P),
                'convergence_rate': self._estimate_convergence_rate()
            }
            
            return metrics
        
        def _estimate_convergence_rate(self) -> float:
            """Estimate the rate of parameter convergence."""
            if len(self.parameter_history) < 20:
                return 0.0
            
            # Calculate parameter change over last 20 steps
            recent_params = np.array(self.parameter_history[-20:])
            param_changes = np.diff(recent_params, axis=0)
            
            # Average magnitude of parameter changes
            avg_change = np.mean(np.linalg.norm(param_changes, axis=1))
            
            return avg_change
        
        def reset(self) -> None:
            """Reset the controller state."""
            self.theta_hat = np.zeros(self.n_params)
            self.P = np.eye(self.n_params) * 1000
            self.integral_error = 0.0
            self.parameter_history.clear()
            self.control_history.clear()
            self.error_history.clear()
    ```

=== "Model Reference Adaptive Control (Advanced)"
    ```python
    class ModelReferenceAdaptiveController(AdaptiveController):
        """
        Model Reference Adaptive Controller (MRAC).
        """
        
        def __init__(self, n_params: int, reference_model: callable, **kwargs):
            super().__init__(n_params, **kwargs)
            self.reference_model = reference_model
            self.reference_output = 0.0
            self.model_error = 0.0
        
        def update_reference_model(self, time: float, input_signal: float) -> float:
            """
            Update reference model output.
            
            Args:
                time: Current time
                input_signal: Reference input signal
                
            Returns:
                Reference model output
            """
            self.reference_output = self.reference_model(time, input_signal)
            return self.reference_output
        
        def compute_control(self, setpoint: float, current_output: float, 
                          regressor: np.ndarray, time: float) -> float:
            """
            Compute control input using MRAC approach.
            """
            # Update reference model
            self.update_reference_model(time, setpoint)
            
            # Model following error
            self.model_error = self.reference_output - current_output
            
            # Adaptive control law
            adaptive_term = self.theta_hat.T @ regressor
            
            # Model reference control
            mrac_term = self.kp * self.model_error + self.ki * self.integral_error
            
            # Total control
            control_input = adaptive_term + mrac_term
            
            # Update integral error
            self.integral_error += self.model_error
            
            # Store history
            self.control_history.append(control_input)
            
            return control_input
    ```

=== "Self-Tuning Regulator"
    ```python
    class SelfTuningRegulator(AdaptiveController):
        """
        Self-Tuning Regulator with online system identification.
        """
        
        def __init__(self, n_params: int, **kwargs):
            super().__init__(n_params, **kwargs)
            
            # System model parameters
            self.A_params = np.zeros(n_params // 2)  # Denominator parameters
            self.B_params = np.zeros(n_params // 2)  # Numerator parameters
            
            # Prediction horizon
            self.prediction_horizon = 5
        
        def identify_system(self, input_history: List[float], 
                          output_history: List[float]) -> Tuple[np.ndarray, np.ndarray]:
            """
            Identify system parameters using least squares.
            
            Args:
                input_history: Recent input values
                output_history: Recent output values
                
            Returns:
                Tuple of (A_params, B_params)
            """
            if len(input_history) < self.n_params or len(output_history) < self.n_params:
                return self.A_params, self.B_params
            
            # Build regression matrix
            n = len(input_history)
            X = []
            y = []
            
            for i in range(self.n_params, n):
                # Output regression
                output_row = [-output_history[i-j] for j in range(1, self.n_params//2 + 1)]
                # Input regression
                input_row = [input_history[i-j] for j in range(1, self.n_params//2 + 1)]
                
                X.append(output_row + input_row)
                y.append(output_history[i])
            
            if len(X) < 2:
                return self.A_params, self.B_params
            
            X = np.array(X)
            y = np.array(y)
            
            try:
                # Solve least squares problem
                params = solve(X.T @ X, X.T @ y)
                
                # Split parameters
                n_a = self.n_params // 2
                self.A_params = params[:n_a]
                self.B_params = params[n_a:]
                
            except np.linalg.LinAlgError:
                # Handle singular matrix
                pass
            
            return self.A_params, self.B_params
        
        def predict_output(self, input_history: List[float], 
                         output_history: List[float]) -> float:
            """
            Predict next output using identified model.
            """
            if len(input_history) < self.n_params or len(output_history) < self.n_params:
                return output_history[-1] if output_history else 0.0
            
            # Use ARX model: y(k) = -a1*y(k-1) - a2*y(k-2) + b1*u(k-1) + b2*u(k-2)
            prediction = 0.0
            
            # Output terms
            for i, a in enumerate(self.A_params):
                if i < len(output_history):
                    prediction -= a * output_history[-(i+1)]
            
            # Input terms
            for i, b in enumerate(self.B_params):
                if i < len(input_history):
                    prediction += b * input_history[-(i+1)]
            
            return prediction
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/control/adaptive_control.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/control/adaptive_control.py)
    - **Tests**: [`tests/unit/control/test_adaptive_control.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/control/test_adaptive_control.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Basic Adaptive** | O(n²) per time step | O(n²) for covariance | RLS parameter estimation |
    **MRAC** | O(n²) per time step | O(n²) for covariance | Additional reference model |
    **Self-Tuning** | O(n³) per time step | O(n²) for regression | Online system identification |

!!! warning "Performance Considerations"
    - **Parameter convergence** depends on persistent excitation
    - **Computational complexity** grows with parameter dimension
    - **Stability** requires careful gain selection
    - **Initialization** affects convergence speed

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Aerospace & Defense"
        - **Flight Control**: Aircraft parameter adaptation, missile guidance
        - **Satellite Control**: Orbit maintenance, attitude control
        - **UAV Systems**: Autonomous navigation, payload stabilization
        - **Missile Systems**: Target tracking, guidance adaptation

    !!! grid-item "Robotics & Automation"
        - **Manipulator Control**: Robot arm dynamics, payload compensation
        - **Mobile Robots**: Terrain adaptation, wheel slip compensation
        - **Industrial Automation**: Process parameter changes, equipment wear
        - **Exoskeletons**: Human-robot interaction, load adaptation

    !!! grid-item "Process Control"
        - **Chemical Processes**: Reaction kinetics, catalyst deactivation
        - **Power Systems**: Load variations, generator dynamics
        - **Manufacturing**: Tool wear, material property changes
        - **Environmental Control**: Weather changes, seasonal variations

    !!! grid-item "Biomedical & Healthcare"
        - **Drug Delivery**: Patient response adaptation, pharmacokinetics
        - **Prosthetics**: User adaptation, terrain changes
        - **Rehabilitation**: Patient progress, therapy adjustment
        - **Medical Devices**: Patient-specific parameters, condition changes

!!! success "Educational Value"
    - **System Identification**: Learning to estimate unknown system parameters
    - **Adaptive Algorithms**: Understanding online learning and adaptation
    - **Stability Analysis**: Lyapunov theory and convergence guarantees
    - **Real-time Control**: Handling changing system dynamics

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Textbooks"
        1. **Åström, K. J., & Wittenmark, B.** (2013). *Adaptive Control*. Dover.
        2. **Ioannou, P. A., & Sun, J.** (2012). *Robust Adaptive Control*. Dover.

    !!! grid-item "Historical & Cultural"
        3. **Landau, Y. D.** (1979). Adaptive control: The model reference approach. *Marcel Dekker*.
        4. **Narendra, K. S., & Annaswamy, A. M.** (1989). *Stable Adaptive Systems*. Prentice Hall.

    !!! grid-item "Online Resources"
        5. [Adaptive Control - Wikipedia](https://en.wikipedia.org/wiki/Adaptive_control)
        6. [Model Reference Adaptive Control](https://www.mathworks.com/help/slcontrol/ug/model-reference-adaptive-control.html)
        7. [Adaptive Control Tutorial](https://www.youtube.com/watch?v=wE7xEgoWu2E)

    !!! grid-item "Implementation & Practice"
        8. [Python Control Library](https://python-control.readthedocs.io/)
        9. [MATLAB Adaptive Control Toolbox](https://www.mathworks.com/help/control/)
        10. [Simulink Adaptive Control](https://www.mathworks.com/help/slcontrol/)

!!! tip "Interactive Learning"
    Try implementing adaptive control yourself! Start with a simple first-order system with unknown parameters, then implement recursive least squares for parameter estimation. Add a reference model for MRAC implementation and experiment with different adaptation gains to see their effects on convergence. Implement a self-tuning regulator that identifies system parameters online. This will give you deep insight into how adaptive controllers learn and adapt to changing system dynamics while maintaining stability and performance.

## Navigation

{{ nav_grid(current_algorithm=config.extra.algorithm_key, current_family=config.extra.family, max_related=5) }}
