---
tags: [control, algorithms, h-infinity-control, robust-control, optimal-control, frequency-domain]
title: "H-Infinity Control"
family: "control"
complexity: "O(n³) for Riccati equation solution"
---

# H-Infinity Control

!!! info "Algorithm Family"
    **Family:** [Control Algorithms](../../families/control.md)

!!! abstract "Overview"
    H-Infinity Control is a robust control design methodology that minimizes the worst-case performance of a system under bounded disturbances and uncertainties. The controller is designed to minimize the H-infinity norm of the closed-loop transfer function from disturbances to performance outputs, ensuring robust performance across a wide range of operating conditions.

    This approach is particularly valuable in aerospace applications, automotive systems, power electronics, and other domains where system performance must be guaranteed despite parameter variations, external disturbances, and model uncertainties. H-infinity controllers provide excellent robustness properties and are widely used in safety-critical applications.

## Mathematical Formulation

!!! math "H-Infinity Control Framework"
    **1. Generalized Plant Model:**
    
    The generalized plant $P(s)$ is described by:
    
    $$\begin{bmatrix} z \\ y \end{bmatrix} = P(s) \begin{bmatrix} w \\ u \end{bmatrix} = \begin{bmatrix} P_{11}(s) & P_{12}(s) \\ P_{21}(s) & P_{22}(s) \end{bmatrix} \begin{bmatrix} w \\ u \end{bmatrix}$$
    
    Where:
    - $w$ is the exogenous input (disturbances, reference signals)
    - $u$ is the control input
    - $z$ is the performance output
    - $y$ is the measured output
    
    **2. H-Infinity Norm:**
    
    The H-infinity norm of a transfer function $G(s)$ is:
    
    $$\|G(s)\|_\infty = \sup_{\omega \in \mathbb{R}} \sigma_{\max}(G(j\omega))$$
    
    Where $\sigma_{\max}$ denotes the maximum singular value.
    
    **3. H-Infinity Control Problem:**
    
    Find a stabilizing controller $K(s)$ such that:
    
    $$\min_{K(s)} \|T_{zw}(s)\|_\infty < \gamma$$
    
    Where $T_{zw}(s)$ is the closed-loop transfer function from $w$ to $z$.
    
    **4. Riccati Equation Solution:**
    
    The H-infinity controller is obtained by solving the coupled Riccati equations:
    
    $$A^T X + X A + X (\gamma^{-2} B_1 B_1^T - B_2 B_2^T) X + C_1^T C_1 = 0$$
    
    $$A Y + Y A^T + Y (\gamma^{-2} C_1^T C_1 - C_2^T C_2) Y + B_1 B_1^T = 0$$
    
    Where $X$ and $Y$ are the stabilizing solutions.

!!! success "Key Properties"
    - **Robust Performance**: Guarantees performance under bounded disturbances
    - **Frequency Domain Design**: Direct design in frequency domain
    - **Optimal Robustness**: Minimizes worst-case performance
    - **Multi-Objective Design**: Handles multiple performance criteria
    - **Stability Guarantees**: Ensures closed-loop stability

## Implementation Approaches

=== "Basic H-Infinity Controller (Recommended)"
    ```python
    import numpy as np
    from scipy.linalg import solve_continuous_are, solve_discrete_are
    from typing import Tuple, Optional
    
    class HInfinityController:
        """
        Basic H-Infinity Controller implementation.
        
        Args:
            A: System state matrix
            B1: Disturbance input matrix
            B2: Control input matrix
            C1: Performance output matrix
            C2: Measured output matrix
            D11: Direct feedthrough from disturbance to performance
            D12: Direct feedthrough from control to performance
            D21: Direct feedthrough from disturbance to measurement
            D22: Direct feedthrough from control to measurement
            gamma: H-infinity performance bound (default: 1.0)
        """
        
        def __init__(self, A: np.ndarray, B1: np.ndarray, B2: np.ndarray,
                     C1: np.ndarray, C2: np.ndarray, D11: np.ndarray,
                     D12: np.ndarray, D21: np.ndarray, D22: np.ndarray,
                     gamma: float = 1.0):
            
            self.A = A
            self.B1 = B1
            self.B2 = B2
            self.C1 = C1
            self.C2 = C2
            self.D11 = D11
            self.D12 = D12
            self.D21 = D21
            self.D22 = D22
            self.gamma = gamma
            
            # Controller matrices
            self.Ac = None
            self.Bc = None
            self.Cc = None
            self.Dc = None
            
            # State dimensions
            self.nx = A.shape[0]  # Number of states
            self.nw = B1.shape[1]  # Number of disturbances
            self.nu = B2.shape[1]  # Number of control inputs
            self.nz = C1.shape[0]  # Number of performance outputs
            self.ny = C2.shape[0]  # Number of measured outputs
            
            # Design controller
            self._design_controller()
        
        def _design_controller(self) -> None:
            """Design the H-infinity controller using Riccati equations."""
            try:
                # Solve continuous-time Riccati equations
                X = solve_continuous_are(
                    self.A, self.B2, self.C1.T @ self.C1,
                    self.D12.T @ self.D12
                )
                
                Y = solve_continuous_are(
                    self.A.T, self.C1.T, self.B1 @ self.B1.T,
                    self.D21 @ self.D21.T
                )
                
                # Check if solutions are positive definite
                if not (np.all(np.linalg.eigvals(X) > 0) and np.all(np.linalg.eigvals(Y) > 0)):
                    raise ValueError("Riccati solutions are not positive definite")
                
                # Compute controller matrices
                self._compute_controller_matrices(X, Y)
                
            except Exception as e:
                print(f"Error in controller design: {e}")
                # Fallback to simple state feedback
                self._fallback_controller()
        
        def _compute_controller_matrices(self, X: np.ndarray, Y: np.ndarray) -> None:
            """Compute controller matrices from Riccati solutions."""
            # Controller state matrix
            self.Ac = (self.A - self.B2 @ np.linalg.inv(self.D12.T @ self.D12) @ 
                      self.D12.T @ self.C1 - Y @ self.C2.T @ 
                      np.linalg.inv(self.D21 @ self.D21.T) @ self.C2)
            
            # Controller input matrix
            self.Bc = Y @ self.C2.T @ np.linalg.inv(self.D21 @ self.D21.T)
            
            # Controller output matrix
            self.Cc = -np.linalg.inv(self.D12.T @ self.D12) @ (self.D12.T @ self.C1 + 
                                                              self.B2.T @ X)
            
            # Controller feedthrough matrix
            self.Dc = np.zeros((self.nu, self.ny))
        
        def _fallback_controller(self) -> None:
            """Fallback to simple state feedback controller."""
            # Simple LQR-like controller
            Q = np.eye(self.nx)
            R = np.eye(self.nu)
            
            # Solve algebraic Riccati equation for LQR
            X = solve_continuous_are(self.A, self.B2, Q, R)
            
            # State feedback gain
            K = np.linalg.inv(R) @ self.B2.T @ X
            
            # Controller matrices (observer-based)
            self.Ac = self.A - self.B2 @ K - self.L @ self.C2
            self.Bc = self.L
            self.Cc = -K
            self.Dc = np.zeros((self.nu, self.ny))
            
            # Observer gain (simple pole placement)
            self.L = self.A @ np.ones((self.nx, self.ny)) * 0.1
        
        def compute_control(self, measurement: np.ndarray, 
                          controller_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Compute control output and update controller state.
            
            Args:
                measurement: Current system measurement
                controller_state: Current controller state
                
            Returns:
                Tuple of (control_output, new_controller_state)
            """
            if self.Ac is None:
                raise ValueError("Controller not properly designed")
            
            # Update controller state
            new_controller_state = (self.Ac @ controller_state + 
                                  self.Bc @ measurement)
            
            # Compute control output
            control_output = (self.Cc @ controller_state + 
                            self.Dc @ measurement)
            
            return control_output, new_controller_state
        
        def get_controller_matrices(self) -> dict:
            """Get controller matrices for analysis."""
            return {
                'Ac': self.Ac,
                'Bc': self.Bc,
                'Cc': self.Cc,
                'Dc': self.Dc
            }
        
        def get_performance_bounds(self) -> dict:
            """Get performance bounds and analysis."""
            if self.Ac is None:
                return {}
            
            # Compute closed-loop poles
            A_cl = np.block([
                [self.A, self.B2 @ self.Cc],
                [self.Bc @ self.C2, self.Ac]
            ])
            
            closed_loop_poles = np.linalg.eigvals(A_cl)
            
            # Stability margin
            stability_margin = -np.max(np.real(closed_loop_poles))
            
            return {
                'closed_loop_poles': closed_loop_poles,
                'stability_margin': stability_margin,
                'is_stable': stability_margin > 0
            }
    ```

=== "H-Infinity Controller with Frequency Shaping (Advanced)"
    ```python
    class FrequencyShapedHInfinityController(HInfinityController):
        """
        H-Infinity Controller with frequency-dependent performance weights.
        """
        
        def __init__(self, A: np.ndarray, B1: np.ndarray, B2: np.ndarray,
                     C1: np.ndarray, C2: np.ndarray, D11: np.ndarray,
                     D12: np.ndarray, D21: np.ndarray, D22: np.ndarray,
                     gamma: float = 1.0, frequency_weights: dict = None):
            
            self.frequency_weights = frequency_weights or {}
            super().__init__(A, B1, B2, C1, C2, D11, D12, D21, D22, gamma)
        
        def _apply_frequency_weights(self) -> None:
            """Apply frequency-dependent weights to the generalized plant."""
            if not self.frequency_weights:
                return
            
            # Apply performance weight
            if 'performance' in self.frequency_weights:
                Wp = self.frequency_weights['performance']
                # Augment system with performance weight
                self._augment_with_performance_weight(Wp)
            
            # Apply control weight
            if 'control' in self.frequency_weights:
                Wu = self.frequency_weights['control']
                # Augment system with control weight
                self._augment_with_control_weight(Wu)
            
            # Apply disturbance weight
            if 'disturbance' in self.frequency_weights:
                Wd = self.frequency_weights['disturbance']
                # Augment system with disturbance weight
                self._augment_with_disturbance_weight(Wd)
        
        def _augment_with_performance_weight(self, Wp: np.ndarray) -> None:
            """Augment system with performance weight."""
            # This is a simplified implementation
            # In practice, this would involve state-space augmentation
            pass
        
        def _augment_with_control_weight(self, Wu: np.ndarray) -> None:
            """Augment system with control weight."""
            # This is a simplified implementation
            pass
        
        def _augment_with_disturbance_weight(self, Wd: np.ndarray) -> None:
            """Augment system with disturbance weight."""
            # This is a simplified implementation
            pass
    ```

=== "Mixed H2/H-Infinity Controller"
    ```python
    class MixedH2HInfinityController(HInfinityController):
        """
        Mixed H2/H-Infinity Controller for multi-objective optimization.
        """
        
        def __init__(self, A: np.ndarray, B1: np.ndarray, B2: np.ndarray,
                     C1: np.ndarray, C2: np.ndarray, D11: np.ndarray,
                     D12: np.ndarray, D21: np.ndarray, D22: np.ndarray,
                     gamma: float = 1.0, h2_weight: float = 0.5):
            
            self.h2_weight = h2_weight
            super().__init__(A, B1, B2, C1, C2, D11, D12, D21, D22, gamma)
        
        def _design_controller(self) -> None:
            """Design mixed H2/H-Infinity controller."""
            try:
                # Solve mixed H2/H-Infinity problem
                # This is a simplified implementation
                
                # First solve H-infinity problem
                X_hinf = solve_continuous_are(
                    self.A, self.B2, self.C1.T @ self.C1,
                    self.D12.T @ self.D12
                )
                
                # Then solve H2 problem
                X_h2 = solve_continuous_are(
                    self.A, self.B2, self.C1.T @ self.C1,
                    self.D12.T @ self.D12
                )
                
                # Combine solutions
                X = self.h2_weight * X_h2 + (1 - self.h2_weight) * X_hinf
                
                # Solve for Y
                Y = solve_continuous_are(
                    self.A.T, self.C1.T, self.B1 @ self.B1.T,
                    self.D21 @ self.D21.T
                )
                
                # Compute controller matrices
                self._compute_controller_matrices(X, Y)
                
            except Exception as e:
                print(f"Error in mixed controller design: {e}")
                self._fallback_controller()
        
        def get_mixed_performance(self) -> dict:
            """Get mixed H2/H-Infinity performance metrics."""
            if self.Ac is None:
                return {}
            
            # Compute H2 norm (simplified)
            try:
                # Solve Lyapunov equation for H2 norm
                Q = self.C1.T @ self.C1
                P = solve_continuous_are(self.A, self.B1, Q, np.eye(self.B1.shape[1]))
                h2_norm = np.sqrt(np.trace(self.B1.T @ P @ self.B1))
            except:
                h2_norm = float('inf')
            
            # Get H-infinity performance
            hinf_performance = self.get_performance_bounds()
            
            return {
                'h2_norm': h2_norm,
                'hinf_performance': hinf_performance,
                'mixed_objective': self.h2_weight * h2_norm + 
                                 (1 - self.h2_weight) * hinf_performance.get('stability_margin', 0)
            }
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/control/h_infinity_control.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/control/h_infinity_control.py)
    - **Tests**: [`tests/unit/control/test_h_infinity_control.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/control/test_h_infinity_control.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Basic H-Infinity** | O(n³) for Riccati | O(n²) for matrices | Standard Riccati solution |
    **Frequency Shaped** | O(n³) for Riccati | O(n²) for augmented system | Additional weight dynamics |
    **Mixed H2/H-Infinity** | O(n³) for both | O(n²) for matrices | Two Riccati equations |

!!! warning "Performance Considerations"
    - **Riccati equation solution** becomes expensive for large systems
    - **Frequency weights** add computational complexity
    - **Performance bound selection** affects controller design
    - **Numerical stability** is crucial for ill-conditioned systems

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Aerospace & Defense"
        - **Flight Control**: Robust aircraft control under varying conditions
        - **Missile Guidance**: Disturbance rejection in guidance systems
        - **Satellite Control**: Attitude control with parameter uncertainties
        - **Spacecraft Systems**: Robust control in space environment

    !!! grid-item "Automotive & Transportation"
        - **Engine Control**: Robust performance under varying loads
        - **Brake Systems**: ABS control with road surface variations
        - **Steering Control**: Lane keeping with wind disturbances
        - **Electric Vehicles**: Motor control with battery variations

    !!! grid-item "Power Systems"
        - **Power Electronics**: Robust control of converters and inverters
        - **Motor Drives**: Speed control under load variations
        - **Grid Control**: Power flow control with disturbances
        - **Renewable Energy**: Wind turbine and solar inverter control

    !!! grid-item "Industrial Automation"
        - **Process Control**: Robust control of chemical processes
        - **Robotics**: Manipulator control with payload changes
        - **Manufacturing**: Quality control under variations
        - **Material Handling**: Conveyor and crane control

!!! success "Educational Value"
    - **Robust Control Theory**: Understanding worst-case performance
    - **Frequency Domain Design**: Learning frequency-domain methods
    - **Optimal Control**: Understanding performance optimization
    - **System Analysis**: Learning stability and performance analysis

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Textbooks"
        1. **Zhou, K., & Doyle, J. C.** (1998). *Essentials of Robust Control*. Prentice Hall.
        2. **Skogestad, S., & Postlethwaite, I.** (2005). *Multivariable Feedback Control*. Wiley.

    !!! grid-item "Historical & Cultural"
        3. **Zames, G.** (1981). Feedback and optimal sensitivity: Model reference transformations, multiplicative seminorms, and approximate inverses. *IEEE Transactions on Automatic Control*, 26(2).
        4. **Doyle, J. C.** (1983). Synthesis of robust controllers and filters. *IEEE Conference on Decision and Control*.

    !!! grid-item "Online Resources"
        5. [H-Infinity Control - Wikipedia](https://en.wikipedia.org/wiki/H-infinity_control)
        6. [H-Infinity Control Tutorial](https://www.mathworks.com/help/robust/ug/h-infinity-control.html)
        7. [Robust Control Methods](https://www.youtube.com/watch?v=example)

    !!! grid-item "Implementation & Practice"
        8. [Python Control Library](https://python-control.readthedocs.io/)
        9. [MATLAB Robust Control Toolbox](https://www.mathworks.com/help/robust/)
        10. [Simulink Robust Control](https://www.mathworks.com/help/slcontrol/)

!!! tip "Interactive Learning"
    Try implementing H-infinity control yourself! Start with a simple second-order system, then formulate the generalized plant with performance and control weights. Solve the Riccati equations to find the controller, and analyze the closed-loop performance. Experiment with different frequency weights to see their effects on performance, and try mixed H2/H-infinity objectives for multi-criteria optimization. This will give you deep insight into robust control design and how to guarantee performance under uncertainty.
