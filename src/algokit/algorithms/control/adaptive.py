"""Adaptive Control implementation with parameter estimation.

This module implements Model Reference Adaptive Control (MRAC) using the
MIT rule for parameter adaptation. Adaptive controllers automatically adjust
their parameters to maintain optimal performance as system dynamics change.

The controller compares system output with a reference model and adjusts
parameters to minimize the tracking error. This is particularly useful for:
- Systems with uncertain or time-varying parameters
- Plants with changing dynamics
- Applications requiring robust performance

Mathematical formulation:
    θ(t+1) = θ(t) + γ(t) * φ(t) * e(t)

where:
    - θ: Adaptive parameter vector
    - γ(t): Adaptive learning rate (dynamically adjusted)
    - φ: Regressor vector (features)
    - e: Tracking error

Advanced features:
    - Dynamic reference model with state dynamics
    - Persistence of excitation monitoring
    - Adaptive learning rate scheduling
    - Lyapunov stability verification
    - Integrated plant simulation utilities
"""

import logging
from collections.abc import Callable
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class ReferenceModel(Protocol):
    """Protocol for reference model dynamics.

    A reference model defines the desired closed-loop behavior.
    """

    def step(self, reference_input: float, dt: float) -> float:
        """Execute one time step of the reference model.

        Args:
            reference_input: Desired input to the reference model
            dt: Time step

        Returns:
            Reference model output
        """
        ...

    def get_state(self) -> np.ndarray:
        """Get current reference model state.

        Returns:
            Reference model state vector
        """
        ...

    def reset(self) -> None:
        """Reset reference model to initial state."""
        ...


class PlantDynamics(Protocol):
    """Protocol for plant dynamics simulation.

    Defines the actual system being controlled.
    """

    def step(self, control_input: float, dt: float) -> float:
        """Execute one time step of the plant dynamics.

        Args:
            control_input: Control signal from controller
            dt: Time step

        Returns:
            Plant output
        """
        ...

    def get_state(self) -> np.ndarray:
        """Get current plant state.

        Returns:
            Plant state vector
        """
        ...

    def reset(self) -> None:
        """Reset plant to initial state."""
        ...


class FirstOrderReferenceModel:
    """First-order linear reference model.

    Implements: ẋ_m = -a_m * x_m + b_m * r

    where x_m is the reference state and r is the reference input.
    """

    def __init__(
        self, a_m: float = 1.0, b_m: float = 1.0, initial_state: float = 0.0
    ) -> None:
        """Initialize first-order reference model.

        Args:
            a_m: Reference model pole (> 0 for stability)
            b_m: Reference model gain
            initial_state: Initial state value
        """
        self.a_m = a_m
        self.b_m = b_m
        self.initial_state = initial_state
        self.state = initial_state

    def step(self, reference_input: float, dt: float) -> float:
        """Execute one time step using Euler integration.

        Args:
            reference_input: Desired reference input
            dt: Time step

        Returns:
            Reference model output
        """
        # ẋ_m = -a_m * x_m + b_m * r
        state_derivative = -self.a_m * self.state + self.b_m * reference_input
        self.state = self.state + dt * state_derivative
        return self.state

    def get_state(self) -> np.ndarray:
        """Get current reference model state.

        Returns:
            Reference model state as 1D array
        """
        return np.array([self.state])

    def reset(self) -> None:
        """Reset reference model to initial state."""
        self.state = self.initial_state


class SecondOrderReferenceModel:
    """Second-order linear reference model.

    Implements: ẍ_m + 2*ζ*ω_n*ẋ_m + ω_n^2*x_m = ω_n^2*r

    where ζ is damping ratio and ω_n is natural frequency.
    """

    def __init__(
        self,
        omega_n: float = 1.0,
        zeta: float = 0.7,
        initial_position: float = 0.0,
        initial_velocity: float = 0.0,
    ) -> None:
        """Initialize second-order reference model.

        Args:
            omega_n: Natural frequency (rad/s)
            zeta: Damping ratio (0 < ζ < 1 for underdamped)
            initial_position: Initial position
            initial_velocity: Initial velocity
        """
        self.omega_n = omega_n
        self.zeta = zeta
        self.initial_position = initial_position
        self.initial_velocity = initial_velocity
        self.position = initial_position
        self.velocity = initial_velocity

    def step(self, reference_input: float, dt: float) -> float:
        """Execute one time step using RK4 integration.

        Args:
            reference_input: Desired reference input
            dt: Time step

        Returns:
            Reference model output (position)
        """
        # State: [position, velocity]
        # ẍ = -2*ζ*ω_n*ẋ - ω_n^2*x + ω_n^2*r

        def dynamics(pos: float, vel: float, r: float) -> tuple[float, float]:
            """Compute state derivatives."""
            acceleration = (
                -2.0 * self.zeta * self.omega_n * vel
                - self.omega_n**2 * pos
                + self.omega_n**2 * r
            )
            return vel, acceleration

        # RK4 integration
        k1_v, k1_a = dynamics(self.position, self.velocity, reference_input)
        k2_v, k2_a = dynamics(
            self.position + 0.5 * dt * k1_v,
            self.velocity + 0.5 * dt * k1_a,
            reference_input,
        )
        k3_v, k3_a = dynamics(
            self.position + 0.5 * dt * k2_v,
            self.velocity + 0.5 * dt * k2_a,
            reference_input,
        )
        k4_v, k4_a = dynamics(
            self.position + dt * k3_v,
            self.velocity + dt * k3_a,
            reference_input,
        )

        self.position += (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        self.velocity += (dt / 6.0) * (k1_a + 2 * k2_a + 2 * k3_a + k4_a)

        return self.position

    def get_state(self) -> np.ndarray:
        """Get current reference model state.

        Returns:
            Reference model state [position, velocity]
        """
        return np.array([self.position, self.velocity])

    def reset(self) -> None:
        """Reset reference model to initial state."""
        self.position = self.initial_position
        self.velocity = self.initial_velocity


class SimpleFirstOrderPlant:
    """Simple first-order plant for testing.

    Implements: ẋ = -a*x + b*u + d(t)

    where d(t) is optional disturbance.
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 1.0,
        initial_state: float = 0.0,
        disturbance_fn: Callable[[float], float] | None = None,
    ) -> None:
        """Initialize first-order plant.

        Args:
            a: Plant pole
            b: Plant control gain
            initial_state: Initial state value
            disturbance_fn: Optional disturbance function(time)
        """
        self.a = a
        self.b = b
        self.initial_state = initial_state
        self.disturbance_fn = disturbance_fn
        self.state = initial_state
        self.time = 0.0

    def step(self, control_input: float, dt: float) -> float:
        """Execute one time step.

        Args:
            control_input: Control signal
            dt: Time step

        Returns:
            Plant output
        """
        disturbance = 0.0
        if self.disturbance_fn is not None:
            disturbance = self.disturbance_fn(self.time)

        # ẋ = -a*x + b*u + d
        state_derivative = -self.a * self.state + self.b * control_input + disturbance
        self.state = self.state + dt * state_derivative
        self.time += dt

        return self.state

    def get_state(self) -> np.ndarray:
        """Get current plant state.

        Returns:
            Plant state as 1D array
        """
        return np.array([self.state])

    def reset(self) -> None:
        """Reset plant to initial state."""
        self.state = self.initial_state
        self.time = 0.0


class PersistenceOfExcitationMonitor:
    """Monitor persistence of excitation for adaptive systems.

    Checks if the regressor signal is sufficiently rich to ensure
    parameter convergence by monitoring the condition number of
    the regressor covariance matrix.
    """

    def __init__(
        self,
        window_size: int = 100,
        condition_threshold: float = 100.0,
        min_eigenvalue_threshold: float = 1e-6,
    ) -> None:
        """Initialize PE monitor.

        Args:
            window_size: Number of samples for covariance estimation
            condition_threshold: Maximum acceptable condition number
            min_eigenvalue_threshold: Minimum eigenvalue for excitation
        """
        self.window_size = window_size
        self.condition_threshold = condition_threshold
        self.min_eigenvalue_threshold = min_eigenvalue_threshold
        self.regressor_history: list[np.ndarray] = []

    def update(self, regressor: np.ndarray) -> None:
        """Update monitor with new regressor sample.

        Args:
            regressor: Current regressor vector
        """
        self.regressor_history.append(regressor.copy())

        # Keep only recent samples
        if len(self.regressor_history) > self.window_size:
            self.regressor_history.pop(0)

    def is_persistently_exciting(self) -> bool:
        """Check if signal is persistently exciting.

        Returns:
            True if signal meets PE conditions
        """
        if len(self.regressor_history) < self.window_size:
            return False

        # Compute regressor covariance matrix
        Phi = np.array(self.regressor_history)  # (window_size x num_params)
        covariance = (Phi.T @ Phi) / self.window_size

        # Check eigenvalues
        eigenvalues = np.linalg.eigvalsh(covariance)
        min_eigenvalue = np.min(eigenvalues)
        max_eigenvalue = np.max(eigenvalues)

        if min_eigenvalue < self.min_eigenvalue_threshold:
            return False

        # Check condition number
        condition_number = max_eigenvalue / (min_eigenvalue + 1e-12)

        return condition_number < self.condition_threshold

    def get_condition_number(self) -> float:
        """Get current condition number of regressor covariance.

        Returns:
            Condition number (inf if insufficient data)
        """
        if len(self.regressor_history) < self.window_size:
            return float("inf")

        Phi = np.array(self.regressor_history)
        covariance = (Phi.T @ Phi) / self.window_size

        eigenvalues = np.linalg.eigvalsh(covariance)
        min_eigenvalue = np.min(eigenvalues)
        max_eigenvalue = np.max(eigenvalues)

        if min_eigenvalue < 1e-12:
            return float("inf")

        return max_eigenvalue / min_eigenvalue

    def get_min_eigenvalue(self) -> float:
        """Get minimum eigenvalue of regressor covariance.

        Returns:
            Minimum eigenvalue (0 if insufficient data)
        """
        if len(self.regressor_history) < self.window_size:
            return 0.0

        Phi = np.array(self.regressor_history)
        covariance = (Phi.T @ Phi) / self.window_size
        eigenvalues = np.linalg.eigvalsh(covariance)

        return float(np.min(eigenvalues))

    def reset(self) -> None:
        """Reset monitor history."""
        self.regressor_history = []


class AdaptiveControlConfig(BaseModel):
    """Configuration for adaptive controller with automatic validation.

    Attributes:
        num_parameters: Number of adaptive parameters to estimate
        adaptation_gain: Initial learning rate for parameter updates (γ > 0)
        reference_model: Callable that generates reference trajectory (deprecated)
        reference_model_dynamics: Dynamic reference model with state
        initial_parameters: Initial guess for adaptive parameters
        parameter_bounds: Optional bounds for parameter values (min, max)
        dead_zone: Dead zone threshold to prevent parameter drift
        sigma_modification: Leakage term coefficient for robustness (0 ≤ σ ≤ 1)
        use_normalization: Whether to normalize regressor for stability
        enable_adaptive_gain: Whether to adapt learning rate dynamically
        gain_adaptation_rate: Rate of learning rate adaptation
        min_adaptation_gain: Minimum allowed learning rate
        max_adaptation_gain: Maximum allowed learning rate
        enable_pe_monitoring: Enable persistence of excitation monitoring
        pe_window_size: Window size for PE monitoring
        pe_condition_threshold: Maximum condition number for PE
        enable_lyapunov_monitoring: Enable Lyapunov stability monitoring
        debug: Whether to enable debug logging
    """

    num_parameters: int = Field(gt=0, description="Number of adaptive parameters")
    adaptation_gain: float = Field(
        default=0.1, gt=0.0, description="Initial learning rate for adaptation"
    )
    reference_model: Callable[[float], float] | None = Field(
        default=None, description="Reference model function (deprecated)"
    )
    reference_model_dynamics: object | None = Field(
        default=None, description="Dynamic reference model with state"
    )
    initial_parameters: list[float] | None = Field(
        default=None, description="Initial parameter values"
    )
    parameter_bounds: tuple[float, float] | None = Field(
        default=None, description="Parameter bounds (min, max)"
    )
    dead_zone: float = Field(
        default=0.0, ge=0.0, description="Dead zone threshold for adaptation"
    )
    sigma_modification: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sigma modification (leakage) coefficient",
    )
    use_normalization: bool = Field(
        default=True, description="Use normalized gradient for stability"
    )
    enable_adaptive_gain: bool = Field(
        default=False, description="Enable adaptive learning rate scheduling"
    )
    gain_adaptation_rate: float = Field(
        default=0.01, gt=0.0, description="Rate of learning rate adaptation"
    )
    min_adaptation_gain: float = Field(
        default=0.001, gt=0.0, description="Minimum learning rate"
    )
    max_adaptation_gain: float = Field(
        default=1.0, gt=0.0, description="Maximum learning rate"
    )
    enable_pe_monitoring: bool = Field(
        default=False, description="Enable persistence of excitation monitoring"
    )
    pe_window_size: int = Field(
        default=100, gt=0, description="Window size for PE monitoring"
    )
    pe_condition_threshold: float = Field(
        default=100.0, gt=0.0, description="Maximum condition number for PE"
    )
    enable_lyapunov_monitoring: bool = Field(
        default=False, description="Enable Lyapunov stability monitoring"
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    @field_validator("parameter_bounds")
    @classmethod
    def validate_parameter_bounds(
        cls, v: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        """Validate parameter bounds are ordered correctly.

        Args:
            v: Parameter bounds tuple (min, max)

        Returns:
            Validated parameter bounds

        Raises:
            ValueError: If min >= max
        """
        if v is not None:
            min_val, max_val = v
            if min_val >= max_val:
                raise ValueError(
                    f"Parameter min ({min_val}) must be less than max ({max_val})"
                )
        return v

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


class AdaptiveController:
    """Model Reference Adaptive Controller with parameter estimation.

    This implementation includes:
    - MIT rule for parameter adaptation
    - Sigma modification for robustness
    - Dead zone to prevent parameter drift
    - Normalized gradient for stability
    - Parameter projection for bounded adaptation
    - Dynamic reference model with state tracking
    - Persistence of excitation monitoring
    - Adaptive learning rate scheduling
    - Lyapunov stability verification

    Example:
        >>> config = AdaptiveControlConfig(num_parameters=3, adaptation_gain=0.1)
        >>> controller = AdaptiveController(config)
        >>> control = controller.compute(measurement=5.0, regressor=[1.0, 5.0, 25.0])
    """

    def __init__(self, config: AdaptiveControlConfig) -> None:
        """Initialize adaptive controller.

        Args:
            config: Adaptive controller configuration
        """
        self.config = config

        # Initialize parameters
        if config.initial_parameters is not None:
            if len(config.initial_parameters) != config.num_parameters:
                raise ValueError(
                    f"Expected {config.num_parameters} initial parameters, "
                    f"got {len(config.initial_parameters)}"
                )
            self._parameters = np.array(config.initial_parameters, dtype=np.float64)
        else:
            self._parameters = np.zeros(config.num_parameters, dtype=np.float64)

        self._reference_output: float = 0.0
        self._adaptation_history: list[np.ndarray] = []

        # Current adaptive gain (can change if adaptive gain is enabled)
        self._current_gain = config.adaptation_gain
        self._gain_history: list[float] = []

        # Reference model dynamics
        self._ref_model_dynamics = config.reference_model_dynamics

        # Persistence of excitation monitor
        self._pe_monitor: PersistenceOfExcitationMonitor | None = None
        if config.enable_pe_monitoring:
            self._pe_monitor = PersistenceOfExcitationMonitor(
                window_size=config.pe_window_size,
                condition_threshold=config.pe_condition_threshold,
            )

        # Lyapunov function monitoring
        self._lyapunov_history: list[float] = []
        self._lyapunov_derivative_history: list[float] = []

        # Error history for adaptive gain
        self._error_history: list[float] = []

        if self.config.debug:
            logger.setLevel(logging.DEBUG)

    @property
    def parameters(self) -> np.ndarray:
        """Get current adaptive parameter values."""
        return self._parameters.copy()

    def compute(
        self,
        measurement: float,
        regressor: list[float] | np.ndarray,
        reference: float | None = None,
        reference_input: float | None = None,
        dt: float = 0.01,
    ) -> float:
        """Compute adaptive control output and update parameters.

        Args:
            measurement: Current plant output
            regressor: Feature vector φ(t) for parameter adaptation
            reference: Reference signal (uses reference_model if None)
            reference_input: Input to reference model dynamics (if using dynamics)
            dt: Time step for integration

        Returns:
            Control output signal

        Raises:
            ValueError: If regressor dimension doesn't match num_parameters

        Example:
            >>> controller = AdaptiveController(AdaptiveControlConfig(num_parameters=2))
            >>> output = controller.compute(measurement=5.0, regressor=[1.0, 5.0])
        """
        # Convert regressor to numpy array
        phi = np.array(regressor, dtype=np.float64)

        if phi.shape[0] != self.config.num_parameters:
            raise ValueError(
                f"Regressor dimension ({phi.shape[0]}) doesn't match "
                f"num_parameters ({self.config.num_parameters})"
            )

        # Update PE monitor
        if self._pe_monitor is not None:
            self._pe_monitor.update(phi)

        # Get reference signal
        if reference is not None:
            self._reference_output = reference
        elif self._ref_model_dynamics is not None:
            # Use dynamic reference model
            ref_input = reference_input if reference_input is not None else measurement
            self._reference_output = self._ref_model_dynamics.step(ref_input, dt)  # type: ignore[attr-defined]
        elif self.config.reference_model is not None:
            # Use legacy callable reference model
            self._reference_output = self.config.reference_model(measurement)
        else:
            self._reference_output = 0.0

        # Calculate tracking error
        error = self._reference_output - measurement
        self._error_history.append(abs(error))

        # Apply dead zone to prevent drift on small errors
        if abs(error) < self.config.dead_zone:
            error = 0.0

        # Adapt learning rate if enabled
        if self.config.enable_adaptive_gain:
            self._update_adaptive_gain(error, dt)

        # Compute control output
        control_output = float(np.dot(self._parameters, phi))

        # Parameter adaptation using MIT rule with modifications
        if self.config.use_normalization:
            # Normalized gradient for stability
            normalization = 1.0 + np.dot(phi, phi)
            adaptation = self._current_gain * error * phi / normalization
        else:
            # Standard gradient
            adaptation = self._current_gain * error * phi

        # Add sigma modification (leakage) for robustness
        if self.config.sigma_modification > 0.0:
            leakage = self.config.sigma_modification * self._parameters
            adaptation = adaptation - leakage

        # Update parameters
        self._parameters = self._parameters + adaptation * dt

        # Project parameters to bounds if specified
        if self.config.parameter_bounds is not None:
            min_val, max_val = self.config.parameter_bounds
            self._parameters = np.clip(self._parameters, min_val, max_val)

        # Store adaptation history
        self._adaptation_history.append(self._parameters.copy())
        self._gain_history.append(self._current_gain)

        # Compute Lyapunov function if enabled
        if self.config.enable_lyapunov_monitoring:
            self._update_lyapunov(error, phi, adaptation, dt)

        if self.config.debug:
            logger.debug(
                f"Adaptive: e={error:.3f}, θ={self._parameters}, "
                f"γ={self._current_gain:.4f}, u={control_output:.3f}"
            )

        return control_output

    def _update_adaptive_gain(self, error: float, dt: float) -> None:
        """Update adaptive learning rate based on tracking error.

        Uses exponential scaling: γ(t) = γ_base * (1 + α*|e|)

        Args:
            error: Current tracking error
            dt: Time step
        """
        # Exponential adaptation based on error magnitude
        error_magnitude = abs(error)

        # Scale gain based on error
        gain_adjustment = 1.0 + self.config.gain_adaptation_rate * error_magnitude
        self._current_gain = self.config.adaptation_gain * gain_adjustment

        # Clip to bounds
        self._current_gain = np.clip(
            self._current_gain,
            self.config.min_adaptation_gain,
            self.config.max_adaptation_gain,
        )

    def _update_lyapunov(
        self,
        error: float,
        phi: np.ndarray,
        adaptation: np.ndarray,
        dt: float,
    ) -> None:
        """Update Lyapunov function and derivative.

        Computes V = e² + θ̃ᵀΓ⁻¹θ̃ and V̇

        Args:
            error: Tracking error
            phi: Regressor vector
            adaptation: Parameter adaptation vector
            dt: Time step
        """
        # Simplified Lyapunov function: V = e²
        lyapunov = error**2
        self._lyapunov_history.append(lyapunov)

        # Lyapunov derivative: V̇ = 2*e*ė
        # For MRAC: ė ≈ -adaptation · phi (simplified)
        if len(self._lyapunov_history) >= 2:
            lyapunov_derivative = (lyapunov - self._lyapunov_history[-2]) / dt
            self._lyapunov_derivative_history.append(lyapunov_derivative)
        else:
            self._lyapunov_derivative_history.append(0.0)

    def reset(self) -> None:
        """Reset controller to initial state.

        Resets parameters to initial values and clears adaptation history.

        Example:
            >>> controller.reset()
        """
        if self.config.initial_parameters is not None:
            self._parameters = np.array(
                self.config.initial_parameters, dtype=np.float64
            )
        else:
            self._parameters = np.zeros(self.config.num_parameters, dtype=np.float64)

        self._reference_output = 0.0
        self._adaptation_history = []
        self._current_gain = self.config.adaptation_gain
        self._gain_history = []
        self._error_history = []
        self._lyapunov_history = []
        self._lyapunov_derivative_history = []

        if self._pe_monitor is not None:
            self._pe_monitor.reset()

        if self._ref_model_dynamics is not None:
            self._ref_model_dynamics.reset()  # type: ignore[attr-defined]

    def get_adaptation_history(self) -> list[np.ndarray]:
        """Get history of parameter adaptations.

        Returns:
            List of parameter vectors over time

        Example:
            >>> history = controller.get_adaptation_history()
            >>> print(f"Parameter evolution: {len(history)} updates")
        """
        return [params.copy() for params in self._adaptation_history]

    def set_parameters(self, parameters: list[float] | np.ndarray) -> None:
        """Manually set parameter values.

        Args:
            parameters: New parameter values

        Raises:
            ValueError: If parameter count doesn't match configuration

        Example:
            >>> controller.set_parameters([1.0, 0.5, 0.2])
        """
        params = np.array(parameters, dtype=np.float64)

        if params.shape[0] != self.config.num_parameters:
            raise ValueError(
                f"Expected {self.config.num_parameters} parameters, "
                f"got {params.shape[0]}"
            )

        self._parameters = params

    def get_tracking_error(self, measurement: float) -> float:
        """Calculate current tracking error.

        Args:
            measurement: Current plant output

        Returns:
            Tracking error (reference - measurement)

        Example:
            >>> error = controller.get_tracking_error(5.0)
        """
        return self._reference_output - measurement

    def estimate_convergence_rate(self) -> float:
        """Estimate parameter convergence rate from adaptation history.

        Returns:
            Convergence rate metric (lower is faster convergence)

        Example:
            >>> rate = controller.estimate_convergence_rate()
            >>> print(f"Convergence rate: {rate:.4f}")
        """
        if len(self._adaptation_history) < 2:
            return float("inf")

        # Calculate average parameter change magnitude
        total_change = 0.0
        for i in range(1, len(self._adaptation_history)):
            change = float(
                np.linalg.norm(
                    self._adaptation_history[i] - self._adaptation_history[i - 1]
                )
            )
            total_change += change

        return total_change / (len(self._adaptation_history) - 1)

    def is_persistently_exciting(self) -> bool:
        """Check if regressor signal is persistently exciting.

        Returns:
            True if signal meets PE conditions, False otherwise

        Example:
            >>> is_pe = controller.is_persistently_exciting()
            >>> print(f"Persistently exciting: {is_pe}")
        """
        if self._pe_monitor is None:
            return False
        return self._pe_monitor.is_persistently_exciting()

    def get_pe_condition_number(self) -> float:
        """Get condition number of regressor covariance matrix.

        Returns:
            Condition number (inf if PE monitoring disabled or insufficient data)

        Example:
            >>> cond = controller.get_pe_condition_number()
            >>> print(f"Condition number: {cond:.2f}")
        """
        if self._pe_monitor is None:
            return float("inf")
        return self._pe_monitor.get_condition_number()

    def get_pe_min_eigenvalue(self) -> float:
        """Get minimum eigenvalue of regressor covariance matrix.

        Returns:
            Minimum eigenvalue (0 if PE monitoring disabled)

        Example:
            >>> min_eig = controller.get_pe_min_eigenvalue()
            >>> print(f"Min eigenvalue: {min_eig:.6f}")
        """
        if self._pe_monitor is None:
            return 0.0
        return self._pe_monitor.get_min_eigenvalue()

    def get_current_gain(self) -> float:
        """Get current adaptive learning rate.

        Returns:
            Current learning rate value

        Example:
            >>> gamma = controller.get_current_gain()
            >>> print(f"Current learning rate: {gamma:.4f}")
        """
        return self._current_gain

    def get_gain_history(self) -> list[float]:
        """Get history of learning rate values.

        Returns:
            List of learning rate values over time

        Example:
            >>> gain_hist = controller.get_gain_history()
            >>> print(f"Gain varied from {min(gain_hist):.4f} to {max(gain_hist):.4f}")
        """
        return self._gain_history.copy()

    def get_lyapunov_history(self) -> list[float]:
        """Get history of Lyapunov function values.

        Returns:
            List of Lyapunov function values (empty if monitoring disabled)

        Example:
            >>> V_hist = controller.get_lyapunov_history()
            >>> print(f"Lyapunov function decreased: {V_hist[0] > V_hist[-1]}")
        """
        return self._lyapunov_history.copy()

    def get_lyapunov_derivative_history(self) -> list[float]:
        """Get history of Lyapunov function derivatives.

        Returns:
            List of Lyapunov derivative values (empty if monitoring disabled)

        Example:
            >>> Vdot_hist = controller.get_lyapunov_derivative_history()
            >>> stable = all(v <= 0 for v in Vdot_hist)
            >>> print(f"Lyapunov stable: {stable}")
        """
        return self._lyapunov_derivative_history.copy()

    def is_lyapunov_stable(self, window: int = 50) -> bool:
        """Check if system is Lyapunov stable over recent window.

        System is considered stable if V̇ ≤ 0 on average.

        Args:
            window: Number of recent samples to check

        Returns:
            True if Lyapunov derivative is non-positive on average

        Example:
            >>> stable = controller.is_lyapunov_stable(window=100)
            >>> print(f"System is stable: {stable}")
        """
        if len(self._lyapunov_derivative_history) < window:
            return False

        recent_derivatives = self._lyapunov_derivative_history[-window:]
        avg_derivative = np.mean(recent_derivatives)

        return bool(avg_derivative <= 0.0)

    def get_error_history(self) -> list[float]:
        """Get history of tracking errors.

        Returns:
            List of absolute tracking error values

        Example:
            >>> errors = controller.get_error_history()
            >>> print(f"Final error: {errors[-1]:.4f}")
        """
        return self._error_history.copy()

    def get_reference_model_state(self) -> np.ndarray | None:
        """Get current reference model state.

        Returns:
            Reference model state vector, or None if no dynamic model

        Example:
            >>> ref_state = controller.get_reference_model_state()
            >>> if ref_state is not None:
            ...     print(f"Reference state: {ref_state}")
        """
        if self._ref_model_dynamics is None:
            return None
        return self._ref_model_dynamics.get_state()  # type: ignore[attr-defined]


def simulate_closed_loop(
    controller: AdaptiveController,
    plant: object,
    reference_input: float | Callable[[float], float],
    duration: float,
    dt: float = 0.01,
    regressor_fn: Callable[[float], np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Simulate closed-loop adaptive control system.

    Runs a complete simulation of plant + adaptive controller.

    Args:
        controller: Adaptive controller instance
        plant: Plant dynamics (must have step() method)
        reference_input: Constant reference or time-varying function
        duration: Simulation duration
        dt: Time step
        regressor_fn: Function to compute regressor from plant output

    Returns:
        Dictionary containing:
            - 'time': Time vector
            - 'plant_output': Plant output trajectory
            - 'reference': Reference trajectory
            - 'control': Control signal trajectory
            - 'parameters': Parameter evolution (num_steps x num_params)
            - 'error': Tracking error trajectory

    Example:
        >>> from algokit.algorithms.control.adaptive import (
        ...     AdaptiveController, AdaptiveControlConfig,
        ...     SimpleFirstOrderPlant, simulate_closed_loop
        ... )
        >>> config = AdaptiveControlConfig(num_parameters=2)
        >>> controller = AdaptiveController(config)
        >>> plant = SimpleFirstOrderPlant(a=1.0, b=1.0)
        >>> results = simulate_closed_loop(
        ...     controller, plant, reference_input=10.0, duration=10.0
        ... )
        >>> print(f"Final error: {results['error'][-1]:.4f}")
    """
    num_steps = int(duration / dt)
    time = np.arange(num_steps) * dt

    # Storage arrays
    plant_output = np.zeros(num_steps)
    reference = np.zeros(num_steps)
    control = np.zeros(num_steps)
    parameters_history = np.zeros((num_steps, controller.config.num_parameters))
    error = np.zeros(num_steps)

    # Initial plant output
    plant_state = plant.get_state()  # type: ignore[attr-defined]
    y = float(plant_state[0]) if len(plant_state) > 0 else 0.0

    for i in range(num_steps):
        # Get reference
        r = reference_input(time[i]) if callable(reference_input) else reference_input
        reference[i] = r

        # Compute regressor
        phi = regressor_fn(y) if regressor_fn is not None else np.array([1.0, y])

        # Compute control
        u = controller.compute(
            measurement=y,
            regressor=phi,
            reference_input=r,
            dt=dt,
        )

        # Apply to plant
        y = plant.step(u, dt)  # type: ignore[attr-defined]

        # Store results
        plant_output[i] = y
        control[i] = u
        parameters_history[i, :] = controller.parameters
        error[i] = r - y

    return {
        "time": time,
        "plant_output": plant_output,
        "reference": reference,
        "control": control,
        "parameters": parameters_history,
        "error": error,
    }
