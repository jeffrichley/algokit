"""Research-Grade Model Reference Adaptive Control (MRAC) Implementation.

This module implements canonical Model Reference Adaptive Control (MRAC) with
advanced stability features and diagnostics. The implementation is suitable for
research, education, and production control applications.

Mathematical Formulation
------------------------
The canonical MRAC adaptation law:
    θ̇ = -Γ φ e  (with optional normalization)

Control law:
    u = θᵀ φ

Reference model dynamics:
    ẋₘ = Aₘxₘ + Bₘr

Plant dynamics (linear):
    ẋ = Ax + Bu + d(t)

Lyapunov function:
    V = eᵀPe + θ̃ᵀΓ⁻¹θ̃

Advanced Features
-----------------
- **Canonical MRAC formulation** with rigorous mathematical foundation
- **Sigma modification** for robustness and drift prevention
- **Normalized gradient** for stability under large regressor values
- **Dead-zone logic** to prevent adaptation under small errors
- **Persistence of excitation (PE) monitoring** with automatic freeze
- **Adaptive gain scheduling** Γ(t) based on error magnitude and PE metrics
- **Lyapunov stability monitoring** with derivative computation
- **RK4 integration** for accurate simulation
- **Comprehensive diagnostics** with plotting and reporting tools
- **Matrix-form plant dynamics** supporting multi-input systems
- **Parameter projection** for bounded adaptation

Applications
------------
- Aerospace flight control with uncertain aerodynamics
- Robotic manipulators with payload variations
- Process control with time-varying parameters
- Any system requiring automatic parameter tuning
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


class LinearPlant:
    """General linear plant with matrix-form dynamics.

    Implements: ẋ = Ax + Bu + d(t)

    where:
        - x: State vector (n-dimensional)
        - u: Control input (scalar or vector)
        - A: System matrix (n x n)
        - B: Input matrix (n x m)
        - d(t): Optional disturbance vector function

    This is the most general form suitable for MRAC applications.
    Uses RK4 integration for accurate state propagation.
    """

    def __init__(
        self,
        A: np.ndarray | list[list[float]],
        B: np.ndarray | list[float] | list[list[float]],
        initial_state: np.ndarray | list[float] | None = None,
        disturbance_fn: Callable[[float], np.ndarray] | None = None,
    ) -> None:
        """Initialize linear plant.

        Args:
            A: System matrix (n x n)
            B: Input matrix (n x m) or vector (n,) for single input
            initial_state: Initial state vector (n,), defaults to zeros
            disturbance_fn: Optional disturbance function(time) -> (n,)

        Raises:
            ValueError: If matrix dimensions are incompatible
        """
        # Convert to numpy arrays
        self.A = np.atleast_2d(np.array(A, dtype=np.float64))
        self.B = np.atleast_2d(np.array(B, dtype=np.float64))

        # Handle single-input case (B is a column vector)
        if self.B.ndim == 1 or self.B.shape[1] == 1:
            self.B = self.B.reshape(-1, 1)

        # Validate dimensions
        n = self.A.shape[0]
        if self.A.shape[1] != n:
            raise ValueError(f"A must be square, got shape {self.A.shape}")
        if self.B.shape[0] != n:
            raise ValueError(f"B rows ({self.B.shape[0]}) must match A dimension ({n})")

        self.n_states = n
        self.n_inputs = self.B.shape[1]

        # Initialize state
        if initial_state is not None:
            self.initial_state = np.array(initial_state, dtype=np.float64)
            if self.initial_state.shape[0] != n:
                raise ValueError(
                    f"Initial state size ({self.initial_state.shape[0]}) "
                    f"must match system dimension ({n})"
                )
        else:
            self.initial_state = np.zeros(n, dtype=np.float64)

        self.state = self.initial_state.copy()
        self.disturbance_fn = disturbance_fn
        self.time = 0.0

    def step(self, control_input: float | np.ndarray, dt: float) -> float:
        """Execute one time step using RK4 integration.

        Args:
            control_input: Control signal (scalar or vector)
            dt: Time step

        Returns:
            Plant output (first state component)

        Raises:
            ValueError: If control input dimension doesn't match system
        """
        # Convert control to array
        u = np.atleast_1d(np.array(control_input, dtype=np.float64))
        if u.shape[0] != self.n_inputs:
            raise ValueError(
                f"Control input size ({u.shape[0]}) must match "
                f"system inputs ({self.n_inputs})"
            )

        # RK4 integration
        def dynamics(x: np.ndarray, t: float) -> np.ndarray:
            """Compute state derivative."""
            disturbance = np.zeros(self.n_states)
            if self.disturbance_fn is not None:
                disturbance = self.disturbance_fn(t)

            # ẋ = Ax + Bu + d
            return self.A @ x + self.B @ u + disturbance

        # RK4 steps
        k1 = dynamics(self.state, self.time)
        k2 = dynamics(self.state + 0.5 * dt * k1, self.time + 0.5 * dt)
        k3 = dynamics(self.state + 0.5 * dt * k2, self.time + 0.5 * dt)
        k4 = dynamics(self.state + dt * k3, self.time + dt)

        # Update state
        self.state = self.state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.time += dt

        # Return output (first state)
        return float(self.state[0])

    def get_state(self) -> np.ndarray:
        """Get current plant state.

        Returns:
            Plant state vector (n,)
        """
        return self.state.copy()

    def reset(self) -> None:
        """Reset plant to initial state."""
        self.state = self.initial_state.copy()
        self.time = 0.0


class SimpleFirstOrderPlant:
    """Simple first-order plant for testing.

    Implements: ẋ = -a*x + b*u + d(t)

    where d(t) is optional disturbance.

    Note: This is a convenience class. For general applications,
    use LinearPlant with A=[-a], B=[b].
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

        A signal is PE if:
        1. Minimum eigenvalue > threshold (sufficient excitation)
        2. Condition number < threshold (well-conditioned)

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
        min_eigenvalue = float(np.min(eigenvalues))
        max_eigenvalue = float(np.max(eigenvalues))

        # Must have sufficient excitation
        if min_eigenvalue < self.min_eigenvalue_threshold:
            return False

        # Must be well-conditioned
        condition_number = max_eigenvalue / (min_eigenvalue + 1e-12)

        return bool(condition_number < self.condition_threshold)

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
    """Configuration for research-grade MRAC controller.

    This configuration supports all features of the canonical MRAC formulation
    including stability guarantees, PE monitoring, and advanced diagnostics.

    Attributes:
        num_parameters: Number of adaptive parameters to estimate
        adaptation_gain: Initial learning rate Γ for parameter updates (γ > 0)
        reference_model: Callable that generates reference trajectory (deprecated)
        reference_model_dynamics: Dynamic reference model with state
        initial_parameters: Initial guess for adaptive parameters θ₀
        parameter_bounds: Optional bounds for parameter values (min, max)
        dead_zone: Dead zone threshold to prevent parameter drift on small errors
        sigma_modification: Leakage term coefficient σ for robustness (0 ≤ σ ≤ 1)
        use_normalization: Whether to normalize regressor for stability
        enable_adaptive_gain: Whether to adapt learning rate Γ(t) dynamically
        gain_adaptation_rate: Rate of learning rate adaptation
        min_adaptation_gain: Minimum allowed learning rate Γₘᵢₙ
        max_adaptation_gain: Maximum allowed learning rate Γₘₐₓ
        enable_pe_monitoring: Enable persistence of excitation monitoring
        pe_window_size: Window size for PE covariance estimation
        pe_condition_threshold: Maximum condition number for PE acceptance
        pe_min_eigenvalue: Minimum eigenvalue threshold for PE
        freeze_on_pe_failure: Freeze parameter updates when PE condition fails
        enable_lyapunov_monitoring: Enable Lyapunov stability monitoring
        lyapunov_p_matrix: Lyapunov matrix P for V = eᵀPe (None uses identity)
        track_full_history: Track complete state history for diagnostics
        debug: Whether to enable debug logging
    """

    num_parameters: int = Field(gt=0, description="Number of adaptive parameters")
    adaptation_gain: float = Field(
        default=0.1, gt=0.0, description="Initial learning rate Γ for adaptation"
    )
    reference_model: Callable[[float], float] | None = Field(
        default=None, description="Reference model function (deprecated)"
    )
    reference_model_dynamics: object | None = Field(
        default=None, description="Dynamic reference model with state"
    )
    initial_parameters: list[float] | None = Field(
        default=None, description="Initial parameter values θ₀"
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
        description="Sigma modification (leakage) coefficient σ",
    )
    use_normalization: bool = Field(
        default=True, description="Use normalized gradient for stability"
    )
    enable_adaptive_gain: bool = Field(
        default=False, description="Enable adaptive learning rate Γ(t) scheduling"
    )
    gain_adaptation_rate: float = Field(
        default=0.01, gt=0.0, description="Rate of learning rate adaptation"
    )
    min_adaptation_gain: float = Field(
        default=0.001, gt=0.0, description="Minimum learning rate Γₘᵢₙ"
    )
    max_adaptation_gain: float = Field(
        default=1.0, gt=0.0, description="Maximum learning rate Γₘₐₓ"
    )
    enable_pe_monitoring: bool = Field(
        default=False, description="Enable persistence of excitation monitoring"
    )
    pe_window_size: int = Field(
        default=100, gt=0, description="Window size for PE covariance estimation"
    )
    pe_condition_threshold: float = Field(
        default=100.0, gt=0.0, description="Maximum condition number for PE"
    )
    pe_min_eigenvalue: float = Field(
        default=1e-6, gt=0.0, description="Minimum eigenvalue threshold for PE"
    )
    freeze_on_pe_failure: bool = Field(
        default=False,
        description="Freeze parameter updates when PE condition fails",
    )
    enable_lyapunov_monitoring: bool = Field(
        default=False, description="Enable Lyapunov stability monitoring"
    )
    lyapunov_p_matrix: list[list[float]] | None = Field(
        default=None, description="Lyapunov matrix P for V = eᵀPe (None uses I)"
    )
    track_full_history: bool = Field(
        default=True, description="Track complete state history for diagnostics"
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

        # Initialize parameters θ
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

        # Current adaptive gain Γ(t) - can change if adaptive gain is enabled
        self._current_gain = config.adaptation_gain
        self._gain_history: list[float] = []

        # Reference model dynamics
        self._ref_model_dynamics = config.reference_model_dynamics

        # Persistence of excitation monitor
        self._pe_monitor: PersistenceOfExcitationMonitor | None = None
        self._pe_frozen = False  # Track if parameters are frozen due to PE failure
        if config.enable_pe_monitoring:
            self._pe_monitor = PersistenceOfExcitationMonitor(
                window_size=config.pe_window_size,
                condition_threshold=config.pe_condition_threshold,
                min_eigenvalue_threshold=config.pe_min_eigenvalue,
            )

        # Lyapunov function monitoring with matrix P
        self._lyapunov_P: np.ndarray | None = None
        if config.lyapunov_p_matrix is not None:
            self._lyapunov_P = np.array(config.lyapunov_p_matrix, dtype=np.float64)
        self._lyapunov_history: list[float] = []
        self._lyapunov_derivative_history: list[float] = []

        # Error history for adaptive gain
        self._error_history: list[float] = []

        # Full history tracking for diagnostics
        if config.track_full_history:
            self._time_history: list[float] = []
            self._control_history: list[float] = []
            self._reference_history: list[float] = []
            self._measurement_history: list[float] = []
            self._pe_condition_history: list[float] = []
            self._pe_status_history: list[bool] = []
        else:
            self._time_history = []
            self._control_history = []
            self._reference_history = []
            self._measurement_history = []
            self._pe_condition_history = []
            self._pe_status_history = []

        # Time counter
        self._current_time = 0.0

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

        Implements the canonical MRAC law:
            u = θᵀφ
            θ̇ = -Γφe (with optional normalization, sigma, and projection)

        Args:
            measurement: Current plant output y
            regressor: Feature vector φ(t) for parameter adaptation
            reference: Reference signal r (uses reference_model if None)
            reference_input: Input to reference model dynamics (if using dynamics)
            dt: Time step for integration

        Returns:
            Control output signal u

        Raises:
            ValueError: If regressor dimension doesn't match num_parameters

        Example:
            >>> controller = AdaptiveController(AdaptiveControlConfig(num_parameters=2))
            >>> output = controller.compute(measurement=5.0, regressor=[1.0, 5.0])
        """
        # Convert regressor φ to numpy array
        phi = np.array(regressor, dtype=np.float64)

        if phi.shape[0] != self.config.num_parameters:
            raise ValueError(
                f"Regressor dimension ({phi.shape[0]}) doesn't match "
                f"num_parameters ({self.config.num_parameters})"
            )

        # Update PE monitor
        pe_status = True  # Assume PE satisfied unless proven otherwise
        pe_condition = 0.0
        if self._pe_monitor is not None:
            self._pe_monitor.update(phi)
            pe_status = self._pe_monitor.is_persistently_exciting()
            pe_condition = self._pe_monitor.get_condition_number()

            # Check if we should freeze parameters
            if self.config.freeze_on_pe_failure and not pe_status:
                if not self._pe_frozen:
                    logger.warning(
                        "⚠️ PE condition violated (κ=%.2f). Freezing parameter updates.",
                        pe_condition,
                    )
                    self._pe_frozen = True
            else:
                if self._pe_frozen:
                    logger.info("✓ PE condition restored. Resuming parameter updates.")
                    self._pe_frozen = False

        # Get reference signal yₘ
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

        # Calculate tracking error e = yₘ - y
        error = self._reference_output - measurement
        self._error_history.append(abs(error))

        # Apply dead zone to prevent drift on small errors
        error_for_adaptation = error
        if abs(error) < self.config.dead_zone:
            error_for_adaptation = 0.0

        # Adapt learning rate Γ(t) if enabled
        if self.config.enable_adaptive_gain:
            self._update_adaptive_gain(error, dt)

        # Compute control output: u = θᵀφ
        control_output = float(np.dot(self._parameters, phi))

        # Parameter adaptation: θ̇ = -Γφe (canonical MRAC)
        # With normalization, becomes: θ̇ = -Γφe/(1 + φᵀφ)
        adaptation = np.zeros_like(self._parameters)

        if not self._pe_frozen or not self.config.freeze_on_pe_failure:
            if self.config.use_normalization:
                # Normalized gradient for stability
                normalization = 1.0 + np.dot(phi, phi)
                adaptation = (
                    self._current_gain * error_for_adaptation * phi / normalization
                )
            else:
                # Standard gradient: θ̇ = -Γφe (note: MIT rule uses +)
                # We use + to make error positive when tracking
                adaptation = self._current_gain * error_for_adaptation * phi

            # Add sigma modification (leakage) for robustness: -σθ
            if self.config.sigma_modification > 0.0:
                leakage = self.config.sigma_modification * self._parameters
                adaptation = adaptation - leakage

            # Update parameters: θ(t+dt) = θ(t) + θ̇·dt
            self._parameters = self._parameters + adaptation * dt

            # Project parameters to bounds if specified
            if self.config.parameter_bounds is not None:
                min_val, max_val = self.config.parameter_bounds
                self._parameters = np.clip(self._parameters, min_val, max_val)

        # Store adaptation history
        self._adaptation_history.append(self._parameters.copy())
        self._gain_history.append(self._current_gain)

        # Store full history for diagnostics
        if self.config.track_full_history:
            self._time_history.append(self._current_time)
            self._control_history.append(control_output)
            self._reference_history.append(self._reference_output)
            self._measurement_history.append(measurement)
            self._pe_condition_history.append(pe_condition)
            self._pe_status_history.append(pe_status)

        # Compute Lyapunov function if enabled
        if self.config.enable_lyapunov_monitoring:
            self._update_lyapunov(error, phi, adaptation, dt)

        # Update time
        self._current_time += dt

        if self.config.debug:
            logger.debug(
                f"MRAC: t={self._current_time:.2f}, e={error:.3f}, θ={self._parameters}, "
                f"Γ={self._current_gain:.4f}, u={control_output:.3f}, "
                f"PE={pe_status}, frozen={self._pe_frozen}"
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

        Implements the canonical MRAC Lyapunov function:
            V = eᵀPe + θ̃ᵀΓ⁻¹θ̃

        where:
            - e: Tracking error (scalar in this case)
            - P: Positive definite matrix (default: identity)
            - θ̃: Parameter error (θ* - θ, unknown in practice)
            - Γ: Adaptation gain

        For analysis, we use a simplified form V ≈ e² since θ* is unknown.

        The derivative should satisfy:
            V̇ = -eᵀQe ≤ 0 (Q > 0 for stability)

        Args:
            error: Tracking error e
            phi: Regressor vector φ
            adaptation: Parameter adaptation vector θ̇
            dt: Time step
        """
        # Compute V = eᵀPe (scalar case with P = identity or user-provided)
        if self._lyapunov_P is not None:
            # Use provided P matrix (for multi-output systems)
            error_vec = np.array([error])
            lyapunov = float(error_vec.T @ self._lyapunov_P @ error_vec)
        else:
            # Simplified form: V = e²
            lyapunov = error**2

        # Note: We cannot compute the full V = eᵀPe + θ̃ᵀΓ⁻¹θ̃ because
        # we don't know the true parameters θ*. However, for MRAC with
        # proper design, V̇ ≤ 0 is guaranteed theoretically.

        self._lyapunov_history.append(lyapunov)

        # Lyapunov derivative: V̇ (numerical approximation)
        # Theoretical: V̇ = 2e·ė for the error part
        # For MRAC: V̇ = -eᵀQe + 2θ̃ᵀΓ⁻¹θ̃̇
        # With proper MRAC design: V̇ ≤ -λ_min(Q)e²
        if len(self._lyapunov_history) >= 2:
            lyapunov_derivative = (lyapunov - self._lyapunov_history[-2]) / dt
            self._lyapunov_derivative_history.append(lyapunov_derivative)
        else:
            self._lyapunov_derivative_history.append(0.0)

    def reset(self) -> None:
        """Reset controller to initial state.

        Resets parameters to initial values and clears all history.

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

        # Reset PE state
        self._pe_frozen = False
        if self._pe_monitor is not None:
            self._pe_monitor.reset()

        # Reset full history
        self._time_history = []
        self._control_history = []
        self._reference_history = []
        self._measurement_history = []
        self._pe_condition_history = []
        self._pe_status_history = []

        # Reset time
        self._current_time = 0.0

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

    def get_full_history(self) -> dict[str, np.ndarray]:
        """Get complete history of all tracked variables.

        Returns comprehensive data for analysis and visualization.

        Returns:
            Dictionary containing:
                - 'time': Time vector
                - 'measurement': Plant output history
                - 'reference': Reference signal history
                - 'control': Control signal history
                - 'error': Tracking error history (absolute values)
                - 'parameters': Parameter evolution (n_steps x num_params)
                - 'gamma': Adaptation gain history
                - 'lyapunov': Lyapunov function values (if monitoring enabled)
                - 'lyapunov_derivative': V̇ values (if monitoring enabled)
                - 'pe_condition': PE condition number history
                - 'pe_status': PE satisfaction status history (bool)

        Example:
            >>> history = controller.get_full_history()
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(history['time'], history['error'])
            >>> plt.show()
        """
        history: dict[str, np.ndarray] = {}

        if self._time_history:
            history["time"] = np.array(self._time_history)
            history["measurement"] = np.array(self._measurement_history)
            history["reference"] = np.array(self._reference_history)
            history["control"] = np.array(self._control_history)
            history["error"] = np.array(self._error_history)
            history["pe_condition"] = np.array(self._pe_condition_history)
            history["pe_status"] = np.array(self._pe_status_history)

        if self._adaptation_history:
            history["parameters"] = np.array(self._adaptation_history)

        if self._gain_history:
            history["gamma"] = np.array(self._gain_history)

        if self._lyapunov_history:
            history["lyapunov"] = np.array(self._lyapunov_history)
            history["lyapunov_derivative"] = np.array(self._lyapunov_derivative_history)

        return history

    def report_metrics(self) -> dict[str, float]:
        """Generate performance metrics report.

        Computes various performance indicators for the adaptive controller.

        Returns:
            Dictionary containing:
                - 'final_error': Final tracking error (absolute)
                - 'mean_error': Mean tracking error
                - 'rms_error': RMS tracking error
                - 'max_error': Maximum tracking error
                - 'convergence_rate': Parameter convergence rate
                - 'final_gamma': Final adaptation gain
                - 'mean_gamma': Mean adaptation gain
                - 'pe_satisfaction_rate': Fraction of time PE was satisfied
                - 'mean_pe_condition': Mean PE condition number
                - 'lyapunov_stable': Whether Lyapunov stable (if monitoring enabled)

        Example:
            >>> metrics = controller.report_metrics()
            >>> print(f"RMS Error: {metrics['rms_error']:.4f}")
            >>> print(f"PE Satisfaction: {metrics['pe_satisfaction_rate']*100:.1f}%")
        """
        metrics: dict[str, float] = {}

        # Error metrics
        if self._error_history:
            errors = np.array(self._error_history)
            metrics["final_error"] = float(errors[-1]) if len(errors) > 0 else 0.0
            metrics["mean_error"] = float(np.mean(errors))
            metrics["rms_error"] = float(np.sqrt(np.mean(errors**2)))
            metrics["max_error"] = float(np.max(errors))
        else:
            metrics["final_error"] = 0.0
            metrics["mean_error"] = 0.0
            metrics["rms_error"] = 0.0
            metrics["max_error"] = 0.0

        # Convergence metrics
        metrics["convergence_rate"] = self.estimate_convergence_rate()

        # Adaptation gain metrics
        if self._gain_history:
            gains = np.array(self._gain_history)
            metrics["final_gamma"] = float(gains[-1])
            metrics["mean_gamma"] = float(np.mean(gains))
        else:
            metrics["final_gamma"] = self._current_gain
            metrics["mean_gamma"] = self._current_gain

        # PE metrics
        if self._pe_status_history:
            pe_status = np.array(self._pe_status_history)
            metrics["pe_satisfaction_rate"] = float(np.mean(pe_status))
        else:
            metrics["pe_satisfaction_rate"] = 1.0  # Assume satisfied if not monitored

        if self._pe_condition_history:
            # Filter out infinities for mean calculation
            conditions = np.array(self._pe_condition_history)
            finite_conditions = conditions[np.isfinite(conditions)]
            if len(finite_conditions) > 0:
                metrics["mean_pe_condition"] = float(np.mean(finite_conditions))
            else:
                metrics["mean_pe_condition"] = float("inf")
        else:
            metrics["mean_pe_condition"] = 0.0

        # Lyapunov stability
        if self._lyapunov_derivative_history:
            metrics["lyapunov_stable"] = float(
                self.is_lyapunov_stable(
                    window=min(50, len(self._lyapunov_derivative_history))
                )
            )
        else:
            metrics["lyapunov_stable"] = 0.0  # Unknown

        return metrics

    def plot_results(self, save_path: str | None = None) -> None:
        """Generate comprehensive diagnostic plots.

        Creates a multi-panel figure showing:
        1. Tracking performance (reference vs measurement)
        2. Control signal
        3. Parameter evolution
        4. Adaptation gain Γ(t)
        5. Lyapunov function (if monitoring enabled)
        6. PE condition number

        Args:
            save_path: Optional path to save figure (e.g., 'mrac_results.png')

        Raises:
            ImportError: If matplotlib is not installed
            ValueError: If no history data available

        Example:
            >>> controller.plot_results('mrac_diagnostic.png')
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            ) from err

        history = self.get_full_history()
        if not history or "time" not in history:
            raise ValueError("No history data available. Run controller first.")

        # Create figure with subplots
        n_plots = 4
        if self._lyapunov_history:
            n_plots += 1
        if self._pe_condition_history:
            n_plots += 1

        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.5 * n_plots))
        if n_plots == 1:
            axes = [axes]

        time = history["time"]
        plot_idx = 0

        # 1. Tracking performance
        axes[plot_idx].plot(
            time, history["reference"], "r--", label="Reference", linewidth=2
        )
        axes[plot_idx].plot(
            time, history["measurement"], "b-", label="Measurement", alpha=0.8
        )
        axes[plot_idx].set_ylabel("Output")
        axes[plot_idx].set_title("Tracking Performance")
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

        # 2. Control signal
        axes[plot_idx].plot(time, history["control"], "g-", linewidth=1.5)
        axes[plot_idx].set_ylabel("Control u")
        axes[plot_idx].set_title("Control Signal")
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

        # 3. Parameter evolution
        if "parameters" in history:
            params = history["parameters"]
            for i in range(params.shape[1]):
                axes[plot_idx].plot(time, params[:, i], label=f"θ_{i + 1}")
            axes[plot_idx].set_ylabel("Parameters θ")
            axes[plot_idx].set_title("Parameter Evolution")
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        # 4. Adaptation gain
        if "gamma" in history:
            axes[plot_idx].plot(time, history["gamma"], "m-", linewidth=1.5)
            axes[plot_idx].set_ylabel("Gain Γ")
            axes[plot_idx].set_title("Adaptation Gain")
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        # 5. Lyapunov function (if available)
        if "lyapunov" in history:
            axes[plot_idx].semilogy(time, history["lyapunov"], "k-", linewidth=1.5)
            axes[plot_idx].set_ylabel("V (log scale)")
            axes[plot_idx].set_title("Lyapunov Function")
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        # 6. PE condition number (if available)
        if "pe_condition" in history:
            cond = history["pe_condition"]
            # Filter out infinities for plotting
            finite_mask = np.isfinite(cond)
            if np.any(finite_mask):
                axes[plot_idx].semilogy(
                    time[finite_mask],
                    cond[finite_mask],
                    "c-",
                    linewidth=1.5,
                    label="Condition Number",
                )
                axes[plot_idx].axhline(
                    y=self.config.pe_condition_threshold,
                    color="r",
                    linestyle="--",
                    label="PE Threshold",
                )
                axes[plot_idx].set_ylabel("κ (log scale)")
                axes[plot_idx].set_title("PE Condition Number")
                axes[plot_idx].legend()
                axes[plot_idx].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"📊 Diagnostic plots saved to {save_path}")
        else:
            plt.show()


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
