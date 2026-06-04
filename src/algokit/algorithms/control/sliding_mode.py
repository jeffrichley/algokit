"""Research-Grade Sliding Mode Control (SMC) with MIMO support and adaptive gains.

This module implements advanced Sliding Mode Control with:
- MIMO sliding surfaces: s = C·x where C can be a matrix
- Proper equivalent control using plant dynamics (A, B matrices)
- Smooth switching functions (sign, saturation, tanh)
- Lyapunov-motivated adaptive gain: dK/dt = η|s| - ρK
- Optional disturbance observer on sliding surface
- Multiple reaching laws (constant, exponential, power)
- RK4 simulation with matched disturbance injection

Mathematical formulation:
    Sliding surface: s = C·x  (r-dimensional for MIMO)
    Control law: u = u_eq + u_sw

    Equivalent control (if A,B provided):
        u_eq = -(CB)^† (CA·x)  where ^† denotes pseudoinverse

    Switching control:
        u_sw = -K(t)·σ(s)·g(s)
        where σ ∈ {sign, sat, tanh} and g is reaching law shaping

    Adaptive gain (Lyapunov-style):
        dK/dt = η|s| - ρK, clamped to [K_min, K_max]

Key features:
    - Research-grade MIMO support
    - Proper equivalent control via pseudoinverse
    - Smooth switching to reduce chattering
    - Adaptive gain with leakage term
    - Disturbance estimation and compensation
    - Comprehensive diagnostics and simulation
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class SlidingModeConfig(BaseModel):
    """Configuration for research-grade sliding mode controller.

    Attributes:
        state_dim: Dimension of state vector (n)
        control_dim: Dimension of control input vector (m)
        C: Sliding surface matrix (r x n) or vector for SISO

        # Optional plant dynamics for proper equivalent control
        A: System dynamics matrix A (n x n) - for x_dot = Ax + Bu
        B: Control input matrix B (n x m) - for x_dot = Ax + Bu

        # Switching configuration
        mode: Switching function type ('sign', 'sat', 'tanh')
        boundary_layer: Boundary layer width φ for saturation (φ > 0)
        tanh_slope: Slope β for tanh approximation (β > 0)

        # Reaching law configuration
        reaching: Reaching law type ('constant', 'exponential', 'power')
        k1: Primary reaching law gain (> 0)
        k2: Secondary reaching law coefficient (≥ 0)
        alpha: Power law exponent (0 < α < 1)

        # Lyapunov-style adaptive gain
        adaptive_gain: Enable adaptive switching gain update
        eta: Growth rate in dK/dt = η|s| - ρK (> 0)
        rho: Leakage/decay rate in adaptive law (≥ 0)
        K_init: Initial switching gain value (> 0)
        K_min: Minimum allowed switching gain (> 0)
        K_max: Maximum allowed switching gain (> 0)

        # Disturbance observer
        enable_surface_disturbance_observer: Enable surface disturbance observer
        observer_gain: Observer update gain ℓ for d_hat update (> 0)

        # Control saturation
        u_min: Minimum control value (optional)
        u_max: Maximum control value (optional)

        # Diagnostics
        debug: Enable debug logging
    """

    # System dimensions
    state_dim: int = Field(gt=0, description="State vector dimension (n)")
    control_dim: int = Field(gt=0, description="Control input dimension (m)")

    # Sliding surface: C matrix (r x n) or vector for SISO
    C: list[float] | list[list[float]] = Field(
        description="Sliding surface matrix or vector: s = C·x"
    )

    # Optional plant dynamics for proper equivalent control
    A: list[list[float]] | None = Field(
        default=None,
        description="System dynamics matrix A (n x n) for proper u_eq computation",
    )
    B: list[list[float]] | None = Field(
        default=None,
        description="Control input matrix B (n x m) for proper u_eq computation",
    )

    # Switching configuration
    mode: str = Field(
        default="tanh",
        description="Switching function: 'sign', 'sat' (saturation), or 'tanh'",
    )
    boundary_layer: float = Field(
        default=0.1, gt=0.0, description="Boundary layer width φ for sat(s/φ)"
    )
    tanh_slope: float = Field(
        default=3.0, gt=0.0, description="Slope β for tanh(β·s) approximation"
    )

    # Reaching law shaping
    reaching: str = Field(
        default="constant",
        description="Reaching law: 'constant', 'exponential', or 'power'",
    )
    k1: float = Field(default=1.0, gt=0.0, description="Primary reaching gain")
    k2: float = Field(default=0.5, ge=0.0, description="Secondary reaching coefficient")
    alpha: float = Field(
        default=0.6,
        gt=0.0,
        lt=1.0,
        description="Power law exponent α ∈ (0,1) for finite-time convergence",
    )

    # Lyapunov-style adaptive gain: dK/dt = η|s| - ρK
    adaptive_gain: bool = Field(
        default=True, description="Enable Lyapunov-motivated adaptive gain"
    )
    eta: float = Field(
        default=2.0, gt=0.0, description="Adaptation growth rate η in dK/dt"
    )
    rho: float = Field(
        default=0.5,
        ge=0.0,
        description="Leakage/decay rate ρ in dK/dt (prevents unbounded growth)",
    )
    K_init: float = Field(
        default=1.0, gt=0.0, description="Initial switching gain K(0)"
    )
    K_min: float = Field(
        default=0.1, gt=0.0, description="Minimum switching gain (lower clamp)"
    )
    K_max: float = Field(
        default=50.0, gt=0.0, description="Maximum switching gain (upper clamp)"
    )

    # Disturbance observer on sliding surface (optional)
    enable_surface_disturbance_observer: bool = Field(
        default=False, description="Enable simple LPF-based disturbance observer on s"
    )
    observer_gain: float = Field(
        default=0.2, gt=0.0, description="Observer update gain ℓ for d_hat dynamics"
    )

    # Control saturation limits
    u_min: float | None = Field(default=None, description="Minimum control value")
    u_max: float | None = Field(default=None, description="Maximum control value")

    # Diagnostics
    debug: bool = Field(default=False, description="Enable debug logging")

    @field_validator("C")
    @classmethod
    def validate_c_not_empty(
        cls, v: list[float] | list[list[float]]
    ) -> list[float] | list[list[float]]:
        """Validate sliding surface matrix C is not empty.

        Args:
            v: Sliding surface matrix C (vector for SISO, matrix for MIMO)

        Returns:
            Validated C matrix/vector

        Raises:
            ValueError: If C is empty or has empty rows
        """
        if not v:
            raise ValueError("Sliding surface matrix C cannot be empty")

        # Check if it's a matrix (MIMO case)
        if v and isinstance(v[0], list):
            n_cols = len(v[0])
            for i, row in enumerate(v):
                if not isinstance(row, list):
                    raise ValueError(f"Row {i} must be a list")
                if not row:
                    raise ValueError(
                        f"Sliding surface matrix C row {i} cannot be empty"
                    )
                if len(row) != n_cols:
                    raise ValueError(
                        f"All rows of C must have same length, row {i} has {len(row)} "
                        f"but expected {n_cols}"
                    )
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate switching function mode.

        Args:
            v: Switching mode

        Returns:
            Validated mode

        Raises:
            ValueError: If mode is not recognized
        """
        valid_modes = {"sign", "sat", "tanh"}
        if v not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{v}'")
        return v

    @field_validator("reaching")
    @classmethod
    def validate_reaching(cls, v: str) -> str:
        """Validate reaching law type.

        Args:
            v: Reaching law type

        Returns:
            Validated reaching law

        Raises:
            ValueError: If reaching law is not recognized
        """
        valid_laws = {"constant", "exponential", "power"}
        if v not in valid_laws:
            raise ValueError(f"reaching must be one of {valid_laws}, got '{v}'")
        return v

    @field_validator("A")
    @classmethod
    def validate_a_matrix(cls, v: list[list[float]] | None) -> list[list[float]] | None:
        """Validate system matrix A is square.

        Args:
            v: System matrix A

        Returns:
            Validated system matrix A

        Raises:
            ValueError: If matrix is not square or empty
        """
        if v is not None:
            if not v:
                raise ValueError("System matrix A cannot be empty if provided")

            n_rows = len(v)
            for i, row in enumerate(v):
                if len(row) != n_rows:
                    raise ValueError(
                        f"System matrix A must be square; row {i} has {len(row)} "
                        f"elements but expected {n_rows}"
                    )
        return v

    @field_validator("B")
    @classmethod
    def validate_b_matrix(cls, v: list[list[float]] | None) -> list[list[float]] | None:
        """Validate control matrix B has consistent dimensions.

        Args:
            v: Control matrix B

        Returns:
            Validated control matrix B

        Raises:
            ValueError: If matrix is empty or has inconsistent dimensions
        """
        if v is not None:
            if not v:
                raise ValueError("Control matrix B cannot be empty if provided")

            n_cols = len(v[0]) if v else 0
            for i, row in enumerate(v):
                if len(row) != n_cols:
                    raise ValueError(
                        f"Control matrix B rows must have same length; row {i} has "
                        f"{len(row)} elements but expected {n_cols}"
                    )
        return v

    @field_validator("K_min", "K_max")
    @classmethod
    def validate_k_bounds(cls, v: float) -> float:
        """Validate K bounds are positive.

        Args:
            v: K bound value

        Returns:
            Validated K bound

        Raises:
            ValueError: If bound is not positive
        """
        if v <= 0.0:
            raise ValueError("K_min and K_max must be positive")
        return v


@dataclass
class SMCState:
    """Internal state for sliding mode controller.

    Attributes:
        K: Adaptive switching gain per surface channel (r,)
        d_hat: Disturbance estimate on surface (r,) if observer enabled
    """

    K: np.ndarray = field(default_factory=lambda: np.array([]))
    d_hat: np.ndarray = field(default_factory=lambda: np.array([]))


class SlidingModeController:
    """Research-grade sliding mode controller with MIMO support and adaptive gains.

    This implementation provides:
    - MIMO sliding surface s = C·x with proper pseudoinverse-based equivalent control
    - Smooth switching functions (sign, saturation, tanh) to reduce chattering
    - Lyapunov-motivated adaptive gain: dK/dt = η|s| - ρK
    - Optional disturbance observer on sliding surface
    - Multiple reaching laws with configurable shaping
    - RK4 simulation with matched disturbance injection
    - Comprehensive diagnostics and history tracking

    Example:
        >>> config = SlidingModeConfig(
        ...     state_dim=2,
        ...     control_dim=1,
        ...     C=[1.0, 2.0],
        ...     K_init=5.0,
        ...     mode="tanh",
        ...     adaptive_gain=True
        ... )
        >>> controller = SlidingModeController(config)
        >>> u = controller.compute(x=np.array([1.0, 0.5]), xdot=None, dt=0.01)
    """

    def __init__(self, cfg: SlidingModeConfig) -> None:
        """Initialize research-grade sliding mode controller.

        Args:
            cfg: Sliding mode controller configuration

        Raises:
            ValueError: If configuration is inconsistent
        """
        self.cfg = cfg

        # Convert sliding surface matrix C: (r x n)
        if isinstance(cfg.C[0], list):
            self.C = np.array(cfg.C, dtype=np.float64)
        else:
            self.C = np.array(cfg.C, dtype=np.float64).reshape(1, -1)

        self.r, self.n = self.C.shape

        # Validate C dimensions
        if self.n != cfg.state_dim:
            raise ValueError(
                f"Sliding surface matrix C has {self.n} columns but state_dim={cfg.state_dim}"
            )

        # Optional A, B for equivalent control
        self.A: np.ndarray | None = None
        self.B: np.ndarray | None = None
        self.CB: np.ndarray | None = None
        self.CB_pinv: np.ndarray | None = None

        if cfg.A is not None and cfg.B is not None:
            self.A = np.array(cfg.A, dtype=np.float64)
            self.B = np.array(cfg.B, dtype=np.float64)

            # Validate A, B dimensions
            if self.A.shape != (cfg.state_dim, cfg.state_dim):
                raise ValueError(
                    f"System matrix A shape {self.A.shape} doesn't match "
                    f"expected ({cfg.state_dim}, {cfg.state_dim})"
                )
            if self.B.shape != (cfg.state_dim, cfg.control_dim):
                raise ValueError(
                    f"Control matrix B shape {self.B.shape} doesn't match "
                    f"expected ({cfg.state_dim}, {cfg.control_dim})"
                )

            # Precompute CB and its pseudoinverse for equivalent control
            self.CB = self.C @ self.B  # (r x m)
            rank_cb = np.linalg.matrix_rank(self.CB)
            if rank_cb < min(self.r, cfg.control_dim):
                logger.warning(
                    f"CB is rank-deficient (rank={rank_cb}, shape={self.CB.shape}); "
                    "using pseudoinverse for u_eq"
                )
            self.CB_pinv = np.linalg.pinv(self.CB, rcond=1e-8)

        elif (cfg.A is None) ^ (cfg.B is None):
            raise ValueError("A and B must both be provided or both be None")

        # Initialize adaptive state
        self.state = SMCState(
            K=np.full(self.r, cfg.K_init, dtype=np.float64),
            d_hat=np.zeros(self.r, dtype=np.float64),
        )

        # History tracking for diagnostics
        self.t_hist: list[float] = []
        self.x_hist: list[np.ndarray] = []
        self.s_hist: list[np.ndarray] = []
        self.u_hist: list[np.ndarray] = []
        self.ueq_hist: list[np.ndarray] = []
        self.usw_hist: list[np.ndarray] = []
        self.K_hist: list[np.ndarray] = []
        self.dhat_hist: list[np.ndarray] = []

        if self.cfg.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug(
                f"SMC initialized: r={self.r}, n={self.n}, m={cfg.control_dim}, "
                f"mode={cfg.mode}, reaching={cfg.reaching}, adaptive={cfg.adaptive_gain}"
            )

    # ----------------- Core API ----------------- #

    def compute(self, x: np.ndarray, xdot: np.ndarray | None, dt: float) -> np.ndarray:
        """Compute SMC control for current state.

        Args:
            x: Current state vector (n,)
            xdot: State derivative (n,) if available; can be None if A,B provided
            dt: Time step for adaptive gain integration

        Returns:
            Control output u (m,)

        Raises:
            ValueError: If dimensions are incorrect or xdot required but not provided

        Example:
            >>> u = controller.compute(x=np.array([1.0, 0.5]), xdot=None, dt=0.01)
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.n:
            raise ValueError(
                f"State dimension {x.shape[0]} doesn't match expected {self.n}"
            )

        # Compute sliding surface s = C @ x  (r,)
        s = self.C @ x

        if self.cfg.debug:
            logger.debug(f"s = {s}, K = {self.state.K}")

        # Equivalent control
        u_eq = self._equivalent_control(x, xdot)

        # Switching term with reaching law shaping
        sigma_s = self._smooth_switch(s)
        gain_shaping = self._reaching_gain(s)
        u_sw_surface = -(self.state.K * gain_shaping) * sigma_s  # (r,)

        # Map surface control to actuator space
        if self.A is not None and self.CB_pinv is not None:
            u_sw = self.CB_pinv @ u_sw_surface
        else:
            u_sw = self._fallback_surface_to_input(u_sw_surface)

        # Total control
        u = u_eq + u_sw

        # Optional disturbance compensation (feedforward on surface)
        if self.cfg.enable_surface_disturbance_observer:
            u -= self._surface_disturbance_feedforward(s, dt)

        # Saturation
        u = self._apply_saturation(u)

        # Adaptive gain update
        self._update_adaptive_gain(s, dt)

        # Update histories (time not tracked here, done in simulate_response)
        self._record_step(x, s, u, u_eq, u_sw)

        return u

    def compute_sliding_surface(self, state: np.ndarray) -> np.ndarray:
        """Compute sliding surface value s(x) = C·x.

        Args:
            state: Current state vector (n,)

        Returns:
            Sliding surface value(s) (r,) - scalar for SISO, vector for MIMO

        Example:
            >>> s = controller.compute_sliding_surface(np.array([1.0, 0.5]))
        """
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        s = self.C @ state
        return s.ravel() if s.ndim > 1 else s

    # ----------------- Building Blocks ----------------- #

    def _equivalent_control(self, x: np.ndarray, xdot: np.ndarray | None) -> np.ndarray:
        """Compute equivalent control u_eq to maintain s_dot ≈ 0 on surface.

        Args:
            x: Current state (n,)
            xdot: State derivative (n,) if available; None if A,B provided

        Returns:
            Equivalent control u_eq (m,)

        Raises:
            ValueError: If xdot required but not provided

        Notes:
            If A,B provided: u_eq = -(CB)^† (CA·x)
            Else fallback: u_eq = -K_eq·C·xdot (requires xdot)
        """
        if self.A is not None and self.B is not None and self.CB_pinv is not None:
            # Proper equivalent control via pseudoinverse
            u_eq_surface = -(self.C @ self.A @ x)  # (r,)
            u_eq = self.CB_pinv @ u_eq_surface  # (m,)
            if self.cfg.debug:
                logger.debug(f"u_eq (AB-based) = {u_eq}")
            return u_eq
        else:
            # Fallback to simple derivative-based equivalent control
            if xdot is None:
                raise ValueError(
                    "xdot is required for equivalent control when A,B are not provided"
                )
            xdot = np.asarray(xdot, dtype=np.float64).reshape(-1)
            K_eq = 1.0  # Could be configurable
            u_eq_surface = -K_eq * (self.C @ xdot)  # (r,)
            return self._fallback_surface_to_input(u_eq_surface)

    def _smooth_switch(self, s: np.ndarray) -> np.ndarray:
        """Smooth switching function σ(s).

        Args:
            s: Sliding surface value (r,)

        Returns:
            Switching function output σ(s) (r,)

        Notes:
            Supports three modes:
            - sign: σ(s) = sign(s)
            - sat: σ(s) = sat(s/φ)
            - tanh: σ(s) = tanh(β·s)
        """
        if self.cfg.mode == "sign":
            return np.sign(s)
        elif self.cfg.mode == "sat":
            phi = self.cfg.boundary_layer
            return np.clip(s / phi, -1.0, 1.0)
        else:  # tanh
            return np.tanh(self.cfg.tanh_slope * s)

    def _reaching_gain(self, s: np.ndarray) -> np.ndarray:
        """Reaching law shaping multiplier g(s) per channel.

        Args:
            s: Sliding surface value (r,)

        Returns:
            Gain shaping g(s) (r,)

        Notes:
            - constant: g = 1
            - exponential: g = 1 + k1·exp(-k2·|s|)
            - power: g = 1 + k1·|s|^α + k2·|s|
        """
        if self.cfg.reaching == "constant":
            return np.ones_like(s)
        elif self.cfg.reaching == "exponential":
            return 1.0 + self.cfg.k1 * np.exp(-self.cfg.k2 * np.abs(s))
        else:  # power
            return (
                1.0
                + self.cfg.k1 * (np.abs(s) ** self.cfg.alpha)
                + self.cfg.k2 * np.abs(s)
            )

    def _fallback_surface_to_input(self, surface_vec: np.ndarray) -> np.ndarray:
        """Distribute surface action to inputs when A,B unavailable.

        Args:
            surface_vec: Surface-space control (r,)

        Returns:
            Input-space control (m,)

        Notes:
            Simple mean distribution across all inputs.
        """
        m = self.cfg.control_dim
        return np.full(m, surface_vec.mean(), dtype=np.float64)

    def _apply_saturation(self, u: np.ndarray) -> np.ndarray:
        """Apply control saturation limits.

        Args:
            u: Control vector (m,)

        Returns:
            Saturated control (m,)
        """
        if self.cfg.u_min is not None:
            u = np.maximum(u, self.cfg.u_min)
        if self.cfg.u_max is not None:
            u = np.minimum(u, self.cfg.u_max)
        return u

    def _update_adaptive_gain(self, s: np.ndarray, dt: float) -> None:
        """Lyapunov-style adaptive gain update: dK/dt = η|s| - ρK.

        Args:
            s: Sliding surface value (r,)
            dt: Time step

        Notes:
            Updates K element-wise with clamping to [K_min, K_max].
            Leakage term ρK prevents unbounded growth.
        """
        if not self.cfg.adaptive_gain:
            return

        K = self.state.K
        dK = self.cfg.eta * np.abs(s) - self.cfg.rho * K
        K_new = K + dt * dK
        self.state.K = np.clip(K_new, self.cfg.K_min, self.cfg.K_max)

        if self.cfg.debug:
            logger.debug(f"K updated: {K} -> {self.state.K} (dK={dK * dt})")

    def _surface_disturbance_feedforward(self, s: np.ndarray, dt: float) -> np.ndarray:
        """Simple LPF observer on surface disturbances; returns input-space correction.

        Args:
            s: Sliding surface value (r,)
            dt: Time step

        Returns:
            Feedforward control correction (m,)

        Notes:
            Observer: d_hat_{k+1} = d_hat_k + ℓ(s - d_hat_k)
            This is a conservative first-order filter; consider SMO/KF for production.
        """
        ell = self.cfg.observer_gain
        self.state.d_hat = self.state.d_hat + ell * (s - self.state.d_hat)

        # Map surface-level compensation to input space
        if self.A is not None and self.CB_pinv is not None:
            return self.CB_pinv @ self.state.d_hat
        return self._fallback_surface_to_input(self.state.d_hat)

    def _record_step(
        self,
        x: np.ndarray,
        s: np.ndarray,
        u: np.ndarray,
        u_eq: np.ndarray,
        u_sw: np.ndarray,
    ) -> None:
        """Record step in history for diagnostics.

        Args:
            x: State (n,)
            s: Sliding surface (r,)
            u: Total control (m,)
            u_eq: Equivalent control (m,)
            u_sw: Switching control (m,)
        """
        self.x_hist.append(x.copy())
        self.s_hist.append(s.copy())
        self.u_hist.append(u.copy())
        self.ueq_hist.append(u_eq.copy())
        self.usw_hist.append(u_sw.copy())
        self.K_hist.append(self.state.K.copy())
        self.dhat_hist.append(self.state.d_hat.copy())

    # ----------------- Diagnostics & Simulation ----------------- #

    def is_on_surface(self, tol: float = 1e-3) -> bool:
        """Check if system is currently on sliding surface.

        Args:
            tol: Tolerance for surface proximity (default: 1e-3)

        Returns:
            True if ||s|| ≤ tol

        Example:
            >>> if controller.is_on_surface():
            ...     print("In sliding mode")
        """
        if not self.s_hist:
            return False
        return bool(np.linalg.norm(self.s_hist[-1], ord=np.inf) <= tol)

    def summary(self) -> dict[str, str | int | float | np.ndarray | None]:
        """Get summary of controller state and configuration.

        Returns:
            Dictionary with controller summary

        Example:
            >>> info = controller.summary()
            >>> print(f"Channels: {info['channels']}, Mode: {info['mode']}")
        """
        return {
            "channels": self.r,
            "adaptive_gain": self.cfg.adaptive_gain,
            "mode": self.cfg.mode,
            "reaching": self.cfg.reaching,
            "K_current": self.state.K.copy() if self.K_hist else None,
            "d_hat_current": self.state.d_hat.copy() if self.dhat_hist else None,
            "on_surface": self.is_on_surface() if self.s_hist else False,
        }

    def reset(self) -> None:
        """Reset controller state and clear history.

        Example:
            >>> controller.reset()
        """
        self.state.K = np.full(self.r, self.cfg.K_init, dtype=np.float64)
        self.state.d_hat = np.zeros(self.r, dtype=np.float64)
        self.t_hist.clear()
        self.x_hist.clear()
        self.s_hist.clear()
        self.u_hist.clear()
        self.ueq_hist.clear()
        self.usw_hist.clear()
        self.K_hist.clear()
        self.dhat_hist.clear()

    def simulate_response(
        self,
        x0: np.ndarray,
        t_final: float,
        dt: float,
        w_func: Callable[[float], np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate closed-loop system with RK4.

        Args:
            x0: Initial state (n,)
            t_final: Final simulation time
            dt: Time step
            w_func: Optional disturbance function w(t) -> (m,) injected via B

        Returns:
            Tuple of (t_axis, xs, us):
                - t_axis: Time vector (steps,)
                - xs: State trajectory (steps, n)
                - us: Control trajectory (steps, m)

        Raises:
            ValueError: If A,B not provided for proper simulation

        Notes:
            If A,B provided: integrates x_dot = Ax + B(u + w)
            Disturbance w(t) is matched (injected through B matrix).

        Example:
            >>> t, x, u = controller.simulate_response(
            ...     x0=np.array([1.0, 0.0]),
            ...     t_final=5.0,
            ...     dt=0.01,
            ...     w_func=lambda t: np.array([0.1*np.sin(t)])
            ... )
        """
        if self.A is None or self.B is None:
            raise ValueError(
                "simulate_response requires A,B matrices for proper dynamics integration"
            )

        steps = int(np.ceil(t_final / dt))
        x = np.asarray(x0, dtype=np.float64).reshape(-1)
        t_axis = np.linspace(0.0, t_final, steps, endpoint=False)

        xs = np.zeros((steps, self.n))
        us = np.zeros((steps, self.cfg.control_dim))
        ss = np.zeros((steps, self.r))

        for i, t in enumerate(t_axis):
            # Compute control at current state
            u = self.compute(x, xdot=None, dt=dt)

            # Disturbance injection (matched via B)
            w = (
                np.zeros(self.cfg.control_dim)
                if w_func is None
                else np.asarray(w_func(t), dtype=np.float64).reshape(-1)
            )

            # RK4 integration: x_dot = A·x + B·(u + w)
            def dynamics(
                x_val: np.ndarray, u_val: np.ndarray = u, w_val: np.ndarray = w
            ) -> np.ndarray:
                # For RK4 intermediate steps, reuse current u (could recompute if needed)
                assert self.A is not None and self.B is not None
                return self.A @ x_val + self.B @ (u_val + w_val)

            k1 = dynamics(x)
            k2 = dynamics(x + 0.5 * dt * k1)
            k3 = dynamics(x + 0.5 * dt * k2)
            k4 = dynamics(x + dt * k3)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            xs[i, :] = x
            us[i, :] = u
            ss[i, :] = self.s_hist[-1] if self.s_hist else np.zeros(self.r)
            self.t_hist.append(t)

        return t_axis, xs, us

    # ----------------- Legacy/Compatibility API ----------------- #

    # Backward compatibility helpers
    def get_sliding_surface_history(self) -> list[np.ndarray]:
        """Get history of sliding surface values (backward compatibility).

        Returns:
            List of sliding surface value arrays over time

        Example:
            >>> history = controller.get_sliding_surface_history()
        """
        return [s.copy() for s in self.s_hist]

    def is_on_sliding_surface(self, tolerance: float | None = None) -> bool:
        """Check if system is on sliding surface (backward compatibility).

        Args:
            tolerance: Tolerance for surface proximity

        Returns:
            True if on sliding surface

        Example:
            >>> if controller.is_on_sliding_surface():
            ...     print("In sliding mode")
        """
        tol = tolerance if tolerance is not None else self.cfg.boundary_layer
        return self.is_on_surface(tol)

    def estimate_chattering_magnitude(self, window_size: int = 10) -> float:
        """Estimate chattering magnitude from recent sliding surface history.

        Args:
            window_size: Number of recent samples to analyze

        Returns:
            Chattering magnitude estimate (variance of sliding surface)

        Example:
            >>> chattering = controller.estimate_chattering_magnitude()
        """
        if len(self.s_hist) < window_size:
            return float("inf")

        recent_s = self.s_hist[-window_size:]
        # For SISO, use raw values; for MIMO, use norms
        if recent_s[0].shape[0] == 1:
            s_values = [float(s[0]) for s in recent_s]
            return float(np.var(s_values))
        else:
            s_norms = [np.linalg.norm(s) for s in recent_s]
            return float(np.var(s_norms))

    def get_reaching_time_estimate(self, initial_s: float) -> float:
        """Estimate time to reach sliding surface.

        Args:
            initial_s: Initial sliding surface value

        Returns:
            Estimated reaching time

        Example:
            >>> time = controller.get_reaching_time_estimate(initial_s=5.0)

        Notes:
            Estimates based on Lyapunov analysis of reaching laws
        """
        K = self.state.K.mean()  # Use mean K for estimate

        if self.cfg.reaching == "constant":
            return abs(initial_s) / K if K > 0 else float("inf")
        elif self.cfg.reaching == "exponential":
            k1, k2 = self.cfg.k1, self.cfg.k2
            denominator = k1 * K + k2 * abs(initial_s)
            return abs(initial_s) / denominator if denominator > 0 else float("inf")
        elif self.cfg.reaching == "power":
            k1, alpha = self.cfg.k1, self.cfg.alpha
            denominator = k1 * K * (1 - alpha)
            if denominator > 0:
                return (abs(initial_s) ** (1 - alpha)) / denominator
            return float("inf")
        else:
            return float("inf")

    def get_current_switching_gain(self) -> float:
        """Get current switching gain (scalar average for backward compatibility).

        Returns:
            Current switching gain value

        Example:
            >>> K = controller.get_current_switching_gain()
        """
        return float(self.state.K.mean()) if self.state.K.size > 0 else self.cfg.K_init

    def get_estimated_disturbance_bound(self) -> float:
        """Estimate disturbance bound from observer (backward compatibility).

        Returns:
            Estimated disturbance upper bound

        Example:
            >>> bound = controller.get_estimated_disturbance_bound()
        """
        return (
            float(np.linalg.norm(self.state.d_hat, ord=np.inf))
            if self.state.d_hat.size > 0
            else 0.0
        )
