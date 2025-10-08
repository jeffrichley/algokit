"""Sliding Mode Control implementation with chattering reduction.

This module implements Sliding Mode Control (SMC), a nonlinear control technique
that uses discontinuous control to drive the system state to a sliding surface.
SMC provides robustness to uncertainties and disturbances.

The sliding mode controller consists of two components:
1. Equivalent control: u_eq drives the system along the sliding surface
2. Switching control: u_sw ensures reaching and staying on the surface

Mathematical formulation:
    Sliding surface: s(x) = cx = 0
    Control law: u = u_eq + u_sw
    u_sw = -K sign(s)  (with chattering reduction)

Key features:
    - Robustness to matched uncertainties
    - Finite-time convergence
    - Insensitivity to parameter variations
    - Chattering reduction via boundary layer
"""

import logging

import numpy as np
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class SlidingModeConfig(BaseModel):
    """Configuration for sliding mode controller with automatic validation.

    Attributes:
        state_dim: Dimension of state vector
        control_dim: Dimension of control input vector
        sliding_surface_coeffs: Coefficients for sliding surface s = Cx (MIMO: matrix rows)
        switching_gain: Gain K for switching control (K > 0)
        boundary_layer_width: Boundary layer thickness φ for chattering reduction
        reaching_law: Type of reaching law ('constant', 'exponential', 'power')
        power_reaching_alpha: Exponent for power reaching law (0 < α < 1)
        use_saturation: Use saturation function instead of sign for smoothness
        control_limits: Optional control saturation limits (min, max)
        debug: Whether to enable debug logging
        system_matrix_A: System dynamics matrix A (for x_dot = Ax + Bu)
        control_matrix_B: Control input matrix B (for x_dot = Ax + Bu)
        exponential_reaching_k1: Coefficient k1 for exponential reaching law
        exponential_reaching_k2: Coefficient k2 for exponential reaching law
        power_reaching_k1: Coefficient k1 for power reaching law
        power_reaching_k2: Coefficient k2 for power reaching law
        use_smooth_approximation: Use tanh-based smooth approximation instead of saturation
        smooth_approximation_slope: Slope parameter for tanh approximation
        adaptive_gain: Enable adaptive switching gain update
        adaptive_gain_rate: Adaptation rate for switching gain (eta > 0)
        disturbance_estimation_window: Window size for disturbance estimation
    """

    state_dim: int = Field(gt=0, description="State vector dimension")
    control_dim: int = Field(gt=0, description="Control input dimension")
    sliding_surface_coeffs: list[float] | list[list[float]] = Field(
        description="Sliding surface coefficients (vector for SISO, matrix for MIMO)"
    )
    switching_gain: float = Field(gt=0.0, description="Switching control gain (K > 0)")
    boundary_layer_width: float = Field(
        default=0.1,
        gt=0.0,
        description="Boundary layer thickness for chattering reduction",
    )
    reaching_law: str = Field(
        default="constant",
        description="Reaching law type (constant, exponential, power)",
    )
    power_reaching_alpha: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="Exponent for power reaching law",
    )
    use_saturation: bool = Field(
        default=True, description="Use saturation instead of sign function"
    )
    control_limits: tuple[float, float] | None = Field(
        default=None, description="Control saturation limits (min, max)"
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    # System dynamics matrices (optional, for full-state feedback)
    system_matrix_A: list[list[float]] | None = Field(
        default=None, description="System dynamics matrix A (state_dim x state_dim)"
    )
    control_matrix_B: list[list[float]] | None = Field(
        default=None, description="Control input matrix B (state_dim x control_dim)"
    )

    # Reaching law coefficients
    exponential_reaching_k1: float = Field(
        default=1.0, gt=0.0, description="Coefficient k1 for exponential reaching law"
    )
    exponential_reaching_k2: float = Field(
        default=0.5, gt=0.0, description="Coefficient k2 for exponential reaching law"
    )
    power_reaching_k1: float = Field(
        default=1.0, gt=0.0, description="Coefficient k1 for power reaching law"
    )
    power_reaching_k2: float = Field(
        default=1.0, gt=0.0, description="Coefficient k2 for power reaching law"
    )

    # Smooth approximation settings
    use_smooth_approximation: bool = Field(
        default=False, description="Use tanh-based smooth approximation"
    )
    smooth_approximation_slope: float = Field(
        default=2.0, gt=0.0, description="Slope parameter for tanh approximation"
    )

    # Adaptive gain settings
    adaptive_gain: bool = Field(
        default=False, description="Enable adaptive switching gain"
    )
    adaptive_gain_rate: float = Field(
        default=0.1, gt=0.0, description="Adaptation rate for switching gain"
    )
    disturbance_estimation_window: int = Field(
        default=10, gt=0, description="Window size for disturbance estimation"
    )

    @field_validator("sliding_surface_coeffs")
    @classmethod
    def validate_coeffs_not_empty(
        cls, v: list[float] | list[list[float]]
    ) -> list[float] | list[list[float]]:
        """Validate sliding surface coefficients are not empty.

        Args:
            v: Sliding surface coefficients (vector for SISO, matrix for MIMO)

        Returns:
            Validated coefficients

        Raises:
            ValueError: If coefficients are empty or invalid
        """
        if not v:
            raise ValueError("Sliding surface coefficients cannot be empty")

        # Check if it's a matrix (MIMO case)
        if v and isinstance(v[0], list):
            for row in v:
                if not row:
                    raise ValueError("Sliding surface coefficient rows cannot be empty")

        return v

    @field_validator("reaching_law")
    @classmethod
    def validate_reaching_law(cls, v: str) -> str:
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
            raise ValueError(f"Reaching law must be one of {valid_laws}, got '{v}'")
        return v

    @field_validator("control_limits")
    @classmethod
    def validate_control_limits(
        cls, v: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        """Validate control limits are ordered correctly.

        Args:
            v: Control limits tuple (min, max)

        Returns:
            Validated control limits

        Raises:
            ValueError: If min >= max
        """
        if v is not None:
            min_val, max_val = v
            if min_val >= max_val:
                raise ValueError(
                    f"Control min ({min_val}) must be less than max ({max_val})"
                )
        return v

    @field_validator("system_matrix_A")
    @classmethod
    def validate_system_matrix_a(
        cls, v: list[list[float]] | None
    ) -> list[list[float]] | None:
        """Validate system matrix A dimensions.

        Args:
            v: System matrix A

        Returns:
            Validated system matrix A

        Raises:
            ValueError: If matrix is not square or empty
        """
        if v is not None:
            if not v:
                raise ValueError("System matrix A cannot be empty")

            n_rows = len(v)
            for i, row in enumerate(v):
                if len(row) != n_rows:
                    raise ValueError(
                        f"System matrix A must be square, row {i} has {len(row)} "
                        f"elements but expected {n_rows}"
                    )
        return v

    @field_validator("control_matrix_B")
    @classmethod
    def validate_control_matrix_b(
        cls, v: list[list[float]] | None
    ) -> list[list[float]] | None:
        """Validate control matrix B dimensions.

        Args:
            v: Control matrix B

        Returns:
            Validated control matrix B

        Raises:
            ValueError: If matrix is empty or has inconsistent dimensions
        """
        if v is not None:
            if not v:
                raise ValueError("Control matrix B cannot be empty")

            n_cols = len(v[0]) if v else 0
            for i, row in enumerate(v):
                if len(row) != n_cols:
                    raise ValueError(
                        f"Control matrix B rows must have same length, row {i} has "
                        f"{len(row)} elements but expected {n_cols}"
                    )
        return v


class SlidingModeController:
    """Sliding Mode Controller with chattering reduction.

    This implementation provides:
    - Configurable sliding surface design
    - Multiple reaching law options
    - Boundary layer for chattering reduction
    - Saturation function for smooth control
    - Lyapunov stability guarantees

    Example:
        >>> config = SlidingModeConfig(
        ...     state_dim=2,
        ...     control_dim=1,
        ...     sliding_surface_coeffs=[1.0, 2.0],
        ...     switching_gain=5.0,
        ...     boundary_layer_width=0.1
        ... )
        >>> controller = SlidingModeController(config)
        >>> control = controller.compute(state=[1.0, 0.5], state_derivative=[0.1, -0.2])
    """

    def __init__(self, config: SlidingModeConfig) -> None:
        """Initialize sliding mode controller.

        Args:
            config: Sliding mode controller configuration

        Raises:
            ValueError: If configuration is inconsistent
        """
        self.config = config

        # Convert sliding surface coefficients to numpy array
        # Support both SISO (vector) and MIMO (matrix) cases
        coeffs = config.sliding_surface_coeffs
        if coeffs and isinstance(coeffs[0], list):
            # MIMO case: C is a matrix (sliding_dim x state_dim)
            self._C = np.array(coeffs, dtype=np.float64)
            self._sliding_dim = self._C.shape[0]

            if self._C.shape[1] != config.state_dim:
                raise ValueError(
                    f"Sliding surface matrix columns ({self._C.shape[1]}) "
                    f"don't match state_dim ({config.state_dim})"
                )
        else:
            # SISO case: C is a row vector
            self._C = np.array(coeffs, dtype=np.float64).reshape(1, -1)
            self._sliding_dim = 1

            if self._C.shape[1] != config.state_dim:
                raise ValueError(
                    f"Sliding surface coefficients dimension ({self._C.shape[1]}) "
                    f"doesn't match state_dim ({config.state_dim})"
                )

        # System dynamics matrices
        self._A: np.ndarray | None
        self._B: np.ndarray | None
        if config.system_matrix_A is not None:
            self._A = np.array(config.system_matrix_A, dtype=np.float64)
            if self._A.shape != (config.state_dim, config.state_dim):
                raise ValueError(
                    f"System matrix A shape {self._A.shape} doesn't match "
                    f"expected ({config.state_dim}, {config.state_dim})"
                )
        else:
            self._A = None

        if config.control_matrix_B is not None:
            self._B = np.array(config.control_matrix_B, dtype=np.float64)
            if self._B.shape != (config.state_dim, config.control_dim):
                raise ValueError(
                    f"Control matrix B shape {self._B.shape} doesn't match "
                    f"expected ({config.state_dim}, {config.control_dim})"
                )
        else:
            self._B = None

        # Validate that both A and B are provided together
        if (self._A is not None) != (self._B is not None):
            raise ValueError(
                "System matrix A and control matrix B must both be provided or both be None"
            )

        # Adaptive gain state
        self._current_switching_gain = config.switching_gain
        self._disturbance_estimates: list[float] = []

        # History tracking
        self._sliding_surface_history: list[np.ndarray] = []
        self._control_history: list[np.ndarray] = []

        if self.config.debug:
            logger.setLevel(logging.DEBUG)

    def compute_sliding_surface(self, state: np.ndarray) -> np.ndarray:
        """Compute sliding surface value s(x) = Cx.

        Args:
            state: Current state vector

        Returns:
            Sliding surface value(s) - scalar for SISO, vector for MIMO

        Example:
            >>> s = controller.compute_sliding_surface(np.array([1.0, 0.5]))
        """
        # Compute s = C @ x (matrix-vector product)
        s = self._C @ state

        # Return as 1D array for consistency
        return s.ravel() if s.ndim > 1 else s

    def _smooth_sign(self, s: np.ndarray) -> np.ndarray:
        """Smooth approximation of sign function using tanh.

        Args:
            s: Input value(s)

        Returns:
            Smooth sign approximation

        Notes:
            Uses tanh(slope * s) for smooth approximation
        """
        slope = self.config.smooth_approximation_slope
        return np.tanh(slope * s)

    def _compute_equivalent_control(
        self, state: np.ndarray, state_derivative: np.ndarray
    ) -> np.ndarray:
        """Compute equivalent control u_eq.

        Args:
            state: Current state vector
            state_derivative: Time derivative of state

        Returns:
            Equivalent control vector

        Notes:
            If A and B matrices are provided:
                u_eq = -(CB)^{-1} (CA x + C x_dot_desired)
            Otherwise:
                u_eq = -C @ x_dot (simplified)
        """
        if self._A is not None and self._B is not None:
            # Full-state dynamics: x_dot = Ax + Bu
            # Sliding surface: s = Cx
            # ds/dt = C(Ax + Bu) = CAx + CBu
            # For ds/dt = 0: u_eq = -(CB)^{-1} (CAx)

            CA = self._C @ self._A
            CB = self._C @ self._B

            # Check if CB is invertible
            try:
                if self._sliding_dim == self.config.control_dim:
                    # Square case: direct inversion
                    CB_inv = np.linalg.inv(CB)
                else:
                    # Non-square case: use pseudoinverse
                    CB_inv = np.linalg.pinv(CB)

                u_eq = -CB_inv @ (CA @ state)
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse if inversion fails
                CB_inv = np.linalg.pinv(CB)
                u_eq = -CB_inv @ (CA @ state)
        else:
            # Simplified equivalent control: u_eq = -C @ x_dot
            # This assumes the user has already accounted for system dynamics in state_derivative
            s_dot = self._C @ state_derivative
            u_eq = -s_dot

            # Expand to control dimension if needed
            if self._sliding_dim == 1 and self.config.control_dim > 1:
                u_eq = np.full(self.config.control_dim, u_eq[0])

        return u_eq

    def _update_adaptive_gain(self, s: np.ndarray) -> None:
        """Update switching gain adaptively based on sliding surface magnitude.

        Args:
            s: Sliding surface value(s)

        Notes:
            Updates switching gain using: K_dot = eta * |s|
        """
        if not self.config.adaptive_gain:
            return

        # Adaptation law: K_dot = eta * ||s||
        s_magnitude = float(np.linalg.norm(s))
        self._disturbance_estimates.append(s_magnitude)

        # Update gain
        eta = self.config.adaptive_gain_rate
        self._current_switching_gain += eta * s_magnitude

        # Keep recent estimates only
        if len(self._disturbance_estimates) > self.config.disturbance_estimation_window:
            self._disturbance_estimates = self._disturbance_estimates[
                -self.config.disturbance_estimation_window :
            ]

    def get_current_switching_gain(self) -> float:
        """Get current switching gain (may be adapted).

        Returns:
            Current switching gain value
        """
        return self._current_switching_gain

    def get_estimated_disturbance_bound(self) -> float:
        """Estimate disturbance bound from recent sliding surface values.

        Returns:
            Estimated disturbance upper bound

        Notes:
            Uses maximum of recent disturbance estimates
        """
        if not self._disturbance_estimates:
            return 0.0

        return float(max(self._disturbance_estimates))

    def compute(
        self,
        state: list[float] | np.ndarray,
        state_derivative: list[float] | np.ndarray,
        disturbance_bound: float = 0.0,
    ) -> np.ndarray:
        """Compute sliding mode control output.

        Args:
            state: Current state vector
            state_derivative: Time derivative of state (dx/dt)
            disturbance_bound: Upper bound on disturbance magnitude

        Returns:
            Control output vector

        Raises:
            ValueError: If dimensions are incorrect

        Example:
            >>> u = controller.compute(
            ...     state=[1.0, 0.5],
            ...     state_derivative=[0.1, -0.2],
            ...     disturbance_bound=0.5
            ... )
        """
        # Convert inputs to numpy arrays
        x = np.array(state, dtype=np.float64)
        x_dot = np.array(state_derivative, dtype=np.float64)

        if x.shape[0] != self.config.state_dim:
            raise ValueError(f"State dimension {x.shape[0]} doesn't match state_dim")

        if x_dot.shape[0] != self.config.state_dim:
            raise ValueError(
                f"State derivative dimension {x_dot.shape[0]} doesn't match state_dim"
            )

        # Compute sliding surface (returns vector for MIMO, scalar for SISO)
        s = self.compute_sliding_surface(x)
        self._sliding_surface_history.append(s.copy())

        # Update adaptive gain if enabled
        self._update_adaptive_gain(s)

        # Compute equivalent control using full-state dynamics if available
        u_eq = self._compute_equivalent_control(x, x_dot)

        # Compute switching control based on reaching law
        if self.config.reaching_law == "constant":
            u_sw = self._constant_reaching_law(s, disturbance_bound)
        elif self.config.reaching_law == "exponential":
            u_sw = self._exponential_reaching_law(s, disturbance_bound)
        elif self.config.reaching_law == "power":
            u_sw = self._power_reaching_law(s, disturbance_bound)
        else:
            u_sw = np.zeros(self.config.control_dim)

        # Total control
        control = u_eq + u_sw

        # Ensure correct dimensions
        if control.shape[0] != self.config.control_dim:
            # Broadcast if needed
            if self._sliding_dim == 1 and self.config.control_dim > 1:
                control = np.full(self.config.control_dim, control[0], dtype=np.float64)
            else:
                raise ValueError(
                    f"Control dimension mismatch: got {control.shape[0]}, "
                    f"expected {self.config.control_dim}"
                )

        # Apply saturation if specified
        if self.config.control_limits is not None:
            min_val, max_val = self.config.control_limits
            control = np.clip(control, min_val, max_val)

        # Store control history
        self._control_history.append(control.copy())

        if self.config.debug:
            logger.debug(
                f"SMC: s={s}, u_eq={u_eq}, u_sw={u_sw}, u={control}, K={self._current_switching_gain:.3f}"
            )

        return control

    def _constant_reaching_law(
        self, s: np.ndarray, disturbance_bound: float
    ) -> np.ndarray:
        """Constant reaching law: ds/dt = -K sign(s).

        Args:
            s: Sliding surface value(s)
            disturbance_bound: Disturbance upper bound

        Returns:
            Switching control value(s)
        """
        K = self._current_switching_gain + disturbance_bound

        if self.config.use_smooth_approximation:
            # Smooth tanh approximation
            sign_s = self._smooth_sign(s)
        elif self.config.use_saturation:
            # Saturation function for smoothness
            sign_s = self._saturate(s, self.config.boundary_layer_width)
        else:
            # Pure sign function
            sign_s = np.sign(s)

        # Compute switching control: u_sw = -K * sign(s)
        # For MIMO: may need to map from sliding space to control space
        u_sw = -K * sign_s

        # Map to control space if dimensions differ
        if self._sliding_dim != self.config.control_dim:
            # Use pseudoinverse of CB to map back
            if self._A is not None and self._B is not None:
                CB = self._C @ self._B
                CB_pinv = np.linalg.pinv(CB)
                u_sw = CB_pinv @ u_sw
            else:
                # Simple broadcast for scalar case
                if self._sliding_dim == 1:
                    u_sw = np.full(self.config.control_dim, u_sw[0])

        return u_sw

    def _exponential_reaching_law(
        self, s: np.ndarray, disturbance_bound: float
    ) -> np.ndarray:
        """Exponential reaching law: ds/dt = -k1 * K sign(s) - k2 * s.

        Args:
            s: Sliding surface value(s)
            disturbance_bound: Disturbance upper bound

        Returns:
            Switching control value(s)

        Notes:
            Uses configurable k1 and k2 coefficients for tuning convergence
        """
        K = self._current_switching_gain + disturbance_bound
        k1 = self.config.exponential_reaching_k1
        k2 = self.config.exponential_reaching_k2

        if self.config.use_smooth_approximation:
            sign_s = self._smooth_sign(s)
        elif self.config.use_saturation:
            sign_s = self._saturate(s, self.config.boundary_layer_width)
        else:
            sign_s = np.sign(s)

        # Exponential reaching: u_sw = -k1*K*sign(s) - k2*s
        u_sw = -k1 * K * sign_s - k2 * s

        # Map to control space if dimensions differ
        if self._sliding_dim != self.config.control_dim:
            if self._A is not None and self._B is not None:
                CB = self._C @ self._B
                CB_pinv = np.linalg.pinv(CB)
                u_sw = CB_pinv @ u_sw
            else:
                if self._sliding_dim == 1:
                    u_sw = np.full(self.config.control_dim, u_sw[0])

        return u_sw

    def _power_reaching_law(
        self, s: np.ndarray, disturbance_bound: float
    ) -> np.ndarray:
        """Power reaching law: ds/dt = -k1 * K |s|^α sign(s) - k2 * s.

        Args:
            s: Sliding surface value(s)
            disturbance_bound: Disturbance upper bound

        Returns:
            Switching control value(s)

        Notes:
            Uses configurable k1 and k2 coefficients for tuning convergence
        """
        K = self._current_switching_gain + disturbance_bound
        k1 = self.config.power_reaching_k1
        k2 = self.config.power_reaching_k2
        alpha = self.config.power_reaching_alpha

        if self.config.use_smooth_approximation:
            sign_s = self._smooth_sign(s)
        elif self.config.use_saturation:
            sign_s = self._saturate(s, self.config.boundary_layer_width)
        else:
            sign_s = np.sign(s)

        # Power reaching: u_sw = -k1*K*|s|^α*sign(s) - k2*s
        # Handle element-wise power for vector s
        s_abs_alpha = np.abs(s) ** alpha
        u_sw = -k1 * K * s_abs_alpha * sign_s - k2 * s

        # Map to control space if dimensions differ
        if self._sliding_dim != self.config.control_dim:
            if self._A is not None and self._B is not None:
                CB = self._C @ self._B
                CB_pinv = np.linalg.pinv(CB)
                u_sw = CB_pinv @ u_sw
            else:
                if self._sliding_dim == 1:
                    u_sw = np.full(self.config.control_dim, u_sw[0])

        return u_sw

    def _saturate(self, s: np.ndarray, phi: float) -> np.ndarray:
        """Saturation function for chattering reduction.

        Args:
            s: Input value(s)
            phi: Boundary layer width

        Returns:
            Saturated value(s) in [-1, 1]

        Notes:
            Vectorized to support MIMO systems
        """
        # Vectorized saturation: sat(s) = s/phi if |s| <= phi, else sign(s)
        return np.where(np.abs(s) <= phi, s / phi, np.sign(s))

    def is_on_sliding_surface(self, tolerance: float | None = None) -> bool:
        """Check if system is on the sliding surface.

        Args:
            tolerance: Tolerance for surface proximity (uses boundary_layer_width if None)

        Returns:
            True if on sliding surface, False otherwise

        Example:
            >>> if controller.is_on_sliding_surface():
            ...     print("System is in sliding mode")

        Notes:
            For MIMO systems, checks if ||s|| <= tolerance
        """
        if not self._sliding_surface_history:
            return False

        tol = tolerance if tolerance is not None else self.config.boundary_layer_width
        current_s = self._sliding_surface_history[-1]

        # For MIMO, check norm of sliding surface vector
        return bool(np.linalg.norm(current_s) <= tol)

    def get_sliding_surface_history(self) -> list[np.ndarray]:
        """Get history of sliding surface values.

        Returns:
            List of sliding surface value arrays over time

        Example:
            >>> history = controller.get_sliding_surface_history()
            >>> print(f"Surface values: {history}")

        Notes:
            Returns list of arrays; each array is scalar for SISO, vector for MIMO
        """
        return [s.copy() for s in self._sliding_surface_history]

    def reset(self) -> None:
        """Reset controller state and clear history.

        Example:
            >>> controller.reset()
        """
        self._sliding_surface_history = []
        self._control_history = []
        self._disturbance_estimates = []
        self._current_switching_gain = self.config.switching_gain

    def estimate_chattering_magnitude(self, window_size: int = 10) -> float:
        """Estimate chattering magnitude from recent control history.

        Args:
            window_size: Number of recent samples to analyze

        Returns:
            Chattering magnitude estimate (variance of sliding surface norm)

        Example:
            >>> chattering = controller.estimate_chattering_magnitude()
            >>> print(f"Chattering level: {chattering:.4f}")

        Notes:
            For MIMO systems, computes variance of ||s|| over time
        """
        if len(self._sliding_surface_history) < window_size:
            return float("inf")

        recent_s = self._sliding_surface_history[-window_size:]
        # For MIMO, use norms; for SISO, use raw values to detect oscillation
        if len(recent_s[0].shape) == 0 or recent_s[0].shape[0] == 1:
            # SISO: use raw values to detect sign oscillations (chattering)
            s_values = [
                float(s.flat[0]) if hasattr(s, "flat") else float(s) for s in recent_s
            ]
            return float(np.var(s_values))
        else:
            # MIMO: use norms of sliding surface vectors
            s_norms = [np.linalg.norm(s) for s in recent_s]
            return float(np.var(s_norms))

    def get_reaching_time_estimate(self, initial_s: float) -> float:
        """Estimate time to reach sliding surface.

        Args:
            initial_s: Initial sliding surface value (magnitude for MIMO)

        Returns:
            Estimated reaching time (finite for constant/exponential laws)

        Example:
            >>> time = controller.get_reaching_time_estimate(initial_s=5.0)
            >>> print(f"Estimated reaching time: {time:.2f}s")

        Notes:
            Estimates are based on Lyapunov analysis of reaching laws
        """
        K = self._current_switching_gain

        if self.config.reaching_law == "constant":
            # T = |s(0)| / K
            return abs(initial_s) / K
        elif self.config.reaching_law == "exponential":
            # With k1 and k2 coefficients
            k1 = self.config.exponential_reaching_k1
            k2 = self.config.exponential_reaching_k2
            # Faster convergence: approximately T ≈ ln(s0) / (k1*K + k2)
            if k1 * K + k2 > 0:
                return abs(initial_s) / (k1 * K + k2 * abs(initial_s))
            else:
                return abs(initial_s) / K
        elif self.config.reaching_law == "power":
            # Finite-time convergence
            k1 = self.config.power_reaching_k1
            alpha = self.config.power_reaching_alpha
            # T = s0^(1-α) / (k1 * K * (1-α))
            return (abs(initial_s) ** (1 - alpha)) / (k1 * K * (1 - alpha))
        else:
            return float("inf")
