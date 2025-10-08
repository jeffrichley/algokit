"""Linear Quadratic Regulator (LQR) optimal control implementation.

This module implements the Linear Quadratic Regulator for optimal state
feedback control of linear time-invariant systems. LQR minimizes a quadratic
cost function while satisfying system dynamics constraints.

The LQR problem solves:
    min J = ∫(x'Qx + u'Ru) dt

for the linear system:
    dx/dt = Ax + Bu

The solution is a state feedback controller:
    u = -Kx

where K is computed by solving the Algebraic Riccati Equation (ARE):
    A'P + PA - PBR⁻¹B'P + Q = 0

Mathematical formulation:
    - Q: State cost matrix (positive semi-definite)
    - R: Control cost matrix (positive definite)
    - K: Optimal feedback gain matrix
    - P: Solution to Algebraic Riccati Equation

Improvements:
    - Regularization for numerical stability
    - Stabilizability verification before solving ARE
    - Runge-Kutta 4th order integration
    - Discrete-time LQR support
    - Cholesky-based positive definiteness checks
"""

import logging
from enum import Enum

import numpy as np
import scipy.linalg
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class LQRType(str, Enum):
    """Type of LQR controller."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


class LQRConfig(BaseModel):
    """Configuration for LQR controller with automatic validation.

    Attributes:
        state_dim: Dimension of state vector
        control_dim: Dimension of control input vector
        A: System dynamics matrix (state_dim × state_dim)
        B: Control input matrix (state_dim × control_dim)
        Q: State cost matrix (state_dim × state_dim, positive semi-definite)
        R: Control cost matrix (control_dim × control_dim, positive definite)
        lqr_type: Type of LQR controller (continuous or discrete)
        dt: Time step for discrete-time systems (required if lqr_type=discrete)
        regularization_epsilon: Regularization for near-singular R matrix
        control_limits: Optional control saturation limits (min, max)
        debug: Whether to enable debug logging
    """

    state_dim: int = Field(gt=0, description="State vector dimension")
    control_dim: int = Field(gt=0, description="Control input dimension")
    A: list[list[float]] = Field(description="System dynamics matrix")
    B: list[list[float]] = Field(description="Control input matrix")
    Q: list[list[float]] = Field(description="State cost matrix")
    R: list[list[float]] = Field(description="Control cost matrix")
    lqr_type: LQRType = Field(
        default=LQRType.CONTINUOUS, description="Type of LQR controller"
    )
    dt: float | None = Field(
        default=None, ge=0, description="Time step for discrete-time systems"
    )
    regularization_epsilon: float = Field(
        default=1e-10, ge=0, description="Regularization for near-singular R matrix"
    )
    control_limits: tuple[float, float] | None = Field(
        default=None, description="Control saturation limits (min, max)"
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    @field_validator("A")
    @classmethod
    def validate_A_dimensions(cls, v: list[list[float]]) -> list[list[float]]:
        """Validate A matrix has correct dimensions.

        Args:
            v: A matrix as list of lists

        Returns:
            Validated A matrix

        Raises:
            ValueError: If dimensions are incorrect
        """
        if not v:
            raise ValueError("A matrix cannot be empty")

        rows = len(v)
        cols = len(v[0])

        if rows != cols:
            raise ValueError(f"A matrix must be square, got {rows}x{cols}")

        for row in v:
            if len(row) != cols:
                raise ValueError("A matrix rows must have consistent length")

        return v

    @field_validator("B")
    @classmethod
    def validate_B_dimensions(cls, v: list[list[float]]) -> list[list[float]]:
        """Validate B matrix has correct dimensions.

        Args:
            v: B matrix as list of lists

        Returns:
            Validated B matrix

        Raises:
            ValueError: If dimensions are incorrect
        """
        if not v:
            raise ValueError("B matrix cannot be empty")

        for row in v:
            if not row:
                raise ValueError("B matrix rows cannot be empty")

        cols = len(v[0])
        for row in v:
            if len(row) != cols:
                raise ValueError("B matrix rows must have consistent length")

        return v

    @field_validator("Q")
    @classmethod
    def validate_Q_matrix(cls, v: list[list[float]]) -> list[list[float]]:
        """Validate Q matrix is square, symmetric, and positive semi-definite.

        Args:
            v: Q matrix as list of lists

        Returns:
            Validated Q matrix

        Raises:
            ValueError: If Q is not valid
        """
        if not v:
            raise ValueError("Q matrix cannot be empty")

        n = len(v)
        for row in v:
            if len(row) != n:
                raise ValueError("Q matrix must be square")

        # Check symmetry
        Q_array = np.array(v)
        if not np.allclose(Q_array, Q_array.T):
            raise ValueError("Q matrix must be symmetric")

        # Check positive semi-definiteness via eigenvalues
        # (Cholesky doesn't work for semi-definite, only definite)
        eigenvalues = np.linalg.eigvalsh(Q_array)
        if not np.all(eigenvalues >= -1e-10):  # Allow small numerical errors
            raise ValueError(
                "Q matrix must be positive semi-definite (non-negative eigenvalues)"
            )

        return v

    @field_validator("R")
    @classmethod
    def validate_R_matrix(cls, v: list[list[float]]) -> list[list[float]]:
        """Validate R matrix is square, symmetric, and positive definite.

        Uses Cholesky decomposition for robust positive definiteness check.

        Args:
            v: R matrix as list of lists

        Returns:
            Validated R matrix

        Raises:
            ValueError: If R is not valid
        """
        if not v:
            raise ValueError("R matrix cannot be empty")

        n = len(v)
        for row in v:
            if len(row) != n:
                raise ValueError("R matrix must be square")

        # Check symmetry
        R_array = np.array(v)
        if not np.allclose(R_array, R_array.T):
            raise ValueError("R matrix must be symmetric")

        # Check positive definiteness via Cholesky decomposition
        # This is more numerically robust than eigenvalue check
        try:
            np.linalg.cholesky(R_array)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "R matrix must be positive definite (Cholesky decomposition failed)"
            ) from e

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

    @model_validator(mode="after")
    def validate_dt_for_discrete(self) -> "LQRConfig":
        """Validate dt is provided for discrete-time systems.

        Returns:
            Validated config

        Raises:
            ValueError: If dt is not provided for discrete systems
        """
        if self.lqr_type == LQRType.DISCRETE and self.dt is None:
            raise ValueError("dt must be provided for discrete-time LQR systems")
        if self.dt is not None and self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        return self


class LQRController:
    """Linear Quadratic Regulator for optimal state feedback control.

    This implementation:
    - Solves continuous or discrete-time Algebraic Riccati Equation
    - Computes optimal feedback gains
    - Provides state feedback control
    - Supports control saturation
    - Validates system stabilizability
    - Uses regularization for numerical stability
    - Supports RK4 integration for simulation

    Example:
        >>> A = [[0, 1], [-1, -0.5]]
        >>> B = [[0], [1]]
        >>> Q = [[1, 0], [0, 1]]
        >>> R = [[0.1]]
        >>> config = LQRConfig(state_dim=2, control_dim=1, A=A, B=B, Q=Q, R=R)
        >>> controller = LQRController(config)
        >>> control = controller.compute(state=[1.0, 0.5])
    """

    def __init__(self, config: LQRConfig) -> None:
        """Initialize LQR controller and solve for optimal gains.

        Args:
            config: LQR controller configuration

        Raises:
            ValueError: If system dimensions are inconsistent or not stabilizable
            RuntimeError: If Riccati equation solution fails
        """
        self.config = config

        # Convert matrices to numpy arrays
        self._A = np.array(config.A, dtype=np.float64)
        self._B = np.array(config.B, dtype=np.float64)
        self._Q = np.array(config.Q, dtype=np.float64)
        self._R = np.array(config.R, dtype=np.float64)

        # Add regularization to R for numerical stability
        self._R_reg = self._R + config.regularization_epsilon * np.eye(
            config.control_dim
        )

        # Validate dimensions
        self._validate_dimensions()

        # Check stabilizability before solving ARE
        if not self.is_stabilizable():
            raise ValueError(
                "System (A, B) is not stabilizable. Cannot solve LQR problem."
            )

        # Solve for optimal gain
        self._K, self._P = self._solve_lqr()

        if self.config.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug(f"LQR gains computed: K = \n{self._K}")
            logger.debug(f"Riccati solution: P = \n{self._P}")

    def _validate_dimensions(self) -> None:
        """Validate matrix dimensions are consistent.

        Raises:
            ValueError: If dimensions don't match configuration
        """
        if self._A.shape != (self.config.state_dim, self.config.state_dim):
            raise ValueError(
                f"A matrix shape {self._A.shape} doesn't match "
                f"state_dim {self.config.state_dim}"
            )

        if self._B.shape != (self.config.state_dim, self.config.control_dim):
            raise ValueError(
                f"B matrix shape {self._B.shape} doesn't match "
                f"state_dim={self.config.state_dim}, control_dim={self.config.control_dim}"
            )

        if self._Q.shape != (self.config.state_dim, self.config.state_dim):
            raise ValueError(
                f"Q matrix shape {self._Q.shape} doesn't match "
                f"state_dim {self.config.state_dim}"
            )

        if self._R.shape != (self.config.control_dim, self.config.control_dim):
            raise ValueError(
                f"R matrix shape {self._R.shape} doesn't match "
                f"control_dim {self.config.control_dim}"
            )

    def _solve_lqr(self) -> tuple[np.ndarray, np.ndarray]:
        """Solve the Algebraic Riccati Equation (continuous or discrete).

        Uses regularized R matrix for numerical stability.

        Returns:
            Tuple of (K, P) where:
                K: Optimal feedback gain matrix
                P: Solution to Riccati equation

        Raises:
            RuntimeError: If Riccati equation has no solution
        """
        try:
            if self.config.lqr_type == LQRType.CONTINUOUS:
                # Solve continuous-time ARE: A'P + PA - PBR⁻¹B'P + Q = 0
                P = scipy.linalg.solve_continuous_are(
                    self._A, self._B, self._Q, self._R_reg
                )
                # Compute optimal gain: K = R⁻¹B'P
                K = np.linalg.solve(self._R_reg, self._B.T @ P)

            else:  # DISCRETE
                # Solve discrete-time ARE
                P = scipy.linalg.solve_discrete_are(
                    self._A, self._B, self._Q, self._R_reg
                )
                # Compute optimal gain: K = (B'PB + R)⁻¹B'PA
                K = np.linalg.solve(
                    self._B.T @ P @ self._B + self._R_reg, self._B.T @ P @ self._A
                )

            return K, P

        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                f"Failed to solve Algebraic Riccati Equation: {e}"
            ) from e

    @property
    def gain_matrix(self) -> np.ndarray:
        """Get optimal feedback gain matrix K."""
        return self._K.copy()

    @property
    def riccati_solution(self) -> np.ndarray:
        """Get Riccati equation solution matrix P."""
        return self._P.copy()

    def compute_raw_control(
        self,
        state: list[float] | np.ndarray,
        reference: list[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute raw optimal control without saturation.

        For reference tracking, computes: u = u_ref - K(x - x_ref)
        where u_ref is the feedforward term that holds x_ref at equilibrium.

        Args:
            state: Current state vector
            reference: Optional reference state (default is zero)

        Returns:
            Raw control input vector (before saturation)

        Raises:
            ValueError: If state dimension is incorrect

        Example:
            >>> controller = LQRController(config)
            >>> u_raw = controller.compute_raw_control(state=[1.0, 0.5])
            >>> # With reference tracking
            >>> u_raw = controller.compute_raw_control(state=[1.0, 0.5], reference=[5.0, 0.0])
        """
        # Convert state to numpy array
        x = np.array(state, dtype=np.float64)

        if x.shape[0] != self.config.state_dim:
            raise ValueError(
                f"State dimension {x.shape[0]} doesn't match "
                f"state_dim {self.config.state_dim}"
            )

        # Handle reference tracking with feedforward
        if reference is not None:
            x_ref = np.array(reference, dtype=np.float64)
            if x_ref.shape[0] != self.config.state_dim:
                raise ValueError(
                    f"Reference dimension {x_ref.shape[0]} doesn't match "
                    f"state_dim {self.config.state_dim}"
                )

            # Compute tracking error
            error = x - x_ref

            # Compute feedforward control to maintain equilibrium at x_ref
            # At equilibrium: 0 = Ax_ref + Bu_ref → u_ref = -B†Ax_ref
            A_np = np.array(self.config.A, dtype=np.float64)
            B_np = np.array(self.config.B, dtype=np.float64)

            try:
                # Use pseudoinverse for numerical stability
                u_ref = -np.linalg.pinv(B_np) @ (A_np @ x_ref)
                u_ref = u_ref.flatten()
            except np.linalg.LinAlgError:
                logger.warning(
                    "Could not compute feedforward term, using feedback only"
                )
                u_ref = np.zeros(self.config.control_dim)

            # Total control: feedforward + feedback
            control = u_ref - self._K @ error

            if self.config.debug:
                logger.debug(
                    f"LQR tracking: x={x}, x_ref={x_ref}, error={error}, "
                    f"u_ref={u_ref}, u_feedback={-self._K @ error}, u_total={control}"
                )
        else:
            # Standard LQR regulation to origin
            control = -self._K @ x

            if self.config.debug:
                logger.debug(f"LQR regulation: x={x}, u={control}")

        return control

    def apply_saturation(self, control: np.ndarray) -> np.ndarray:
        """Apply control saturation limits.

        Args:
            control: Raw control input vector

        Returns:
            Saturated control input vector

        Example:
            >>> controller = LQRController(config)
            >>> u_sat = controller.apply_saturation(u_raw)
        """
        if self.config.control_limits is None:
            return control

        min_val, max_val = self.config.control_limits
        saturated = np.clip(control, min_val, max_val)

        if self.config.debug and not np.allclose(saturated, control):
            logger.debug(f"Control saturated: {control} -> {saturated}")

        return saturated

    def compute(
        self,
        state: list[float] | np.ndarray,
        reference: list[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute optimal control for given state with saturation.

        Args:
            state: Current state vector
            reference: Optional reference state (default is zero)

        Returns:
            Control input vector (with saturation if configured)

        Raises:
            ValueError: If state dimension is incorrect

        Example:
            >>> controller = LQRController(config)
            >>> u = controller.compute(state=[1.0, 0.5])
        """
        # Compute raw control
        control = self.compute_raw_control(state, reference)

        # Apply saturation
        control = self.apply_saturation(control)

        return control

    def compute_cost(
        self, state: list[float] | np.ndarray, control: list[float] | np.ndarray
    ) -> float:
        """Compute LQR cost for given state and control.

        Args:
            state: State vector
            control: Control vector

        Returns:
            Cost value J = x'Qx + u'Ru

        Example:
            >>> cost = controller.compute_cost(state=[1.0, 0.5], control=[0.3])
        """
        x = np.array(state, dtype=np.float64)
        u = np.array(control, dtype=np.float64)

        state_cost = x.T @ self._Q @ x
        control_cost = u.T @ self._R @ u

        return float(state_cost + control_cost)

    def is_controllable(self) -> bool:
        """Check if the system (A, B) is controllable.

        Uses controllability matrix C = [B AB A²B ... A^(n-1)B].
        More efficient computation than matrix_power.

        Returns:
            True if system is controllable, False otherwise

        Example:
            >>> if controller.is_controllable():
            ...     print("System is controllable")
        """
        # Compute controllability matrix [B AB A²B ... A^(n-1)B]
        n = self.config.state_dim
        controllability_matrix = self._B.copy()
        A_power_B = self._B.copy()

        for _ in range(1, n):
            A_power_B = self._A @ A_power_B
            controllability_matrix = np.hstack([controllability_matrix, A_power_B])

        # System is controllable if rank = n
        rank = np.linalg.matrix_rank(controllability_matrix)
        return rank == n

    def is_stabilizable(self) -> bool:
        """Check if the system (A, B) is stabilizable.

        A system is stabilizable if all unstable modes are controllable.
        This is a weaker condition than full controllability and is
        sufficient for LQR.

        Returns:
            True if system is stabilizable, False otherwise

        Example:
            >>> if controller.is_stabilizable():
            ...     print("System is stabilizable")
        """
        # Compute eigenvalues and eigenvectors of A
        eigenvalues, eigenvectors = np.linalg.eig(self._A)

        # Check unstable modes (Re(λ) >= 0 for continuous, |λ| >= 1 for discrete)
        for i, eigenvalue in enumerate(eigenvalues):
            if self.config.lqr_type == LQRType.CONTINUOUS:
                is_unstable = np.real(eigenvalue) >= -1e-10
            else:  # DISCRETE
                is_unstable = np.abs(eigenvalue) >= 1.0 - 1e-10

            if is_unstable:
                # Check if this mode is controllable
                # Mode i is controllable if B'v_i ≠ 0, where v_i is left eigenvector
                v_i = eigenvectors[:, i]
                controllability = self._B.T @ v_i
                if np.allclose(controllability, 0):
                    return False

        return True

    def get_closed_loop_eigenvalues(self) -> np.ndarray:
        """Compute eigenvalues of closed-loop system A - BK.

        Returns:
            Array of closed-loop eigenvalues

        Example:
            >>> eigenvalues = controller.get_closed_loop_eigenvalues()
            >>> print(f"All stable: {np.all(np.real(eigenvalues) < 0)}")
        """
        closed_loop_A = self._A - self._B @ self._K
        return np.linalg.eigvals(closed_loop_A)

    def simulate(
        self,
        initial_state: list[float] | np.ndarray,
        time_steps: int,
        dt: float = 0.01,
        integration_method: str = "rk4",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate closed-loop system response.

        Args:
            initial_state: Initial state vector
            time_steps: Number of simulation steps
            dt: Time step size (ignored for discrete-time systems)
            integration_method: Integration method ('euler' or 'rk4')

        Returns:
            Tuple of (states, controls) arrays

        Example:
            >>> states, controls = controller.simulate([1.0, 0.0], time_steps=100)
        """
        x = np.array(initial_state, dtype=np.float64)
        states = np.zeros((time_steps + 1, self.config.state_dim))
        controls = np.zeros((time_steps, self.config.control_dim))

        states[0] = x

        # Use configured dt for discrete systems
        if self.config.lqr_type == LQRType.DISCRETE:
            if self.config.dt is None:
                raise ValueError("dt must be set for discrete-time simulation")
            dt = self.config.dt

        for t in range(time_steps):
            # Compute control
            u = self.compute(x)
            controls[t] = u

            # Integrate system dynamics
            if self.config.lqr_type == LQRType.DISCRETE:
                # Discrete update: x[k+1] = Ax[k] + Bu[k]
                x = self._A @ x + self._B @ u
            else:
                # Continuous-time integration
                if integration_method == "rk4":
                    x = self._rk4_step(x, u, dt)
                else:  # euler
                    dx_dt = self._A @ x + self._B @ u
                    x = x + dx_dt * dt

            states[t + 1] = x

        return states, controls

    def _rk4_step(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """Perform one RK4 integration step.

        Args:
            x: Current state
            u: Current control
            dt: Time step

        Returns:
            Next state
        """

        def dynamics(state: np.ndarray) -> np.ndarray:
            """System dynamics dx/dt = Ax + Bu."""
            return self._A @ state + self._B @ u

        # RK4 coefficients
        k1 = dynamics(x)
        k2 = dynamics(x + 0.5 * dt * k1)
        k3 = dynamics(x + 0.5 * dt * k2)
        k4 = dynamics(x + dt * k3)

        # Weighted average
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return x_next
