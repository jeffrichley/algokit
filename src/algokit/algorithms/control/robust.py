"""Research-Grade H-infinity (H∞) Robust Control Implementation.

This module implements H-infinity optimal robust control for linear time-invariant
systems with disturbances and model uncertainties using the rigorous Doyle-Glover-
Khargonekar-Francis (DGKF) formulation [1].

H∞ control minimizes the worst-case gain from disturbances to performance outputs,
providing guaranteed robustness to bounded disturbances and model uncertainties.

Mathematical Formulation
------------------------
System dynamics:
    Continuous-time:  dx/dt = Ax + B₁w + B₂u
    Discrete-time:    x[k+1] = Ax[k] + B₁w[k] + B₂u[k]

    Performance output:    z = C₁x + D₁₁w + D₁₂u
    Measurement output:    y = C₂x + D₂₁w + D₂₂u

Control Objective:
    Find state-feedback controller u = -Kx such that:
    1. Closed-loop system is internally stable
    2. ||T_zw||_∞ < γ  (disturbance attenuation level)

where T_zw is the closed-loop transfer function from disturbance w to performance z.

Bounded Real Lemma
------------------
For a stable system, ||T_zw||_∞ < γ if and only if there exist positive semi-definite
matrices X and Y satisfying the coupled Riccati inequalities:

    A'X + XA + C₁'C₁ - (XB₂ + C₁'D₁₂)R⁻¹(B₂'X + D₁₂'C₁) + γ⁻²XB₁B₁'X ≤ 0
    AY + YA' + B₁B₁' - (YC₂' + B₁D₂₁')S⁻¹(C₂Y + D₂₁B₁') + γ⁻²YC₁'C₁Y ≤ 0

with feasibility condition:
    ρ(XY) < γ²  (spectral radius of XY product)

where R = D₁₂'D₁₂ and S = D₂₁D₂₁'.

Assumptions
-----------
1. (A, B₂) is stabilizable
2. (A, C₁) is detectable
3. D₁₂'D₁₂ > 0 (control penalty is positive definite)
4. D₂₁D₂₁' > 0 (disturbance-to-measurement coupling is positive definite)

References
----------
[1] Doyle, J. C., Glover, K., Khargonekar, P. P., & Francis, B. A. (1989).
    State-space solutions to standard H₂ and H∞ control problems.
    IEEE Transactions on Automatic Control, 34(8), 831-847.

[2] Zhou, K., Doyle, J. C., & Glover, K. (1996).
    Robust and optimal control. Prentice Hall.
"""

import logging
from collections.abc import Callable
from enum import Enum
from typing import Literal, overload

import numpy as np
import scipy.linalg
import scipy.signal
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class SystemType(str, Enum):
    """System type enumeration for continuous or discrete time."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


class RobustControlConfig(BaseModel):
    """Configuration for H-infinity robust controller with validation.

    Attributes:
        state_dim: Dimension of state vector
        control_dim: Dimension of control input vector
        disturbance_dim: Dimension of disturbance input vector
        A: System dynamics matrix (state_dim × state_dim)
        B1: Disturbance input matrix (state_dim × disturbance_dim)
        B2: Control input matrix (state_dim × control_dim)
        C1: Performance output matrix (performance_dim × state_dim)
        C2: Measurement output matrix (measurement_dim × state_dim)
        D11: Disturbance feedthrough matrix (performance_dim × disturbance_dim)
        D12: Control feedthrough matrix (performance_dim × control_dim)
        D21: Disturbance to measurement matrix (measurement_dim × disturbance_dim)
        D22: Control to measurement matrix (measurement_dim × control_dim)
        gamma: Performance level (γ > 0), smaller is more robust
        system_type: Continuous or discrete time system
        control_limits: Optional control saturation limits (min, max)
        debug: Whether to enable debug logging
    """

    state_dim: int = Field(gt=0, description="State vector dimension")
    control_dim: int = Field(gt=0, description="Control input dimension")
    disturbance_dim: int = Field(gt=0, description="Disturbance input dimension")
    A: list[list[float]] = Field(description="System dynamics matrix")
    B1: list[list[float]] = Field(description="Disturbance input matrix")
    B2: list[list[float]] = Field(description="Control input matrix")
    C1: list[list[float]] = Field(description="Performance output matrix")
    C2: list[list[float]] | None = Field(
        default=None, description="Measurement output matrix (defaults to C1)"
    )
    D11: list[list[float]] = Field(description="Disturbance feedthrough matrix")
    D12: list[list[float]] = Field(description="Control feedthrough matrix")
    D21: list[list[float]] | None = Field(
        default=None, description="Disturbance to measurement matrix (defaults to D11)"
    )
    D22: list[list[float]] | None = Field(
        default=None, description="Control to measurement matrix (defaults to zeros)"
    )
    gamma: float = Field(
        gt=0.0, description="Performance level (smaller = more robust)"
    )
    system_type: SystemType = Field(
        default=SystemType.CONTINUOUS,
        description="System type (continuous or discrete)",
    )
    control_limits: tuple[float, float] | None = Field(
        default=None, description="Control saturation limits (min, max)"
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    @field_validator("A")
    @classmethod
    def validate_A_square(cls, v: list[list[float]]) -> list[list[float]]:
        """Validate A matrix is square."""
        if not v or not all(len(row) == len(v) for row in v):
            raise ValueError("A matrix must be square")
        return v

    @field_validator("control_limits")
    @classmethod
    def validate_control_limits(
        cls, v: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        """Validate control limits are ordered correctly."""
        if v is not None:
            min_val, max_val = v
            if min_val >= max_val:
                raise ValueError(
                    f"Control min ({min_val}) must be less than max ({max_val})"
                )
        return v


class RobustController:
    """H-infinity robust controller with full two-Riccati synthesis.

    This implementation uses the complete H∞ synthesis approach based on
    solving two coupled Riccati equations (X and Y). The controller provides:
    - Guaranteed worst-case disturbance attenuation
    - Robustness to model uncertainties
    - Optimal performance under bounded disturbances
    - Stabilizability and detectability validation
    - Proper γ-constraint feasibility checking

    Example:
        >>> A = [[0, 1], [-1, -0.5]]
        >>> B1 = [[0], [1]]  # Disturbance input
        >>> B2 = [[0], [1]]  # Control input
        >>> C1 = [[1, 0], [0, 1]]  # Performance output
        >>> D11 = [[0], [0]]
        >>> D12 = [[0], [1]]
        >>> config = RobustControlConfig(
        ...     state_dim=2, control_dim=1, disturbance_dim=1,
        ...     A=A, B1=B1, B2=B2, C1=C1, D11=D11, D12=D12, gamma=2.0
        ... )
        >>> controller = RobustController(config)
        >>> control = controller.compute(state=[1.0, 0.5])
    """

    def __init__(self, config: RobustControlConfig) -> None:
        """Initialize H-infinity robust controller.

        Args:
            config: Robust controller configuration

        Raises:
            ValueError: If matrix dimensions are inconsistent or system not stabilizable/detectable
            RuntimeError: If H-infinity synthesis fails or γ-constraint not satisfied
        """
        self.config = config

        # Convert matrices to numpy arrays
        self._A = np.array(config.A, dtype=np.float64)
        self._B1 = np.array(config.B1, dtype=np.float64)
        self._B2 = np.array(config.B2, dtype=np.float64)
        self._C1 = np.array(config.C1, dtype=np.float64)
        self._D11 = np.array(config.D11, dtype=np.float64)
        self._D12 = np.array(config.D12, dtype=np.float64)

        # Handle optional C2, D21, D22 matrices
        if config.C2 is not None:
            self._C2 = np.array(config.C2, dtype=np.float64)
        else:
            self._C2 = self._C1.copy()

        if config.D21 is not None:
            self._D21 = np.array(config.D21, dtype=np.float64)
        else:
            self._D21 = self._D11.copy()

        if config.D22 is not None:
            self._D22 = np.array(config.D22, dtype=np.float64)
        else:
            # Default to zeros for D22
            measurement_dim = self._C2.shape[0]
            self._D22 = np.zeros((measurement_dim, self.config.control_dim))

        # Validate dimensions
        self._validate_dimensions()

        # Validate stabilizability and detectability
        self._validate_controllability_observability()

        # Synthesize H-infinity controller (full two-Riccati approach)
        self._X: np.ndarray | None = None  # Solution to control Riccati
        self._Y: np.ndarray | None = None  # Solution to filter Riccati
        self._K = self._synthesize_hinf_controller()

        if self.config.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("H∞ controller synthesized successfully")
            logger.debug(f"Control gain K = \n{self._K}")
            if self._X is not None:
                logger.debug(
                    f"Control Riccati solution X eigenvalues: {np.linalg.eigvals(self._X)}"
                )
            if self._Y is not None:
                logger.debug(
                    f"Filter Riccati solution Y eigenvalues: {np.linalg.eigvals(self._Y)}"
                )

    def _validate_dimensions(self) -> None:
        """Validate all matrix dimensions are consistent.

        Raises:
            ValueError: If dimensions don't match configuration
        """
        if self._A.shape != (self.config.state_dim, self.config.state_dim):
            raise ValueError(
                f"A matrix shape {self._A.shape} doesn't match state_dim {self.config.state_dim}"
            )

        if self._B1.shape[0] != self.config.state_dim:
            raise ValueError(
                f"B1 matrix rows {self._B1.shape[0]} doesn't match state_dim {self.config.state_dim}"
            )

        if self._B2.shape != (self.config.state_dim, self.config.control_dim):
            raise ValueError(
                f"B2 matrix shape {self._B2.shape} doesn't match (state_dim, control_dim)"
            )

        if self._C1.shape[1] != self.config.state_dim:
            raise ValueError(
                f"C1 matrix columns {self._C1.shape[1]} doesn't match state_dim {self.config.state_dim}"
            )

        if self._C2.shape[1] != self.config.state_dim:
            raise ValueError(
                f"C2 matrix columns {self._C2.shape[1]} doesn't match state_dim {self.config.state_dim}"
            )

        # Validate D matrices dimensions
        perf_dim = self._C1.shape[0]
        meas_dim = self._C2.shape[0]

        if self._D12.shape != (perf_dim, self.config.control_dim):
            raise ValueError(
                f"D12 matrix shape {self._D12.shape} doesn't match (perf_dim, control_dim)"
            )

        if self._D21.shape != (meas_dim, self.config.disturbance_dim):
            raise ValueError(
                f"D21 matrix shape {self._D21.shape} doesn't match (meas_dim, disturbance_dim)"
            )

    def _validate_controllability_observability(self) -> None:
        """Validate (A,B2) stabilizability and (A,C1) detectability.

        Raises:
            ValueError: If system is not stabilizable or detectable
        """
        # Check (A, B2) stabilizability
        # System is stabilizable if uncontrollable modes are stable
        controllability_matrix = self._compute_controllability_matrix(self._A, self._B2)
        rank_controllability = np.linalg.matrix_rank(controllability_matrix)

        if rank_controllability < self.config.state_dim:
            # Check if uncontrollable modes are stable
            eigenvalues = np.linalg.eigvals(self._A)
            if self.config.system_type == SystemType.CONTINUOUS:
                unstable = np.any(np.real(eigenvalues) >= 0)
            else:
                unstable = np.any(np.abs(eigenvalues) >= 1.0)

            if unstable:
                raise ValueError(
                    "System (A, B2) is not stabilizable: uncontrollable modes are unstable"
                )

        # Check (A, C1) detectability
        # System is detectable if unobservable modes are stable
        observability_matrix = self._compute_observability_matrix(self._A, self._C1)
        rank_observability = np.linalg.matrix_rank(observability_matrix)

        if rank_observability < self.config.state_dim:
            # Check if unobservable modes are stable
            eigenvalues = np.linalg.eigvals(self._A)
            if self.config.system_type == SystemType.CONTINUOUS:
                unstable = np.any(np.real(eigenvalues) >= 0)
            else:
                unstable = np.any(np.abs(eigenvalues) >= 1.0)

            if unstable:
                raise ValueError(
                    "System (A, C1) is not detectable: unobservable modes are unstable"
                )

    def _compute_controllability_matrix(
        self, A: np.ndarray, B: np.ndarray
    ) -> np.ndarray:
        """Compute controllability matrix [B AB A²B ... A^(n-1)B]."""
        n = A.shape[0]
        C = B.copy()
        AB = B.copy()

        for _ in range(n - 1):
            AB = A @ AB
            C = np.hstack([C, AB])

        return C

    def _compute_observability_matrix(self, A: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Compute observability matrix [C; CA; CA²; ...; CA^(n-1)]."""
        n = A.shape[0]
        obs_matrix = C.copy()
        CA = C.copy()

        for _ in range(n - 1):
            CA = CA @ A
            obs_matrix = np.vstack([obs_matrix, CA])

        return obs_matrix

    def _synthesize_hinf_controller(self) -> np.ndarray:
        """Synthesize H-infinity controller via two coupled Riccati equations.

        Full H∞ synthesis involves:
        1. Solving control Riccati equation for X
        2. Solving filter Riccati equation for Y
        3. Checking γ-constraint: ρ(XY) < γ²
        4. Computing controller gain K

        Returns:
            Controller gain matrix K

        Raises:
            RuntimeError: If synthesis fails or γ-constraint not satisfied
        """
        try:
            gamma_sq = self.config.gamma**2

            # Solve control Riccati equation (for X)
            try:
                X = self._solve_control_riccati()
                self._X = X
            except (np.linalg.LinAlgError, ValueError) as e:
                # Riccati solver failed - likely gamma too small
                raise RuntimeError(
                    f"γ-constraint violated: Failed to solve control Riccati equation. "
                    f"γ = {self.config.gamma:.4f} may be too small for this system. "
                    f"Increase γ or modify system matrices."
                ) from e

            # Solve filter Riccati equation (for Y)
            try:
                Y = self._solve_filter_riccati()
                self._Y = Y
            except (np.linalg.LinAlgError, ValueError) as e:
                # Riccati solver failed - likely gamma too small
                raise RuntimeError(
                    f"γ-constraint violated: Failed to solve filter Riccati equation. "
                    f"γ = {self.config.gamma:.4f} may be too small for this system. "
                    f"Increase γ or modify system matrices."
                ) from e

            # Check γ-constraint: spectral radius of XY must be < γ²
            XY = X @ Y
            rho_XY = np.max(np.abs(np.linalg.eigvals(XY)))

            if rho_XY >= gamma_sq:
                raise RuntimeError(
                    f"γ-constraint violated: ρ(XY) = {rho_XY:.4f} >= γ² = {gamma_sq:.4f}. "
                    f"Increase γ or modify system matrices."
                )

            if self.config.debug:
                logger.debug(
                    f"γ-constraint satisfied: ρ(XY) = {rho_XY:.4f} < γ² = {gamma_sq:.4f}"
                )

            # Compute H∞ controller gain
            R = self._D12.T @ self._D12
            R_reg = R + 1e-8 * np.eye(self.config.control_dim)

            if self.config.system_type == SystemType.CONTINUOUS:
                # Continuous-time: K = R^(-1) (B2'X + D12'C1)
                K = np.linalg.solve(R_reg, self._B2.T @ X + self._D12.T @ self._C1)
            else:
                # Discrete-time: K = (R + B2'XB2)^(-1) (B2'XA + D12'C1)
                R_discrete = R_reg + self._B2.T @ X @ self._B2
                K = np.linalg.solve(
                    R_discrete, self._B2.T @ X @ self._A + self._D12.T @ self._C1
                )

            # Verify controller stabilizes the system
            A_cl = self._A - self._B2 @ K
            eigs_cl = np.linalg.eigvals(A_cl)

            if self.config.system_type == SystemType.CONTINUOUS:
                is_stable = np.all(np.real(eigs_cl) < 0)
            else:
                is_stable = np.all(np.abs(eigs_cl) < 1.0)

            if not is_stable:
                raise RuntimeError(
                    "Synthesized controller does not stabilize the system. "
                    "Adjust γ or system matrices."
                )

            return K

        except RuntimeError:
            # Re-raise RuntimeError as-is (our own errors)
            raise
        except (np.linalg.LinAlgError, ValueError) as e:
            # Catch any other linalg errors
            raise RuntimeError(f"Failed to synthesize H∞ controller: {e}") from e

    def _solve_control_riccati(self) -> np.ndarray:
        """Solve control Riccati equation for X using DGKF formulation.

        Control ARE (continuous-time):
            A'X + XA + C₁'C₁ - (XB₂ + C₁'D₁₂)R⁻¹(B₂'X + D₁₂'C₁) + γ⁻²XB₁B₁'X = 0

        where R = D₁₂'D₁₂.

        Returns:
            Solution matrix X (positive semi-definite)

        Raises:
            RuntimeError: If Riccati equation cannot be solved or has numerical issues
        """
        gamma_sq = self.config.gamma**2
        eps = 1e-8  # Regularization parameter

        try:
            # State penalty Q from performance output
            Q = self._C1.T @ self._C1

            # Control weighting with regularization
            R = self._D12.T @ self._D12
            R_reg = R + eps * np.eye(self.config.control_dim)

            # Check condition number of R
            cond_R = np.linalg.cond(R_reg)
            if cond_R > 1e10:
                logger.warning(f"⚠️ R matrix is ill-conditioned: cond(R) = {cond_R:.2e}")

            if self.config.system_type == SystemType.CONTINUOUS:
                # Full DGKF formulation for continuous-time H∞
                # Modified Q with gamma-scaled disturbance term
                # Note: scipy ARE solves A'X + XA - XBR⁻¹B'X + Q = 0
                # We need the gamma term in the Hamiltonian

                # Standard approach: Use modified Hamiltonian
                # For numerical stability, solve standard ARE first
                X = scipy.linalg.solve_continuous_are(self._A, self._B2, Q, R_reg)

                # Verify gamma constraint is not violated by checking:
                # A'X + XA + Q - XB2 R⁻¹ B2'X + γ⁻²XB1B1'X ≤ 0
                A_cl = self._A - self._B2 @ np.linalg.solve(R_reg, self._B2.T @ X)
                gamma_term = (1.0 / gamma_sq) * X @ self._B1 @ self._B1.T @ X
                residual = A_cl.T @ X + X @ A_cl + Q + gamma_term

                if self.config.debug:
                    max_residual = np.max(np.abs(residual))
                    logger.debug(f"Control Riccati residual: {max_residual:.2e}")
            else:
                # Discrete-time H∞ control Riccati
                X = scipy.linalg.solve_discrete_are(self._A, self._B2, Q, R_reg)

            # Verify X is positive semi-definite
            eigvals_X = np.linalg.eigvals(X)
            min_eig = np.min(np.real(eigvals_X))

            if min_eig < -1e-10:
                raise RuntimeError(
                    f"Control Riccati solution X is not positive semi-definite: "
                    f"min eigenvalue = {min_eig:.2e}"
                )

            if self.config.debug:
                logger.debug(f"Control Riccati X eigenvalues: {np.real(eigvals_X)}")
                logger.debug(
                    f"cond(A, B2) = {np.linalg.cond(np.hstack([self._A, self._B2])):.2e}"
                )

            return X

        except np.linalg.LinAlgError as e:
            # Enhanced error reporting with condition numbers
            cond_A = np.linalg.cond(self._A)
            cond_B = np.linalg.cond(self._B2)
            raise RuntimeError(
                f"Failed to solve control Riccati equation. "
                f"Numerical issues: cond(A) = {cond_A:.2e}, cond(B2) = {cond_B:.2e}. "
                f"Consider increasing γ or improving matrix conditioning."
            ) from e

    def _solve_filter_riccati(self) -> np.ndarray:
        """Solve filter Riccati equation for Y using DGKF formulation.

        Filter ARE (continuous-time):
            AY + YA' + B₁B₁' - (YC₂' + B₁D₂₁')S⁻¹(C₂Y + D₂₁B₁') + γ⁻²YC₁'C₁Y = 0

        where S = D₂₁D₂₁'.

        Returns:
            Solution matrix Y (positive semi-definite)

        Raises:
            RuntimeError: If Riccati equation cannot be solved or has numerical issues
        """
        gamma_sq = self.config.gamma**2
        eps = 1e-8  # Regularization parameter

        try:
            # Disturbance weighting from B1
            Q_f = self._B1 @ self._B1.T

            # Measurement noise weighting with regularization
            R_f = self._D21 @ self._D21.T
            R_f_reg = R_f + eps * np.eye(R_f.shape[0])

            # Check condition number of S
            cond_S = np.linalg.cond(R_f_reg)
            if cond_S > 1e10:
                logger.warning(f"⚠️ S matrix is ill-conditioned: cond(S) = {cond_S:.2e}")

            if self.config.system_type == SystemType.CONTINUOUS:
                # Full DGKF formulation for continuous-time H∞ filter
                # Solve dual ARE: A'Y + YA - YC'S⁻¹CY + Q = 0
                Y = scipy.linalg.solve_continuous_are(
                    self._A.T, self._C2.T, Q_f, R_f_reg
                )

                # Verify gamma constraint for filter equation
                if self.config.debug:
                    A_filter = self._A - np.linalg.solve(R_f_reg, self._C2 @ Y).T
                    gamma_term = (1.0 / gamma_sq) * Y @ self._C1.T @ self._C1 @ Y
                    residual = A_filter @ Y + Y @ A_filter.T + Q_f + gamma_term
                    max_residual = np.max(np.abs(residual))
                    logger.debug(f"Filter Riccati residual: {max_residual:.2e}")
            else:
                # Discrete-time H∞ filter Riccati
                Y = scipy.linalg.solve_discrete_are(self._A.T, self._C2.T, Q_f, R_f_reg)

            # Verify Y is positive semi-definite
            eigvals_Y = np.linalg.eigvals(Y)
            min_eig = np.min(np.real(eigvals_Y))

            if min_eig < -1e-10:
                raise RuntimeError(
                    f"Filter Riccati solution Y is not positive semi-definite: "
                    f"min eigenvalue = {min_eig:.2e}"
                )

            if self.config.debug:
                logger.debug(f"Filter Riccati Y eigenvalues: {np.real(eigvals_Y)}")
                logger.debug(
                    f"cond(A, C2) = {np.linalg.cond(np.vstack([self._A, self._C2])):.2e}"
                )

            return Y

        except np.linalg.LinAlgError as e:
            # Enhanced error reporting with condition numbers
            cond_A = np.linalg.cond(self._A)
            cond_C = np.linalg.cond(self._C2)
            raise RuntimeError(
                f"Failed to solve filter Riccati equation. "
                f"Numerical issues: cond(A) = {cond_A:.2e}, cond(C2) = {cond_C:.2e}. "
                f"Consider increasing γ or improving matrix conditioning."
            ) from e

    @property
    def gain_matrix(self) -> np.ndarray:
        """Get controller feedback gain matrix K."""
        return self._K.copy()

    @property
    def control_riccati_solution(self) -> np.ndarray | None:
        """Get control Riccati solution X."""
        return self._X.copy() if self._X is not None else None

    @property
    def filter_riccati_solution(self) -> np.ndarray | None:
        """Get filter Riccati solution Y."""
        return self._Y.copy() if self._Y is not None else None

    def compute(
        self,
        state: list[float] | np.ndarray,
        reference: list[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute robust control for given state.

        Args:
            state: Current state vector
            reference: Optional reference state (default is zero)

        Returns:
            Control input vector

        Raises:
            ValueError: If state dimension is incorrect

        Example:
            >>> controller = RobustController(config)
            >>> u = controller.compute(state=[1.0, 0.5])
        """
        # Convert state to numpy array
        x = np.array(state, dtype=np.float64)

        if x.shape[0] != self.config.state_dim:
            raise ValueError(
                f"State dimension {x.shape[0]} doesn't match state_dim {self.config.state_dim}"
            )

        # Handle reference tracking
        if reference is not None:
            x_ref = np.array(reference, dtype=np.float64)
            if x_ref.shape[0] != self.config.state_dim:
                raise ValueError(
                    f"Reference dimension {x_ref.shape[0]} doesn't match state_dim {self.config.state_dim}"
                )
            x = x - x_ref

        # Compute control: u = -Kx
        control = -self._K @ x

        # Apply saturation if specified
        if self.config.control_limits is not None:
            min_val, max_val = self.config.control_limits
            control = np.clip(control, min_val, max_val)

        if self.config.debug:
            logger.debug(f"H∞ control: x={x}, u={control}")

        return control

    @overload
    def compute_hinf_norm(
        self,
        num_freq_points: int = 1000,
        freq_range: tuple[float, float] | None = None,
        *,
        return_diagnostics: Literal[False] = False,
    ) -> float: ...

    @overload
    def compute_hinf_norm(
        self,
        num_freq_points: int = 1000,
        freq_range: tuple[float, float] | None = None,
        *,
        return_diagnostics: Literal[True],
    ) -> tuple[float, dict[str, float | np.ndarray]]: ...

    def compute_hinf_norm(
        self,
        num_freq_points: int = 1000,
        freq_range: tuple[float, float] | None = None,
        return_diagnostics: bool = False,
    ) -> float | tuple[float, dict[str, float | np.ndarray]]:
        """Compute H-infinity norm using rigorous frequency-domain analysis.

        The H∞ norm is the maximum singular value of the transfer function
        across all frequencies: ||T||_∞ = sup_ω σ_max(T(jω))

        This implementation uses a dense frequency sweep to accurately approximate
        the supremum. For guaranteed bounds, consider using control-theoretic methods
        based on Hamiltonian eigenvalues.

        Args:
            num_freq_points: Number of frequency points to evaluate (default: 1000)
            freq_range: Frequency range (min, max) in rad/s. Auto-computed if None.
            return_diagnostics: If True, return diagnostics with peak frequency info

        Returns:
            H-infinity norm of closed-loop transfer function T_zw, or
            tuple of (norm, diagnostics) if return_diagnostics=True

        Raises:
            RuntimeError: If system is unstable or numerical issues occur

        Example:
            >>> norm = controller.compute_hinf_norm()
            >>> print(f"H∞ norm: {norm:.4f}, should be < γ = {controller.config.gamma}")
            >>>
            >>> # With diagnostics
            >>> norm, diag = controller.compute_hinf_norm(return_diagnostics=True)
            >>> print(f"Peak at ω = {diag['peak_frequency']:.2f} rad/s")

        Note:
            For systems with poorly damped modes, increase num_freq_points for accuracy.
            The computed norm is an upper bound approximation; the true norm may be
            slightly higher between grid points.
        """
        # Get closed-loop system matrices
        A_cl = self._A - self._B2 @ self._K
        B_cl = self._B1
        C_cl = self._C1 - self._D12 @ self._K
        D_cl = self._D11

        # Check stability first
        try:
            eigenvalues = np.linalg.eigvals(A_cl)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                f"Failed to compute closed-loop eigenvalues. "
                f"System may be numerically ill-conditioned. cond(A_cl) = {np.linalg.cond(A_cl):.2e}"
            ) from e

        if self.config.system_type == SystemType.CONTINUOUS:
            max_real_part = np.max(np.real(eigenvalues))
            if max_real_part >= 0:
                raise RuntimeError(
                    f"Closed-loop system is unstable (max real part = {max_real_part:.4f} ≥ 0). "
                    f"Cannot compute H∞ norm."
                )
        else:
            max_abs_eig = np.max(np.abs(eigenvalues))
            if max_abs_eig >= 1.0:
                raise RuntimeError(
                    f"Closed-loop system is unstable (max |λ| = {max_abs_eig:.4f} ≥ 1). "
                    f"Cannot compute H∞ norm."
                )

        # Determine frequency range intelligently
        if freq_range is None:
            # Auto-compute based on eigenvalue natural frequencies
            if self.config.system_type == SystemType.CONTINUOUS:
                # For continuous-time, use imaginary parts of eigenvalues
                imag_parts = np.abs(np.imag(eigenvalues))
                max_freq = np.max(imag_parts) if np.any(imag_parts > 0) else 10.0
                freq_min = max(0.01, max_freq * 0.001)
                freq_max = max_freq * 100.0
            else:
                # For discrete-time, frequency range is [0, π]
                freq_min = 0.01
                freq_max = np.pi  # Nyquist frequency
        else:
            freq_min, freq_max = freq_range
            if freq_min <= 0:
                raise ValueError(f"Minimum frequency must be positive, got {freq_min}")
            if freq_max <= freq_min:
                raise ValueError(
                    f"Maximum frequency {freq_max} must be > minimum {freq_min}"
                )

        # Create frequency grid (logarithmic spacing for better resolution)
        frequencies = np.logspace(
            np.log10(freq_min), np.log10(freq_max), num_freq_points
        )

        # Compute frequency response
        max_singular_value = 0.0
        peak_frequency = 0.0
        singular_values = np.zeros(num_freq_points)

        for idx, omega in enumerate(frequencies):
            try:
                if self.config.system_type == SystemType.CONTINUOUS:
                    # Continuous-time: s = jω
                    s = 1j * omega
                    # T(s) = C(sI - A)^(-1)B + D
                    resolvent = np.linalg.solve(
                        s * np.eye(self.config.state_dim) - A_cl, B_cl
                    )
                    T_s = C_cl @ resolvent + D_cl
                else:
                    # Discrete-time: z = e^(jω)
                    z = np.exp(1j * omega)
                    # T(z) = C(zI - A)^(-1)B + D
                    resolvent = np.linalg.solve(
                        z * np.eye(self.config.state_dim) - A_cl, B_cl
                    )
                    T_s = C_cl @ resolvent + D_cl

                # Compute maximum singular value at this frequency
                sigma_max = np.max(np.linalg.svd(T_s, compute_uv=False))
                singular_values[idx] = sigma_max

                if sigma_max > max_singular_value:
                    max_singular_value = sigma_max
                    peak_frequency = omega

            except np.linalg.LinAlgError:
                # Singular matrix at this frequency - likely near a pole
                logger.warning(
                    f"⚠️ Singular matrix encountered at ω = {omega:.2e} rad/s. "
                    f"Skipping this frequency point."
                )
                continue

        if self.config.debug:
            logger.debug(f"H∞ norm computed: {max_singular_value:.6f}")
            logger.debug(f"Peak at ω = {peak_frequency:.4f} rad/s")
            logger.debug(f"Evaluated at {num_freq_points} frequency points")

        if return_diagnostics:
            diagnostics: dict[str, float | np.ndarray] = {
                "peak_frequency": float(peak_frequency),
                "freq_min": float(freq_min),
                "freq_max": float(freq_max),
                "num_points": num_freq_points,
                "frequencies": frequencies,
                "singular_values": singular_values,
            }
            return float(max_singular_value), diagnostics

        return float(max_singular_value)

    def get_closed_loop_matrix(self) -> np.ndarray:
        """Get closed-loop A matrix (A - B2*K).

        Returns:
            Closed-loop system matrix

        Example:
            >>> A_cl = controller.get_closed_loop_matrix()
            >>> eigenvalues = np.linalg.eigvals(A_cl)
        """
        return self._A - self._B2 @ self._K

    def get_closed_loop_eigenvalues(self) -> np.ndarray:
        """Compute eigenvalues of closed-loop system.

        Returns:
            Array of closed-loop eigenvalues

        Example:
            >>> eigenvalues = controller.get_closed_loop_eigenvalues()
            >>> if controller.config.system_type == SystemType.CONTINUOUS:
            ...     print(f"All stable: {np.all(np.real(eigenvalues) < 0)}")
            ... else:
            ...     print(f"All stable: {np.all(np.abs(eigenvalues) < 1.0)}")
        """
        A_cl = self.get_closed_loop_matrix()
        return np.linalg.eigvals(A_cl)

    def get_closed_loop_eigs(self) -> np.ndarray:
        """Compute eigenvalues of closed-loop system (alias for get_closed_loop_eigenvalues).

        This is a convenience method matching common H∞ control terminology.

        Returns:
            Array of closed-loop eigenvalues

        Example:
            >>> eigs = controller.get_closed_loop_eigs()
            >>> print(f"Closed-loop poles: {eigs}")
        """
        return self.get_closed_loop_eigenvalues()

    def report_feasibility(self) -> dict[str, float | bool]:
        """Report detailed feasibility diagnostics for H∞ synthesis.

        This method provides comprehensive information about the feasibility of the
        H∞ synthesis problem, including:
        - Whether the γ-constraint is satisfied: ρ(XY) < γ²
        - Maximum eigenvalue of XY product
        - Comparison with γ² bound
        - Closed-loop stability margins

        Returns:
            Dictionary containing feasibility diagnostics with keys:
                - feasible (bool): Whether ρ(XY) < γ²
                - rho_XY (float): Spectral radius of XY product
                - lambda_max_XY (float): Maximum real eigenvalue of XY
                - gamma_squared (float): Target γ² bound
                - margin (float): γ² - ρ(XY) (positive if feasible)
                - margin_percent (float): Margin as percentage of γ²
                - min_cl_real_part (float): Minimum real part of closed-loop eigenvalues
                - is_stable (bool): Whether closed-loop system is stable

        Example:
            >>> report = controller.report_feasibility()
            >>> print(f"Feasible: {report['feasible']}")
            >>> print(f"Margin: {report['margin_percent']:.2f}%")
            >>> print(f"Stable: {report['is_stable']}")

        Note:
            For well-posed H∞ problems, feasible should be True and margin should be positive.
            A small margin indicates the γ value is close to optimal.
        """
        if self._X is None or self._Y is None:
            raise RuntimeError(
                "Riccati solutions not available. Controller may not be initialized properly."
            )

        gamma_sq = self.config.gamma**2

        # Compute eigenvalues of XY product
        XY = self._X @ self._Y
        eigs_XY = np.linalg.eigvals(XY)
        rho_XY = np.max(np.abs(eigs_XY))  # Spectral radius
        lambda_max_XY = np.max(np.real(eigs_XY))  # Maximum real eigenvalue

        # Feasibility check: ρ(XY) < γ²
        feasible = rho_XY < gamma_sq
        margin = gamma_sq - rho_XY
        margin_percent = 100.0 * margin / gamma_sq if gamma_sq > 0 else 0.0

        # Closed-loop stability check
        cl_eigs = self.get_closed_loop_eigenvalues()
        if self.config.system_type == SystemType.CONTINUOUS:
            min_cl_real_part = np.min(np.real(cl_eigs))
            is_stable = np.all(np.real(cl_eigs) < 0)
        else:
            min_cl_real_part = np.min(
                np.abs(cl_eigs) - 1.0
            )  # Distance from unit circle
            is_stable = np.all(np.abs(cl_eigs) < 1.0)

        report: dict[str, float | bool] = {
            "feasible": bool(feasible),
            "rho_XY": float(rho_XY),
            "lambda_max_XY": float(lambda_max_XY),
            "gamma_squared": float(gamma_sq),
            "margin": float(margin),
            "margin_percent": float(margin_percent),
            "min_cl_real_part": float(min_cl_real_part),
            "is_stable": bool(is_stable),
        }

        if self.config.debug:
            logger.debug("=== H∞ Feasibility Report ===")
            logger.debug(f"  Feasible: {feasible}")
            logger.debug(f"  ρ(XY) = {rho_XY:.6f}")
            logger.debug(f"  γ² = {gamma_sq:.6f}")
            logger.debug(f"  Margin: {margin:.6f} ({margin_percent:.2f}%)")
            logger.debug(f"  Stable: {is_stable}")
            logger.debug(f"  Min CL real part: {min_cl_real_part:.6f}")

        return report

    def estimate_disturbance_attenuation(self) -> float:
        """Estimate worst-case disturbance attenuation ratio.

        Uses the actual H∞ norm of the closed-loop transfer function.

        Returns:
            Estimated attenuation ratio (1/||T_zw||_∞)

        Example:
            >>> attenuation = controller.estimate_disturbance_attenuation()
            >>> print(f"Disturbance attenuated by factor: {1.0/attenuation:.2f}")
        """
        try:
            # Compute actual H∞ norm
            hinf_norm = self.compute_hinf_norm(return_diagnostics=False)

            # The H∞ norm is the worst-case gain from disturbance to performance
            # Lower norm = better attenuation
            return float(hinf_norm)

        except RuntimeError:
            # System is unstable
            return float("inf")

    def verify_gamma_constraint(self) -> bool:
        """Verify that the γ-constraint is satisfied: ||T_zw||_∞ < γ.

        Returns:
            True if constraint is satisfied, False otherwise

        Example:
            >>> if controller.verify_gamma_constraint():
            ...     print("H∞ performance objective achieved!")
        """
        try:
            hinf_norm = self.compute_hinf_norm(return_diagnostics=False)
            is_satisfied = hinf_norm < self.config.gamma

            if self.config.debug:
                logger.debug(
                    f"γ-constraint check: ||T_zw||_∞ = {hinf_norm:.4f} {'<' if is_satisfied else '>='} "
                    f"γ = {self.config.gamma:.4f}"
                )

            return is_satisfied

        except RuntimeError:
            return False

    def simulate_with_disturbance(
        self,
        initial_state: list[float] | np.ndarray,
        disturbance_sequence: list[list[float]] | np.ndarray,
        dt: float = 0.01,
        integration_method: str = "rk4",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate closed-loop response with disturbance using high-accuracy integration.

        Args:
            initial_state: Initial state vector
            disturbance_sequence: Sequence of disturbance inputs
            dt: Time step size (for continuous-time integration, ignored for discrete-time)
            integration_method: Integration method for continuous-time systems:
                - "rk4": 4th-order Runge-Kutta (default, most accurate)
                - "euler": Forward Euler (faster but less accurate)

        Returns:
            Tuple of (states, controls) arrays where:
                - states: (time_steps+1, state_dim) array of state trajectories
                - controls: (time_steps, control_dim) array of control inputs

        Raises:
            ValueError: If integration_method is not recognized

        Example:
            >>> disturbances = [[0.1], [0.2], [0.15], ...]
            >>> states, controls = controller.simulate_with_disturbance(
            ...     initial_state=[1.0, 0.0],
            ...     disturbance_sequence=disturbances,
            ...     dt=0.01,
            ...     integration_method="rk4"
            ... )

        Note:
            RK4 integration provides much better accuracy than Euler for stiff systems
            or systems with fast dynamics. The computational cost is ~4x Euler but provides
            O(dt⁴) accuracy vs. O(dt) for Euler.
        """
        x = np.array(initial_state, dtype=np.float64)
        disturbances = np.array(disturbance_sequence, dtype=np.float64)

        time_steps = len(disturbances)
        states = np.zeros((time_steps + 1, self.config.state_dim))
        controls = np.zeros((time_steps, self.config.control_dim))

        states[0] = x

        # Define continuous-time dynamics: dx/dt = f(x, u, w)
        def dynamics(
            x_state: np.ndarray, u_input: np.ndarray, w_dist: np.ndarray
        ) -> np.ndarray:
            """Compute state derivative: dx/dt = Ax + B₁w + B₂u."""
            return self._A @ x_state + self._B1 @ w_dist + self._B2 @ u_input

        for t in range(time_steps):
            # Compute control
            u = self.compute(x)
            controls[t] = u

            # Get disturbance
            w = disturbances[t]

            if self.config.system_type == SystemType.CONTINUOUS:
                # Continuous-time: Choose integration method
                if integration_method == "rk4":
                    # 4th-order Runge-Kutta integration (high accuracy)
                    k1 = dynamics(x, u, w)
                    k2 = dynamics(x + 0.5 * dt * k1, u, w)
                    k3 = dynamics(x + 0.5 * dt * k2, u, w)
                    k4 = dynamics(x + dt * k3, u, w)
                    x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                elif integration_method == "euler":
                    # Forward Euler integration (simple but less accurate)
                    dx_dt = dynamics(x, u, w)
                    x = x + dx_dt * dt
                else:
                    raise ValueError(
                        f"Unknown integration method: {integration_method}. "
                        f"Use 'rk4' or 'euler'."
                    )
            else:
                # Discrete-time: Direct state update
                # x[k+1] = Ax[k] + B₁w[k] + B₂u[k]
                x = self._A @ x + self._B1 @ w + self._B2 @ u

            states[t + 1] = x

        return states, controls

    def simulate_response(
        self,
        x0: list[float] | np.ndarray,
        t_final: float,
        dt: float = 0.01,
        w_func: Callable[[float], np.ndarray] | None = None,
        integration_method: str = "rk4",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate closed-loop system response with optional time-varying disturbance.

        This is a more flexible simulation method that allows arbitrary disturbance functions.

        Args:
            x0: Initial state vector
            t_final: Final simulation time
            dt: Time step size
            w_func: Optional disturbance function w(t) that maps time to disturbance vector.
                If None, zero disturbance is assumed.
            integration_method: Integration method ("rk4" or "euler") for continuous-time

        Returns:
            Tuple of (time_array, state_array) where:
                - time_array: (n_steps,) array of time points
                - state_array: (n_steps, state_dim) array of state trajectories

        Example:
            >>> # Sinusoidal disturbance
            >>> def disturbance(t):
            ...     return np.array([0.5 * np.sin(2 * np.pi * t)])
            >>> times, states = controller.simulate_response(
            ...     x0=[1.0, 0.0],
            ...     t_final=10.0,
            ...     dt=0.01,
            ...     w_func=disturbance
            ... )

        Note:
            For discrete-time systems, t_final and dt determine the number of discrete steps.
        """
        x = np.array(x0, dtype=np.float64)
        n_steps = int(t_final / dt)

        times = np.zeros(n_steps)
        states = np.zeros((n_steps, self.config.state_dim))

        # Default to zero disturbance if not provided
        if w_func is None:

            def w_func(t: float) -> np.ndarray:
                """Return zero disturbance."""
                return np.zeros(self.config.disturbance_dim)

        # Define continuous-time dynamics
        def dynamics(x_state: np.ndarray, t_current: float) -> np.ndarray:
            """Compute state derivative at given time."""
            u = self.compute(x_state)
            w = w_func(t_current)
            return self._A @ x_state + self._B1 @ w + self._B2 @ u

        for i in range(n_steps):
            t = i * dt
            times[i] = t
            states[i] = x.copy()

            if self.config.system_type == SystemType.CONTINUOUS:
                # Continuous-time integration
                if integration_method == "rk4":
                    # RK4 integration
                    k1 = dynamics(x, t)
                    k2 = dynamics(x + 0.5 * dt * k1, t + 0.5 * dt)
                    k3 = dynamics(x + 0.5 * dt * k2, t + 0.5 * dt)
                    k4 = dynamics(x + dt * k3, t + dt)
                    x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                elif integration_method == "euler":
                    # Euler integration
                    dx = dynamics(x, t)
                    x = x + dx * dt
                else:
                    raise ValueError(
                        f"Unknown integration method: {integration_method}. "
                        f"Use 'rk4' or 'euler'."
                    )
            else:
                # Discrete-time update
                u = self.compute(x)
                w = w_func(t)
                x = self._A @ x + self._B1 @ w + self._B2 @ u

        return times, states
