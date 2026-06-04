"""Tests for Robust H-infinity Control implementation."""

import numpy as np
import pytest
from pydantic import ValidationError

from algokit.algorithms.control.robust import (
    RobustControlConfig,
    RobustController,
    SystemType,
)


class TestRobustControlConfig:
    """Test robust control configuration model."""

    @pytest.mark.unit
    def test_config_initialization(self) -> None:
        """Test RobustControlConfig initializes with valid matrices."""
        # Arrange - Create H-infinity system matrices for 2-state system
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]

        # Act - Initialize robust control configuration
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=1.0,
        )

        # Assert - Configuration stores all parameters correctly
        assert config.state_dim == 2
        assert config.control_dim == 1
        assert config.disturbance_dim == 1
        assert config.gamma == 1.0
        assert config.control_limits is None

    @pytest.mark.unit
    def test_config_validates_negative_state_dim(self) -> None:
        """Test config raises error for non-positive state_dim."""
        # Arrange - Set up configuration with invalid state dimension
        A = [[0]]
        B1 = [[1]]
        B2 = [[1]]
        C1 = [[1]]
        D11 = [[0]]
        D12 = [[1]]

        # Act - Attempt to create config with zero state_dim
        # Assert - Should raise ValidationError
        with pytest.raises(ValidationError, match="state_dim"):
            RobustControlConfig(
                state_dim=0,
                control_dim=1,
                disturbance_dim=1,
                A=A,
                B1=B1,
                B2=B2,
                C1=C1,
                D11=D11,
                D12=D12,
                gamma=1.0,
            )

    @pytest.mark.unit
    def test_config_validates_negative_gamma(self) -> None:
        """Test config raises error for non-positive gamma."""
        # Arrange - Create configuration with zero gamma value
        A = [[0]]
        B1 = [[1]]
        B2 = [[1]]
        C1 = [[1]]
        D11 = [[0]]
        D12 = [[1]]

        # Act - Attempt to create config with invalid gamma
        # Assert - Should raise validation error for gamma
        with pytest.raises(ValidationError, match="gamma"):
            RobustControlConfig(
                state_dim=1,
                control_dim=1,
                disturbance_dim=1,
                A=A,
                B1=B1,
                B2=B2,
                C1=C1,
                D11=D11,
                D12=D12,
                gamma=0.0,
            )

    @pytest.mark.unit
    def test_config_validates_non_square_A(self) -> None:
        """Test config raises error for non-square A matrix."""
        # Arrange - Create configuration with non-square A matrix (2x3)
        A = [[0, 1, 2], [-1, -0.5, 0]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]

        # Act - Attempt to create config with invalid A matrix
        # Assert - Should raise validation error for non-square A
        with pytest.raises(ValidationError, match="A matrix must be square"):
            RobustControlConfig(
                state_dim=2,
                control_dim=1,
                disturbance_dim=1,
                A=A,
                B1=B1,
                B2=B2,
                C1=C1,
                D11=D11,
                D12=D12,
                gamma=1.0,
            )

    @pytest.mark.unit
    def test_config_validates_control_limits(self) -> None:
        """Test config raises error for invalid control limits."""
        # Arrange - Create configuration with reversed control limits (min > max)
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]

        # Act - Attempt to create config with invalid limits
        # Assert - Should raise validation error for control limits
        with pytest.raises(ValidationError, match="Control min"):
            RobustControlConfig(
                state_dim=2,
                control_dim=1,
                disturbance_dim=1,
                A=A,
                B1=B1,
                B2=B2,
                C1=C1,
                D11=D11,
                D12=D12,
                gamma=1.0,
                control_limits=(10.0, 5.0),
            )


class TestRobustController:
    """Test Robust H-infinity Controller implementation."""

    @pytest.mark.unit
    def test_controller_initialization(self) -> None:
        """Test RobustController initializes and computes gains."""
        # Arrange - Create H-infinity configuration for second-order system
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=1.0,
        )

        # Act - Initialize controller and solve Riccati equations
        controller = RobustController(config)

        # Assert - Gain matrix should match expected dimensions
        assert controller.gain_matrix.shape == (1, 2)

    @pytest.mark.unit
    def test_controller_dimension_validation(self) -> None:
        """Test controller raises error for dimension mismatch."""
        # Arrange - Create configuration with mismatched B1 matrix dimensions
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1], [0]]  # Wrong size (3x1 instead of 2x1)
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=1.0,
        )

        # Act - Attempt to create controller with invalid B1 dimensions
        # Assert - Should raise ValueError for dimension mismatch
        with pytest.raises(ValueError, match="B1 matrix rows"):
            RobustController(config)

    @pytest.mark.unit
    def test_compute_basic(self) -> None:
        """Test compute returns control for given state."""
        # Arrange - Create robust controller with standard configuration
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=1.0,
        )
        controller = RobustController(config)

        # Act - Compute control signal for non-zero state
        control = controller.compute(state=[1.0, 0.0])

        # Assert - Control should be scalar with correct shape
        assert control.shape == (1,)
        assert isinstance(control[0], (float, np.floating))

    @pytest.mark.unit
    def test_compute_zero_state(self) -> None:
        """Test compute returns zero control for zero state."""
        # Arrange - Create robust controller with standard configuration
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=1.0,
        )
        controller = RobustController(config)

        # Act - Compute control for equilibrium (zero) state
        control = controller.compute(state=[0.0, 0.0])

        # Assert - Control should be zero at equilibrium
        assert np.allclose(control, [0.0])

    @pytest.mark.unit
    def test_compute_wrong_state_dimension(self) -> None:
        """Test compute raises error for wrong state dimension."""
        # Arrange - Create 2-state controller
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=1.0,
        )
        controller = RobustController(config)

        # Act - Attempt to compute with 3-element state
        # Assert - Should raise ValueError for dimension mismatch
        with pytest.raises(ValueError, match="State dimension"):
            controller.compute(state=[1.0, 2.0, 3.0])

    @pytest.mark.unit
    def test_compute_with_control_limits(self) -> None:
        """Test control saturation respects limits."""
        # Arrange - Create controller with tight control limits and large state
        A = [[0, 1], [-10, -5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[100, 0], [0, 100]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=0.1,
            control_limits=(-1.0, 1.0),
        )
        controller = RobustController(config)

        # Act - Compute control for large state that would saturate
        control = controller.compute(state=[10.0, 10.0])

        # Assert - Control should be clipped within specified limits
        assert control[0] >= -1.0
        assert control[0] <= 1.0

    @pytest.mark.unit
    def test_get_closed_loop_eigenvalues(self) -> None:
        """Test closed-loop eigenvalues are stable."""
        # Arrange - Create robust controller for stability analysis
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=1.0,
        )
        controller = RobustController(config)

        # Act - Get closed-loop system eigenvalues
        eigenvalues = controller.get_closed_loop_eigenvalues()

        # Assert - All eigenvalues should have negative real parts (stable)
        assert np.all(np.real(eigenvalues) < 0)

    @pytest.mark.unit
    def test_get_closed_loop_matrix(self) -> None:
        """Test closed-loop matrix computation."""
        # Arrange - Create robust controller for closed-loop analysis
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=1.0,
        )
        controller = RobustController(config)

        # Act - Compute closed-loop system matrix A_cl = A - B2*K
        A_cl = controller.get_closed_loop_matrix()

        # Assert - Closed-loop matrix should be square matching state dimension
        assert A_cl.shape == (2, 2)

    @pytest.mark.unit
    def test_estimate_disturbance_attenuation(self) -> None:
        """Test disturbance attenuation estimation."""
        # Arrange - Create robust controller with gamma constraint
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=1.0,
        )
        controller = RobustController(config)

        # Act - Estimate disturbance-to-output attenuation level
        attenuation = controller.estimate_disturbance_attenuation()

        # Assert - Attenuation should be non-negative
        assert attenuation >= 0.0

    @pytest.mark.unit
    def test_simulate_with_disturbance(self) -> None:
        """Test simulation with disturbance inputs."""
        # Arrange - Create controller and constant disturbance sequence
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=1.0,
        )
        controller = RobustController(config)
        disturbances = [[0.1] for _ in range(100)]

        # Act - Simulate system with disturbance inputs
        states, controls = controller.simulate_with_disturbance(
            initial_state=[1.0, 0.0], disturbance_sequence=disturbances, dt=0.01
        )

        # Assert - Should return 101 states and 100 controls
        assert states.shape == (101, 2)
        assert controls.shape == (100, 1)

    @pytest.mark.unit
    def test_gamma_constraint_violation(self) -> None:
        """Test controller raises error when gamma is too small."""
        # Arrange - Create configuration with gamma too small for system
        # Use highly unstable system with large performance weights
        A = [[2, 1], [-1, 1]]  # Unstable eigenvalues
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[100, 0], [0, 100]]  # Very large performance weights
        D11 = [[0], [0]]
        D12 = [[0], [0.01]]  # Small control weight
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=0.001,  # Extremely small - will violate rho(XY) < gamma^2
        )

        # Act - Attempt to create controller with infeasible gamma
        # Assert - Should raise RuntimeError for violated gamma constraint
        with pytest.raises(RuntimeError, match="γ-constraint violated"):
            RobustController(config)

    @pytest.mark.unit
    def test_unstabilizable_system(self) -> None:
        """Test controller raises error for unstabilizable system."""
        # Arrange - Create unstabilizable system (unstable mode uncontrollable)
        A = [[1, 0], [0, -1]]  # One unstable eigenvalue at +1
        B1 = [[0], [1]]
        B2 = [[0], [1]]  # Cannot control first state (unstable mode)
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=2.0,
        )

        # Act - Attempt to create controller for unstabilizable system
        # Assert - Should raise ValueError for stabilizability violation
        with pytest.raises(ValueError, match="not stabilizable"):
            RobustController(config)

    @pytest.mark.unit
    def test_undetectable_system(self) -> None:
        """Test controller raises error for undetectable system."""
        # Arrange - Create undetectable system (unstable mode unobservable)
        A = [[1, 0], [0, -1]]  # One unstable eigenvalue at +1
        B1 = [[1], [0]]
        B2 = [[1], [0]]
        C1 = [[0, 1]]  # Cannot observe first state (unstable mode)
        D11 = [[0]]
        D12 = [[1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=2.0,
        )

        # Act - Attempt to create controller for undetectable system
        # Assert - Should raise ValueError for detectability or stabilizability violation
        # Note: System may fail stabilizability check before detectability check
        with pytest.raises(ValueError, match="not stabilizable|not detectable"):
            RobustController(config)

    @pytest.mark.unit
    def test_discrete_time_system(self) -> None:
        """Test controller works with discrete-time systems."""
        # Arrange - Create discrete-time robust control configuration
        # Use well-conditioned system with strong control authority
        A = [[0.5, 0.0], [0.0, 0.6]]  # Diagonal stable system
        B1 = [[0.1], [0.1]]
        B2 = [[1.0], [1.0]]  # Strong control authority
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[1.0], [1.0]]  # Significant control penalty for conditioning
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=10.0,  # Large gamma for easier feasibility
            system_type=SystemType.DISCRETE,
        )

        # Act - Initialize discrete-time H-infinity controller
        controller = RobustController(config)

        # Assert - Closed-loop eigenvalues should be inside unit circle (discrete stability)
        eigenvalues = controller.get_closed_loop_eigenvalues()
        assert np.all(np.abs(eigenvalues) < 1.0)

    @pytest.mark.unit
    def test_verify_gamma_constraint(self) -> None:
        """Test gamma constraint verification method."""
        # Arrange - Create controller with sufficiently large gamma
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=10.0,  # Large enough to satisfy constraint
        )
        controller = RobustController(config)

        # Act - Verify gamma constraint is satisfied
        is_satisfied = controller.verify_gamma_constraint()

        # Assert - Constraint should be satisfied with large gamma
        assert is_satisfied is True

    @pytest.mark.unit
    def test_compute_hinf_norm_frequency_domain(self) -> None:
        """Test H-infinity norm computation using frequency domain."""
        # Arrange - Create robust controller with gamma=5.0
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=5.0,
        )
        controller = RobustController(config)

        # Act - Compute H-infinity norm via frequency sweep
        hinf_norm = controller.compute_hinf_norm(
            num_freq_points=100, return_diagnostics=False
        )

        # Assert - Norm should be positive and less than gamma
        assert hinf_norm > 0.0
        assert hinf_norm < config.gamma  # Should satisfy constraint

    @pytest.mark.unit
    def test_riccati_solution_properties(self) -> None:
        """Test access to Riccati solution matrices."""
        # Arrange - Create robust controller for Riccati analysis
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=2.0,
        )
        controller = RobustController(config)

        # Act - Access control and filter Riccati solutions
        X = controller.control_riccati_solution
        Y = controller.filter_riccati_solution

        # Assert - Both solutions should exist and be positive semi-definite
        assert X is not None
        assert Y is not None
        assert X.shape == (2, 2)
        assert Y.shape == (2, 2)
        # Both should be positive semi-definite
        assert np.all(np.linalg.eigvals(X) >= -1e-10)
        assert np.all(np.linalg.eigvals(Y) >= -1e-10)

    @pytest.mark.unit
    def test_get_closed_loop_eigs_alias(self) -> None:
        """Test get_closed_loop_eigs() is an alias for get_closed_loop_eigenvalues()."""
        # Arrange - Create robust controller
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=1.0,
        )
        controller = RobustController(config)

        # Act - Get eigenvalues using both methods
        eigs1 = controller.get_closed_loop_eigenvalues()
        eigs2 = controller.get_closed_loop_eigs()

        # Assert - Both methods should return identical results
        assert np.allclose(eigs1, eigs2)

    @pytest.mark.unit
    def test_report_feasibility(self) -> None:
        """Test feasibility report provides complete diagnostics."""
        # Arrange - Create robust controller with known feasible system
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=5.0,  # Large gamma ensures feasibility
        )
        controller = RobustController(config)

        # Act - Get feasibility report
        report = controller.report_feasibility()

        # Assert - Report should contain all required keys
        assert "feasible" in report
        assert "rho_XY" in report
        assert "lambda_max_XY" in report
        assert "gamma_squared" in report
        assert "margin" in report
        assert "margin_percent" in report
        assert "min_cl_real_part" in report
        assert "is_stable" in report

        # Verify types
        assert isinstance(report["feasible"], bool)
        assert isinstance(report["rho_XY"], float)
        assert isinstance(report["is_stable"], bool)

        # Verify feasibility conditions
        assert report["feasible"] is True
        assert report["is_stable"] is True
        assert report["rho_XY"] < report["gamma_squared"]
        assert report["margin"] > 0

    @pytest.mark.unit
    def test_simulate_response_with_rk4(self) -> None:
        """Test simulate_response method with RK4 integration."""
        # Arrange - Create robust controller
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=1.0,
        )
        controller = RobustController(config)

        # Act - Simulate with zero disturbance
        times, states = controller.simulate_response(
            x0=[1.0, 0.0], t_final=1.0, dt=0.01, integration_method="rk4"
        )

        # Assert - Output shapes should be correct
        assert times.shape[0] == 100
        assert states.shape == (100, 2)
        # States should converge to zero
        assert np.linalg.norm(states[-1]) < np.linalg.norm(states[0])

    @pytest.mark.unit
    def test_simulate_response_with_disturbance_function(self) -> None:
        """Test simulate_response with time-varying disturbance function."""
        # Arrange - Create robust controller and sinusoidal disturbance
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=2.0,
        )
        controller = RobustController(config)

        # Define time-varying disturbance
        def disturbance(t: float) -> np.ndarray:
            return np.array([0.1 * np.sin(2 * np.pi * t)])

        # Act - Simulate with disturbance
        times, states = controller.simulate_response(
            x0=[0.0, 0.0], t_final=2.0, dt=0.01, w_func=disturbance
        )

        # Assert - States should remain bounded due to H∞ control
        max_state_norm = np.max(np.linalg.norm(states, axis=1))
        assert max_state_norm < 5.0  # Should be reasonably bounded

    @pytest.mark.unit
    def test_simulate_with_disturbance_euler_integration(self) -> None:
        """Test simulate_with_disturbance using Euler integration method."""
        # Arrange - Create controller and disturbance sequence
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=1.0,
        )
        controller = RobustController(config)
        disturbances = [[0.1] for _ in range(50)]

        # Act - Simulate with Euler integration
        states, controls = controller.simulate_with_disturbance(
            initial_state=[1.0, 0.0],
            disturbance_sequence=disturbances,
            dt=0.01,
            integration_method="euler",
        )

        # Assert - Should return correct shapes
        assert states.shape == (51, 2)
        assert controls.shape == (50, 1)

    @pytest.mark.unit
    def test_simulate_with_disturbance_invalid_method(self) -> None:
        """Test simulate_with_disturbance raises error for invalid integration method."""
        # Arrange - Create controller
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=1.0,
        )
        controller = RobustController(config)
        disturbances = [[0.1] for _ in range(10)]

        # Act & Assert - Should raise ValueError for invalid method
        with pytest.raises(ValueError, match="Unknown integration method"):
            controller.simulate_with_disturbance(
                initial_state=[1.0, 0.0],
                disturbance_sequence=disturbances,
                integration_method="invalid_method",
            )

    @pytest.mark.unit
    def test_compute_hinf_norm_with_diagnostics(self) -> None:
        """Test compute_hinf_norm returns diagnostics when requested."""
        # Arrange - Create robust controller
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=5.0,
        )
        controller = RobustController(config)

        # Act - Compute norm with diagnostics
        result = controller.compute_hinf_norm(
            num_freq_points=100, return_diagnostics=True
        )
        assert isinstance(result, tuple)
        norm, diagnostics = result

        # Assert - Should return norm and diagnostics dictionary
        assert isinstance(norm, float)
        assert isinstance(diagnostics, dict)
        assert "peak_frequency" in diagnostics
        assert "freq_min" in diagnostics
        assert "freq_max" in diagnostics
        assert "num_points" in diagnostics
        assert "frequencies" in diagnostics
        assert "singular_values" in diagnostics
        assert diagnostics["num_points"] == 100

    @pytest.mark.unit
    def test_compute_hinf_norm_custom_frequency_range(self) -> None:
        """Test compute_hinf_norm with custom frequency range."""
        # Arrange - Create robust controller
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=5.0,
        )
        controller = RobustController(config)

        # Act - Compute norm with custom frequency range
        norm = controller.compute_hinf_norm(
            num_freq_points=50, freq_range=(0.1, 10.0), return_diagnostics=False
        )

        # Assert - Should return valid norm
        assert norm > 0.0
        assert norm < config.gamma

    @pytest.mark.unit
    def test_compute_hinf_norm_invalid_frequency_range(self) -> None:
        """Test compute_hinf_norm raises error for invalid frequency range."""
        # Arrange - Create robust controller
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=5.0,
        )
        controller = RobustController(config)

        # Act & Assert - Should raise ValueError for invalid range
        with pytest.raises(ValueError, match="must be positive"):
            controller.compute_hinf_norm(freq_range=(0.0, 10.0))

        with pytest.raises(ValueError, match="must be >"):
            controller.compute_hinf_norm(freq_range=(10.0, 1.0))

    @pytest.mark.unit
    def test_optional_c2_d21_d22_matrices(self) -> None:
        """Test controller with optional C2, D21, D22 matrices."""
        # Arrange - Create config with custom measurement and noise matrices
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        C2 = [[1, 0]]  # Different measurement matrix
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        D21 = [[0.1]]  # Measurement noise coupling
        D22 = [[0]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            C2=C2,
            D11=D11,
            D12=D12,
            D21=D21,
            D22=D22,
            gamma=3.0,
        )

        # Act - Initialize controller with optional matrices
        controller = RobustController(config)

        # Assert - Gain matrix should be computed successfully
        assert controller.gain_matrix.shape == (1, 2)


class TestRobustIntegration:
    """Integration tests for robust controller."""

    @pytest.mark.integration
    def test_robust_controller_rejects_disturbance(self) -> None:
        """Test robust controller attenuates disturbances."""
        # Arrange - Create robust controller and persistent disturbance
        A = [[0, 1], [-1, -0.5]]
        B1 = [[0], [1]]
        B2 = [[0], [1]]
        C1 = [[1, 0], [0, 1]]
        D11 = [[0], [0]]
        D12 = [[0], [1]]
        config = RobustControlConfig(
            state_dim=2,
            control_dim=1,
            disturbance_dim=1,
            A=A,
            B1=B1,
            B2=B2,
            C1=C1,
            D11=D11,
            D12=D12,
            gamma=0.5,  # Strong disturbance rejection requirement
        )
        controller = RobustController(config)
        disturbances = [[1.0] for _ in range(200)]

        # Act - Simulate with constant disturbance applied throughout
        states, _ = controller.simulate_with_disturbance(
            initial_state=[0.0, 0.0], disturbance_sequence=disturbances, dt=0.01
        )

        # Assert - State norm should remain bounded by gamma constraint
        max_state_norm = np.max(np.linalg.norm(states, axis=1))
        assert max_state_norm < 5.0  # Should be reasonably bounded
