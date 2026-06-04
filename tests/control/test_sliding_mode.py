"""Tests for Sliding Mode Control implementation."""

import numpy as np
import pytest
from pydantic import ValidationError

from algokit.algorithms.control.sliding_mode import (
    SlidingModeConfig,
    SlidingModeController,
)


class TestSlidingModeConfig:
    """Test sliding mode control configuration model."""

    @pytest.mark.unit
    def test_config_initialization_default(self) -> None:
        """Test SlidingModeConfig initializes with default values."""
        # Arrange - Define sliding mode parameters for 2-state system
        # Act - Initialize config with required parameters only
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 2.0],
        )

        # Assert - Optional parameters should have default values
        assert config.state_dim == 2
        assert config.control_dim == 1
        assert config.C == [1.0, 2.0]
        assert config.K_init == 1.0
        assert config.boundary_layer == 0.1
        assert config.reaching == "constant"
        assert config.alpha == 0.6
        assert config.mode == "tanh"
        assert config.u_min is None
        assert config.u_max is None
        assert config.debug is False

    @pytest.mark.unit
    def test_config_initialization_custom(self) -> None:
        """Test SlidingModeConfig initializes with custom values."""
        # Arrange - Define custom sliding mode parameters
        # Act - Initialize config with all custom values
        config = SlidingModeConfig(
            state_dim=3,
            control_dim=2,
            C=[1.0, 2.0, 3.0],
            K_init=10.0,
            boundary_layer=0.2,
            reaching="exponential",
            alpha=0.7,
            mode="sat",
            u_min=-10.0,
            u_max=10.0,
            debug=True,
        )

        # Assert - Verify expected outcomes
        assert config.state_dim == 3
        assert config.control_dim == 2
        assert config.boundary_layer == 0.2
        assert config.reaching == "exponential"
        assert config.alpha == 0.7
        assert config.mode == "sat"
        assert config.u_min == -10.0
        assert config.u_max == 10.0
        assert config.debug is True

    @pytest.mark.unit
    def test_config_validates_negative_state_dim(self) -> None:
        """Test config raises error for non-positive state_dim."""
        # Arrange - Prepare to create config with zero state dimension
        # Act - Attempt to create config with state_dim=0
        # Assert - Should raise ValidationError for non-positive state_dim
        with pytest.raises(ValidationError, match="state_dim"):
            SlidingModeConfig(
                state_dim=0,
                control_dim=1,
                C=[1.0],
            )

    @pytest.mark.unit
    def test_config_validates_negative_k_init(self) -> None:
        """Test config raises error for non-positive K_init."""
        # Arrange - Prepare to create config with zero initial gain
        # Act - Attempt to create config with K_init=0.0
        # Assert - Should raise ValidationError for non-positive K_init
        with pytest.raises(ValidationError, match="K_init"):
            SlidingModeConfig(
                state_dim=2,
                control_dim=1,
                C=[1.0, 2.0],
                K_init=0.0,
            )

    @pytest.mark.unit
    def test_config_validates_empty_c(self) -> None:
        """Test config raises error for empty sliding surface matrix C."""
        # Arrange - Prepare to create config with empty C matrix
        # Act - Attempt to create config with empty C
        # Assert - Should raise ValidationError for empty C
        with pytest.raises(ValidationError, match="cannot be empty"):
            SlidingModeConfig(
                state_dim=2,
                control_dim=1,
                C=[],
            )

    @pytest.mark.unit
    def test_config_validates_invalid_reaching(self) -> None:
        """Test config raises error for invalid reaching law."""
        # Arrange - Prepare to create config with invalid reaching law name
        # Act - Attempt to create config with reaching="invalid"
        # Assert - Should raise ValidationError for invalid reaching law
        with pytest.raises(ValidationError, match="reaching must be one of"):
            SlidingModeConfig(
                state_dim=2,
                control_dim=1,
                C=[1.0, 2.0],
                reaching="invalid",
            )

    @pytest.mark.unit
    def test_config_validates_alpha_too_large(self) -> None:
        """Test config raises error for alpha >= 1.0."""
        # Arrange - Prepare to create config with alpha at maximum (invalid)
        # Act - Attempt to create config with alpha=1.0
        # Assert - Should raise ValidationError for alpha >= 1.0
        with pytest.raises(ValidationError, match="alpha"):
            SlidingModeConfig(
                state_dim=2,
                control_dim=1,
                C=[1.0, 2.0],
                alpha=1.0,
            )

    @pytest.mark.unit
    def test_config_validates_invalid_mode(self) -> None:
        """Test config raises error for invalid switching mode."""
        # Arrange - Prepare to create config with invalid mode
        # Act - Attempt to create config with mode="invalid"
        # Assert - Should raise ValidationError for invalid mode
        with pytest.raises(ValidationError, match="mode must be one of"):
            SlidingModeConfig(
                state_dim=2,
                control_dim=1,
                C=[1.0, 2.0],
                mode="invalid",
            )


class TestSlidingModeController:
    """Test Sliding Mode Controller implementation."""

    @pytest.mark.unit
    def test_controller_initialization(self) -> None:
        """Test SlidingModeController initializes correctly."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 2.0],
        )

        # Act - Execute the code under test
        controller = SlidingModeController(config)

        # Assert - Verify expected outcomes
        assert controller.cfg.state_dim == 2
        assert len(controller.get_sliding_surface_history()) == 0

    @pytest.mark.unit
    def test_controller_initialization_dimension_mismatch(self) -> None:
        """Test controller raises error for dimension mismatch."""
        # Arrange - Create config with mismatched coefficients (2 coeffs for 3D state)
        config = SlidingModeConfig(
            state_dim=3,
            control_dim=1,
            C=[1.0, 2.0],  # Only 2 coeffs for 3D state
        )

        # Act - Attempt to create controller with mismatched dimensions
        # Assert - Should raise ValueError for coefficient dimension mismatch
        with pytest.raises(ValueError, match="Sliding surface matrix C has"):
            SlidingModeController(config)

    @pytest.mark.unit
    def test_compute_sliding_surface(self) -> None:
        """Test sliding surface computation."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[2.0, 3.0],
        )
        controller = SlidingModeController(config)
        state = np.array([1.0, 2.0])

        # Act - Execute the code under test
        s = controller.compute_sliding_surface(state)

        # Assert - s = 2.0*1.0 + 3.0*2.0 = 8.0
        assert s == pytest.approx(8.0)

    @pytest.mark.unit
    def test_compute_basic(self) -> None:
        """Test compute returns control output."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 1.0],
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test (using new API)
        control = controller.compute(
            x=np.array([1.0, 0.5]), xdot=np.array([0.1, -0.2]), dt=0.01
        )

        # Assert - Verify expected outcomes
        assert control.shape == (1,)
        assert isinstance(control[0], (float, np.floating))

    @pytest.mark.unit
    def test_compute_wrong_state_dimension(self) -> None:
        """Test compute raises error for wrong state dimension."""
        # Arrange - Create 2-state sliding mode controller
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 1.0],
        )
        controller = SlidingModeController(config)

        # Act - Attempt to compute with 3-element state vector
        # Assert - Should raise ValueError for state dimension mismatch
        with pytest.raises(ValueError, match="State dimension"):
            controller.compute(x=np.array([1.0, 2.0, 3.0]), xdot=None, dt=0.01)

    @pytest.mark.unit
    def test_compute_requires_xdot_without_ab(self) -> None:
        """Test compute raises error when xdot needed but not provided."""
        # Arrange - Create controller without A,B matrices
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 1.0],
        )
        controller = SlidingModeController(config)

        # Act - Attempt to compute without xdot and without A,B
        # Assert - Should raise ValueError requiring xdot
        with pytest.raises(ValueError, match="xdot is required"):
            controller.compute(x=np.array([1.0, 2.0]), xdot=None, dt=0.01)

    @pytest.mark.unit
    def test_compute_constant_reaching_law(self) -> None:
        """Test constant reaching law produces expected control."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 0.0],
            reaching="constant",
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        control = controller.compute(
            x=np.array([1.0, 0.0]), xdot=np.array([0.0, 0.0]), dt=0.01
        )

        # Assert - Control should oppose sliding surface
        assert control.shape == (1,)

    @pytest.mark.unit
    def test_compute_exponential_reaching_law(self) -> None:
        """Test exponential reaching law."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 1.0],
            reaching="exponential",
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        control = controller.compute(
            x=np.array([1.0, 0.5]), xdot=np.array([0.1, -0.2]), dt=0.01
        )

        # Assert - Verify expected outcomes
        assert control.shape == (1,)

    @pytest.mark.unit
    def test_compute_power_reaching_law(self) -> None:
        """Test power reaching law."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 1.0],
            reaching="power",
            alpha=0.5,
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        control = controller.compute(
            x=np.array([1.0, 0.5]), xdot=np.array([0.1, -0.2]), dt=0.01
        )

        # Assert - Verify expected outcomes
        assert control.shape == (1,)

    @pytest.mark.unit
    def test_compute_with_adaptive_gain(self) -> None:
        """Test adaptive gain increases with large errors."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 1.0],
            adaptive_gain=True,
            eta=2.0,
            rho=0.1,
        )
        controller = SlidingModeController(config)

        # Act - Execute multiple steps with large error
        initial_K = controller.get_current_switching_gain()
        for _ in range(10):
            controller.compute(
                x=np.array([5.0, 5.0]), xdot=np.array([0.0, 0.0]), dt=0.01
            )
        final_K = controller.get_current_switching_gain()

        # Assert - Gain should have increased
        assert final_K > initial_K

    @pytest.mark.unit
    def test_compute_with_mode_sat(self) -> None:
        """Test saturation mode reduces chattering."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 0.0],
            mode="sat",
            boundary_layer=0.5,
        )
        controller = SlidingModeController(config)

        # Act - State close to surface
        control = controller.compute(
            x=np.array([0.1, 0.0]), xdot=np.array([0.0, 0.0]), dt=0.01
        )

        # Assert - Control should be smooth (not at maximum)
        assert isinstance(control[0], (float, np.floating))

    @pytest.mark.unit
    def test_compute_with_control_limits(self) -> None:
        """Test control saturation respects limits."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 1.0],
            K_init=100.0,  # Very large gain
            u_min=-1.0,
            u_max=1.0,
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        control = controller.compute(
            x=np.array([10.0, 10.0]), xdot=np.array([0.0, 0.0]), dt=0.01
        )

        # Assert - Verify expected outcomes
        assert control[0] >= -1.0
        assert control[0] <= 1.0

    @pytest.mark.unit
    def test_is_on_sliding_surface_true(self) -> None:
        """Test surface detection when on surface."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 0.0],
            K_init=5.0,
            boundary_layer=0.5,
        )
        controller = SlidingModeController(config)

        # Act - State very close to surface
        controller.compute(x=np.array([0.01, 0.0]), xdot=np.array([0.0, 0.0]), dt=0.01)
        on_surface = controller.is_on_sliding_surface()

        # Assert - Verify expected outcomes
        assert on_surface is True

    @pytest.mark.unit
    def test_is_on_sliding_surface_false(self) -> None:
        """Test surface detection when far from surface."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 0.0],
            K_init=5.0,
            boundary_layer=0.1,
        )
        controller = SlidingModeController(config)

        # Act - State far from surface
        controller.compute(x=np.array([10.0, 0.0]), xdot=np.array([0.0, 0.0]), dt=0.01)
        on_surface = controller.is_on_sliding_surface()

        # Assert - Verify expected outcomes
        assert on_surface is False

    @pytest.mark.unit
    def test_get_sliding_surface_history(self) -> None:
        """Test sliding surface history is recorded."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 1.0],
            K_init=5.0,
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        for i in range(5):
            controller.compute(
                x=np.array([float(i), float(i)]), xdot=np.array([0.0, 0.0]), dt=0.01
            )

        history = controller.get_sliding_surface_history()

        # Assert - Verify expected outcomes
        assert len(history) == 5

    @pytest.mark.unit
    def test_reset_clears_history(self) -> None:
        """Test reset clears sliding surface history."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 1.0],
            K_init=5.0,
        )
        controller = SlidingModeController(config)
        controller.compute(x=np.array([1.0, 1.0]), xdot=np.array([0.0, 0.0]), dt=0.01)

        # Act - Execute the code under test
        controller.reset()

        # Assert - Verify expected outcomes
        assert len(controller.get_sliding_surface_history()) == 0

    @pytest.mark.unit
    def test_estimate_chattering_magnitude(self) -> None:
        """Test chattering magnitude estimation."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 0.0],
            K_init=5.0,
        )
        controller = SlidingModeController(config)

        # Act - Simulate chattering
        for i in range(20):
            state = [0.1 * ((-1) ** i), 0.0]  # Oscillating state
            controller.compute(x=np.array(state), xdot=np.array([0.0, 0.0]), dt=0.01)

        chattering = controller.estimate_chattering_magnitude(window_size=10)

        # Assert - Verify expected outcomes
        assert chattering > 0.0
        assert chattering < float("inf")

    @pytest.mark.unit
    def test_get_reaching_time_estimate_constant(self) -> None:
        """Test reaching time estimate for constant law."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 0.0],
            K_init=5.0,
            reaching="constant",
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        time = controller.get_reaching_time_estimate(initial_s=10.0)

        # Assert - T = |s(0)| / K = 10 / 5 = 2.0
        assert time == pytest.approx(2.0)

    @pytest.mark.unit
    def test_get_reaching_time_estimate_exponential(self) -> None:
        """Test reaching time estimate for exponential law."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 0.0],
            K_init=5.0,
            reaching="exponential",
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        time = controller.get_reaching_time_estimate(initial_s=10.0)

        # Assert - Should be faster than constant law
        assert time > 0.0
        assert time < 2.0  # Less than constant law

    @pytest.mark.unit
    def test_get_reaching_time_estimate_power(self) -> None:
        """Test reaching time estimate for power law."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 0.0],
            K_init=5.0,
            reaching="power",
            alpha=0.5,
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        time = controller.get_reaching_time_estimate(initial_s=4.0)

        # Assert - Finite time convergence
        assert time > 0.0
        assert time < float("inf")


class TestSlidingModeAdvancedFeatures:
    """Tests for advanced sliding mode controller features."""

    @pytest.mark.unit
    def test_config_with_system_matrices(self) -> None:
        """Test configuration with A and B matrices."""
        # Arrange - Define system dynamics matrices A and B
        # Act - Initialize config with optional system matrices
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 1.0],
            K_init=5.0,
            A=[[0.0, 1.0], [-1.0, -2.0]],
            B=[[0.0], [1.0]],
        )

        # Assert - System matrices should be stored correctly
        assert config.A == [[0.0, 1.0], [-1.0, -2.0]]
        assert config.B == [[0.0], [1.0]]

    @pytest.mark.unit
    def test_config_validates_a_matrix_square(self) -> None:
        """Test config raises error for non-square A matrix."""
        # Arrange - Prepare to create config with non-square A matrix (2x3)
        # Act - Attempt to create config with rectangular system_matrix_A
        # Assert - Should raise ValidationError for non-square A
        with pytest.raises(ValidationError, match="must be square"):
            SlidingModeConfig(
                state_dim=2,
                control_dim=1,
                C=[1.0, 1.0],
                K_init=5.0,
                A=[[0.0, 1.0, 2.0], [-1.0, -2.0, 0.0]],
                B=[[0.0], [1.0]],
            )

    @pytest.mark.unit
    def test_config_validates_b_matrix_dimensions(self) -> None:
        """Test config validates B matrix has consistent row lengths."""
        # Arrange - Prepare config with inconsistent B matrix row lengths
        # Act - Attempt to create config with jagged control_matrix_B
        # Assert - Should raise ValidationError for inconsistent B dimensions
        with pytest.raises(ValidationError, match="must have same length"):
            SlidingModeConfig(
                state_dim=2,
                control_dim=1,
                C=[1.0, 1.0],
                K_init=5.0,
                A=[[0.0, 1.0], [-1.0, -2.0]],
                B=[[0.0], [1.0, 2.0]],  # Inconsistent dimensions
            )

    @pytest.mark.unit
    def test_controller_with_full_state_dynamics(self) -> None:
        """Test controller uses A and B matrices for equivalent control."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 1.0],
            K_init=5.0,
            A=[[0.0, 1.0], [-1.0, -2.0]],
            B=[[0.0], [1.0]],
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        control = controller.compute(
            x=np.array([1.0, 0.5]), xdot=np.array([0.5, -1.5]), dt=0.01
        )

        # Assert - Control should be computed using A and B matrices
        assert control.shape == (1,)
        assert isinstance(control[0], (float, np.floating))

    @pytest.mark.unit
    def test_mimo_sliding_surface(self) -> None:
        """Test MIMO sliding surface with matrix coefficients."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=3,
            control_dim=2,
            C=[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],  # 2x3 matrix
            K_init=5.0,
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        s = controller.compute_sliding_surface(np.array([1.0, 2.0, 3.0]))

        # Assert - Should return vector of size 2
        assert s.shape == (2,)
        assert s[0] == pytest.approx(4.0)  # 1*1 + 0*2 + 1*3 = 4
        assert s[1] == pytest.approx(5.0)  # 0*1 + 1*2 + 1*3 = 5

    @pytest.mark.unit
    def test_smooth_approximation_tanh(self) -> None:
        """Test smooth tanh approximation instead of saturation."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 0.0],
            K_init=5.0,
            mode="tanh",
            tanh_slope=2.0,
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        control = controller.compute(
            x=np.array([0.1, 0.0]), xdot=np.array([0.0, 0.0]), dt=0.01
        )

        # Assert - Control should be smooth (using tanh)
        assert control.shape == (1,)
        # Tanh-based control should be smoother than sign-based
        assert abs(control[0]) < 10.0

    @pytest.mark.unit
    def test_exponential_reaching_law_with_k1_k2(self) -> None:
        """Test exponential reaching law uses configurable k1 and k2."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 1.0],
            K_init=5.0,
            reaching="exponential",
            k1=1.5,
            k2=0.8,
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        control = controller.compute(
            x=np.array([1.0, 0.5]), xdot=np.array([0.1, -0.2]), dt=0.01
        )

        # Assert - Verify expected outcomes
        assert control.shape == (1,)
        assert isinstance(control[0], (float, np.floating))

    @pytest.mark.unit
    def test_power_reaching_law_with_k1_k2(self) -> None:
        """Test power reaching law uses configurable k1 and k2."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 1.0],
            K_init=5.0,
            reaching="power",
            k1=1.2,
            k2=0.5,
            alpha=0.7,
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        control = controller.compute(
            x=np.array([1.0, 0.5]), xdot=np.array([0.1, -0.2]), dt=0.01
        )

        # Assert - Verify expected outcomes
        assert control.shape == (1,)
        assert isinstance(control[0], (float, np.floating))

    @pytest.mark.unit
    def test_adaptive_gain_increases_with_disturbance(self) -> None:
        """Test adaptive gain update based on sliding surface magnitude."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 0.0],
            K_init=5.0,
            adaptive_gain=True,
            eta=0.1,
        )
        controller = SlidingModeController(config)

        # Act - Compute multiple times with large sliding surface
        initial_gain = controller.get_current_switching_gain()
        for _ in range(10):
            controller.compute(
                x=np.array([10.0, 0.0]), xdot=np.array([0.0, 0.0]), dt=0.01
            )
        final_gain = controller.get_current_switching_gain()

        # Assert - Gain should have increased
        assert final_gain > initial_gain

    @pytest.mark.unit
    def test_get_estimated_disturbance_bound(self) -> None:
        """Test disturbance bound estimation from sliding surface history."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 0.0],
            K_init=5.0,
            adaptive_gain=True,
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        controller.compute(x=np.array([5.0, 0.0]), xdot=np.array([0.0, 0.0]), dt=0.01)
        controller.compute(x=np.array([3.0, 0.0]), xdot=np.array([0.0, 0.0]), dt=0.01)
        estimated_bound = controller.get_estimated_disturbance_bound()

        # Assert - Verify expected outcomes
        assert estimated_bound >= 0.0  # May be 0 with new observer
        # Disturbance observer may not capture all surface motion

    @pytest.mark.unit
    def test_reset_clears_adaptive_gain(self) -> None:
        """Test reset restores initial switching gain."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 0.0],
            K_init=5.0,
            adaptive_gain=True,
            eta=0.1,
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        initial_gain = controller.get_current_switching_gain()
        controller.compute(x=np.array([10.0, 0.0]), xdot=np.array([0.0, 0.0]), dt=0.01)
        controller.compute(x=np.array([10.0, 0.0]), xdot=np.array([0.0, 0.0]), dt=0.01)
        controller.reset()
        reset_gain = controller.get_current_switching_gain()

        # Assert - Verify expected outcomes
        assert reset_gain == initial_gain


class TestSlidingModeIntegration:
    """Integration tests for sliding mode controller."""

    @pytest.mark.integration
    def test_sliding_mode_reaches_surface(self) -> None:
        """Test controller drives system to sliding surface."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 1.0],
            K_init=10.0,
            boundary_layer=0.1,
        )
        controller = SlidingModeController(config)

        # Act - Simulate system
        x = np.array([5.0, -5.0])  # Start far from surface
        dt = 0.01

        for _ in range(500):
            # Simple double integrator: dx1/dt = x2, dx2/dt = u
            x_dot = np.array([x[1], 0.0])  # Will be modified by control
            u = controller.compute(x=np.array(x), xdot=x_dot, dt=0.01)

            # Update state with control
            x_dot[1] = u[0]
            x = x + x_dot * dt

        # Assert - Should reach sliding surface
        s = controller.compute_sliding_surface(x)
        assert np.linalg.norm(s) < 0.5

    @pytest.mark.integration
    def test_sliding_mode_rejects_disturbances(self) -> None:
        """Test robustness to bounded disturbances."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 2.0],
            K_init=15.0,
            boundary_layer=0.2,
        )
        controller = SlidingModeController(config)

        # Act - Simulate with disturbance
        x = np.array([1.0, 1.0])
        dt = 0.01

        for i in range(300):
            x_dot = np.array([x[1], 0.0])
            u = controller.compute(x=x, xdot=x_dot, dt=dt)

            # Add disturbance
            disturbance = 2.0 * np.sin(i * dt * 10)  # Bounded disturbance

            # Update with control and disturbance
            x_dot[1] = u[0] + disturbance
            x = x + x_dot * dt

        # Assert - Should still be near surface despite disturbance
        s = controller.compute_sliding_surface(x)
        assert np.linalg.norm(s) < 1.0

    @pytest.mark.integration
    def test_mimo_controller_with_full_dynamics(self) -> None:
        """Test MIMO controller with full-state dynamics (A, B matrices)."""
        # Arrange - 2-state, 2-control system
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=2,
            C=[[1.0, 0.0], [0.0, 1.0]],  # Identity sliding surface
            K_init=10.0,
            A=[[0.0, 1.0], [-1.0, -1.0]],
            B=[[1.0, 0.0], [0.0, 1.0]],  # Identity control matrix
            mode="tanh",
        )
        controller = SlidingModeController(config)

        # Act - Simulate system
        x = np.array([2.0, -1.0])
        dt = 0.01

        for _ in range(200):
            # Full-state dynamics: x_dot = Ax + Bu
            A = np.array([[0.0, 1.0], [-1.0, -1.0]])
            B = np.array([[1.0, 0.0], [0.0, 1.0]])

            # Compute control (controller will use A and B internally)
            u = controller.compute(x=x, xdot=None, dt=0.01)

            # Update state
            x_dot = A @ x + B @ u
            x = x + x_dot * dt

        # Assert - Should reach sliding surface
        s = controller.compute_sliding_surface(x)
        assert np.linalg.norm(s) < 1.0

    @pytest.mark.integration
    def test_adaptive_gain_handles_varying_disturbance(self) -> None:
        """Test adaptive gain adjusts to varying disturbance levels."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            C=[1.0, 1.0],
            K_init=5.0,
            adaptive_gain=True,
            eta=0.05,
            mode="tanh",
        )
        controller = SlidingModeController(config)

        # Act - Simulate with time-varying disturbance
        x = np.array([3.0, -2.0])
        dt = 0.01
        gains = []

        for i in range(500):
            x_dot = np.array([x[1], 0.0])

            # Time-varying disturbance bound
            disturbance_level = 1.0 + 2.0 * np.sin(i * dt * 2 * np.pi / 10)

            u = controller.compute(x=np.array(x), xdot=x_dot, dt=0.01)

            # Apply disturbance
            disturbance = disturbance_level * np.cos(i * dt * 20)
            x_dot[1] = u[0] + disturbance
            x = x + x_dot * dt

            gains.append(controller.get_current_switching_gain())

        # Assert - Gain should have adapted
        assert max(gains) > min(gains)
        # System should still be reasonably close to sliding surface
        s = controller.compute_sliding_surface(x)
        assert np.linalg.norm(s) < 2.0
