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
            sliding_surface_coeffs=[1.0, 2.0],
            switching_gain=5.0,
        )

        # Assert - Optional parameters should have default values
        assert config.state_dim == 2
        assert config.control_dim == 1
        assert config.sliding_surface_coeffs == [1.0, 2.0]
        assert config.switching_gain == 5.0
        assert config.boundary_layer_width == 0.1
        assert config.reaching_law == "constant"
        assert config.power_reaching_alpha == 0.5
        assert config.use_saturation is True
        assert config.control_limits is None
        assert config.debug is False

    @pytest.mark.unit
    def test_config_initialization_custom(self) -> None:
        """Test SlidingModeConfig initializes with custom values."""
        # Arrange - Define custom sliding mode parameters
        # Act - Initialize config with all custom values
        config = SlidingModeConfig(
            state_dim=3,
            control_dim=2,
            sliding_surface_coeffs=[1.0, 2.0, 3.0],
            switching_gain=10.0,
            boundary_layer_width=0.2,
            reaching_law="exponential",
            power_reaching_alpha=0.7,
            use_saturation=False,
            control_limits=(-10.0, 10.0),
            debug=True,
        )

        # Assert - Verify expected outcomes
        assert config.state_dim == 3
        assert config.control_dim == 2
        assert config.boundary_layer_width == 0.2
        assert config.reaching_law == "exponential"
        assert config.power_reaching_alpha == 0.7
        assert config.use_saturation is False
        assert config.control_limits == (-10.0, 10.0)
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
                sliding_surface_coeffs=[1.0],
                switching_gain=5.0,
            )

    @pytest.mark.unit
    def test_config_validates_negative_switching_gain(self) -> None:
        """Test config raises error for non-positive switching_gain."""
        # Arrange - Prepare to create config with zero switching gain
        # Act - Attempt to create config with switching_gain=0.0
        # Assert - Should raise ValidationError for non-positive switching_gain
        with pytest.raises(ValidationError, match="switching_gain"):
            SlidingModeConfig(
                state_dim=2,
                control_dim=1,
                sliding_surface_coeffs=[1.0, 2.0],
                switching_gain=0.0,
            )

    @pytest.mark.unit
    def test_config_validates_empty_coeffs(self) -> None:
        """Test config raises error for empty sliding surface coefficients."""
        # Arrange - Prepare to create config with empty coefficient list
        # Act - Attempt to create config with empty sliding_surface_coeffs
        # Assert - Should raise ValidationError for empty coefficients
        with pytest.raises(ValidationError, match="cannot be empty"):
            SlidingModeConfig(
                state_dim=2,
                control_dim=1,
                sliding_surface_coeffs=[],
                switching_gain=5.0,
            )

    @pytest.mark.unit
    def test_config_validates_invalid_reaching_law(self) -> None:
        """Test config raises error for invalid reaching law."""
        # Arrange - Prepare to create config with invalid reaching law name
        # Act - Attempt to create config with reaching_law="invalid"
        # Assert - Should raise ValidationError for invalid reaching law
        with pytest.raises(ValidationError, match="Reaching law must be one of"):
            SlidingModeConfig(
                state_dim=2,
                control_dim=1,
                sliding_surface_coeffs=[1.0, 2.0],
                switching_gain=5.0,
                reaching_law="invalid",
            )

    @pytest.mark.unit
    def test_config_validates_power_alpha_too_large(self) -> None:
        """Test config raises error for power_reaching_alpha >= 1.0."""
        # Arrange - Prepare to create config with alpha at maximum (invalid)
        # Act - Attempt to create config with power_reaching_alpha=1.0
        # Assert - Should raise ValidationError for alpha >= 1.0
        with pytest.raises(ValidationError, match="power_reaching_alpha"):
            SlidingModeConfig(
                state_dim=2,
                control_dim=1,
                sliding_surface_coeffs=[1.0, 2.0],
                switching_gain=5.0,
                power_reaching_alpha=1.0,
            )

    @pytest.mark.unit
    def test_config_validates_invalid_control_limits(self) -> None:
        """Test config raises error for invalid control limits."""
        # Arrange - Prepare to create config with reversed control limits
        # Act - Attempt to create config with min > max (10.0, 5.0)
        # Assert - Should raise ValidationError for invalid control limits
        with pytest.raises(ValidationError, match="Control min"):
            SlidingModeConfig(
                state_dim=2,
                control_dim=1,
                sliding_surface_coeffs=[1.0, 2.0],
                switching_gain=5.0,
                control_limits=(10.0, 5.0),
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
            sliding_surface_coeffs=[1.0, 2.0],
            switching_gain=5.0,
        )

        # Act - Execute the code under test
        controller = SlidingModeController(config)

        # Assert - Verify expected outcomes
        assert controller.config.state_dim == 2
        assert len(controller.get_sliding_surface_history()) == 0

    @pytest.mark.unit
    def test_controller_initialization_dimension_mismatch(self) -> None:
        """Test controller raises error for dimension mismatch."""
        # Arrange - Create config with mismatched coefficients (2 coeffs for 3D state)
        config = SlidingModeConfig(
            state_dim=3,
            control_dim=1,
            sliding_surface_coeffs=[1.0, 2.0],  # Only 2 coeffs for 3D state
            switching_gain=5.0,
        )

        # Act - Attempt to create controller with mismatched dimensions
        # Assert - Should raise ValueError for coefficient dimension mismatch
        with pytest.raises(ValueError, match="Sliding surface coefficients dimension"):
            SlidingModeController(config)

    @pytest.mark.unit
    def test_compute_sliding_surface(self) -> None:
        """Test sliding surface computation."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            sliding_surface_coeffs=[2.0, 3.0],
            switching_gain=5.0,
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
            sliding_surface_coeffs=[1.0, 1.0],
            switching_gain=5.0,
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        control = controller.compute(state=[1.0, 0.5], state_derivative=[0.1, -0.2])

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
            sliding_surface_coeffs=[1.0, 1.0],
            switching_gain=5.0,
        )
        controller = SlidingModeController(config)

        # Act - Attempt to compute with 3-element state vector
        # Assert - Should raise ValueError for state dimension mismatch
        with pytest.raises(ValueError, match="State dimension"):
            controller.compute(state=[1.0, 2.0, 3.0], state_derivative=[0.1, 0.2])

    @pytest.mark.unit
    def test_compute_wrong_derivative_dimension(self) -> None:
        """Test compute raises error for wrong state_derivative dimension."""
        # Arrange - Create 2-state sliding mode controller
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            sliding_surface_coeffs=[1.0, 1.0],
            switching_gain=5.0,
        )
        controller = SlidingModeController(config)

        # Act - Attempt to compute with 1-element derivative (expects 2)
        # Assert - Should raise ValueError for derivative dimension mismatch
        with pytest.raises(ValueError, match="State derivative dimension"):
            controller.compute(state=[1.0, 2.0], state_derivative=[0.1])

    @pytest.mark.unit
    def test_compute_constant_reaching_law(self) -> None:
        """Test constant reaching law produces expected control."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            sliding_surface_coeffs=[1.0, 0.0],
            switching_gain=5.0,
            reaching_law="constant",
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        control = controller.compute(state=[1.0, 0.0], state_derivative=[0.0, 0.0])

        # Assert - Control should oppose sliding surface
        assert control.shape == (1,)

    @pytest.mark.unit
    def test_compute_exponential_reaching_law(self) -> None:
        """Test exponential reaching law."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            sliding_surface_coeffs=[1.0, 1.0],
            switching_gain=5.0,
            reaching_law="exponential",
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        control = controller.compute(state=[1.0, 0.5], state_derivative=[0.1, -0.2])

        # Assert - Verify expected outcomes
        assert control.shape == (1,)

    @pytest.mark.unit
    def test_compute_power_reaching_law(self) -> None:
        """Test power reaching law."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            sliding_surface_coeffs=[1.0, 1.0],
            switching_gain=5.0,
            reaching_law="power",
            power_reaching_alpha=0.5,
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        control = controller.compute(state=[1.0, 0.5], state_derivative=[0.1, -0.2])

        # Assert - Verify expected outcomes
        assert control.shape == (1,)

    @pytest.mark.unit
    def test_compute_with_disturbance_bound(self) -> None:
        """Test control compensates for disturbance bound."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            sliding_surface_coeffs=[1.0, 1.0],
            switching_gain=5.0,
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        control_no_dist = controller.compute(
            state=[1.0, 0.5], state_derivative=[0.1, -0.2], disturbance_bound=0.0
        )
        controller.reset()
        control_with_dist = controller.compute(
            state=[1.0, 0.5], state_derivative=[0.1, -0.2], disturbance_bound=2.0
        )

        # Assert - Control with disturbance should be larger
        assert abs(control_with_dist[0]) > abs(control_no_dist[0])

    @pytest.mark.unit
    def test_compute_with_saturation(self) -> None:
        """Test saturation function reduces chattering."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            sliding_surface_coeffs=[1.0, 0.0],
            switching_gain=5.0,
            use_saturation=True,
            boundary_layer_width=0.5,
        )
        controller = SlidingModeController(config)

        # Act - State close to surface
        control = controller.compute(state=[0.1, 0.0], state_derivative=[0.0, 0.0])

        # Assert - Control should be smooth (not at maximum)
        assert abs(control[0]) < 5.0  # Less than full switching gain

    @pytest.mark.unit
    def test_compute_with_control_limits(self) -> None:
        """Test control saturation respects limits."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            sliding_surface_coeffs=[1.0, 1.0],
            switching_gain=100.0,  # Very large gain
            control_limits=(-1.0, 1.0),
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        control = controller.compute(state=[10.0, 10.0], state_derivative=[0.0, 0.0])

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
            sliding_surface_coeffs=[1.0, 0.0],
            switching_gain=5.0,
            boundary_layer_width=0.5,
        )
        controller = SlidingModeController(config)

        # Act - State very close to surface
        controller.compute(state=[0.01, 0.0], state_derivative=[0.0, 0.0])
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
            sliding_surface_coeffs=[1.0, 0.0],
            switching_gain=5.0,
            boundary_layer_width=0.1,
        )
        controller = SlidingModeController(config)

        # Act - State far from surface
        controller.compute(state=[10.0, 0.0], state_derivative=[0.0, 0.0])
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
            sliding_surface_coeffs=[1.0, 1.0],
            switching_gain=5.0,
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        for i in range(5):
            controller.compute(state=[float(i), float(i)], state_derivative=[0.0, 0.0])

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
            sliding_surface_coeffs=[1.0, 1.0],
            switching_gain=5.0,
        )
        controller = SlidingModeController(config)
        controller.compute(state=[1.0, 1.0], state_derivative=[0.0, 0.0])

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
            sliding_surface_coeffs=[1.0, 0.0],
            switching_gain=5.0,
        )
        controller = SlidingModeController(config)

        # Act - Simulate chattering
        for i in range(20):
            state = [0.1 * ((-1) ** i), 0.0]  # Oscillating state
            controller.compute(state=state, state_derivative=[0.0, 0.0])

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
            sliding_surface_coeffs=[1.0, 0.0],
            switching_gain=5.0,
            reaching_law="constant",
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
            sliding_surface_coeffs=[1.0, 0.0],
            switching_gain=5.0,
            reaching_law="exponential",
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
            sliding_surface_coeffs=[1.0, 0.0],
            switching_gain=5.0,
            reaching_law="power",
            power_reaching_alpha=0.5,
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
            sliding_surface_coeffs=[1.0, 1.0],
            switching_gain=5.0,
            system_matrix_A=[[0.0, 1.0], [-1.0, -2.0]],
            control_matrix_B=[[0.0], [1.0]],
        )

        # Assert - System matrices should be stored correctly
        assert config.system_matrix_A == [[0.0, 1.0], [-1.0, -2.0]]
        assert config.control_matrix_B == [[0.0], [1.0]]

    @pytest.mark.unit
    def test_config_validates_a_matrix_square(self) -> None:
        """Test config raises error for non-square A matrix."""
        # Arrange - Prepare to create config with non-square A matrix (2x3)
        # Act - Attempt to create config with rectangular system_matrix_A
        # Assert - Should raise ValidationError for non-square A
        with pytest.raises(ValidationError, match="System matrix A must be square"):
            SlidingModeConfig(
                state_dim=2,
                control_dim=1,
                sliding_surface_coeffs=[1.0, 1.0],
                switching_gain=5.0,
                system_matrix_A=[[0.0, 1.0, 2.0], [-1.0, -2.0, 0.0]],
                control_matrix_B=[[0.0], [1.0]],
            )

    @pytest.mark.unit
    def test_config_validates_b_matrix_dimensions(self) -> None:
        """Test config validates B matrix has consistent row lengths."""
        # Arrange - Prepare config with inconsistent B matrix row lengths
        # Act - Attempt to create config with jagged control_matrix_B
        # Assert - Should raise ValidationError for inconsistent B dimensions
        with pytest.raises(
            ValidationError, match="Control matrix B rows must have same length"
        ):
            SlidingModeConfig(
                state_dim=2,
                control_dim=1,
                sliding_surface_coeffs=[1.0, 1.0],
                switching_gain=5.0,
                system_matrix_A=[[0.0, 1.0], [-1.0, -2.0]],
                control_matrix_B=[[0.0], [1.0, 2.0]],  # Inconsistent dimensions
            )

    @pytest.mark.unit
    def test_controller_with_full_state_dynamics(self) -> None:
        """Test controller uses A and B matrices for equivalent control."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            sliding_surface_coeffs=[1.0, 1.0],
            switching_gain=5.0,
            system_matrix_A=[[0.0, 1.0], [-1.0, -2.0]],
            control_matrix_B=[[0.0], [1.0]],
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        control = controller.compute(state=[1.0, 0.5], state_derivative=[0.5, -1.5])

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
            sliding_surface_coeffs=[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],  # 2x3 matrix
            switching_gain=5.0,
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
            sliding_surface_coeffs=[1.0, 0.0],
            switching_gain=5.0,
            use_smooth_approximation=True,
            smooth_approximation_slope=2.0,
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        control = controller.compute(state=[0.1, 0.0], state_derivative=[0.0, 0.0])

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
            sliding_surface_coeffs=[1.0, 1.0],
            switching_gain=5.0,
            reaching_law="exponential",
            exponential_reaching_k1=1.5,
            exponential_reaching_k2=0.8,
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        control = controller.compute(state=[1.0, 0.5], state_derivative=[0.1, -0.2])

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
            sliding_surface_coeffs=[1.0, 1.0],
            switching_gain=5.0,
            reaching_law="power",
            power_reaching_k1=1.2,
            power_reaching_k2=0.5,
            power_reaching_alpha=0.7,
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        control = controller.compute(state=[1.0, 0.5], state_derivative=[0.1, -0.2])

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
            sliding_surface_coeffs=[1.0, 0.0],
            switching_gain=5.0,
            adaptive_gain=True,
            adaptive_gain_rate=0.1,
        )
        controller = SlidingModeController(config)

        # Act - Compute multiple times with large sliding surface
        initial_gain = controller.get_current_switching_gain()
        for _ in range(10):
            controller.compute(state=[10.0, 0.0], state_derivative=[0.0, 0.0])
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
            sliding_surface_coeffs=[1.0, 0.0],
            switching_gain=5.0,
            adaptive_gain=True,
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        controller.compute(state=[5.0, 0.0], state_derivative=[0.0, 0.0])
        controller.compute(state=[3.0, 0.0], state_derivative=[0.0, 0.0])
        estimated_bound = controller.get_estimated_disturbance_bound()

        # Assert - Verify expected outcomes
        assert estimated_bound > 0.0
        assert estimated_bound <= 5.0  # Should be at most the max sliding surface

    @pytest.mark.unit
    def test_reset_clears_adaptive_gain(self) -> None:
        """Test reset restores initial switching gain."""
        # Arrange - Set up test fixtures and inputs
        config = SlidingModeConfig(
            state_dim=2,
            control_dim=1,
            sliding_surface_coeffs=[1.0, 0.0],
            switching_gain=5.0,
            adaptive_gain=True,
            adaptive_gain_rate=0.1,
        )
        controller = SlidingModeController(config)

        # Act - Execute the code under test
        initial_gain = controller.get_current_switching_gain()
        controller.compute(state=[10.0, 0.0], state_derivative=[0.0, 0.0])
        controller.compute(state=[10.0, 0.0], state_derivative=[0.0, 0.0])
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
            sliding_surface_coeffs=[1.0, 1.0],
            switching_gain=10.0,
            boundary_layer_width=0.1,
        )
        controller = SlidingModeController(config)

        # Act - Simulate system
        x = np.array([5.0, -5.0])  # Start far from surface
        dt = 0.01

        for _ in range(500):
            # Simple double integrator: dx1/dt = x2, dx2/dt = u
            x_dot = np.array([x[1], 0.0])  # Will be modified by control
            u = controller.compute(state=x, state_derivative=x_dot)

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
            sliding_surface_coeffs=[1.0, 2.0],
            switching_gain=15.0,
            boundary_layer_width=0.2,
        )
        controller = SlidingModeController(config)

        # Act - Simulate with disturbance
        x = np.array([1.0, 1.0])
        dt = 0.01

        for i in range(300):
            x_dot = np.array([x[1], 0.0])
            u = controller.compute(
                state=x, state_derivative=x_dot, disturbance_bound=2.0
            )

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
            sliding_surface_coeffs=[[1.0, 0.0], [0.0, 1.0]],  # Identity sliding surface
            switching_gain=10.0,
            system_matrix_A=[[0.0, 1.0], [-1.0, -1.0]],
            control_matrix_B=[[1.0, 0.0], [0.0, 1.0]],  # Identity control matrix
            use_smooth_approximation=True,
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
            u = controller.compute(state=x, state_derivative=np.zeros(2))

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
            sliding_surface_coeffs=[1.0, 1.0],
            switching_gain=5.0,
            adaptive_gain=True,
            adaptive_gain_rate=0.05,
            use_smooth_approximation=True,
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

            u = controller.compute(state=x, state_derivative=x_dot)

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
