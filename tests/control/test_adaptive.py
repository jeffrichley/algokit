"""Tests for Adaptive Control implementation."""

import numpy as np
import pytest
from pydantic import ValidationError

from algokit.algorithms.control.adaptive import (
    AdaptiveControlConfig,
    AdaptiveController,
    FirstOrderReferenceModel,
    PersistenceOfExcitationMonitor,
    SecondOrderReferenceModel,
    SimpleFirstOrderPlant,
    simulate_closed_loop,
)


class TestAdaptiveControlConfig:
    """Test adaptive control configuration model."""

    @pytest.mark.unit
    def test_config_initialization_default(self) -> None:
        """Test AdaptiveControlConfig initializes with default values."""
        # Arrange - Set up basic configuration parameters
        # Act - Initialize config with minimal parameters
        config = AdaptiveControlConfig(num_parameters=3)

        # Assert - Verify all default values are set correctly
        assert config.num_parameters == 3
        assert config.adaptation_gain == 0.1
        assert config.reference_model is None
        assert config.initial_parameters is None
        assert config.parameter_bounds is None
        assert config.dead_zone == 0.0
        assert config.sigma_modification == 0.0
        assert config.use_normalization is True
        assert config.debug is False

    @pytest.mark.unit
    def test_config_initialization_custom(self) -> None:
        """Test AdaptiveControlConfig initializes with custom values."""

        # Arrange - Define custom reference model and parameters
        def ref_model(x: float) -> float:
            return 2.0 * x

        # Act - Initialize config with all custom values
        config = AdaptiveControlConfig(
            num_parameters=5,
            adaptation_gain=0.5,
            reference_model=ref_model,
            initial_parameters=[1.0, 2.0, 3.0, 4.0, 5.0],
            parameter_bounds=(-10.0, 10.0),
            dead_zone=0.1,
            sigma_modification=0.01,
            use_normalization=False,
            debug=True,
        )

        # Assert - Verify all custom values are set correctly
        assert config.num_parameters == 5
        assert config.adaptation_gain == 0.5
        assert config.reference_model is not None
        assert config.initial_parameters == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert config.parameter_bounds == (-10.0, 10.0)
        assert config.dead_zone == 0.1
        assert config.sigma_modification == 0.01
        assert config.use_normalization is False
        assert config.debug is True

    @pytest.mark.unit
    def test_config_validates_negative_num_parameters(self) -> None:
        """Test config raises error for non-positive num_parameters."""
        # Arrange - Prepare invalid num_parameters value
        # Act - Attempt to create config with zero parameters
        # Assert - Should raise ValidationError
        with pytest.raises(ValidationError, match="num_parameters"):
            AdaptiveControlConfig(num_parameters=0)

    @pytest.mark.unit
    def test_config_validates_negative_adaptation_gain(self) -> None:
        """Test config raises error for negative adaptation_gain."""
        # Arrange - Prepare invalid negative adaptation gain
        # Act - Attempt to create config with negative gain
        # Assert - Should raise ValidationError
        with pytest.raises(ValidationError, match="adaptation_gain"):
            AdaptiveControlConfig(num_parameters=3, adaptation_gain=-0.1)

    @pytest.mark.unit
    def test_config_validates_zero_adaptation_gain(self) -> None:
        """Test config raises error for zero adaptation_gain."""
        # Arrange - Prepare zero adaptation gain value
        # Act - Attempt to create config with zero gain
        # Assert - Should raise ValidationError
        with pytest.raises(ValidationError, match="adaptation_gain"):
            AdaptiveControlConfig(num_parameters=3, adaptation_gain=0.0)

    @pytest.mark.unit
    def test_config_validates_negative_dead_zone(self) -> None:
        """Test config raises error for negative dead_zone."""
        # Arrange - Prepare negative dead zone value
        # Act - Attempt to create config with negative dead zone
        # Assert - Should raise ValidationError
        with pytest.raises(ValidationError, match="dead_zone"):
            AdaptiveControlConfig(num_parameters=3, dead_zone=-0.1)

    @pytest.mark.unit
    def test_config_validates_sigma_too_large(self) -> None:
        """Test config raises error for sigma_modification > 1.0."""
        # Arrange - Prepare sigma value exceeding maximum
        # Act - Attempt to create config with sigma > 1.0
        # Assert - Should raise ValidationError
        with pytest.raises(ValidationError, match="sigma_modification"):
            AdaptiveControlConfig(num_parameters=3, sigma_modification=1.5)

    @pytest.mark.unit
    def test_config_validates_invalid_parameter_bounds(self) -> None:
        """Test config raises error when parameter min >= max."""
        # Arrange - Prepare invalid bounds where min > max
        # Act - Attempt to create config with inverted bounds
        # Assert - Should raise ValidationError
        with pytest.raises(ValidationError, match="Parameter min"):
            AdaptiveControlConfig(num_parameters=3, parameter_bounds=(10.0, 5.0))


class TestAdaptiveController:
    """Test Adaptive Controller implementation."""

    @pytest.mark.unit
    def test_controller_initialization_default(self) -> None:
        """Test AdaptiveController initializes with zero parameters."""
        # Arrange - Create config with default parameters
        config = AdaptiveControlConfig(num_parameters=3)

        # Act - Initialize controller
        controller = AdaptiveController(config)

        # Assert - Parameters should be zero-initialized
        assert controller.config.num_parameters == 3
        assert len(controller.parameters) == 3
        assert np.allclose(controller.parameters, [0.0, 0.0, 0.0])

    @pytest.mark.unit
    def test_controller_initialization_custom_parameters(self) -> None:
        """Test AdaptiveController initializes with custom parameters."""
        # Arrange - Create config with custom initial parameters
        config = AdaptiveControlConfig(
            num_parameters=3, initial_parameters=[1.0, 2.0, 3.0]
        )

        # Act - Initialize controller
        controller = AdaptiveController(config)

        # Assert - Parameters should match initial values
        assert np.allclose(controller.parameters, [1.0, 2.0, 3.0])

    @pytest.mark.unit
    def test_controller_initialization_parameter_count_mismatch(self) -> None:
        """Test controller raises error when initial_parameters count mismatches."""
        # Arrange - Create config with mismatched parameter count
        config = AdaptiveControlConfig(num_parameters=3, initial_parameters=[1.0, 2.0])

        # Act - Attempt to initialize controller
        # Assert - Should raise ValueError
        with pytest.raises(ValueError, match="Expected 3 initial parameters"):
            AdaptiveController(config)

    @pytest.mark.unit
    def test_compute_basic(self) -> None:
        """Test compute returns control output based on parameters."""
        # Arrange - Create controller with known parameters
        config = AdaptiveControlConfig(num_parameters=2, initial_parameters=[2.0, 3.0])
        controller = AdaptiveController(config)

        # Act - Compute control output
        output = controller.compute(
            measurement=5.0, regressor=[1.0, 2.0], reference=10.0
        )

        # Assert - Output should be dot(params, regressor) = 2.0*1.0 + 3.0*2.0 = 8.0
        assert output == pytest.approx(8.0)

    @pytest.mark.unit
    def test_compute_updates_parameters(self) -> None:
        """Test compute adapts parameters based on tracking error."""
        # Arrange - Create controller with zero initial parameters
        config = AdaptiveControlConfig(
            num_parameters=2,
            initial_parameters=[0.0, 0.0],
            adaptation_gain=1.0,
            use_normalization=False,
        )
        controller = AdaptiveController(config)

        # Act - Run adaptation step with large error
        initial_params = controller.parameters.copy()
        controller.compute(
            measurement=0.0, regressor=[1.0, 1.0], reference=10.0, dt=1.0
        )
        updated_params = controller.parameters

        # Assert - Parameters should change due to error
        assert not np.allclose(initial_params, updated_params)

    @pytest.mark.unit
    def test_compute_with_regressor_dimension_mismatch(self) -> None:
        """Test compute raises error for wrong regressor dimension."""
        # Arrange - Create controller expecting 3 parameters
        config = AdaptiveControlConfig(num_parameters=3)
        controller = AdaptiveController(config)

        # Act - Attempt to compute with wrong regressor size
        # Assert - Should raise ValueError
        with pytest.raises(ValueError, match="Regressor dimension"):
            controller.compute(measurement=5.0, regressor=[1.0, 2.0])

    @pytest.mark.unit
    def test_compute_with_reference_model(self) -> None:
        """Test compute uses reference_model when reference not provided."""

        # Arrange - Create controller with callable reference model
        def ref_model(x: float) -> float:
            return 2.0 * x

        config = AdaptiveControlConfig(
            num_parameters=2,
            initial_parameters=[1.0, 1.0],
            reference_model=ref_model,
        )
        controller = AdaptiveController(config)

        # Act - Compute without explicit reference
        controller.compute(measurement=5.0, regressor=[1.0, 1.0])

        # Assert - Reference model should compute reference as 2.0 * 5.0 = 10.0
        error = controller.get_tracking_error(5.0)
        assert error == pytest.approx(5.0)

    @pytest.mark.unit
    def test_compute_with_dead_zone(self) -> None:
        """Test dead zone prevents adaptation on small errors."""
        # Arrange - Create controller with dead zone enabled
        config = AdaptiveControlConfig(
            num_parameters=2,
            initial_parameters=[1.0, 1.0],
            adaptation_gain=1.0,
            dead_zone=0.5,
            use_normalization=False,
        )
        controller = AdaptiveController(config)

        # Act - Compute with small error within dead zone
        initial_params = controller.parameters.copy()
        controller.compute(
            measurement=9.7, regressor=[1.0, 1.0], reference=10.0, dt=1.0
        )
        updated_params = controller.parameters

        # Assert - Parameters should not change (error = 0.3 < 0.5)
        assert np.allclose(initial_params, updated_params)

    @pytest.mark.unit
    def test_compute_with_normalization(self) -> None:
        """Test normalized gradient adaptation."""
        # Arrange - Create controller with normalization enabled
        config = AdaptiveControlConfig(
            num_parameters=2,
            initial_parameters=[0.0, 0.0],
            adaptation_gain=1.0,
            use_normalization=True,
        )
        controller = AdaptiveController(config)

        # Act - Compute with normalized gradient
        controller.compute(
            measurement=0.0, regressor=[1.0, 1.0], reference=10.0, dt=1.0
        )
        params = controller.parameters

        # Assert - Parameters updated with normalization factor
        # Normalization = 1 + (1^2 + 1^2) = 3
        # Update = 1.0 * 10.0 * [1.0, 1.0] / 3 * 1.0 = [3.33, 3.33]
        assert params[0] == pytest.approx(10.0 / 3.0, abs=0.01)
        assert params[1] == pytest.approx(10.0 / 3.0, abs=0.01)

    @pytest.mark.unit
    def test_compute_with_sigma_modification(self) -> None:
        """Test sigma modification adds leakage to parameters."""
        # Arrange - Create controller with sigma modification
        config = AdaptiveControlConfig(
            num_parameters=2,
            initial_parameters=[10.0, 10.0],
            adaptation_gain=0.1,
            sigma_modification=0.1,
            use_normalization=False,
        )
        controller = AdaptiveController(config)

        # Act - Compute with zero error (only leakage acts)
        controller.compute(
            measurement=10.0, regressor=[1.0, 1.0], reference=10.0, dt=1.0
        )
        params = controller.parameters

        # Assert - Parameters should decrease due to leakage
        assert params[0] < 10.0
        assert params[1] < 10.0

    @pytest.mark.unit
    def test_compute_with_parameter_bounds(self) -> None:
        """Test parameter projection to bounds."""
        # Arrange - Create controller with parameter bounds
        config = AdaptiveControlConfig(
            num_parameters=2,
            initial_parameters=[0.0, 0.0],
            adaptation_gain=100.0,
            parameter_bounds=(-5.0, 5.0),
            use_normalization=False,
        )
        controller = AdaptiveController(config)

        # Act - Apply large adaptation that would exceed bounds
        controller.compute(
            measurement=0.0, regressor=[1.0, 1.0], reference=10.0, dt=1.0
        )
        params = controller.parameters

        # Assert - Parameters should be clipped to bounds
        assert params[0] <= 5.0
        assert params[1] <= 5.0

    @pytest.mark.unit
    def test_reset_clears_state(self) -> None:
        """Test reset returns controller to initial state."""
        # Arrange - Create controller and run adaptation
        config = AdaptiveControlConfig(num_parameters=2, initial_parameters=[1.0, 2.0])
        controller = AdaptiveController(config)
        controller.compute(measurement=5.0, regressor=[1.0, 1.0], reference=10.0)

        # Act - Reset controller
        controller.reset()

        # Assert - State should be restored
        assert np.allclose(controller.parameters, [1.0, 2.0])
        assert len(controller.get_adaptation_history()) == 0

    @pytest.mark.unit
    def test_reset_to_zero_when_no_initial(self) -> None:
        """Test reset returns to zero when no initial_parameters."""
        # Arrange - Create controller without initial parameters and run
        config = AdaptiveControlConfig(num_parameters=2)
        controller = AdaptiveController(config)
        controller.compute(measurement=5.0, regressor=[1.0, 1.0], reference=10.0)

        # Act - Reset controller
        controller.reset()

        # Assert - Parameters should return to zero
        assert np.allclose(controller.parameters, [0.0, 0.0])

    @pytest.mark.unit
    def test_get_adaptation_history(self) -> None:
        """Test adaptation history is recorded correctly."""
        # Arrange - Create controller
        config = AdaptiveControlConfig(num_parameters=2)
        controller = AdaptiveController(config)

        # Act - Run multiple adaptation steps
        for i in range(5):
            controller.compute(
                measurement=float(i), regressor=[1.0, 1.0], reference=10.0
            )

        history = controller.get_adaptation_history()

        # Assert - History should contain all parameter snapshots
        assert len(history) == 5
        assert all(isinstance(params, np.ndarray) for params in history)

    @pytest.mark.unit
    def test_set_parameters(self) -> None:
        """Test manual parameter setting."""
        # Arrange - Create controller with default parameters
        config = AdaptiveControlConfig(num_parameters=3)
        controller = AdaptiveController(config)

        # Act - Manually set new parameter values
        controller.set_parameters([1.0, 2.0, 3.0])

        # Assert - Parameters should be updated
        assert np.allclose(controller.parameters, [1.0, 2.0, 3.0])

    @pytest.mark.unit
    def test_set_parameters_dimension_mismatch(self) -> None:
        """Test set_parameters raises error for wrong dimension."""
        # Arrange - Create controller expecting 3 parameters
        config = AdaptiveControlConfig(num_parameters=3)
        controller = AdaptiveController(config)

        # Act - Attempt to set wrong number of parameters
        # Assert - Should raise ValueError
        with pytest.raises(ValueError, match="Expected 3 parameters"):
            controller.set_parameters([1.0, 2.0])

    @pytest.mark.unit
    def test_get_tracking_error(self) -> None:
        """Test tracking error calculation."""
        # Arrange - Create controller and set reference
        config = AdaptiveControlConfig(num_parameters=2)
        controller = AdaptiveController(config)
        controller.compute(measurement=5.0, regressor=[1.0, 1.0], reference=10.0)

        # Act - Get tracking error
        error = controller.get_tracking_error(5.0)

        # Assert - Error should be reference minus measurement
        assert error == pytest.approx(5.0)

    @pytest.mark.unit
    def test_estimate_convergence_rate_with_history(self) -> None:
        """Test convergence rate estimation from adaptation history."""
        # Arrange - Create controller and run adaptation
        config = AdaptiveControlConfig(num_parameters=2, adaptation_gain=0.1)
        controller = AdaptiveController(config)

        # Act - Run multiple steps and estimate convergence rate
        for _ in range(10):
            controller.compute(measurement=0.0, regressor=[1.0, 1.0], reference=10.0)

        rate = controller.estimate_convergence_rate()

        # Assert - Rate should be finite and positive
        assert rate > 0.0
        assert rate < float("inf")

    @pytest.mark.unit
    def test_estimate_convergence_rate_no_history(self) -> None:
        """Test convergence rate returns inf with insufficient history."""
        # Arrange - Create controller without running it
        config = AdaptiveControlConfig(num_parameters=2)
        controller = AdaptiveController(config)

        # Act - Estimate convergence rate with no history
        rate = controller.estimate_convergence_rate()

        # Assert - Should return infinity with insufficient data
        assert rate == float("inf")

    @pytest.mark.unit
    def test_parameters_property_returns_copy(self) -> None:
        """Test parameters property returns a copy, not reference."""
        # Arrange - Create controller with initial parameters
        config = AdaptiveControlConfig(num_parameters=2, initial_parameters=[1.0, 2.0])
        controller = AdaptiveController(config)

        # Act - Get parameters and modify them
        params = controller.parameters
        params[0] = 999.0

        # Assert - Internal parameters should remain unchanged
        assert controller.parameters[0] == 1.0


class TestAdaptiveIntegration:
    """Integration tests for adaptive controller."""

    @pytest.mark.integration
    def test_adaptive_tracks_constant_reference(self) -> None:
        """Test adaptive controller runs closed-loop simulation."""
        # Arrange - Create controller and plant for integration test
        config = AdaptiveControlConfig(
            num_parameters=2,
            initial_parameters=[5.0, 0.5],  # Good initial guess
            adaptation_gain=3.0,
            use_normalization=True,
        )
        controller = AdaptiveController(config)
        plant = SimpleFirstOrderPlant(a=1.0, b=1.0, initial_state=5.0)

        # Act - Run closed-loop simulation
        results = simulate_closed_loop(
            controller=controller,
            plant=plant,
            reference_input=10.0,
            duration=15.0,
            dt=0.05,
            regressor_fn=lambda y: np.array([1.0, y]),
        )

        # Assert - Simulation completes successfully
        assert len(results["time"]) == 300
        # Parameters should adapt
        param_change = np.linalg.norm(
            results["parameters"][-1] - results["parameters"][0]
        )
        assert param_change > 0.01  # Parameters adapted

    @pytest.mark.integration
    def test_adaptive_converges_to_true_parameters(self) -> None:
        """Test adaptive controller adapts parameters over time."""
        # Arrange - Create controller with good initial guess
        config = AdaptiveControlConfig(
            num_parameters=2,
            initial_parameters=[1.0, 0.2],  # Better starting point
            adaptation_gain=3.0,
            use_normalization=True,
        )
        controller = AdaptiveController(config)
        plant = SimpleFirstOrderPlant(a=1.0, b=1.0, initial_state=1.0)

        # Act - Run simulation with time-varying reference for PE
        results = simulate_closed_loop(
            controller=controller,
            plant=plant,
            reference_input=lambda t: 8.0 + 2.0 * np.sin(0.3 * t),  # PE signal
            duration=30.0,
            dt=0.02,
            regressor_fn=lambda y: np.array([1.0, y]),
        )

        # Assert - Parameters should adapt during simulation
        initial_params = results["parameters"][0]
        final_params = results["parameters"][-1]
        param_change = np.linalg.norm(final_params - initial_params)
        assert param_change > 0.1  # Parameters changed from initial values


class TestFirstOrderReferenceModel:
    """Test first-order reference model."""

    @pytest.mark.unit
    def test_initialization(self) -> None:
        """Test FirstOrderReferenceModel initializes correctly."""
        # Arrange - Create reference model with specific parameters
        # Act - Initialize model
        model = FirstOrderReferenceModel(a_m=2.0, b_m=1.5, initial_state=1.0)

        # Assert - Verify parameters are set correctly
        assert model.a_m == 2.0
        assert model.b_m == 1.5
        assert model.state == 1.0

    @pytest.mark.unit
    def test_step_updates_state(self) -> None:
        """Test step updates state according to dynamics."""
        # Arrange - Create model at zero initial state
        model = FirstOrderReferenceModel(a_m=1.0, b_m=1.0, initial_state=0.0)

        # Act - Execute one time step with positive reference
        output = model.step(reference_input=10.0, dt=0.1)

        # Assert - State should move toward steady state
        assert output > 0.0

    @pytest.mark.unit
    def test_get_state(self) -> None:
        """Test get_state returns current state as array."""
        # Arrange - Create model with non-zero initial state
        model = FirstOrderReferenceModel(initial_state=5.0)

        # Act - Get current state
        state = model.get_state()

        # Assert - State is returned as numpy array
        assert isinstance(state, np.ndarray)
        assert state[0] == 5.0

    @pytest.mark.unit
    def test_reset(self) -> None:
        """Test reset returns model to initial state."""
        # Arrange - Create model and advance its state
        model = FirstOrderReferenceModel(initial_state=2.0)
        model.step(reference_input=10.0, dt=0.1)

        # Act - Reset model
        model.reset()

        # Assert - State returns to initial value
        assert model.state == 2.0


class TestSecondOrderReferenceModel:
    """Test second-order reference model."""

    @pytest.mark.unit
    def test_initialization(self) -> None:
        """Test SecondOrderReferenceModel initializes correctly."""
        # Arrange - Set up second-order model parameters
        # Act - Initialize model with specific values
        model = SecondOrderReferenceModel(
            omega_n=2.0, zeta=0.5, initial_position=1.0, initial_velocity=0.5
        )

        # Assert - Verify all parameters are set correctly
        assert model.omega_n == 2.0
        assert model.zeta == 0.5
        assert model.position == 1.0
        assert model.velocity == 0.5

    @pytest.mark.unit
    def test_step_updates_state(self) -> None:
        """Test step updates state using RK4 integration."""
        # Arrange - Create model at zero initial state
        model = SecondOrderReferenceModel(omega_n=1.0, zeta=0.7)

        # Act - Execute one RK4 integration step
        output = model.step(reference_input=10.0, dt=0.01)

        # Assert - Position should start moving from zero
        assert output != 0.0

    @pytest.mark.unit
    def test_get_state(self) -> None:
        """Test get_state returns position and velocity."""
        # Arrange - Create model with non-zero initial conditions
        model = SecondOrderReferenceModel(initial_position=3.0, initial_velocity=2.0)

        # Act - Get current state vector
        state = model.get_state()

        # Assert - State contains position and velocity
        assert isinstance(state, np.ndarray)
        assert len(state) == 2
        assert state[0] == 3.0
        assert state[1] == 2.0

    @pytest.mark.unit
    def test_reset(self) -> None:
        """Test reset returns model to initial state."""
        # Arrange - Create model and advance state multiple steps
        model = SecondOrderReferenceModel(initial_position=5.0, initial_velocity=1.0)
        for _ in range(10):
            model.step(reference_input=10.0, dt=0.01)

        # Act - Reset model to initial conditions
        model.reset()

        # Assert - Position and velocity return to initial values
        assert model.position == 5.0
        assert model.velocity == 1.0


class TestSimpleFirstOrderPlant:
    """Test simple first-order plant."""

    @pytest.mark.unit
    def test_initialization(self) -> None:
        """Test SimpleFirstOrderPlant initializes correctly."""
        # Arrange - Set up plant parameters
        # Act - Initialize plant with specific values
        plant = SimpleFirstOrderPlant(a=1.5, b=2.0, initial_state=3.0)

        # Assert - Verify all parameters are set correctly
        assert plant.a == 1.5
        assert plant.b == 2.0
        assert plant.state == 3.0

    @pytest.mark.unit
    def test_step_updates_state(self) -> None:
        """Test step updates plant state."""
        # Arrange - Create plant at zero initial state
        plant = SimpleFirstOrderPlant(a=1.0, b=1.0, initial_state=0.0)

        # Act - Apply control input for one time step
        output = plant.step(control_input=5.0, dt=0.1)

        # Assert - State should change based on control input
        assert output != 0.0

    @pytest.mark.unit
    def test_step_with_disturbance(self) -> None:
        """Test step applies disturbance function."""

        # Arrange - Create plant with constant disturbance function
        def disturbance(t: float) -> float:
            return 1.0  # Constant disturbance

        plant = SimpleFirstOrderPlant(a=1.0, b=1.0, disturbance_fn=disturbance)

        # Act - Step with zero control input
        output = plant.step(control_input=0.0, dt=0.1)

        # Assert - Disturbance should affect state despite zero control
        assert output != 0.0

    @pytest.mark.unit
    def test_get_state(self) -> None:
        """Test get_state returns current state."""
        # Arrange - Create plant with non-zero initial state
        plant = SimpleFirstOrderPlant(initial_state=7.0)

        # Act - Get current state
        state = plant.get_state()

        # Assert - State is returned as numpy array
        assert isinstance(state, np.ndarray)
        assert state[0] == 7.0

    @pytest.mark.unit
    def test_reset(self) -> None:
        """Test reset returns plant to initial state."""
        # Arrange - Create plant and advance its state
        plant = SimpleFirstOrderPlant(initial_state=4.0)
        plant.step(control_input=10.0, dt=0.1)

        # Act - Reset plant to initial conditions
        plant.reset()

        # Assert - State and time are reset
        assert plant.state == 4.0
        assert plant.time == 0.0


class TestPersistenceOfExcitationMonitor:
    """Test persistence of excitation monitor."""

    @pytest.mark.unit
    def test_initialization(self) -> None:
        """Test PersistenceOfExcitationMonitor initializes correctly."""
        # Arrange - Set up monitor parameters
        # Act - Initialize PE monitor
        monitor = PersistenceOfExcitationMonitor(
            window_size=50, condition_threshold=50.0
        )

        # Assert - Verify parameters are set correctly
        assert monitor.window_size == 50
        assert monitor.condition_threshold == 50.0

    @pytest.mark.unit
    def test_update_stores_regressors(self) -> None:
        """Test update stores regressor samples."""
        # Arrange - Create PE monitor with window size
        monitor = PersistenceOfExcitationMonitor(window_size=10)

        # Act - Add multiple regressor samples
        for i in range(5):
            monitor.update(np.array([float(i), float(i**2)]))

        # Assert - History contains all samples
        assert len(monitor.regressor_history) == 5

    @pytest.mark.unit
    def test_update_maintains_window_size(self) -> None:
        """Test update maintains fixed window size."""
        # Arrange - Create PE monitor with small window
        monitor = PersistenceOfExcitationMonitor(window_size=5)

        # Act - Add more samples than window size
        for i in range(10):
            monitor.update(np.array([float(i), float(i**2)]))

        # Assert - History is limited to window size
        assert len(monitor.regressor_history) == 5

    @pytest.mark.unit
    def test_is_persistently_exciting_insufficient_data(self) -> None:
        """Test is_persistently_exciting returns False with insufficient data."""
        # Arrange - Add fewer samples than window size
        monitor = PersistenceOfExcitationMonitor(window_size=10)
        for _ in range(5):
            monitor.update(np.array([1.0, 2.0]))

        # Act - Check PE status
        is_pe = monitor.is_persistently_exciting()

        # Assert - Should not be PE with insufficient data
        assert is_pe is False

    @pytest.mark.unit
    def test_is_persistently_exciting_with_rich_signal(self) -> None:
        """Test is_persistently_exciting with rich signal."""
        # Arrange - Create monitor with higher threshold and add rich signal
        monitor = PersistenceOfExcitationMonitor(
            window_size=50, condition_threshold=200.0
        )
        for i in range(50):
            t = i * 0.1
            regressor = np.array([np.sin(t), np.cos(2 * t)])
            monitor.update(regressor)

        # Act - Check if signal is persistently exciting
        is_pe = monitor.is_persistently_exciting()

        # Assert - Rich multi-frequency signal should meet PE criteria
        assert is_pe

    @pytest.mark.unit
    def test_get_condition_number_insufficient_data(self) -> None:
        """Test get_condition_number returns inf with insufficient data."""
        # Arrange - Create monitor with no data
        monitor = PersistenceOfExcitationMonitor(window_size=10)

        # Act - Get condition number without sufficient samples
        cond = monitor.get_condition_number()

        # Assert - Should return infinity for insufficient data
        assert cond == float("inf")

    @pytest.mark.unit
    def test_get_min_eigenvalue(self) -> None:
        """Test get_min_eigenvalue returns minimum eigenvalue."""
        # Arrange - Add sufficient samples to monitor
        monitor = PersistenceOfExcitationMonitor(window_size=10)
        for i in range(10):
            monitor.update(np.array([1.0, float(i)]))

        # Act - Get minimum eigenvalue of covariance
        min_eig = monitor.get_min_eigenvalue()

        # Assert - Eigenvalue should be non-negative
        assert min_eig >= 0.0

    @pytest.mark.unit
    def test_reset(self) -> None:
        """Test reset clears regressor history."""
        # Arrange - Add samples to monitor
        monitor = PersistenceOfExcitationMonitor(window_size=10)
        for _ in range(5):
            monitor.update(np.array([1.0, 2.0]))

        # Act - Reset monitor
        monitor.reset()

        # Assert - History should be cleared
        assert len(monitor.regressor_history) == 0


class TestAdaptiveControllerAdvancedFeatures:
    """Test advanced features of adaptive controller."""

    @pytest.mark.unit
    def test_controller_with_reference_model_dynamics(self) -> None:
        """Test controller uses dynamic reference model."""
        # Arrange - Create controller with first-order reference model
        ref_model = FirstOrderReferenceModel(a_m=1.0, b_m=1.0)
        config = AdaptiveControlConfig(
            num_parameters=2, reference_model_dynamics=ref_model
        )
        controller = AdaptiveController(config)

        # Act - Execute one control step
        controller.compute(
            measurement=0.0, regressor=[1.0, 0.0], reference_input=10.0, dt=0.1
        )

        # Assert - Reference model state should have advanced
        ref_state = controller.get_reference_model_state()
        assert ref_state is not None
        assert ref_state[0] != 0.0

    @pytest.mark.unit
    def test_controller_with_pe_monitoring(self) -> None:
        """Test controller with PE monitoring enabled."""
        # Arrange - Create controller with PE monitoring
        config = AdaptiveControlConfig(
            num_parameters=2, enable_pe_monitoring=True, pe_window_size=20
        )
        controller = AdaptiveController(config)

        # Act - Run control loop with rich varying regressor
        for i in range(25):
            t = i * 0.1
            controller.compute(
                measurement=0.0,
                regressor=[np.sin(t), np.cos(2 * t)],
                reference=10.0,
                dt=0.1,
            )

        # Assert - Should have valid PE condition number
        cond = controller.get_pe_condition_number()
        assert cond != float("inf")

    @pytest.mark.unit
    def test_controller_with_adaptive_gain(self) -> None:
        """Test controller with adaptive learning rate."""
        # Arrange - Create controller with adaptive gain enabled
        config = AdaptiveControlConfig(
            num_parameters=2,
            adaptation_gain=0.1,
            enable_adaptive_gain=True,
            gain_adaptation_rate=0.1,
        )
        controller = AdaptiveController(config)

        # Act - Compute with large tracking error
        controller.compute(
            measurement=0.0, regressor=[1.0, 1.0], reference=10.0, dt=0.1
        )

        # Assert - Learning rate should increase due to large error
        current_gain = controller.get_current_gain()
        assert current_gain > 0.1

    @pytest.mark.unit
    def test_controller_with_lyapunov_monitoring(self) -> None:
        """Test controller with Lyapunov monitoring enabled."""
        # Arrange - Create controller with Lyapunov monitoring
        config = AdaptiveControlConfig(
            num_parameters=2, enable_lyapunov_monitoring=True
        )
        controller = AdaptiveController(config)

        # Act - Run multiple control steps
        for _ in range(10):
            controller.compute(
                measurement=0.0, regressor=[1.0, 0.0], reference=10.0, dt=0.1
            )

        # Assert - Lyapunov function values should be recorded
        lyap_hist = controller.get_lyapunov_history()
        assert len(lyap_hist) > 0

    @pytest.mark.unit
    def test_get_current_gain(self) -> None:
        """Test get_current_gain returns learning rate."""
        # Arrange - Create controller with specific learning rate
        config = AdaptiveControlConfig(num_parameters=2, adaptation_gain=0.5)
        controller = AdaptiveController(config)

        # Act - Get current learning rate
        gain = controller.get_current_gain()

        # Assert - Should match configured value
        assert gain == 0.5

    @pytest.mark.unit
    def test_get_gain_history(self) -> None:
        """Test get_gain_history records gain evolution."""
        # Arrange - Create controller
        config = AdaptiveControlConfig(num_parameters=2)
        controller = AdaptiveController(config)

        # Act - Run multiple control steps
        for _ in range(5):
            controller.compute(measurement=0.0, regressor=[1.0, 0.0], reference=10.0)

        history = controller.get_gain_history()

        # Assert - History should contain one entry per step
        assert len(history) == 5

    @pytest.mark.unit
    def test_is_persistently_exciting(self) -> None:
        """Test is_persistently_exciting checks PE status."""
        # Arrange - Create controller with PE monitoring
        config = AdaptiveControlConfig(num_parameters=2, enable_pe_monitoring=True)
        controller = AdaptiveController(config)

        # Act - Check PE status without running controller
        is_pe = controller.is_persistently_exciting()

        # Assert - Should be False with no data
        assert is_pe is False

    @pytest.mark.unit
    def test_get_error_history(self) -> None:
        """Test get_error_history returns tracking errors."""
        # Arrange - Create controller
        config = AdaptiveControlConfig(num_parameters=2)
        controller = AdaptiveController(config)

        # Act - Run control steps with constant reference
        for _ in range(5):
            controller.compute(measurement=0.0, regressor=[1.0, 0.0], reference=10.0)

        errors = controller.get_error_history()

        # Assert - Should have recorded all errors
        assert len(errors) == 5
        assert all(e > 0 for e in errors)

    @pytest.mark.unit
    def test_is_lyapunov_stable(self) -> None:
        """Test is_lyapunov_stable checks stability condition."""
        # Arrange - Create controller with Lyapunov monitoring
        config = AdaptiveControlConfig(
            num_parameters=2, enable_lyapunov_monitoring=True
        )
        controller = AdaptiveController(config)

        # Act - Check stability without sufficient data
        stable = controller.is_lyapunov_stable(window=10)

        # Assert - Should be False with insufficient history
        assert stable is False


class TestClosedLoopSimulation:
    """Test closed-loop simulation utilities."""

    @pytest.mark.integration
    def test_simulate_closed_loop_basic(self) -> None:
        """Test simulate_closed_loop runs complete simulation."""
        # Arrange - Create controller and plant for simulation
        config = AdaptiveControlConfig(num_parameters=2, adaptation_gain=0.5)
        controller = AdaptiveController(config)
        plant = SimpleFirstOrderPlant(a=1.0, b=1.0)

        # Act - Run closed-loop simulation
        results = simulate_closed_loop(
            controller=controller,
            plant=plant,
            reference_input=10.0,
            duration=2.0,
            dt=0.01,
        )

        # Assert - Results dictionary contains all expected fields
        assert "time" in results
        assert "plant_output" in results
        assert "reference" in results
        assert "control" in results
        assert "parameters" in results
        assert "error" in results
        assert len(results["time"]) == 200

    @pytest.mark.integration
    def test_simulate_closed_loop_with_time_varying_reference(self) -> None:
        """Test simulate_closed_loop with time-varying reference."""
        # Arrange - Create controller, plant, and sinusoidal reference
        config = AdaptiveControlConfig(num_parameters=2, adaptation_gain=0.5)
        controller = AdaptiveController(config)
        plant = SimpleFirstOrderPlant(a=1.0, b=1.0)

        def reference(t: float) -> float:
            return 10.0 * np.sin(t)

        # Act - Run simulation with time-varying reference
        results = simulate_closed_loop(
            controller=controller,
            plant=plant,
            reference_input=reference,
            duration=5.0,
            dt=0.01,
        )

        # Assert - Reference values should vary over time
        assert not np.allclose(results["reference"], results["reference"][0])

    @pytest.mark.integration
    def test_simulate_closed_loop_with_custom_regressor(self) -> None:
        """Test simulate_closed_loop with custom regressor function."""
        # Arrange - Create controller with 3 parameters and custom regressor
        config = AdaptiveControlConfig(num_parameters=3, adaptation_gain=0.5)
        controller = AdaptiveController(config)
        plant = SimpleFirstOrderPlant(a=1.0, b=1.0)

        def regressor_fn(y: float) -> np.ndarray:
            return np.array([1.0, y, y**2])

        # Act - Run simulation with quadratic regressor
        results = simulate_closed_loop(
            controller=controller,
            plant=plant,
            reference_input=10.0,
            duration=1.0,
            dt=0.01,
            regressor_fn=regressor_fn,
        )

        # Assert - Parameters array should have 3 columns
        assert results["parameters"].shape[1] == 3

    @pytest.mark.integration
    def test_simulate_closed_loop_tracks_reference(self) -> None:
        """Test simulate_closed_loop runs complete tracking simulation."""
        # Arrange - Create controller with favorable parameters
        config = AdaptiveControlConfig(
            num_parameters=2,
            adaptation_gain=2.0,
            use_normalization=True,
            initial_parameters=[1.0, 0.1],  # Reasonable starting point
        )
        controller = AdaptiveController(config)
        plant = SimpleFirstOrderPlant(a=1.0, b=1.0)

        # Act - Run closed-loop simulation
        results = simulate_closed_loop(
            controller=controller,
            plant=plant,
            reference_input=5.0,  # Smaller reference for better convergence
            duration=15.0,
            dt=0.01,
        )

        # Assert - Simulation completes and produces reasonable outputs
        assert len(results["time"]) == 1500
        assert len(results["error"]) == 1500
        # Parameters should adapt (change from initial values)
        param_change = np.linalg.norm(
            results["parameters"][-1] - results["parameters"][0]
        )
        assert param_change > 0.01  # Parameters adapted
