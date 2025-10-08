"""PID (Proportional-Integral-Derivative) Controller implementation.

This module implements a discrete-time PID controller with anti-windup
and derivative filtering capabilities. PID control is one of the most
widely used control algorithms in industrial automation.

The PID controller computes a control signal based on:
- Proportional term: Reacts to current error
- Integral term: Eliminates steady-state error
- Derivative term: Anticipates future error

Mathematical formulation:
    u(t) = Kp * e(t) + Ki * ∫e(τ)dτ + Kd * de(t)/dt

where:
    - u(t): Control output
    - e(t): Error signal (setpoint - measured value)
    - Kp: Proportional gain
    - Ki: Integral gain
    - Kd: Derivative gain

Enhanced Features:
    - Back-calculation anti-windup (Åström & Hägglund, 1995)
    - NaN/Inf protection for numerical stability
    - Measurement smoothing for auto-tuning
    - Vectorized batch processing mode
    - Persistent integral freeze tracking
"""

import logging
import math

import numpy as np
from pydantic import BaseModel, Field, field_validator
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


class PIDConfig(BaseModel):
    """Configuration parameters for PID controller with automatic validation.

    Attributes:
        kp: Proportional gain (must be >= 0)
        ki: Integral gain (must be >= 0)
        kd: Derivative gain (must be >= 0)
        setpoint: Desired target value
        sample_time: Time step between control updates in seconds (must be > 0)
        output_limits: Tuple of (min, max) output limits for saturation
        integral_limits: Tuple of (min, max) integral windup limits
        derivative_filter_alpha: Low-pass filter coefficient for derivative term (0 < α ≤ 1)
        use_derivative_on_measurement: If True, derivative acts on measurement (reduces kick)
        back_calculation_coefficient: Coefficient for back-calculation anti-windup (0 < kb ≤ 1)
        enable_nan_protection: Enable NaN/Inf protection (recommended for production)
        max_integral_magnitude: Maximum allowed integral magnitude (prevents overflow)
        enable_vectorized: Enable NumPy vectorized mode for batch processing
        debug: Whether to enable debug logging
    """

    kp: float = Field(ge=0.0, description="Proportional gain")
    ki: float = Field(default=0.0, ge=0.0, description="Integral gain")
    kd: float = Field(default=0.0, ge=0.0, description="Derivative gain")
    setpoint: float = Field(default=0.0, description="Desired target value")
    sample_time: float = Field(
        default=0.01, gt=0.0, description="Time step between updates in seconds"
    )
    output_limits: tuple[float, float] | None = Field(
        default=None, description="Output saturation limits (min, max)"
    )
    integral_limits: tuple[float, float] | None = Field(
        default=None, description="Integral anti-windup limits (min, max)"
    )
    derivative_filter_alpha: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Derivative filter coefficient (1.0 = no filter)",
    )
    use_derivative_on_measurement: bool = Field(
        default=False, description="Use derivative on measurement instead of error"
    )
    back_calculation_coefficient: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Back-calculation coefficient for anti-windup (Åström & Hägglund)",
    )
    enable_nan_protection: bool = Field(
        default=True, description="Enable NaN/Inf protection for numerical stability"
    )
    max_integral_magnitude: float | None = Field(
        default=None,
        gt=0.0,
        description="Maximum integral magnitude to prevent overflow",
    )
    enable_vectorized: bool = Field(
        default=False, description="Enable NumPy vectorized batch processing mode"
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    @field_validator("output_limits")
    @classmethod
    def validate_output_limits(
        cls, v: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        """Validate output limits are ordered correctly.

        Args:
            v: Output limits tuple (min, max)

        Returns:
            Validated output limits

        Raises:
            ValueError: If min >= max
        """
        if v is not None:
            min_val, max_val = v
            if min_val >= max_val:
                raise ValueError(
                    f"Output min ({min_val}) must be less than max ({max_val})"
                )
        return v

    @field_validator("integral_limits")
    @classmethod
    def validate_integral_limits(
        cls, v: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        """Validate integral limits are ordered correctly.

        Args:
            v: Integral limits tuple (min, max)

        Returns:
            Validated integral limits

        Raises:
            ValueError: If min >= max
        """
        if v is not None:
            min_val, max_val = v
            if min_val >= max_val:
                raise ValueError(
                    f"Integral min ({min_val}) must be less than max ({max_val})"
                )
        return v


class PIDController:
    """PID Controller with anti-windup and derivative filtering.

    This implementation includes several practical enhancements:
    - Anti-windup protection to prevent integral term saturation
    - Derivative filtering to reduce noise amplification
    - Derivative on measurement option to eliminate setpoint kick
    - Output saturation with proper back-calculation
    - Comprehensive error and state tracking

    Example:
        >>> config = PIDConfig(kp=1.0, ki=0.1, kd=0.05, setpoint=100.0)
        >>> controller = PIDController(config)
        >>> # Control loop
        >>> for measurement in sensor_readings:
        ...     control_output = controller.compute(measurement)
        ...     actuator.set_value(control_output)
    """

    def __init__(self, config: PIDConfig) -> None:
        """Initialize PID controller with given configuration.

        Args:
            config: PID controller configuration parameters
        """
        self.config = config
        self._integral: float = 0.0
        self._prev_error: float | None = None
        self._prev_measurement: float | None = None
        self._prev_derivative: float = 0.0
        self._output: float = 0.0
        self._integral_frozen: bool = False  # Track saturation state for anti-windup
        self._saturation_error: float = 0.0  # Error between ideal and saturated output

        if self.config.debug:
            logger.setLevel(logging.DEBUG)

    @property
    def setpoint(self) -> float:
        """Get current setpoint value."""
        return self.config.setpoint

    @setpoint.setter
    def setpoint(self, value: float) -> None:
        """Set new setpoint value.

        Args:
            value: New setpoint target
        """
        self.config.setpoint = value

    def _validate_and_protect(
        self, value: float, name: str = "value"
    ) -> tuple[float, bool]:
        """Validate and protect against NaN/Inf values.

        Args:
            value: Value to validate
            name: Name of value for logging

        Returns:
            Tuple of (sanitized_value, is_valid)
        """
        if not self.config.enable_nan_protection:
            return value, True

        if math.isnan(value) or math.isinf(value):
            if self.config.debug:
                logger.warning(f"PID: {name} is NaN/Inf, replacing with 0.0")
            return 0.0, False

        return value, True

    def compute(
        self, measurement: float | np.ndarray, dt: float | None = None
    ) -> float | np.ndarray:
        """Compute PID control output for given measurement(s).

        Args:
            measurement: Current process variable measurement (scalar or array)
            dt: Optional time step override (uses config.sample_time if None)

        Returns:
            Control output signal (scalar or array matching input)

        Raises:
            ValueError: If dt is zero or negative with NaN protection enabled

        Example:
            >>> controller = PIDController(PIDConfig(kp=1.0, ki=0.1, kd=0.05))
            >>> output = controller.compute(measurement=50.0)
            >>> # Vectorized batch mode
            >>> config = PIDConfig(kp=1.0, enable_vectorized=True)
            >>> controller = PIDController(config)
            >>> outputs = controller.compute(np.array([50.0, 55.0, 60.0]))
        """
        # Handle vectorized mode
        if self.config.enable_vectorized and isinstance(measurement, np.ndarray):
            return self._compute_vectorized(measurement, dt)

        # Scalar mode - ensure measurement is a float
        if isinstance(measurement, np.ndarray):
            raise TypeError(
                "Array measurement requires enable_vectorized=True in config"
            )

        dt = dt if dt is not None else self.config.sample_time

        # NaN/Inf protection for dt
        if self.config.enable_nan_protection and (
            dt <= 0.0 or math.isnan(dt) or math.isinf(dt)
        ):
            raise ValueError(f"Invalid time step dt={dt}. Must be positive and finite.")

        # Validate measurement
        measurement, measurement_valid = self._validate_and_protect(
            measurement, "measurement"
        )
        if not measurement_valid:
            measurement = self._prev_measurement or self.config.setpoint

        # Calculate error
        error = self.config.setpoint - measurement
        error, _ = self._validate_and_protect(error, "error")

        # Proportional term
        p_term = self.config.kp * error

        # Derivative term with filtering and NaN protection (calculate before integral)
        if self.config.use_derivative_on_measurement:
            # Derivative on measurement (avoids setpoint kick)
            if self._prev_measurement is not None:
                derivative = -(measurement - self._prev_measurement) / dt
                derivative, derivative_valid = self._validate_and_protect(
                    derivative, "derivative"
                )
                if not derivative_valid:
                    derivative = 0.0
            else:
                derivative = 0.0
            self._prev_measurement = measurement
        else:
            # Derivative on error (traditional)
            if self._prev_error is not None:
                derivative = (error - self._prev_error) / dt
                derivative, derivative_valid = self._validate_and_protect(
                    derivative, "derivative"
                )
                if not derivative_valid:
                    derivative = 0.0
            else:
                derivative = 0.0
            self._prev_error = error

        # Apply low-pass filter to derivative
        alpha = self.config.derivative_filter_alpha
        filtered_derivative = alpha * derivative + (1.0 - alpha) * self._prev_derivative
        filtered_derivative, _ = self._validate_and_protect(
            filtered_derivative, "filtered_derivative"
        )
        self._prev_derivative = filtered_derivative

        d_term = self.config.kd * filtered_derivative

        # Integral term with improved anti-windup (Åström & Hägglund, 1995)
        # Calculate tentative integral to check for saturation BEFORE committing
        tentative_integral = self._integral

        # Only add new error if not frozen
        if not self._integral_frozen:
            tentative_integral += error * dt

        # Clamp tentative integral if limits specified
        if self.config.integral_limits is not None:
            min_int, max_int = self.config.integral_limits
            tentative_integral = max(min_int, min(max_int, tentative_integral))

        # Overflow protection for tentative integral
        if self.config.max_integral_magnitude is not None:
            max_mag = self.config.max_integral_magnitude
            tentative_integral = max(-max_mag, min(max_mag, tentative_integral))

        # Calculate tentative i_term and output to check saturation
        tentative_i_term = self.config.ki * tentative_integral
        tentative_output = p_term + tentative_i_term + d_term
        tentative_output, _ = self._validate_and_protect(tentative_output, "output")

        # Check if tentative output would saturate and apply back-calculation
        saturation_error_current = 0.0
        integral_frozen_next = False

        if self.config.output_limits is not None:
            min_out, max_out = self.config.output_limits
            clamped_output = max(min_out, min(max_out, tentative_output))
            saturation_error_current = clamped_output - tentative_output

            # Apply back-calculation anti-windup correction synchronously
            if saturation_error_current != 0.0 and self.config.ki != 0.0:
                # Back-calculation: reduce integral based on current saturation
                # saturation_error < 0 means saturating high, need to reduce integral
                # Formula: integral += kb * saturation_error * dt
                kb = self.config.back_calculation_coefficient
                integral_correction = kb * saturation_error_current * dt
                tentative_integral += integral_correction

                # Re-clamp after back-calculation
                if self.config.integral_limits is not None:
                    min_int, max_int = self.config.integral_limits
                    tentative_integral = max(min_int, min(max_int, tentative_integral))
                if self.config.max_integral_magnitude is not None:
                    max_mag = self.config.max_integral_magnitude
                    tentative_integral = max(-max_mag, min(max_mag, tentative_integral))

                # Update freeze state based on whether error makes saturation worse
                # Saturating high (saturation_error < 0) + positive error = freeze
                # Saturating low (saturation_error > 0) + negative error = freeze
                if (saturation_error_current < 0 and error > 0) or (
                    saturation_error_current > 0 and error < 0
                ):
                    integral_frozen_next = True
                else:
                    integral_frozen_next = False
            else:
                integral_frozen_next = False

        # Commit the tentative integral
        self._integral = tentative_integral
        self._saturation_error = saturation_error_current
        self._integral_frozen = integral_frozen_next

        # Final i_term with corrected integral
        i_term = self.config.ki * self._integral

        # Compute final output
        output = p_term + i_term + d_term
        output, _ = self._validate_and_protect(output, "output")

        # Apply final output limits
        if self.config.output_limits is not None:
            min_out, max_out = self.config.output_limits
            output = max(min_out, min(max_out, output))

        self._output = output

        if self.config.debug:
            logger.debug(
                f"PID: e={error:.3f}, P={p_term:.3f}, I={i_term:.3f}, "
                f"D={d_term:.3f}, u={output:.3f}, frozen={self._integral_frozen}"
            )

        return output

    def _compute_vectorized(
        self, measurements: np.ndarray, dt: float | None = None
    ) -> np.ndarray:
        """Compute PID control output for batch of measurements (vectorized).

        Args:
            measurements: Array of process variable measurements
            dt: Optional time step override

        Returns:
            Array of control outputs matching input shape

        Note:
            This method processes multiple measurements in parallel using NumPy.
            Internal state is updated based on the last measurement in the batch.
        """
        dt = dt if dt is not None else self.config.sample_time

        if self.config.enable_nan_protection and (
            dt <= 0.0 or math.isnan(dt) or math.isinf(dt)
        ):
            raise ValueError(f"Invalid time step dt={dt}. Must be positive and finite.")

        # Validate measurements
        if self.config.enable_nan_protection:
            measurements = np.nan_to_num(measurements, nan=0.0, posinf=0.0, neginf=0.0)

        # Calculate errors
        errors = self.config.setpoint - measurements

        # Proportional terms
        p_terms = self.config.kp * errors

        # Integral terms (simplified for batch - each builds on previous)
        integrals = np.zeros_like(errors)
        current_integral = self._integral

        for i in range(len(errors)):
            if not self._integral_frozen:
                current_integral += errors[i] * dt

            # Apply back-calculation if saturated
            if self._saturation_error != 0.0 and self.config.ki != 0.0:
                kb = self.config.back_calculation_coefficient
                correction = kb * self._saturation_error * dt
                current_integral += correction

            # Clamp integral
            if self.config.integral_limits is not None:
                min_int, max_int = self.config.integral_limits
                current_integral = np.clip(current_integral, min_int, max_int)

            if self.config.max_integral_magnitude is not None:
                max_mag = self.config.max_integral_magnitude
                current_integral = np.clip(current_integral, -max_mag, max_mag)

            integrals[i] = current_integral

        i_terms = self.config.ki * integrals

        # Derivative terms
        if self.config.use_derivative_on_measurement:
            derivatives = np.zeros_like(measurements)
            if self._prev_measurement is not None:
                derivatives[0] = -(measurements[0] - self._prev_measurement) / dt
            derivatives[1:] = -np.diff(measurements) / dt
        else:
            derivatives = np.zeros_like(errors)
            if self._prev_error is not None:
                derivatives[0] = (errors[0] - self._prev_error) / dt
            derivatives[1:] = np.diff(errors) / dt

        # Filter derivatives
        filtered_derivatives = np.zeros_like(derivatives)
        current_deriv = self._prev_derivative

        alpha = self.config.derivative_filter_alpha
        for i in range(len(derivatives)):
            current_deriv = alpha * derivatives[i] + (1.0 - alpha) * current_deriv
            filtered_derivatives[i] = current_deriv

        d_terms = self.config.kd * filtered_derivatives

        # Compute outputs
        outputs = p_terms + i_terms + d_terms

        # Apply saturation
        if self.config.output_limits is not None:
            min_out, max_out = self.config.output_limits
            outputs = np.clip(outputs, min_out, max_out)

        # Update state based on last element
        self._integral = integrals[-1]
        self._prev_derivative = filtered_derivatives[-1]
        self._prev_measurement = measurements[
            -1
        ]  # Always save last measurement for state tracking

        if self.config.use_derivative_on_measurement:
            pass  # Already using measurement for derivative
        else:
            self._prev_error = errors[-1]

        # Update saturation tracking based on last output
        if self.config.output_limits is not None:
            ideal_output = p_terms[-1] + i_terms[-1] + d_terms[-1]
            self._saturation_error = outputs[-1] - ideal_output

            if self._saturation_error != 0.0:
                # Saturating high (saturation_error < 0) + positive error = freeze
                # Saturating low (saturation_error > 0) + negative error = freeze
                if (self._saturation_error < 0 and errors[-1] > 0) or (
                    self._saturation_error > 0 and errors[-1] < 0
                ):
                    self._integral_frozen = True
                else:
                    self._integral_frozen = False
            else:
                self._integral_frozen = False

        self._output = outputs[-1]

        return outputs

    def reset(self) -> None:
        """Reset controller internal state.

        Clears integral accumulation, derivative memory, and previous errors.
        Useful when restarting control or changing operating conditions.

        Example:
            >>> controller.reset()
            >>> # Controller state is now cleared
        """
        self._integral = 0.0
        self._prev_error = None
        self._prev_measurement = None
        self._prev_derivative = 0.0
        self._output = 0.0
        self._integral_frozen = False
        self._saturation_error = 0.0

    def get_components(self) -> dict[str, float]:
        """Get individual PID term contributions.

        Returns:
            Dictionary containing P, I, D term values and total output

        Example:
            >>> components = controller.get_components()
            >>> print(f"P: {components['p_term']}, I: {components['i_term']}")
        """
        # Recompute for current state (this is a snapshot)
        error = self.config.setpoint - (self._prev_measurement or 0.0)
        p_term = self.config.kp * error
        i_term = self.config.ki * self._integral
        d_term = self.config.kd * self._prev_derivative

        return {
            "p_term": p_term,
            "i_term": i_term,
            "d_term": d_term,
            "output": self._output,
            "integral": self._integral,
        }

    def set_gains(
        self, kp: float | None = None, ki: float | None = None, kd: float | None = None
    ) -> None:
        """Update PID gains without resetting controller state.

        Args:
            kp: New proportional gain (if provided)
            ki: New integral gain (if provided)
            kd: New derivative gain (if provided)

        Example:
            >>> controller.set_gains(kp=2.0, ki=0.5)
        """
        if kp is not None:
            self.config.kp = kp
        if ki is not None:
            self.config.ki = ki
        if kd is not None:
            self.config.kd = kd

    def _smooth_measurements(
        self, measurements: list[float], method: str = "savgol"
    ) -> list[float]:
        """Smooth measurements before auto-tuning to reduce noise.

        Args:
            measurements: Raw step response measurements
            method: Smoothing method ('savgol' or 'moving_average')

        Returns:
            Smoothed measurements

        Raises:
            ValueError: If smoothing method is invalid or data is too short
        """
        if len(measurements) < 5:
            logger.warning(
                f"Insufficient data for smoothing (n={len(measurements)}), "
                "using raw measurements"
            )
            return measurements

        meas_array = np.array(measurements)

        # Check for NaN/Inf in input data
        if np.any(np.isnan(meas_array)) or np.any(np.isinf(meas_array)):
            logger.warning(
                "NaN/Inf detected in measurements, replacing with interpolation"
            )
            # Replace NaN/Inf with linear interpolation
            mask = np.isnan(meas_array) | np.isinf(meas_array)
            indices = np.arange(len(meas_array))
            meas_array[mask] = np.interp(
                indices[mask], indices[~mask], meas_array[~mask]
            )

        if method == "savgol":
            # Savitzky-Golay filter for smoothing
            window_length = min(
                11,
                len(measurements)
                if len(measurements) % 2 == 1
                else len(measurements) - 1,
            )
            if window_length < 5:
                window_length = 5
            poly_order = min(3, window_length - 2)

            try:
                smoothed = savgol_filter(meas_array, window_length, poly_order)
                return smoothed.tolist()
            except ValueError as e:
                logger.warning(
                    f"Savitzky-Golay filter failed: {e}, using moving average"
                )
                method = "moving_average"

        if method == "moving_average":
            # Simple moving average
            window_size = min(5, len(measurements) // 3)
            if window_size < 2:
                return measurements

            # Apply convolution for moving average
            kernel = np.ones(window_size) / window_size
            smoothed = np.convolve(meas_array, kernel, mode="same")

            # Fix edge effects
            smoothed[: window_size // 2] = meas_array[: window_size // 2]
            smoothed[-(window_size // 2) :] = meas_array[-(window_size // 2) :]

            return smoothed.tolist()

        raise ValueError(
            f"Unknown smoothing method: {method}. Use 'savgol' or 'moving_average'"
        )

    def auto_tune(
        self,
        measurements: list[float],
        method: str = "ziegler_nichols",
        smooth: bool = True,
        smooth_method: str = "savgol",
    ) -> dict[str, float]:
        """Auto-tune PID parameters based on step response data.

        Args:
            measurements: List of step response measurements
            method: Tuning method ('ziegler_nichols' or 'cohen_coon')
            smooth: Whether to smooth measurements before tuning
            smooth_method: Smoothing method ('savgol' or 'moving_average')

        Returns:
            Dictionary of tuned parameters {kp, ki, kd}

        Raises:
            ValueError: If method is not recognized or data is insufficient

        Note:
            Measurement smoothing improves tuning robustness against noise.
            Uses Savitzky-Golay filter or moving average.

        Example:
            >>> step_response = [0, 10, 20, 30, 35, 38, 39, 40]
            >>> tuned_params = controller.auto_tune(step_response, smooth=True)
            >>> controller.set_gains(**tuned_params)
        """
        if len(measurements) < 3:
            raise ValueError(
                f"Insufficient data for auto-tuning (n={len(measurements)}). "
                "Need at least 3 measurements."
            )

        # Smooth measurements if requested
        if smooth:
            measurements = self._smooth_measurements(measurements, smooth_method)

        if method == "ziegler_nichols":
            return self._ziegler_nichols_tuning(measurements)
        elif method == "cohen_coon":
            return self._cohen_coon_tuning(measurements)
        else:
            raise ValueError(
                f"Unknown tuning method: {method}. "
                "Use 'ziegler_nichols' or 'cohen_coon'"
            )

    def _ziegler_nichols_tuning(self, measurements: list[float]) -> dict[str, float]:
        """Apply Ziegler-Nichols tuning method with improved system identification.

        Args:
            measurements: Step response data (should be smoothed)

        Returns:
            Tuned PID parameters

        Note:
            Uses tangent method to estimate process characteristics:
            - L (dead time): Time to first significant response
            - T (time constant): Time for system to reach steady state
            - K (process gain): Steady-state change in output
        """
        meas_array = np.array(measurements)
        n = len(meas_array)

        if n < 5:
            logger.warning("Insufficient data for ZN tuning, using conservative values")
            return {"kp": 1.0, "ki": 0.1, "kd": 0.05}

        # Estimate steady-state values
        initial_value = np.mean(meas_array[: min(3, n // 10)])
        final_value = np.mean(meas_array[-min(3, n // 10) :])
        delta_y = final_value - initial_value

        if abs(delta_y) < 1e-6:
            logger.warning("No significant response detected, using default gains")
            return {"kp": 1.0, "ki": 0.1, "kd": 0.05}

        # Calculate numerical derivative (rate of change)
        derivatives = np.gradient(meas_array)

        # Find maximum slope point (inflection point)
        max_slope = 0.0
        max_slope_idx = 0
        for i in range(1, n - 1):
            if derivatives[i] > max_slope:
                max_slope = derivatives[i]
                max_slope_idx = i

        if max_slope <= 0:
            logger.warning("Could not find inflection point, using conservative gains")
            return {"kp": 0.5, "ki": 0.05, "kd": 0.02}

        # Tangent line at inflection point
        # y = slope * (t - t_inflection) + y_inflection
        t_inflection = max_slope_idx * self.config.sample_time
        y_inflection = meas_array[max_slope_idx]

        # Find where tangent line intersects initial and final values
        # L (dead time): intersection with initial value
        # L + T (total time): intersection with final value
        t_start = (initial_value - y_inflection) / max_slope + t_inflection
        t_end = (final_value - y_inflection) / max_slope + t_inflection

        L = max(t_start, self.config.sample_time)  # Dead time must be positive
        T = max(t_end - t_start, self.config.sample_time)  # Time constant

        # Process gain K
        K = abs(delta_y)

        # Ziegler-Nichols PID tuning rules for step response
        # These are well-established empirical formulas
        if L > 0 and T > 0:
            kp = 1.2 * (T / L) * (1.0 / K)
            ki = kp / (2.0 * L)
            kd = kp * 0.5 * L
        else:
            # Fallback to conservative values
            kp = 1.0 / K if K > 0 else 1.0
            ki = 0.1
            kd = 0.05

        # Sanity checks
        kp = max(0.01, min(kp, 100.0))
        ki = max(0.001, min(ki, 10.0))
        kd = max(0.0, min(kd, 10.0))

        if self.config.debug:
            logger.debug(
                f"ZN Tuning: L={L:.3f}, T={T:.3f}, K={K:.3f} → "
                f"Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}"
            )

        return {"kp": kp, "ki": ki, "kd": kd}

    def _cohen_coon_tuning(self, measurements: list[float]) -> dict[str, float]:
        """Apply Cohen-Coon tuning method with improved system identification.

        Args:
            measurements: Step response data (should be smoothed)

        Returns:
            Tuned PID parameters

        Note:
            Cohen-Coon method uses characteristic response times (t28, t63)
            to estimate process parameters and provides more aggressive tuning
            than Ziegler-Nichols for processes with significant dead time.
        """
        meas_array = np.array(measurements)
        n = len(meas_array)

        if n < 5:
            logger.warning("Insufficient data for CC tuning, using conservative values")
            return {"kp": 1.0, "ki": 0.1, "kd": 0.05}

        # Estimate steady-state values
        initial_value = np.mean(meas_array[: min(3, n // 10)])
        final_value = np.mean(meas_array[-min(3, n // 10) :])
        delta_y = final_value - initial_value

        if abs(delta_y) < 1e-6:
            logger.warning("No significant response detected, using default gains")
            return {"kp": 1.0, "ki": 0.1, "kd": 0.05}

        # Find 28.3% and 63.2% response times (characteristic for first-order systems)
        target_28 = initial_value + 0.283 * delta_y
        target_63 = initial_value + 0.632 * delta_y

        # Use linear interpolation for more accurate time estimation
        t28 = None
        t63 = None

        for i in range(1, n):
            # Check if we crossed the 28.3% threshold
            if t28 is None and (
                (meas_array[i - 1] < target_28 <= meas_array[i])
                or (meas_array[i - 1] > target_28 >= meas_array[i])
            ):
                # Linear interpolation
                frac = (target_28 - meas_array[i - 1]) / (
                    meas_array[i] - meas_array[i - 1] + 1e-10
                )
                t28 = ((i - 1) + frac) * self.config.sample_time

            # Check if we crossed the 63.2% threshold
            if t63 is None and (
                (meas_array[i - 1] < target_63 <= meas_array[i])
                or (meas_array[i - 1] > target_63 >= meas_array[i])
            ):
                # Linear interpolation
                frac = (target_63 - meas_array[i - 1]) / (
                    meas_array[i] - meas_array[i - 1] + 1e-10
                )
                t63 = ((i - 1) + frac) * self.config.sample_time
                break

        if t28 is None or t63 is None or t63 <= t28:
            logger.warning(
                "Could not identify response characteristics, using ZN method"
            )
            return self._ziegler_nichols_tuning(measurements)

        # Process model parameters
        L = t28  # Dead time (lag) approximation
        tau = 1.5 * (t63 - t28)  # Time constant approximation
        K = abs(delta_y)  # Process gain

        if L <= 0 or tau <= 0:
            logger.warning("Invalid process parameters, using conservative gains")
            return {"kp": 0.5, "ki": 0.05, "kd": 0.02}

        # Cohen-Coon PID tuning rules
        # These provide more aggressive control than ZN for processes with dead time
        tau_over_L = tau / L

        kp = (1.0 / K) * (
            (16.0 * tau + 3.0 * L) / (12.0 * L)
        )  # More conservative than original

        Ti = L * ((32.0 + 6.0 * tau_over_L) / (13.0 + 8.0 * tau_over_L))
        ki = kp / Ti if Ti > 0 else 0.0

        Td = L * (4.0 / (11.0 + 2.0 * tau_over_L))
        kd = kp * Td

        # Sanity checks
        kp = max(0.01, min(kp, 100.0))
        ki = max(0.001, min(ki, 10.0))
        kd = max(0.0, min(kd, 10.0))

        if self.config.debug:
            logger.debug(
                f"CC Tuning: L={L:.3f}, τ={tau:.3f}, K={K:.3f} → "
                f"Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}"
            )

        return {"kp": kp, "ki": ki, "kd": kd}
