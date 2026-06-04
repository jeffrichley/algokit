"""Control Systems algorithms package.

This package contains implementations of various control algorithms for
regulating and controlling dynamic systems. These algorithms are widely
used in robotics, aerospace, industrial automation, and process control.

Modules:
    pid: Proportional-Integral-Derivative (PID) control
    adaptive: Adaptive control with parameter estimation
    lqr: Linear Quadratic Regulator (LQR) optimal control
    robust: Robust H-infinity control
    sliding_mode: Sliding mode control with chattering reduction
"""

from algokit.algorithms.control.adaptive import (
    AdaptiveControlConfig,
    AdaptiveController,
    FirstOrderReferenceModel,
    LinearPlant,
    PersistenceOfExcitationMonitor,
    SecondOrderReferenceModel,
    SimpleFirstOrderPlant,
    simulate_closed_loop,
)
from algokit.algorithms.control.lqr import LQRConfig, LQRController
from algokit.algorithms.control.pid import PIDConfig, PIDController
from algokit.algorithms.control.robust import (
    RobustControlConfig,
    RobustController,
    SystemType,
)
from algokit.algorithms.control.sliding_mode import (
    SlidingModeConfig,
    SlidingModeController,
)

__all__ = [
    "PIDController",
    "PIDConfig",
    "AdaptiveController",
    "AdaptiveControlConfig",
    "FirstOrderReferenceModel",
    "SecondOrderReferenceModel",
    "LinearPlant",
    "SimpleFirstOrderPlant",
    "PersistenceOfExcitationMonitor",
    "simulate_closed_loop",
    "LQRController",
    "LQRConfig",
    "RobustController",
    "RobustControlConfig",
    "SystemType",
    "SlidingModeController",
    "SlidingModeConfig",
]
