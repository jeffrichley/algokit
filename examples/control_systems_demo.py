"""Comprehensive demonstration of all Control Systems algorithms.

This script demonstrates the usage of all control algorithms in the control
systems family:
1. PID Control - Classic feedback controller
2. Adaptive Control - Parameter estimation and adaptation
3. LQR Control - Optimal state feedback
4. Robust H-infinity Control - Disturbance rejection
5. Sliding Mode Control - Robust nonlinear control

Each demo shows typical usage patterns, configuration, and visualization.
"""

import matplotlib.pyplot as plt
import numpy as np

from algokit.algorithms.control import (
    AdaptiveControlConfig,
    AdaptiveController,
    LQRConfig,
    LQRController,
    PIDConfig,
    PIDController,
    RobustControlConfig,
    RobustController,
    SlidingModeConfig,
    SlidingModeController,
)


def demo_pid_control() -> None:
    """Demonstrate PID controller for temperature control."""
    print("\n" + "=" * 80)
    print("🎮 PID CONTROL DEMO - Temperature Control System")
    print("=" * 80)

    # Configure PID controller for temperature control
    config = PIDConfig(
        kp=2.0,  # Proportional gain
        ki=0.5,  # Integral gain
        kd=0.1,  # Derivative gain
        setpoint=75.0,  # Target temperature (°F)
        sample_time=0.1,  # 100ms control loop
        output_limits=(-100.0, 100.0),  # Heater power limits
        use_derivative_on_measurement=True,  # Avoid setpoint kick
    )

    controller = PIDController(config)

    # Simulate temperature control
    temperature = 20.0  # Start at room temperature
    time_steps = 200
    dt = 0.1

    temperatures = []
    control_outputs = []
    times = []

    print(f"Initial temperature: {temperature:.1f}°F")
    print(f"Target temperature: {config.setpoint:.1f}°F")
    print(f"\nSimulating {time_steps * dt:.1f} seconds...")

    for i in range(time_steps):
        # Compute PID control output
        control = controller.compute(temperature, dt=dt)

        # Simple thermal model: dT/dt = -0.1*(T - T_ambient) + 0.01*u
        ambient_temp = 20.0
        dT_dt = -0.1 * (temperature - ambient_temp) + 0.01 * control

        # Update temperature
        temperature += dT_dt * dt

        # Record data
        temperatures.append(temperature)
        control_outputs.append(control)
        times.append(i * dt)

    print(f"Final temperature: {temperature:.1f}°F")
    print(f"Settling time: {times[np.argmax(np.array(temperatures) > 74.5)]:.1f}s")

    # Get PID component breakdown
    components = controller.get_components()
    print(f"\nPID Components at end:")
    print(f"  P-term: {components['p_term']:.2f}")
    print(f"  I-term: {components['i_term']:.2f}")
    print(f"  D-term: {components['d_term']:.2f}")
    print(f"  Total output: {components['output']:.2f}")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(times, temperatures, "b-", label="Temperature")
    ax1.axhline(
        config.setpoint, color="r", linestyle="--", alpha=0.7, label="Setpoint"
    )
    ax1.fill_between(
        times, 74.5, 75.5, alpha=0.2, color="g", label="Acceptable range"
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Temperature (°F)")
    ax1.set_title("PID Temperature Control")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(times, control_outputs, "g-", label="Control output")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Heater power (%)")
    ax2.set_title("PID Control Signal")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pid_control_demo.png", dpi=150)
    print("\n✅ Plot saved as 'pid_control_demo.png'")


def demo_adaptive_control() -> None:
    """Demonstrate adaptive control with unknown system parameters."""
    print("\n" + "=" * 80)
    print("🔧 ADAPTIVE CONTROL DEMO - Parameter Estimation")
    print("=" * 80)

    # Configure adaptive controller
    config = AdaptiveControlConfig(
        num_parameters=2,
        adaptation_gain=0.5,
        initial_parameters=[0.0, 0.0],  # Unknown initial parameters
        parameter_bounds=(-10.0, 10.0),
        use_normalization=True,
    )

    controller = AdaptiveController(config)

    # True (unknown) system parameters
    true_params = np.array([2.0, -0.5])
    print(f"True system parameters: {true_params}")
    print(f"Initial parameter estimate: {controller.parameters}")

    # Simulate adaptive control
    x = 1.0
    time_steps = 500
    dt = 0.01

    states = []
    param_estimates = []
    times = []

    for i in range(time_steps):
        # Regressor (feature vector)
        regressor = np.array([1.0, x])

        # True system output
        true_output = np.dot(true_params, regressor)

        # Adaptive control with reference
        reference = 5.0 * np.sin(i * dt * 2)  # Varying reference
        control = controller.compute(
            measurement=x, regressor=regressor, reference=reference, dt=dt
        )

        # Update state (using true system)
        x = true_output + 0.1 * np.random.randn()  # Add small noise

        # Record data
        states.append(x)
        param_estimates.append(controller.parameters.copy())
        times.append(i * dt)

    final_params = controller.parameters
    print(f"Final parameter estimate: {final_params}")
    print(f"Parameter error: {np.linalg.norm(final_params - true_params):.4f}")

    # Plot parameter convergence
    param_estimates = np.array(param_estimates)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(times, param_estimates[:, 0], "b-", label="θ₁ estimate")
    ax1.axhline(true_params[0], color="b", linestyle="--", alpha=0.7, label="θ₁ true")
    ax1.plot(times, param_estimates[:, 1], "r-", label="θ₂ estimate")
    ax1.axhline(true_params[1], color="r", linestyle="--", alpha=0.7, label="θ₂ true")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Parameter value")
    ax1.set_title("Adaptive Parameter Estimation")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(times, states, "g-", label="System state")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("State value")
    ax2.set_title("System Response")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("adaptive_control_demo.png", dpi=150)
    print("\n✅ Plot saved as 'adaptive_control_demo.png'")


def demo_lqr_control() -> None:
    """Demonstrate LQR optimal control for inverted pendulum."""
    print("\n" + "=" * 80)
    print("⚖️  LQR CONTROL DEMO - Inverted Pendulum Stabilization")
    print("=" * 80)

    # Linearized inverted pendulum model
    # State: [angle, angular_velocity]
    # dx/dt = Ax + Bu
    A = [[0, 1], [9.8, -0.1]]  # Unstable system (positive eigenvalue)
    B = [[0], [1.0]]

    # LQR cost matrices
    Q = [[100, 0], [0, 1]]  # Penalize angle heavily
    R = [[0.1]]  # Small control cost (allow aggressive control)

    config = LQRConfig(
        state_dim=2,
        control_dim=1,
        A=A,
        B=B,
        Q=Q,
        R=R,
        control_limits=(-20.0, 20.0),
    )

    controller = LQRController(config)

    print("System properties:")
    print(f"  Controllable: {controller.is_controllable()}")
    eigenvalues = controller.get_closed_loop_eigenvalues()
    print(f"  Closed-loop eigenvalues: {eigenvalues}")
    print(f"  All stable: {np.all(np.real(eigenvalues) < 0)}")
    print(f"  Feedback gain K: {controller.gain_matrix}")

    # Simulate stabilization
    initial_state = [0.3, 0.0]  # 0.3 rad ≈ 17° initial angle
    print(f"\nInitial angle: {initial_state[0] * 180 / np.pi:.1f}°")

    states, controls = controller.simulate(
        initial_state=initial_state, time_steps=500, dt=0.01
    )

    times = np.arange(len(states)) * 0.01

    print(f"Final angle: {states[-1, 0] * 180 / np.pi:.3f}°")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(times, states[:, 0] * 180 / np.pi, "b-", label="Angle (°)")
    ax1.plot(times, states[:, 1] * 180 / np.pi, "r-", label="Angular velocity (°/s)")
    ax1.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("State")
    ax1.set_title("LQR Inverted Pendulum Stabilization")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(times[:-1], controls, "g-", label="Control torque")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Control (Nm)")
    ax2.set_title("LQR Control Signal")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lqr_control_demo.png", dpi=150)
    print("\n✅ Plot saved as 'lqr_control_demo.png'")


def demo_robust_control() -> None:
    """Demonstrate robust H-infinity control with disturbances."""
    print("\n" + "=" * 80)
    print("🛡️  ROBUST H∞ CONTROL DEMO - Disturbance Rejection")
    print("=" * 80)

    # System with disturbance input
    A = [[0, 1], [-1, -0.5]]
    B1 = [[0], [1]]  # Disturbance input
    B2 = [[0], [1]]  # Control input
    C1 = [[1, 0], [0, 1]]  # Performance output
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
        gamma=0.8,  # Strong disturbance rejection
        control_limits=(-50.0, 50.0),
    )

    controller = RobustController(config)

    print(f"Performance level γ: {config.gamma}")
    print(f"Disturbance attenuation: {controller.estimate_disturbance_attenuation():.2f}x")
    print(f"Closed-loop stable: {np.all(np.real(controller.get_closed_loop_eigenvalues()) < 0)}")

    # Create disturbance sequence (sinusoidal + noise)
    time_steps = 500
    dt = 0.01
    disturbances = []

    for i in range(time_steps):
        t = i * dt
        # Combination of sinusoidal and random disturbances
        dist = 2.0 * np.sin(2 * np.pi * t) + 0.5 * np.random.randn()
        disturbances.append([dist])

    # Simulate with disturbances
    states, controls = controller.simulate_with_disturbance(
        initial_state=[1.0, 0.0], disturbance_sequence=disturbances, dt=dt
    )

    times = np.arange(len(states)) * dt
    disturbances_array = np.array(disturbances)

    print(f"\nMax state deviation: {np.max(np.linalg.norm(states, axis=1)):.3f}")

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    ax1.plot(times[:-1], disturbances_array[:, 0], "r-", alpha=0.7, label="Disturbance")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Disturbance")
    ax1.set_title("External Disturbance")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(times, states[:, 0], "b-", label="State x₁")
    ax2.plot(times, states[:, 1], "g-", label="State x₂")
    ax2.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("State")
    ax2.set_title("H∞ Robust Control - State Response")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(times[:-1], controls, "m-", label="Control")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Control")
    ax3.set_title("Robust Control Signal")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("robust_control_demo.png", dpi=150)
    print("\n✅ Plot saved as 'robust_control_demo.png'")


def demo_sliding_mode_control() -> None:
    """Demonstrate sliding mode control for nonlinear system."""
    print("\n" + "=" * 80)
    print("🎯 SLIDING MODE CONTROL DEMO - Robust Nonlinear Control")
    print("=" * 80)

    # Configure sliding mode controller
    config = SlidingModeConfig(
        state_dim=2,
        control_dim=1,
        sliding_surface_coeffs=[1.0, 2.0],  # s = x₁ + 2x₂
        switching_gain=10.0,
        boundary_layer_width=0.2,
        reaching_law="exponential",
        use_saturation=True,
        control_limits=(-50.0, 50.0),
    )

    controller = SlidingModeController(config)

    print(f"Sliding surface: s = {config.sliding_surface_coeffs[0]}·x₁ + {config.sliding_surface_coeffs[1]}·x₂")
    print(f"Reaching law: {config.reaching_law}")
    print(f"Boundary layer: φ = {config.boundary_layer_width}")

    # Simulate double integrator with disturbance
    x = np.array([5.0, -2.0])  # Initial state far from surface
    time_steps = 300
    dt = 0.01

    states = []
    controls = []
    sliding_surface = []
    times = []

    print(f"\nInitial state: {x}")
    print(f"Initial sliding surface value: {controller.compute_sliding_surface(x):.3f}")
    print(f"Estimated reaching time: {controller.get_reaching_time_estimate(controller.compute_sliding_surface(x)):.2f}s")

    for i in range(time_steps):
        # State derivative (double integrator)
        x_dot = np.array([x[1], 0.0])

        # Add matched uncertainty/disturbance
        disturbance = 3.0 * np.sin(i * dt * 5)  # Time-varying disturbance

        # Compute sliding mode control
        u = controller.compute(
            state=x, state_derivative=x_dot, disturbance_bound=3.0
        )

        # Update dynamics with disturbance
        x_dot[1] = u[0] + disturbance
        x = x + x_dot * dt

        # Record data
        states.append(x.copy())
        controls.append(u[0])
        sliding_surface.append(controller.compute_sliding_surface(x))
        times.append(i * dt)

    states_array = np.array(states)

    print(f"\nFinal state: {states_array[-1]}")
    print(f"Final sliding surface value: {sliding_surface[-1]:.3f}")
    print(f"On sliding surface: {controller.is_on_sliding_surface()}")
    print(f"Chattering magnitude: {controller.estimate_chattering_magnitude():.4f}")

    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # State trajectory
    ax1.plot(states_array[:, 0], states_array[:, 1], "b-", linewidth=2, label="Trajectory")
    ax1.plot(states_array[0, 0], states_array[0, 1], "go", markersize=10, label="Start")
    ax1.plot(states_array[-1, 0], states_array[-1, 1], "ro", markersize=10, label="End")

    # Plot sliding surface
    x1_range = np.linspace(-6, 6, 100)
    x2_surface = -config.sliding_surface_coeffs[0] / config.sliding_surface_coeffs[1] * x1_range
    ax1.plot(x1_range, x2_surface, "r--", alpha=0.7, label="Sliding surface")
    ax1.fill_between(
        x1_range,
        x2_surface - 0.2,
        x2_surface + 0.2,
        alpha=0.2,
        color="r",
        label="Boundary layer",
    )

    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")
    ax1.set_title("State Space Trajectory")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Sliding surface over time
    ax2.plot(times, sliding_surface, "g-", linewidth=2)
    ax2.axhline(0, color="r", linestyle="--", alpha=0.7, label="Target (s=0)")
    ax2.fill_between(
        times, -0.2, 0.2, alpha=0.2, color="g", label="Boundary layer"
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Sliding surface s")
    ax2.set_title("Sliding Surface Evolution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # States over time
    ax3.plot(times, states_array[:, 0], "b-", label="x₁")
    ax3.plot(times, states_array[:, 1], "r-", label="x₂")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("State")
    ax3.set_title("State Evolution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Control signal
    ax4.plot(times, controls, "m-", linewidth=1.5)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Control u")
    ax4.set_title("Sliding Mode Control Signal")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sliding_mode_control_demo.png", dpi=150)
    print("\n✅ Plot saved as 'sliding_mode_control_demo.png'")


def main() -> None:
    """Run all control systems demonstrations."""
    print("\n" + "=" * 80)
    print("🎮 CONTROL SYSTEMS FAMILY - COMPREHENSIVE DEMO")
    print("=" * 80)
    print("\nThis demo showcases all 5 control algorithms:")
    print("  1. PID Control")
    print("  2. Adaptive Control")
    print("  3. LQR Control")
    print("  4. Robust H-infinity Control")
    print("  5. Sliding Mode Control")
    print("\n" + "=" * 80)

    # Run all demos
    demo_pid_control()
    demo_adaptive_control()
    demo_lqr_control()
    demo_robust_control()
    demo_sliding_mode_control()

    print("\n" + "=" * 80)
    print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated plots:")
    print("  - pid_control_demo.png")
    print("  - adaptive_control_demo.png")
    print("  - lqr_control_demo.png")
    print("  - robust_control_demo.png")
    print("  - sliding_mode_control_demo.png")
    print("\n")


if __name__ == "__main__":
    main()

