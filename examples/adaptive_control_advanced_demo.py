"""Advanced Adaptive Control Demo.

This example demonstrates the upgraded adaptive controller with:
- Dynamic reference models with state tracking
- Persistence of excitation (PE) monitoring
- Adaptive learning rate scheduling
- Lyapunov stability verification
- Closed-loop simulation utilities
"""

import matplotlib.pyplot as plt
import numpy as np

from algokit.algorithms.control.adaptive import (
    AdaptiveControlConfig,
    AdaptiveController,
    FirstOrderReferenceModel,
    SecondOrderReferenceModel,
    PersistenceOfExcitationMonitor,
    SimpleFirstOrderPlant,
    simulate_closed_loop,
)


def demo_basic_adaptive_control() -> None:
    """Demonstrate basic adaptive control with all advanced features."""
    print("\n" + "=" * 80)
    print("🎯 Demo 1: Basic Adaptive Control with Advanced Features")
    print("=" * 80)

    # Configure controller with all advanced features enabled
    config = AdaptiveControlConfig(
        num_parameters=2,
        adaptation_gain=0.5,
        initial_parameters=[0.0, 0.0],
        enable_adaptive_gain=True,  # Learning rate adapts based on error
        gain_adaptation_rate=0.05,
        min_adaptation_gain=0.01,
        max_adaptation_gain=2.0,
        enable_pe_monitoring=True,  # Monitor persistence of excitation
        pe_window_size=100,
        enable_lyapunov_monitoring=True,  # Track stability
        use_normalization=True,
        debug=False,
    )

    controller = AdaptiveController(config)
    plant = SimpleFirstOrderPlant(a=1.0, b=1.0, initial_state=0.0)

    print(f"✅ Controller initialized with {config.num_parameters} parameters")
    print(f"   Initial learning rate: {config.adaptation_gain}")
    print(f"   PE monitoring: {config.enable_pe_monitoring}")
    print(f"   Lyapunov monitoring: {config.enable_lyapunov_monitoring}")

    # Run closed-loop simulation
    results = simulate_closed_loop(
        controller=controller,
        plant=plant,
        reference_input=10.0,
        duration=15.0,
        dt=0.01,
    )

    # Display results
    final_error = abs(results["error"][-1])
    print(f"\n📊 Simulation Results:")
    print(f"   Final tracking error: {final_error:.4f}")
    print(f"   Final plant output: {results['plant_output'][-1]:.4f}")
    print(f"   Final parameters: {results['parameters'][-1]}")

    # Check PE status
    is_pe = controller.is_persistently_exciting()
    cond_number = controller.get_pe_condition_number()
    print(f"\n🔍 Persistence of Excitation:")
    print(f"   PE Status: {'✅ Yes' if is_pe else '❌ No'}")
    print(f"   Condition Number: {cond_number:.2f}")

    # Check stability
    is_stable = controller.is_lyapunov_stable(window=50)
    print(f"\n⚡ Lyapunov Stability:")
    print(f"   System is stable: {'✅ Yes' if is_stable else '❌ No'}")

    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Adaptive Control with Advanced Features", fontsize=14, fontweight="bold")

    # Plot 1: Tracking performance
    ax1.plot(results["time"], results["reference"], "r--", label="Reference", linewidth=2)
    ax1.plot(results["time"], results["plant_output"], "b-", label="Output", alpha=0.7)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Signal")
    ax1.set_title("Tracking Performance")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Parameter evolution
    for i in range(config.num_parameters):
        ax2.plot(results["time"], results["parameters"][:, i], label=f"θ_{i+1}")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Parameter Value")
    ax2.set_title("Parameter Convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Adaptive learning rate
    gain_history = controller.get_gain_history()
    ax3.plot(results["time"], gain_history, "g-", linewidth=2)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Learning Rate γ(t)")
    ax3.set_title("Adaptive Learning Rate")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Lyapunov function
    lyap_history = controller.get_lyapunov_history()
    ax4.plot(results["time"], lyap_history, "m-", linewidth=2)
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("V(e)")
    ax4.set_title("Lyapunov Function (Decreasing = Stable)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def demo_reference_model_dynamics() -> None:
    """Demonstrate adaptive control with dynamic reference models."""
    print("\n" + "=" * 80)
    print("🎯 Demo 2: Adaptive Control with Dynamic Reference Models")
    print("=" * 80)

    # First-order reference model
    print("\n📐 Using First-Order Reference Model:")
    print("   ẋ_m = -a_m * x_m + b_m * r")

    ref_model = FirstOrderReferenceModel(a_m=2.0, b_m=2.0, initial_state=0.0)

    config = AdaptiveControlConfig(
        num_parameters=2,
        adaptation_gain=1.0,
        reference_model_dynamics=ref_model,
        enable_lyapunov_monitoring=True,
    )

    controller = AdaptiveController(config)
    plant = SimpleFirstOrderPlant(a=1.5, b=1.5, initial_state=0.0)

    results = simulate_closed_loop(
        controller=controller,
        plant=plant,
        reference_input=10.0,
        duration=10.0,
        dt=0.01,
    )

    print(f"   Final tracking error: {abs(results['error'][-1]):.4f}")

    # Second-order reference model
    print("\n📐 Using Second-Order Reference Model:")
    print("   ẍ_m + 2ζω_n ẋ_m + ω_n² x_m = ω_n² r")

    ref_model_2nd = SecondOrderReferenceModel(
        omega_n=2.0, zeta=0.7, initial_position=0.0, initial_velocity=0.0
    )

    config2 = AdaptiveControlConfig(
        num_parameters=2,
        adaptation_gain=1.0,
        reference_model_dynamics=ref_model_2nd,
        enable_lyapunov_monitoring=True,
    )

    controller2 = AdaptiveController(config2)
    plant2 = SimpleFirstOrderPlant(a=1.0, b=1.0, initial_state=0.0)

    results2 = simulate_closed_loop(
        controller=controller2,
        plant=plant2,
        reference_input=10.0,
        duration=10.0,
        dt=0.01,
    )

    print(f"   Final tracking error: {abs(results2['error'][-1]):.4f}")

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Reference Model Comparison", fontsize=14, fontweight="bold")

    ax1.plot(results["time"], results["reference"], "r--", label="1st-order ref", linewidth=2)
    ax1.plot(results["time"], results["plant_output"], "b-", label="Output", alpha=0.7)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Signal")
    ax1.set_title("First-Order Reference Model")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(results2["time"], results2["reference"], "r--", label="2nd-order ref", linewidth=2)
    ax2.plot(results2["time"], results2["plant_output"], "b-", label="Output", alpha=0.7)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Signal")
    ax2.set_title("Second-Order Reference Model (Critically Damped)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def demo_persistence_of_excitation() -> None:
    """Demonstrate PE monitoring for parameter convergence."""
    print("\n" + "=" * 80)
    print("🎯 Demo 3: Persistence of Excitation (PE) Monitoring")
    print("=" * 80)

    # Create PE monitor
    pe_monitor = PersistenceOfExcitationMonitor(
        window_size=100, condition_threshold=100.0, min_eigenvalue_threshold=1e-6
    )

    print("\n🔬 Testing different input signals:")

    # Test 1: Constant signal (NOT PE)
    print("\n1️⃣  Constant Signal (should NOT be PE):")
    for _ in range(100):
        pe_monitor.update(np.array([1.0, 1.0]))

    is_pe = pe_monitor.is_persistently_exciting()
    cond = pe_monitor.get_condition_number()
    min_eig = pe_monitor.get_min_eigenvalue()
    print(f"   PE: {is_pe}")
    print(f"   Condition number: {cond:.2e}")
    print(f"   Min eigenvalue: {min_eig:.2e}")

    # Test 2: Rich signal (IS PE)
    print("\n2️⃣  Rich Multi-Frequency Signal (should be PE):")
    pe_monitor.reset()
    for i in range(100):
        t = i * 0.1
        regressor = np.array([np.sin(t), np.cos(2 * t) + 0.5 * np.sin(5 * t)])
        pe_monitor.update(regressor)

    is_pe = pe_monitor.is_persistently_exciting()
    cond = pe_monitor.get_condition_number()
    min_eig = pe_monitor.get_min_eigenvalue()
    print(f"   PE: {is_pe}")
    print(f"   Condition number: {cond:.2f}")
    print(f"   Min eigenvalue: {min_eig:.6f}")

    print("\n💡 PE is required for parameter convergence!")
    print("   Without PE, parameters may drift or fail to converge.")


def demo_time_varying_tracking() -> None:
    """Demonstrate tracking of time-varying references."""
    print("\n" + "=" * 80)
    print("🎯 Demo 4: Time-Varying Reference Tracking")
    print("=" * 80)

    # Sinusoidal reference
    def reference_signal(t: float) -> float:
        return 10.0 * np.sin(0.5 * t) + 5.0

    config = AdaptiveControlConfig(
        num_parameters=2,
        adaptation_gain=2.0,
        enable_adaptive_gain=True,
        gain_adaptation_rate=0.1,
        max_adaptation_gain=5.0,
        enable_pe_monitoring=True,
        use_normalization=True,
    )

    controller = AdaptiveController(config)
    plant = SimpleFirstOrderPlant(a=1.0, b=1.0)

    results = simulate_closed_loop(
        controller=controller,
        plant=plant,
        reference_input=reference_signal,
        duration=20.0,
        dt=0.01,
    )

    rms_error = np.sqrt(np.mean(results["error"] ** 2))
    print(f"\n📊 Results:")
    print(f"   RMS tracking error: {rms_error:.4f}")
    print(f"   Max tracking error: {np.max(np.abs(results['error'])):.4f}")

    # Check if signal is PE
    is_pe = controller.is_persistently_exciting()
    print(f"   Regressor is PE: {'✅ Yes' if is_pe else '❌ No'}")

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("Time-Varying Reference Tracking", fontsize=14, fontweight="bold")

    ax1.plot(results["time"], results["reference"], "r--", label="Reference", linewidth=2)
    ax1.plot(results["time"], results["plant_output"], "b-", label="Output", alpha=0.7)
    ax1.fill_between(
        results["time"],
        results["reference"] - 0.5,
        results["reference"] + 0.5,
        alpha=0.2,
        color="red",
        label="±0.5 tolerance",
    )
    ax1.set_ylabel("Signal")
    ax1.set_title("Tracking Performance")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(results["time"], results["error"], "g-", linewidth=1.5)
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Tracking Error")
    ax2.set_title("Error Evolution")
    ax2.grid(True, alpha=0.3)

    ax3.plot(results["time"], controller.get_gain_history(), "m-", linewidth=2)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Learning Rate γ(t)")
    ax3.set_title("Adaptive Learning Rate (increases with error)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def demo_plant_with_disturbance() -> None:
    """Demonstrate adaptive control with plant disturbances."""
    print("\n" + "=" * 80)
    print("🎯 Demo 5: Adaptive Control with Plant Disturbance")
    print("=" * 80)

    # Plant with sinusoidal disturbance
    def disturbance(t: float) -> float:
        return 2.0 * np.sin(3.0 * t)

    plant = SimpleFirstOrderPlant(
        a=1.0, b=1.0, initial_state=0.0, disturbance_fn=disturbance
    )

    config = AdaptiveControlConfig(
        num_parameters=2,
        adaptation_gain=1.5,
        sigma_modification=0.01,  # Leakage for robustness
        enable_lyapunov_monitoring=True,
        use_normalization=True,
    )

    controller = AdaptiveController(config)

    results = simulate_closed_loop(
        controller=controller,
        plant=plant,
        reference_input=10.0,
        duration=15.0,
        dt=0.01,
    )

    print(f"\n📊 Performance under disturbance:")
    print(f"   Final tracking error: {abs(results['error'][-1]):.4f}")
    print(f"   Mean absolute error: {np.mean(np.abs(results['error'])):.4f}")

    # Check stability
    is_stable = controller.is_lyapunov_stable(window=100)
    lyap_deriv = controller.get_lyapunov_derivative_history()
    mean_deriv = np.mean(lyap_deriv[-100:]) if len(lyap_deriv) >= 100 else float("inf")
    print(f"\n⚡ Stability Analysis:")
    print(f"   Lyapunov stable: {'✅ Yes' if is_stable else '❌ No'}")
    print(f"   Mean V̇ (last 100 steps): {mean_deriv:.6f}")

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(results["time"], results["reference"], "r--", label="Reference", linewidth=2)
    plt.plot(results["time"], results["plant_output"], "b-", label="Output (with disturbance)", alpha=0.7)
    plt.ylabel("Signal")
    plt.title("Tracking Performance Under Disturbance")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(results["time"], results["control"], "orange", linewidth=1.5)
    plt.ylabel("Control Signal")
    plt.title("Control Effort")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    lyap_hist = controller.get_lyapunov_history()
    plt.plot(results["time"], lyap_hist, "m-", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("V(e)")
    plt.title("Lyapunov Function (Should Decrease for Stability)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("🚀 Advanced Adaptive Control Demonstrations")
    print("=" * 80)
    print("\nThis demo showcases the upgraded Model Reference Adaptive Controller (MRAC)")
    print("with advanced features:")
    print("  ✨ Dynamic reference models (1st and 2nd order)")
    print("  ✨ Persistence of excitation (PE) monitoring")
    print("  ✨ Adaptive learning rate scheduling")
    print("  ✨ Lyapunov stability verification")
    print("  ✨ Integrated closed-loop simulation")

    try:
        demo_basic_adaptive_control()
        demo_reference_model_dynamics()
        demo_persistence_of_excitation()
        demo_time_varying_tracking()
        demo_plant_with_disturbance()

        print("\n" + "=" * 80)
        print("✅ All demonstrations completed successfully!")
        print("=" * 80)
        print("\n💡 Key Takeaways:")
        print("   1. Adaptive learning rate improves convergence speed")
        print("   2. PE monitoring ensures parameter identifiability")
        print("   3. Lyapunov monitoring verifies closed-loop stability")
        print("   4. Dynamic reference models provide better transient response")
        print("   5. Sigma modification adds robustness to disturbances")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()

