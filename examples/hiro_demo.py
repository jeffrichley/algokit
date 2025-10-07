"""Demo script for HIRO (Hierarchical Reinforcement Learning) algorithm.

This script demonstrates both initialization styles:
1. New config-based approach (recommended)
2. Legacy kwargs-based approach (backwards compatible)

The HIRO algorithm learns hierarchical policies where a higher-level
policy proposes goals for a lower-level policy to achieve.
"""

from __future__ import annotations

import gymnasium as gym

from algokit.algorithms.hierarchical_rl.hiro import HIROAgent, HIROConfig


def demo_config_based_initialization() -> None:
    """Demonstrate HIRO initialization with config object (recommended)."""
    print("\n" + "=" * 60)
    print("HIRO Demo: Config-based Initialization (Recommended)")
    print("=" * 60 + "\n")

    # Create environment
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create config object with validation
    config = HIROConfig(
        state_size=state_size,
        action_size=action_size,
        goal_size=16,
        hidden_size=256,
        goal_horizon=10,
        learning_rate=0.0003,
        gamma=0.99,
        tau=0.005,
        device="cpu",
        seed=42,
        policy_noise=0.2,
        noise_clip=0.5,
        intrinsic_scale=1.0,
    )

    print("Configuration created:")
    print(f"  State size: {config.state_size}")
    print(f"  Action size: {config.action_size}")
    print(f"  Goal size: {config.goal_size}")
    print(f"  Goal horizon: {config.goal_horizon}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Gamma: {config.gamma}")
    print(f"  Device: {config.device}")

    # Initialize agent with config
    agent = HIROAgent(config=config)

    print("\nAgent initialized successfully!")
    print(f"  Higher-level policy: {type(agent.higher_policy).__name__}")
    print(f"  Lower-level policy: {type(agent.lower_policy).__name__}")

    # Train for a few episodes
    print("\nTraining for 3 episodes...")
    for episode in range(3):
        metrics = agent.train_episode(env, max_steps=200, epsilon=0.1)
        print(f"  Episode {episode + 1}:")
        print(f"    Reward: {metrics['reward']:.2f}")
        print(f"    Steps: {metrics['steps']}")
        print(f"    Lower critic loss: {metrics['avg_lower_critic_loss']:.4f}")
        print(f"    Higher critic loss: {metrics['avg_higher_critic_loss']:.4f}")

    # Get statistics
    stats = agent.get_statistics()
    print("\nFinal Statistics:")
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Average reward: {stats['avg_reward']:.2f}")
    print(f"  Intrinsic reward: {stats['avg_intrinsic_reward']:.4f}")
    print(f"  Extrinsic reward: {stats['avg_extrinsic_reward']:.4f}")

    env.close()


def demo_kwargs_based_initialization() -> None:
    """Demonstrate HIRO initialization with kwargs (backwards compatible)."""
    print("\n" + "=" * 60)
    print("HIRO Demo: Kwargs-based Initialization (Backwards Compatible)")
    print("=" * 60 + "\n")

    # Create environment
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize agent with individual parameters (old style)
    agent = HIROAgent(
        state_size=state_size,
        action_size=action_size,
        goal_size=16,
        hidden_size=256,
        goal_horizon=10,
        learning_rate=0.0003,
        gamma=0.99,
        tau=0.005,
        device="cpu",
        seed=42,
    )

    print("Agent initialized with kwargs (old style)")
    print(f"  State size: {agent.state_size}")
    print(f"  Action size: {agent.action_size}")
    print(f"  Goal size: {agent.goal_size}")

    # Note: Config is automatically created internally
    print("\nConfig object created automatically:")
    print(f"  Config type: {type(agent.config).__name__}")
    print(f"  Config.state_size: {agent.config.state_size}")
    print(f"  Config.action_size: {agent.config.action_size}")

    # Train for a few episodes
    print("\nTraining for 3 episodes...")
    for episode in range(3):
        metrics = agent.train_episode(env, max_steps=200, epsilon=0.1)
        print(f"  Episode {episode + 1}: Reward = {metrics['reward']:.2f}, Steps = {metrics['steps']}")

    env.close()


def demo_parameter_validation() -> None:
    """Demonstrate automatic parameter validation with Pydantic."""
    print("\n" + "=" * 60)
    print("HIRO Demo: Automatic Parameter Validation")
    print("=" * 60 + "\n")

    # Valid configuration
    print("Creating valid configuration...")
    try:
        config = HIROConfig(state_size=4, action_size=2)
        print("  ✓ Valid config created successfully")
        print(f"    State size: {config.state_size}")
        print(f"    Action size: {config.action_size}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Invalid state_size (negative)
    print("\nTrying to create config with negative state_size...")
    try:
        config = HIROConfig(state_size=-1, action_size=2)
        print("  ✗ Should have raised ValidationError!")
    except Exception as e:
        print(f"  ✓ Caught ValidationError: {type(e).__name__}")

    # Invalid learning_rate (out of range)
    print("\nTrying to create config with invalid learning_rate (> 1.0)...")
    try:
        config = HIROConfig(state_size=4, action_size=2, learning_rate=1.5)
        print("  ✗ Should have raised ValidationError!")
    except Exception as e:
        print(f"  ✓ Caught ValidationError: {type(e).__name__}")

    # Invalid device
    print("\nTrying to create config with invalid device...")
    try:
        config = HIROConfig(state_size=4, action_size=2, device="gpu")
        print("  ✗ Should have raised ValidationError!")
    except Exception as e:
        print(f"  ✓ Caught ValidationError: {type(e).__name__}")

    # Invalid seed (negative)
    print("\nTrying to create config with negative seed...")
    try:
        config = HIROConfig(state_size=4, action_size=2, seed=-1)
        print("  ✗ Should have raised ValidationError!")
    except Exception as e:
        print(f"  ✓ Caught ValidationError: {type(e).__name__}")

    print("\n" + "=" * 60)
    print("Validation demos complete!")
    print("=" * 60)


def main() -> None:
    """Run all HIRO demos."""
    print("\n" + "=" * 60)
    print("HIRO Algorithm Demonstration")
    print("=" * 60)

    # Demo 1: Config-based initialization (recommended)
    demo_config_based_initialization()

    # Demo 2: Kwargs-based initialization (backwards compatible)
    demo_kwargs_based_initialization()

    # Demo 3: Parameter validation
    demo_parameter_validation()

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
