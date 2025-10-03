#!/usr/bin/env python3
"""Demonstration of Policy Gradient improvements: GAE fix, KL tracking, and reward normalization.

This demo shows:
1. GAE indexing bug fix - properly handles non-terminal final states
2. KL divergence tracking for stability monitoring
3. Reward normalization for learning stability
4. Enhanced training statistics
"""

import numpy as np
import torch
from algokit.algorithms.reinforcement_learning.policy_gradient import (
    PolicyGradientAgent,
    RolloutExperience,
)


def create_mock_environment():
    """Create a simple mock environment for testing."""
    class MockEnv:
        def __init__(self):
            self.state = np.array([0.0, 0.0, 0.0, 0.0])
            self.step_count = 0

        def reset(self):
            self.state = np.array([0.0, 0.0, 0.0, 0.0])
            self.step_count = 0
            return self.state

        def step(self, action):
            self.step_count += 1

            # Simple reward structure
            if action == 0:
                reward = 1.0
            else:
                reward = -0.5

            # Update state
            self.state += np.array([0.1, 0.1, 0.1, 0.1])

            # Terminal condition
            done = self.step_count >= 5

            return self.state.copy(), reward, done, {}

    return MockEnv()


def demonstrate_gae_fix():
    """Demonstrate the GAE indexing bug fix."""
    print("=== GAE Indexing Bug Fix Demonstration ===")

    # Create agent with GAE enabled
    agent = PolicyGradientAgent(
        state_size=4,
        action_size=2,
        use_gae=True,
        use_baseline=True,
        gae_lambda=0.95,
        continuous_actions=False,
    )

    # Create test data that would cause the original bug
    # Final step is NOT terminal - this would crash the old implementation
    rollout_data = [
        RolloutExperience(
            state=np.array([1.0, 0.0, 0.0, 0.0]),
            action=0,
            reward=1.0,
            log_prob=-0.5,
            value=0.8,
            done=False,
        ),
        RolloutExperience(
            state=np.array([0.0, 1.0, 0.0, 0.0]),
            action=1,
            reward=0.5,
            log_prob=-0.6,
            value=0.6,
            done=False,
        ),
        RolloutExperience(
            state=np.array([0.0, 0.0, 1.0, 0.0]),
            action=0,
            reward=-0.5,
            log_prob=-0.7,
            value=0.4,
            done=False,  # Final step is NOT terminal - this was the bug!
        ),
    ]

    print("Testing GAE computation with non-terminal final step...")
    print("This would have crashed the old implementation!")

    # This should work without crashing
    try:
        metrics = agent.learn(rollout_data)
        print("✅ GAE computation successful!")
        print(f"Policy loss: {metrics['policy_loss']:.4f}")
        print(f"KL divergence: {metrics['kl_divergence']:.4f}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print()


def demonstrate_kl_tracking():
    """Demonstrate KL divergence tracking."""
    print("=== KL Divergence Tracking Demonstration ===")

    # Create agent with enhanced features
    agent = PolicyGradientAgent(
        state_size=4,
        action_size=2,
        continuous_actions=False,
    )

    print("Training with KL divergence tracking...")

    # Simulate multiple training steps
    for episode in range(3):
        rollout_data = [
            RolloutExperience(
                state=np.random.randn(4),
                action=np.random.randint(0, 2),
                reward=np.random.randn() * 10,  # High variance rewards
                log_prob=np.random.randn() * 0.1,
                value=np.random.randn() * 0.1,
                done=True,
            ),
            RolloutExperience(
                state=np.random.randn(4),
                action=np.random.randint(0, 2),
                reward=np.random.randn() * 10,
                log_prob=np.random.randn() * 0.1,
                value=np.random.randn() * 0.1,
                done=True,
            ),
        ]

        metrics = agent.learn(rollout_data)
        kl_div = metrics['kl_divergence']

        print(f"Episode {episode + 1}: KL divergence = {kl_div:.4f}")

    # Check KL tracking statistics
    stats = agent.get_training_stats()
    if agent.kl_divergence_history:
        print(f"\nKL Divergence Statistics:")
        print(f"  Mean: {stats['mean_kl_divergence']:.4f}")
        if 'std_kl_divergence' in stats:
            print(f"  Std: {stats['std_kl_divergence']:.4f}")
        if 'recent_kl_divergence' in stats:
            print(f"  Recent: {stats['recent_kl_divergence']:.4f}")
        print(f"  History length: {len(agent.kl_divergence_history)}")

    print()


def demonstrate_reward_normalization():
    """Demonstrate reward normalization."""
    print("=== Reward Normalization Demonstration ===")

    # Create agent with reward normalization enabled
    agent = PolicyGradientAgent(
        state_size=4,
        action_size=2,
        normalize_rewards=True,
        continuous_actions=False,
    )

    print("Training with reward normalization...")
    print("Initial reward stats:", agent.reward_mean, agent.reward_std)

    # Simulate training with high variance rewards
    for episode in range(3):
        # Create rewards with high variance
        rewards = [100.0, -50.0, 200.0, -75.0]  # High variance

        rollout_data = [
            RolloutExperience(
                state=np.random.randn(4),
                action=np.random.randint(0, 2),
                reward=reward,
                log_prob=np.random.randn() * 0.1,
                value=np.random.randn() * 0.1,
                done=True,
            )
            for reward in rewards
        ]

        metrics = agent.learn(rollout_data)

        print(f"Episode {episode + 1}:")
        print(f"  Raw rewards: {rewards}")
        print(f"  Reward mean: {metrics['reward_mean']:.2f}")
        print(f"  Reward std: {metrics['reward_std']:.2f}")
        print(f"  Update count: {agent.reward_update_count}")

    print()


def demonstrate_enhanced_statistics():
    """Demonstrate enhanced training statistics."""
    print("=== Enhanced Training Statistics Demonstration ===")

    # Create agent with all new features
    agent = PolicyGradientAgent(
        state_size=4,
        action_size=2,
        use_gae=True,
        normalize_rewards=True,
        continuous_actions=False,
    )

    # Add some episode data
    agent.episode_rewards = [10.0, 15.0, 8.0, 20.0, 12.0]
    agent.episode_lengths = [50, 60, 45, 70, 55]

    # Add some KL divergence history
    agent.kl_divergence_history = [0.05, 0.08, 0.06, 0.09, 0.07]

    # Get enhanced statistics
    stats = agent.get_training_stats()

    print("Enhanced Training Statistics:")
    print(f"  Episode Statistics:")
    print(f"    Mean reward: {stats['mean_reward']:.2f}")
    print(f"    Std reward: {stats['std_reward']:.2f}")
    print(f"    Mean length: {stats['mean_length']:.2f}")
    print(f"    Total episodes: {stats['total_episodes']}")

    print(f"  KL Divergence Statistics:")
    print(f"    Mean: {stats['mean_kl_divergence']:.4f}")
    if 'std_kl_divergence' in stats:
        print(f"    Std: {stats['std_kl_divergence']:.4f}")
    if 'recent_kl_divergence' in stats:
        print(f"    Recent: {stats['recent_kl_divergence']:.4f}")

    print(f"  Reward Normalization Statistics:")
    print(f"    Mean: {stats['reward_normalization_stats']['mean']:.2f}")
    print(f"    Std: {stats['reward_normalization_stats']['std']:.2f}")

    print()


def main():
    """Run all demonstrations."""
    print("Policy Gradient Algorithm Improvements Demo")
    print("=" * 50)
    print()

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        demonstrate_gae_fix()
        demonstrate_kl_tracking()
        demonstrate_reward_normalization()
        demonstrate_enhanced_statistics()

        print("✅ All demonstrations completed successfully!")
        print("\nKey Improvements Implemented:")
        print("1. ✅ GAE indexing bug fix - handles non-terminal final states")
        print("2. ✅ KL divergence tracking for stability monitoring")
        print("3. ✅ Reward normalization for learning stability")
        print("4. ✅ Enhanced training statistics")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
