#!/usr/bin/env python3
"""Policy Gradient Algorithm Demo.

This script demonstrates the Policy Gradient (REINFORCE) algorithm for both
discrete and continuous action spaces. It shows how to train a policy
directly using gradient ascent on the expected return.

The demo includes:
1. Discrete action space example (CartPole-like environment)
2. Continuous action space example (MountainCar-like environment)
3. Comparison with and without baseline for variance reduction
4. Training progress visualization
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from algokit.algorithms.reinforcement_learning.policy_gradient import (
    Experience,
    PolicyGradientAgent,
)


class SimpleDiscreteEnvironment:
    """Simple discrete environment for testing Policy Gradient."""

    def __init__(self, state_size: int = 4, action_size: int = 2) -> None:
        """Initialize the environment.

        Args:
            state_size: Dimension of state space
            action_size: Number of discrete actions
        """
        self.state_size = state_size
        self.action_size = action_size
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state.

        Returns:
            Initial state
        """
        self.state = np.random.randn(self.state_size)
        self.step_count = 0
        self.max_steps = 100
        return self.state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.step_count += 1

        # Simple reward structure: encourage staying in center
        center_reward = -np.sum(self.state**2) * 0.1

        # Action reward: encourage action 0
        action_reward = 1.0 if action == 0 else -0.5

        # Step reward: encourage longer episodes
        step_reward = 0.1

        reward = center_reward + action_reward + step_reward

        # Update state with some dynamics
        self.state += np.random.randn(self.state_size) * 0.1

        # Episode ends after max steps
        done = self.step_count >= self.max_steps

        info = {"step_count": self.step_count}

        return self.state.copy(), reward, done, info


class SimpleContinuousEnvironment:
    """Simple continuous environment for testing Policy Gradient."""

    def __init__(self, state_size: int = 4, action_size: int = 2) -> None:
        """Initialize the environment.

        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
        """
        self.state_size = state_size
        self.action_size = action_size
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state.

        Returns:
            Initial state
        """
        self.state = np.random.randn(self.state_size)
        self.step_count = 0
        self.max_steps = 100
        return self.state.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Take a step in the environment.

        Args:
            action: Action to take (continuous)

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.step_count += 1

        # Simple reward structure: encourage staying in center
        center_reward = -np.sum(self.state**2) * 0.1

        # Action reward: encourage small actions
        action_reward = -np.sum(action**2) * 0.1

        # Step reward: encourage longer episodes
        step_reward = 0.1

        reward = center_reward + action_reward + step_reward

        # Update state with some dynamics
        self.state += action * 0.1 + np.random.randn(self.state_size) * 0.05

        # Episode ends after max steps
        done = self.step_count >= self.max_steps

        info = {"step_count": self.step_count}

        return self.state.copy(), reward, done, info


def run_episode(
    agent: PolicyGradientAgent,
    env,
    max_steps: int = 100,
    render: bool = False,
) -> tuple[list[Experience], float, int]:
    """Run a single episode with the agent.

    Args:
        agent: Policy gradient agent
        env: Environment to interact with
        max_steps: Maximum steps per episode
        render: Whether to render the environment

    Returns:
        Tuple of (trajectory, total_reward, episode_length)
    """
    state = env.reset()
    trajectory = []
    total_reward = 0.0

    for step in range(max_steps):
        # Get action and log probability
        action, log_prob = agent.get_action_with_log_prob(state)

        # Take step in environment
        next_state, reward, done, info = env.step(action)

        # Store experience
        trajectory.append(Experience(state, action, reward, log_prob))

        total_reward += reward
        state = next_state

        if render:
            print(f"Step {step}: State={state[:2]}, Action={action}, Reward={reward:.3f}")

        if done:
            break

    return trajectory, total_reward, len(trajectory)


def train_agent(
    agent: PolicyGradientAgent,
    env,
    num_episodes: int = 100,
    episodes_per_update: int = 5,
    render_frequency: int = 20,
) -> tuple[list[float], list[float], list[float]]:
    """Train the agent for multiple episodes.

    Args:
        agent: Policy gradient agent
        env: Environment to train on
        num_episodes: Total number of episodes to run
        episodes_per_update: Number of episodes to collect before updating
        render_frequency: How often to render episodes

    Returns:
        Tuple of (episode_rewards, episode_lengths, policy_losses)
    """
    episode_rewards = []
    episode_lengths = []
    policy_losses = []

    trajectories_buffer = []

    print(f"Training for {num_episodes} episodes...")
    print(f"Updating every {episodes_per_update} episodes")
    print("-" * 50)

    for episode in range(num_episodes):
        # Run episode
        render = (episode + 1) % render_frequency == 0
        trajectory, total_reward, episode_length = run_episode(
            agent, env, render=render
        )

        trajectories_buffer.append(trajectory)
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)

        # Update agent if we have enough episodes
        if len(trajectories_buffer) >= episodes_per_update:
            metrics = agent.update(trajectories_buffer)
            policy_losses.append(metrics["policy_loss"])
            trajectories_buffer = []

            # Print progress
            if (episode + 1) % 10 == 0:
                recent_rewards = episode_rewards[-10:]
                avg_reward = np.mean(recent_rewards)
                avg_length = np.mean(episode_lengths[-10:])
                print(f"Episode {episode + 1:3d}: "
                      f"Avg Reward = {avg_reward:6.2f}, "
                      f"Avg Length = {avg_length:5.1f}")

    return episode_rewards, episode_lengths, policy_losses


def plot_training_progress(
    episode_rewards: list[float],
    episode_lengths: list[float],
    policy_losses: list[float],
    title: str = "Policy Gradient Training Progress",
) -> None:
    """Plot training progress.

    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        policy_losses: List of policy losses
        title: Plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot episode rewards
    axes[0].plot(episode_rewards, alpha=0.7, label="Episode Rewards")
    if len(episode_rewards) > 10:
        # Plot moving average
        window = min(20, len(episode_rewards) // 4)
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode="valid")
        axes[0].plot(range(window-1, len(episode_rewards)), moving_avg,
                    color="red", linewidth=2, label=f"Moving Avg ({window})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("Episode Rewards")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot episode lengths
    axes[1].plot(episode_lengths, alpha=0.7, color="green")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode Length")
    axes[1].set_title("Episode Lengths")
    axes[1].grid(True, alpha=0.3)

    # Plot policy losses
    if policy_losses:
        axes[2].plot(policy_losses, alpha=0.7, color="orange")
        axes[2].set_xlabel("Update")
        axes[2].set_ylabel("Policy Loss")
        axes[2].set_title("Policy Loss")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "No Policy Loss Data",
                    ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title("Policy Loss")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def demo_discrete_actions() -> None:
    """Demonstrate Policy Gradient with discrete actions."""
    print("=" * 60)
    print("DISCRETE ACTION SPACE DEMO")
    print("=" * 60)

    # Create environment and agent
    env = SimpleDiscreteEnvironment(state_size=4, action_size=2)
    agent = PolicyGradientAgent(
        state_size=4,
        action_size=2,
        learning_rate=0.001,
        gamma=0.99,
        use_baseline=True,
        continuous_actions=False,
        seed=42,
    )

    print("Environment: Simple Discrete (4D state, 2 actions)")
    print("Agent: Policy Gradient with Baseline")
    print("Actions: Discrete (0, 1)")
    print()

    # Train agent
    episode_rewards, episode_lengths, policy_losses = train_agent(
        agent, env, num_episodes=100, episodes_per_update=5
    )

    # Show results
    print("\nTraining Results:")
    print(f"Final Average Reward (last 10 episodes): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Final Average Length (last 10 episodes): {np.mean(episode_lengths[-10:]):.1f}")
    print(f"Total Episodes: {len(episode_rewards)}")

    # Plot results
    plot_training_progress(
        episode_rewards, episode_lengths, policy_losses,
        "Policy Gradient - Discrete Actions"
    )


def demo_continuous_actions() -> None:
    """Demonstrate Policy Gradient with continuous actions."""
    print("=" * 60)
    print("CONTINUOUS ACTION SPACE DEMO")
    print("=" * 60)

    # Create environment and agent
    env = SimpleContinuousEnvironment(state_size=4, action_size=2)
    agent = PolicyGradientAgent(
        state_size=4,
        action_size=2,
        learning_rate=0.001,
        gamma=0.99,
        use_baseline=True,
        continuous_actions=True,
        seed=42,
    )

    print("Environment: Simple Continuous (4D state, 2D actions)")
    print("Agent: Policy Gradient with Baseline")
    print("Actions: Continuous (2D vector)")
    print()

    # Train agent
    episode_rewards, episode_lengths, policy_losses = train_agent(
        agent, env, num_episodes=100, episodes_per_update=5
    )

    # Show results
    print("\nTraining Results:")
    print(f"Final Average Reward (last 10 episodes): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Final Average Length (last 10 episodes): {np.mean(episode_lengths[-10:]):.1f}")
    print(f"Total Episodes: {len(episode_rewards)}")

    # Plot results
    plot_training_progress(
        episode_rewards, episode_lengths, policy_losses,
        "Policy Gradient - Continuous Actions"
    )


def demo_baseline_comparison() -> None:
    """Compare Policy Gradient with and without baseline."""
    print("=" * 60)
    print("BASELINE COMPARISON DEMO")
    print("=" * 60)

    # Create environments
    env_with_baseline = SimpleDiscreteEnvironment(state_size=4, action_size=2)
    env_without_baseline = SimpleDiscreteEnvironment(state_size=4, action_size=2)

    # Create agents
    agent_with_baseline = PolicyGradientAgent(
        state_size=4,
        action_size=2,
        use_baseline=True,
        continuous_actions=False,
        seed=42,
    )

    agent_without_baseline = PolicyGradientAgent(
        state_size=4,
        action_size=2,
        use_baseline=False,
        continuous_actions=False,
        seed=42,
    )

    print("Training two agents:")
    print("1. Policy Gradient WITH baseline (variance reduction)")
    print("2. Policy Gradient WITHOUT baseline (pure REINFORCE)")
    print()

    # Train both agents
    print("Training agent WITH baseline...")
    rewards_with, lengths_with, losses_with = train_agent(
        agent_with_baseline, env_with_baseline, num_episodes=50
    )

    print("\nTraining agent WITHOUT baseline...")
    rewards_without, lengths_without, losses_without = train_agent(
        agent_without_baseline, env_without_baseline, num_episodes=50
    )

    # Compare results
    print("\nComparison Results:")
    print(f"With Baseline - Final Avg Reward: {np.mean(rewards_with[-10:]):.2f}")
    print(f"Without Baseline - Final Avg Reward: {np.mean(rewards_without[-10:]):.2f}")
    print(f"With Baseline - Final Avg Length: {np.mean(lengths_with[-10:]):.1f}")
    print(f"Without Baseline - Final Avg Length: {np.mean(lengths_without[-10:]):.1f}")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot rewards comparison
    axes[0].plot(rewards_with, alpha=0.7, label="With Baseline", color="blue")
    axes[0].plot(rewards_without, alpha=0.7, label="Without Baseline", color="red")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("Reward Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot lengths comparison
    axes[1].plot(lengths_with, alpha=0.7, label="With Baseline", color="blue")
    axes[1].plot(lengths_without, alpha=0.7, label="Without Baseline", color="red")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode Length")
    axes[1].set_title("Length Comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Policy Gradient: With vs Without Baseline", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Policy Gradient Algorithm Demo")
    parser.add_argument(
        "--demo",
        choices=["discrete", "continuous", "baseline", "all"],
        default="all",
        help="Which demo to run"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to train"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting"
    )

    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    print("Policy Gradient Algorithm Demo")
    print("=" * 60)
    print("This demo shows the Policy Gradient (REINFORCE) algorithm")
    print("for both discrete and continuous action spaces.")
    print()

    start_time = time.time()

    if args.demo in ["discrete", "all"]:
        demo_discrete_actions()
        print()

    if args.demo in ["continuous", "all"]:
        demo_continuous_actions()
        print()

    if args.demo in ["baseline", "all"]:
        demo_baseline_comparison()
        print()

    total_time = time.time() - start_time
    print(f"Demo completed in {total_time:.2f} seconds")

    if not args.no_plot:
        print("\nNote: Close plot windows to continue or use --no-plot to disable plotting")


if __name__ == "__main__":
    main()
