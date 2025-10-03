"""Demo script for PPO (Proximal Policy Optimization) algorithm.

This script demonstrates how to use the PPO algorithm to solve a simple
reinforcement learning problem using a custom environment.
"""

import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from algokit.algorithms.reinforcement_learning.ppo import PPOAgent


class SimpleEnvironment:
    """Simple environment for testing PPO algorithm.

    This is a simple grid world where the agent needs to navigate
    from start to goal while avoiding obstacles.
    """

    def __init__(self, size: int = 5) -> None:
        """Initialize the environment.

        Args:
            size: Size of the grid world
        """
        self.size = size
        self.agent_pos = [0, 0]
        self.goal_pos = [size - 1, size - 1]
        self.obstacles = [(1, 1), (2, 2), (3, 1)]
        self.max_steps = 100
        self.current_step = 0

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state.

        Returns:
            Initial state observation
        """
        self.agent_pos = [0, 0]
        self.current_step = 0
        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Action to take (0=up, 1=down, 2=left, 3=right)

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.current_step += 1

        # Move agent based on action
        if action == 0:  # Up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # Down
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # Left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # Right
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)

        # Check if hit obstacle
        if tuple(self.agent_pos) in self.obstacles:
            reward = -10.0
            done = True
        # Check if reached goal
        elif self.agent_pos == self.goal_pos:
            reward = 100.0
            done = True
        # Check if max steps reached
        elif self.current_step >= self.max_steps:
            reward = -1.0
            done = True
        else:
            # Small negative reward for each step
            reward = -0.1

        return self._get_state(), reward, done, {}

    def _get_state(self) -> np.ndarray:
        """Get current state representation.

        Returns:
            State vector
        """
        # Flatten position and goal information
        state = np.zeros(self.size * self.size * 2)

        # Agent position (one-hot)
        agent_idx = self.agent_pos[1] * self.size + self.agent_pos[0]
        state[agent_idx] = 1.0

        # Goal position (one-hot)
        goal_idx = self.goal_pos[1] * self.size + self.goal_pos[0]
        state[self.size * self.size + goal_idx] = 1.0

        return state

    def render(self) -> None:
        """Render the current state of the environment."""
        grid = np.zeros((self.size, self.size))

        # Mark obstacles
        for obs in self.obstacles:
            grid[obs[1], obs[0]] = -1

        # Mark agent
        grid[self.agent_pos[1], self.agent_pos[0]] = 1

        # Mark goal
        grid[self.goal_pos[1], self.goal_pos[0]] = 2

        print("Environment:")
        for row in grid:
            print(" ".join(["A" if x == 1 else "G" if x == 2 else "X" if x == -1 else "." for x in row]))
        print()


def train_ppo_agent(
    episodes: int = 1000,
    max_steps: int = 100,
    update_frequency: int = 10,
    learning_rate: float = 3e-4,
    hidden_sizes: list[int] | None = None,
    batch_size: int = 64,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    random_seed: int | None = None,
) -> tuple[PPOAgent, list[float], list[float]]:
    """Train a PPO agent on the simple environment.

    Args:
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        update_frequency: How often to update the agent
        learning_rate: Learning rate for the agent
        hidden_sizes: Hidden layer sizes for networks
        batch_size: Batch size for training
        clip_ratio: PPO clipping ratio
        value_coef: Value function loss coefficient
        entropy_coef: Entropy bonus coefficient
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (trained_agent, episode_rewards, episode_lengths)
    """
    if hidden_sizes is None:
        hidden_sizes = [128, 64]

    # Set random seeds
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    # Initialize environment and agent
    env = SimpleEnvironment(size=5)
    state_size = env.size * env.size * 2  # One-hot encoding for agent and goal positions
    action_size = 4  # Up, down, left, right

    agent = PPOAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        hidden_sizes=hidden_sizes,
        batch_size=batch_size,
        clip_ratio=clip_ratio,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        random_seed=random_seed,
    )

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    training_metrics = []

    print("Starting PPO training...")
    print(f"Environment: {env.size}x{env.size} grid world")
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Episodes: {episodes}, Max steps: {max_steps}")
    print(f"Update frequency: {update_frequency}")
    print("-" * 50)

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0

        for _step in range(max_steps):
            # Get action from agent
            action, log_prob, value = agent.get_action(state)

            # Take action in environment
            next_state, reward, done, _ = env.step(action)

            # Store experience
            agent.add_experience(state, action, reward, log_prob, value, done)

            # Update metrics
            episode_reward += reward
            episode_length += 1
            state = next_state

            if done:
                break

        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Update agent periodically
        if episode % update_frequency == 0 and episode > 0:
            metrics = agent.update(epochs=4)
            training_metrics.append(metrics)

            # Print progress
            avg_reward = np.mean(episode_rewards[-update_frequency:])
            avg_length = np.mean(episode_lengths[-update_frequency:])

            print(f"Episode {episode:4d} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Avg Length: {avg_length:5.1f} | "
                  f"Policy Loss: {metrics['policy_loss']:6.4f} | "
                  f"Value Loss: {metrics['value_loss']:6.4f}")

        # Early stopping if solved
        if len(episode_rewards) >= 100:
            recent_rewards = episode_rewards[-100:]
            if np.mean(recent_rewards) > 50.0:  # Good performance threshold
                print(f"\nSolved in {episode} episodes!")
                break

    print("\nTraining completed!")
    return agent, episode_rewards, episode_lengths


def evaluate_agent(agent: PPOAgent, episodes: int = 10) -> None:
    """Evaluate the trained agent.

    Args:
        agent: Trained PPO agent
        episodes: Number of evaluation episodes
    """
    env = SimpleEnvironment(size=5)

    print(f"\nEvaluating agent over {episodes} episodes...")
    print("-" * 30)

    total_rewards = []
    total_lengths = []
    successes = 0

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0

        for _step in range(100):  # Max steps for evaluation
            action, _, _ = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward
            episode_length += 1
            state = next_state

            if done:
                if reward > 0:  # Positive reward means success
                    successes += 1
                break

        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)

        print(f"Episode {episode + 1:2d} | "
              f"Reward: {episode_reward:6.2f} | "
              f"Length: {episode_length:3d} | "
              f"Success: {'Yes' if episode_reward > 0 else 'No'}")

    # Summary statistics
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(total_lengths)
    success_rate = successes / episodes

    print("-" * 30)
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Length: {avg_length:.1f}")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Successes: {successes}/{episodes}")


def plot_training_progress(
    episode_rewards: list[float],
    episode_lengths: list[int],
    window_size: int = 100,
) -> None:
    """Plot training progress.

    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        window_size: Window size for moving average
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot rewards
    ax1.plot(episode_rewards, alpha=0.3, color="blue", label="Episode Rewards")

    # Moving average
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode="valid")
        ax1.plot(range(window_size-1, len(episode_rewards)), moving_avg, color="red", linewidth=2, label=f"Moving Average ({window_size})")

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("PPO Training Progress - Episode Rewards")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot episode lengths
    ax2.plot(episode_lengths, alpha=0.3, color="green", label="Episode Lengths")

    # Moving average for lengths
    if len(episode_lengths) >= window_size:
        moving_avg_lengths = np.convolve(episode_lengths, np.ones(window_size)/window_size, mode="valid")
        ax2.plot(range(window_size-1, len(episode_lengths)), moving_avg_lengths, color="orange", linewidth=2, label=f"Moving Average ({window_size})")

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Length")
    ax2.set_title("PPO Training Progress - Episode Lengths")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Main function to run the PPO demo."""
    print("PPO (Proximal Policy Optimization) Demo")
    print("=" * 50)

    # Training parameters
    episodes = 2000
    max_steps = 100
    update_frequency = 50
    learning_rate = 3e-4
    hidden_sizes = [128, 64]
    batch_size = 64
    clip_ratio = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    random_seed = 42

    # Train the agent
    agent, episode_rewards, episode_lengths = train_ppo_agent(
        episodes=episodes,
        max_steps=max_steps,
        update_frequency=update_frequency,
        learning_rate=learning_rate,
        hidden_sizes=hidden_sizes,
        batch_size=batch_size,
        clip_ratio=clip_ratio,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        random_seed=random_seed,
    )

    # Evaluate the agent
    evaluate_agent(agent, episodes=20)

    # Plot training progress
    try:
        plot_training_progress(episode_rewards, episode_lengths)
    except ImportError:
        print("\nMatplotlib not available for plotting. Install with: pip install matplotlib")

    # Save the trained model
    model_path = "ppo_model.pth"
    agent.save(model_path)
    print(f"\nTrained model saved to: {model_path}")

    # Load and test the saved model
    print("\nTesting model loading...")
    new_agent = PPOAgent(
        state_size=50,  # 5x5 grid * 2 (agent + goal positions)
        action_size=4,
        learning_rate=learning_rate,
        hidden_sizes=hidden_sizes,
        batch_size=batch_size,
        clip_ratio=clip_ratio,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
    )
    new_agent.load(model_path)

    # Quick test of loaded model
    env = SimpleEnvironment(size=5)
    state = env.reset()
    action, _, _ = new_agent.get_action(state)
    print(f"Loaded model test - Action: {action}")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
