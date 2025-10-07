#!/usr/bin/env python3
"""Actor-Critic demonstration script.

This script demonstrates how to use the Actor-Critic agent to solve
a continuous state space problem using neural networks.
"""

import matplotlib.pyplot as plt
import numpy as np

from algokit.algorithms.reinforcement_learning.actor_critic import (
    ActorCriticAgent,
    ActorCriticConfig,
)


class ContinuousGridWorld:
    """A continuous version of the grid world for Actor-Critic demonstration.

    This environment has continuous states (x, y coordinates) and discrete actions.
    The agent needs to navigate from start to goal while avoiding obstacles.
    """

    def __init__(self, width: float = 10.0, height: float = 10.0) -> None:
        """Initialize continuous grid world.

        Args:
            width: Width of the world
            height: Height of the world
        """
        self.width = width
        self.height = height
        self.state_size = 2  # x, y coordinates

        # Define start and goal positions
        self.start = np.array([1.0, 1.0])
        self.goal = np.array([9.0, 9.0])

        # Define obstacles (circles)
        self.obstacles = [
            {"center": [3.0, 3.0], "radius": 1.0},
            {"center": [7.0, 5.0], "radius": 1.2},
            {"center": [5.0, 8.0], "radius": 0.8},
        ]

        # Action space: [up, down, left, right] with step size
        self.action_size = 4
        self.step_size = 0.2

    def reset(self) -> np.ndarray:
        """Reset environment to initial state.

        Returns:
            Initial state (x, y coordinates)
        """
        self.position = self.start.copy()
        return self.position.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Take a step in the environment.

        Args:
            action: Action to take (0=up, 1=down, 2=left, 3=right)

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Define action directions
        directions = [
            np.array([0, self.step_size]),   # up
            np.array([0, -self.step_size]),  # down
            np.array([-self.step_size, 0]),   # left
            np.array([self.step_size, 0])     # right
        ]

        # Calculate new position
        new_position = self.position + directions[action]

        # Check bounds
        new_position[0] = np.clip(new_position[0], 0, self.width)
        new_position[1] = np.clip(new_position[1], 0, self.height)

        # Check for obstacle collision
        collision = False
        for obstacle in self.obstacles:
            distance = np.linalg.norm(new_position - obstacle["center"])
            if distance < obstacle["radius"]:
                collision = True
                break

        # If collision, don't move
        if collision:
            new_position = self.position.copy()

        # Update position
        self.position = new_position

        # Calculate reward
        reward = self._calculate_reward()

        # Check if done
        done = self._is_done()

        info = {"collision": collision}

        return self.position.copy(), reward, done, info

    def _calculate_reward(self) -> float:
        """Calculate reward based on current position.

        Returns:
            Reward value
        """
        # Distance to goal
        distance_to_goal = np.linalg.norm(self.position - self.goal)

        # Base reward (negative distance to encourage getting closer)
        reward = -distance_to_goal * 0.1

        # Bonus for reaching goal
        if distance_to_goal < 0.5:
            reward += 10.0

        # Penalty for being out of bounds (shouldn't happen with clipping)
        if (self.position[0] < 0 or self.position[0] > self.width or
            self.position[1] < 0 or self.position[1] > self.height):
            reward -= 5.0

        return reward

    def _is_done(self) -> bool:
        """Check if episode is done.

        Returns:
            True if episode is done
        """
        distance_to_goal = np.linalg.norm(self.position - self.goal)
        return distance_to_goal < 0.5

    def render(self, agent_position: np.ndarray | None = None) -> None:
        """Render the environment.

        Args:
            agent_position: Current agent position to highlight
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw obstacles
        for obstacle in self.obstacles:
            circle = plt.Circle(
                obstacle["center"],
                obstacle["radius"],
                color="red",
                alpha=0.7
            )
            ax.add_patch(circle)

        # Draw start and goal
        ax.plot(self.start[0], self.start[1], "go", markersize=10, label="Start")
        ax.plot(self.goal[0], self.goal[1], "bo", markersize=10, label="Goal")

        # Draw agent position if provided
        if agent_position is not None:
            ax.plot(agent_position[0], agent_position[1], "ko", markersize=8, label="Agent")

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title("Continuous Grid World")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

        plt.tight_layout()
        plt.show()


def train_actor_critic_agent(
    episodes: int = 1000,
    max_steps: int = 200,
    learning_frequency: int = 10,
    render_frequency: int = 100
) -> tuple[ActorCriticAgent, list[float]]:
    """Train Actor-Critic agent on continuous grid world.

    Args:
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        learning_frequency: How often to update the networks
        render_frequency: How often to render the environment

    Returns:
        Tuple of (trained_agent, episode_rewards)
    """
    # Create environment and agent
    env = ContinuousGridWorld()

    # New style with config (recommended)
    config = ActorCriticConfig(
        state_size=env.state_size,
        action_size=env.action_size,
        learning_rate_actor=0.001,
        learning_rate_critic=0.001,
        discount_factor=0.99,
        hidden_sizes=[64, 64],
        entropy_coefficient=0.01
    )
    agent = ActorCriticAgent(config=config)

    # Old style still works (backwards compatible)
    # agent = ActorCriticAgent(
    #     state_size=env.state_size,
    #     action_size=env.action_size,
    #     learning_rate_actor=0.001,
    #     learning_rate_critic=0.001,
    #     discount_factor=0.99,
    #     hidden_sizes=[64, 64],
    #     entropy_coefficient=0.01
    # )

    episode_rewards = []
    recent_rewards = []

    print("Training Actor-Critic agent...")
    print("=" * 50)

    for episode in range(episodes):
        # Collect rollout data (full episode)
        rollout_data = agent.collect_rollout(env, n_steps=None, max_episode_length=max_steps)

        # Calculate episode reward
        episode_reward = sum(exp.reward for exp in rollout_data)

        # Learn from rollout data
        if rollout_data and episode % learning_frequency == 0:
            agent.learn(rollout_data)

        # Record episode reward
        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)

        # Keep only recent rewards for moving average
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        # Print progress
        if episode % 50 == 0:
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:6.2f} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Rollout Length: {len(rollout_data):4d}")

        # Render environment occasionally
        if episode % render_frequency == 0 and episode > 0:
            print(f"\nRendering episode {episode}...")
            # Get final position from rollout
            if rollout_data:
                final_state = rollout_data[-1].state
                env.render(final_state)

    return agent, episode_rewards


def test_trained_agent(agent: ActorCriticAgent, num_episodes: int = 5) -> None:
    """Test the trained agent.

    Args:
        agent: Trained Actor-Critic agent
        num_episodes: Number of test episodes
    """
    env = ContinuousGridWorld()
    agent.set_training(False)  # Set to evaluation mode

    print(f"\nTesting trained agent for {num_episodes} episodes...")
    print("=" * 50)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        path = [state.copy()]

        for _step in range(200):  # Max steps
            action, _, _ = agent.get_action(state, training=False)
            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            path.append(next_state.copy())

            if done:
                break

            state = next_state

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {len(path)}")

        # Render the path
        if episode == 0:  # Render first episode
            fig, ax = plt.subplots(figsize=(8, 8))

            # Draw obstacles
            for obstacle in env.obstacles:
                circle = plt.Circle(
                    obstacle["center"],
                    obstacle["radius"],
                    color="red",
                    alpha=0.7
                )
                ax.add_patch(circle)

            # Draw start and goal
            ax.plot(env.start[0], env.start[1], "go", markersize=10, label="Start")
            ax.plot(env.goal[0], env.goal[1], "bo", markersize=10, label="Goal")

            # Draw path
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], "k-", alpha=0.7, linewidth=2, label="Path")
            ax.plot(path_array[0, 0], path_array[0, 1], "ko", markersize=8, label="Start")
            ax.plot(path_array[-1, 0], path_array[-1, 1], "ks", markersize=8, label="End")

            ax.set_xlim(0, env.width)
            ax.set_ylim(0, env.height)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_title(f"Actor-Critic Agent Path (Episode {episode + 1})")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")

            plt.tight_layout()
            plt.show()


def plot_training_progress(episode_rewards: list[float], window_size: int = 50) -> None:
    """Plot training progress.

    Args:
        episode_rewards: List of episode rewards
        window_size: Window size for moving average
    """
    # Calculate moving average
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode="valid")
        episodes_avg = range(window_size-1, len(episode_rewards))
    else:
        moving_avg = episode_rewards
        episodes_avg = range(len(episode_rewards))

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot raw rewards
    ax1.plot(episode_rewards, alpha=0.3, color="blue", label="Raw Rewards")
    ax1.plot(episodes_avg, moving_avg, color="red", linewidth=2, label=f"Moving Average ({window_size})")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Actor-Critic Training Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot smoothed rewards only
    ax2.plot(episodes_avg, moving_avg, color="red", linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Average Reward")
    ax2.set_title("Smoothed Training Progress")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Main demonstration function."""
    print("Actor-Critic Algorithm Demonstration")
    print("=" * 50)
    print("This demo shows how the Actor-Critic algorithm can learn to navigate")
    print("a continuous grid world with obstacles using neural networks.")
    print()

    # Train the agent
    agent, episode_rewards = train_actor_critic_agent(
        episodes=500,
        max_steps=200,
        learning_frequency=5,
        render_frequency=100
    )

    # Plot training progress
    print("\nPlotting training progress...")
    plot_training_progress(episode_rewards)

    # Test the trained agent
    test_trained_agent(agent, num_episodes=3)

    print("\nDemonstration completed!")
    print("The Actor-Critic agent successfully learned to navigate the environment.")


if __name__ == "__main__":
    main()
