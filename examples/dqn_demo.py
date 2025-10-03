#!/usr/bin/env python3
"""DQN demonstration script.

This script demonstrates how to use the DQN agent to solve
a continuous state space problem using neural networks.
"""

import matplotlib.pyplot as plt
import numpy as np

from algokit.algorithms.reinforcement_learning.dqn import DQNAgent


class ContinuousGridWorld:
    """A continuous version of the grid world for DQN demonstration.

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
            Initial state (start position)
        """
        return self.start.copy()

    def step(self, state: np.ndarray, action: int) -> tuple[np.ndarray, float, bool]:
        """Take a step in the environment.

        Args:
            state: Current state [x, y]
            action: Action to take (0=up, 1=down, 2=left, 3=right)

        Returns:
            Tuple of (next_state, reward, done)
        """
        # Action mappings
        actions = [
            [0, self.step_size],   # up
            [0, -self.step_size],  # down
            [-self.step_size, 0],  # left
            [self.step_size, 0],   # right
        ]

        # Calculate next state
        next_state = state + actions[action]

        # Keep within bounds
        next_state[0] = np.clip(next_state[0], 0, self.width)
        next_state[1] = np.clip(next_state[1], 0, self.height)

        # Check for obstacle collision
        collision = self._check_collision(next_state)
        if collision:
            next_state = state  # Stay in current state if collision

        # Check if reached goal
        goal_distance = np.linalg.norm(next_state - self.goal)
        if goal_distance < 0.3:
            reward = 100.0
            done = True
        elif collision:
            reward = -10.0
            done = False
        else:
            # Reward based on distance to goal (closer is better)
            current_distance = np.linalg.norm(state - self.goal)
            next_distance = np.linalg.norm(next_state - self.goal)
            reward = (current_distance - next_distance) * 5.0 - 0.1  # Small step penalty

        return next_state, reward, done

    def _check_collision(self, state: np.ndarray) -> bool:
        """Check if state collides with any obstacle.

        Args:
            state: State to check

        Returns:
            True if collision detected
        """
        for obstacle in self.obstacles:
            distance = np.linalg.norm(state - obstacle["center"])
            if distance < obstacle["radius"]:
                return True
        return False

    def render(self, agent_state: np.ndarray, path: list[np.ndarray]) -> None:
        """Render the environment and agent path.

        Args:
            agent_state: Current agent state
            path: List of states in the path
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw obstacles
        for obstacle in self.obstacles:
            circle = plt.Circle(
                obstacle["center"],
                obstacle["radius"],
                color="red",
                alpha=0.7
            )
            ax.add_patch(circle)

        # Draw start position
        ax.plot(self.start[0], self.start[1], "go", markersize=15, label="Start")

        # Draw goal position
        ax.plot(self.goal[0], self.goal[1], "ro", markersize=15, label="Goal")

        # Draw path
        if len(path) > 1:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], "b-", alpha=0.7, linewidth=2, label="Path")

        # Draw current agent position
        ax.plot(agent_state[0], agent_state[1], "bo", markersize=10, label="Agent")

        # Set limits and labels
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title("DQN Continuous Grid World")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.show()


def train_dqn_agent(agent: DQNAgent, env: ContinuousGridWorld, episodes: int = 500) -> list[float]:
    """Train the DQN agent.

    Args:
        agent: DQN agent
        env: Environment
        episodes: Number of training episodes

    Returns:
        List of episode rewards
    """
    episode_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 200

        while steps < max_steps:
            # Get action from agent
            action = agent.get_action(state)

            # Take step in environment
            next_state, reward, done = env.step(state, action)

            # Store experience and train
            agent.step(state, action, reward, next_state, done)

            # Update state and tracking variables
            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        # Decay epsilon using step-based linear decay
        agent.decay_epsilon_by_step(episode)
        episode_rewards.append(total_reward)

        # Print progress every 50 episodes
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1}: Average reward = {avg_reward:.2f}, Epsilon = {agent.get_epsilon():.3f}")

    return episode_rewards


def test_dqn_agent(agent: DQNAgent, env: ContinuousGridWorld) -> tuple[list[np.ndarray], float]:
    """Test the trained DQN agent.

    Args:
        agent: Trained DQN agent
        env: Environment

    Returns:
        Tuple of (path, total_reward)
    """
    state = env.reset()
    path = [state.copy()]
    total_reward = 0
    steps = 0
    max_steps = 200

    # Set epsilon to 0 for greedy policy
    original_epsilon = agent.get_epsilon()
    agent.set_epsilon(0.0)

    while steps < max_steps:
        # Get action from agent
        action = agent.get_action(state)

        # Take step in environment
        next_state, reward, done = env.step(state, action)

        # Update state and tracking variables
        state = next_state
        path.append(state.copy())
        total_reward += reward
        steps += 1

        if done:
            break

    # Restore original epsilon
    agent.set_epsilon(original_epsilon)

    return path, total_reward


def plot_training_progress(episode_rewards: list[float]) -> None:
    """Plot training progress.

    Args:
        episode_rewards: List of episode rewards
    """
    plt.figure(figsize=(12, 4))

    # Plot raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.3, color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("DQN Training Progress (Raw)")
    plt.grid(True, alpha=0.3)

    # Plot smoothed rewards
    plt.subplot(1, 2, 2)
    window_size = 50
    if len(episode_rewards) >= window_size:
        smoothed_rewards = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode="valid")
        plt.plot(range(window_size-1, len(episode_rewards)), smoothed_rewards, color="red", linewidth=2)
    else:
        plt.plot(episode_rewards, color="red", linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Smoothed Episode Reward")
    plt.title(f"DQN Training Progress (Smoothed, window={window_size})")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Main demonstration function."""
    print("DQN Continuous Grid World Demo")
    print("=" * 40)

    # Create environment
    env = ContinuousGridWorld()

    print(f"Environment: {env.width}x{env.height} continuous world")
    print(f"State space: {env.state_size}D (x, y coordinates)")
    print(f"Action space: {env.action_size} actions (up, down, left, right)")
    print(f"Start position: {env.start}")
    print(f"Goal position: {env.goal}")
    print(f"Number of obstacles: {len(env.obstacles)}")

    # Create DQN agent with enhanced features
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        hidden_sizes=[64, 64, 32],
        learning_rate=0.001,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        batch_size=64,
        memory_size=10000,
        target_update=100,
        dropout_rate=0.1,
        random_seed=42,
        dqn_variant="double",  # Use Double DQN
        use_huber_loss=True,   # Use Huber loss for robustness
        gradient_clip_norm=1.0,  # Gradient clipping
        tau=0.01,  # Soft target updates
        epsilon_decay_type="linear",  # Linear epsilon decay
        epsilon_decay_steps=400,  # Decay over 400 episodes
    )

    print(f"\nDQN Agent: {agent}")
    print(f"Network architecture: {env.state_size} -> {agent.q_network.network[0].out_features} -> {agent.q_network.network[-1].out_features}")
    print(f"Device: {agent.device}")
    print(f"DQN Variant: {agent.dqn_variant}")
    print(f"Loss Function: {'Huber' if agent.use_huber_loss else 'MSE'}")
    print(f"Target Updates: {'Soft (τ=' + str(agent.tau) + ')' if agent.tau > 0 else 'Hard'}")
    print(f"Epsilon Decay: {agent.epsilon_decay_type} ({agent.epsilon_decay_steps} steps)")

    # Train the agent
    print("\nTraining DQN agent...")
    episode_rewards = train_dqn_agent(agent, env, episodes=500)

    # Test the agent
    print("\nTesting trained agent...")
    path, total_reward = test_dqn_agent(agent, env)

    # Display results
    print("\nResults:")
    print(f"  Final path length: {len(path)} steps")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final epsilon: {agent.get_epsilon():.3f}")
    print(f"  Memory size: {len(agent.memory)}")

    # Plot training progress
    plot_training_progress(episode_rewards)

    # Show Q-values for start state
    start_q_values = agent.get_q_values(env.start)
    action_names = ["Up", "Down", "Left", "Right"]
    print("\nQ-values for start state:")
    for action, q_value in enumerate(start_q_values):
        print(f"  {action_names[action]}: {q_value:.3f}")

    # Render the environment with the learned path
    print("\nRendering environment with learned path...")
    env.render(env.goal, path)

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
