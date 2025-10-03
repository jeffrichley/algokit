#!/usr/bin/env python3
"""Q-Learning demonstration script.

This script demonstrates how to use the Q-Learning agent to solve
a simple grid world environment.
"""

import numpy as np
from algokit.algorithms.reinforcement_learning.q_learning import QLearningAgent


def create_grid_world(size: int = 5) -> tuple[np.ndarray, int, int]:
    """Create a simple grid world environment.

    Args:
        size: Size of the grid (size x size)

    Returns:
        Tuple of (grid, start_state, goal_state)
    """
    grid = np.zeros((size, size))
    start_state = 0  # Top-left corner
    goal_state = size * size - 1  # Bottom-right corner

    # Add some obstacles (represented as -1)
    if size >= 5:
        grid[1, 1] = -1  # Obstacle
        grid[2, 3] = -1  # Obstacle
        grid[3, 1] = -1  # Obstacle

    return grid, start_state, goal_state


def state_to_position(state: int, size: int) -> tuple[int, int]:
    """Convert state number to grid position.

    Args:
        state: State number
        size: Grid size

    Returns:
        Tuple of (row, col)
    """
    return state // size, state % size


def position_to_state(row: int, col: int, size: int) -> int:
    """Convert grid position to state number.

    Args:
        row: Row position
        col: Column position
        size: Grid size

    Returns:
        State number
    """
    return row * size + col


def get_valid_actions(state: int, grid: np.ndarray) -> list[int]:
    """Get valid actions for a given state.

    Args:
        state: Current state
        grid: Grid world

    Returns:
        List of valid action indices (0=up, 1=down, 2=left, 3=right)
    """
    size = grid.shape[0]
    row, col = state_to_position(state, size)
    valid_actions = []

    # Check each direction
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    for i, (dr, dc) in enumerate(directions):
        new_row, new_col = row + dr, col + dc

        # Check if new position is valid and not an obstacle
        if (0 <= new_row < size and
            0 <= new_col < size and
            grid[new_row, new_col] != -1):
            valid_actions.append(i)

    return valid_actions


def take_action(state: int, action: int, grid: np.ndarray) -> tuple[int, float, bool]:
    """Take an action in the environment.

    Args:
        state: Current state
        action: Action to take
        grid: Grid world

    Returns:
        Tuple of (new_state, reward, done)
    """
    size = grid.shape[0]
    row, col = state_to_position(state, size)

    # Action mappings
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    if action < len(directions):
        dr, dc = directions[action]
        new_row, new_col = row + dr, col + dc

        # Check if action is valid
        if (0 <= new_row < size and
            0 <= new_col < size and
            grid[new_row, new_col] != -1):

            new_state = position_to_state(new_row, new_col, size)

            # Check if reached goal
            if new_state == size * size - 1:
                return new_state, 100.0, True
            else:
                return new_state, -0.1, False  # Small negative reward for each step

    # Invalid action - stay in same state with penalty
    return state, -1.0, False


def train_agent(agent: QLearningAgent, grid: np.ndarray, episodes: int = 1000) -> list[float]:
    """Train the Q-Learning agent.

    Args:
        agent: Q-Learning agent
        grid: Grid world environment
        episodes: Number of training episodes

    Returns:
        List of episode rewards
    """
    size = grid.shape[0]
    episode_rewards = []

    for episode in range(episodes):
        state = 0  # Start state
        total_reward = 0
        steps = 0
        max_steps = 100

        while steps < max_steps:
            # Get valid actions
            valid_actions = get_valid_actions(state, grid)

            if not valid_actions:
                break

            # Choose action (only from valid actions)
            action = agent.get_action(state)
            while action not in valid_actions:
                action = agent.get_action(state)

            # Take action
            next_state, reward, done = take_action(state, action, grid)

            # Update Q-values
            agent.update_q_value(state, action, reward, next_state, done)

            # Update state and tracking variables
            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        # Decay epsilon
        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}: Average reward = {avg_reward:.2f}, Epsilon = {agent.get_epsilon():.3f}")

    return episode_rewards


def test_agent(agent: QLearningAgent, grid: np.ndarray) -> list[int]:
    """Test the trained agent.

    Args:
        agent: Trained Q-Learning agent
        grid: Grid world environment

    Returns:
        List of states in the path
    """
    size = grid.shape[0]
    state = 0  # Start state
    path = [state]
    steps = 0
    max_steps = 50

    # Set epsilon to 0 for greedy policy
    original_epsilon = agent.get_epsilon()
    agent.set_epsilon(0.0)

    while steps < max_steps:
        # Get valid actions
        valid_actions = get_valid_actions(state, grid)

        if not valid_actions:
            break

        # Choose best action (only from valid actions)
        action = agent.get_action(state)
        while action not in valid_actions:
            action = agent.get_action(state)

        # Take action
        next_state, reward, done = take_action(state, action, grid)

        state = next_state
        path.append(state)
        steps += 1

        if done:
            break

    # Restore original epsilon
    agent.set_epsilon(original_epsilon)

    return path


def print_grid_with_path(grid: np.ndarray, path: list[int], size: int) -> None:
    """Print the grid with the agent's path.

    Args:
        grid: Grid world
        path: List of states in the path
        size: Grid size
    """
    print("\nGrid World with Agent Path:")
    print("S = Start, G = Goal, X = Obstacle, * = Path")
    print("-" * (size * 2 + 1))

    for row in range(size):
        line = "|"
        for col in range(size):
            state = position_to_state(row, col, size)

            if state == 0:  # Start
                char = "S"
            elif state == size * size - 1:  # Goal
                char = "G"
            elif grid[row, col] == -1:  # Obstacle
                char = "X"
            elif state in path:  # Path
                char = "*"
            else:  # Empty
                char = " "

            line += f"{char}|"
        print(line)
        print("-" * (size * 2 + 1))


def main() -> None:
    """Main demonstration function."""
    print("Q-Learning Grid World Demo")
    print("=" * 40)

    # Create environment
    grid, start_state, goal_state = create_grid_world(5)
    state_space_size = grid.size
    action_space_size = 4  # up, down, left, right

    print(f"Grid size: {grid.shape[0]}x{grid.shape[0]}")
    print(f"State space size: {state_space_size}")
    print(f"Action space size: {action_space_size}")
    print(f"Start state: {start_state}")
    print(f"Goal state: {goal_state}")

    # Create Q-Learning agent
    agent = QLearningAgent(
        state_space_size=state_space_size,
        action_space_size=action_space_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        random_seed=42,
    )

    print(f"\nAgent created: {agent}")

    # Train the agent
    print("\nTraining agent...")
    episode_rewards = train_agent(agent, grid, episodes=1000)

    # Test the agent
    print("\nTesting trained agent...")
    path = test_agent(agent, grid)

    # Display results
    print(f"\nFinal path length: {len(path)} steps")
    print(f"Path: {path}")

    print_grid_with_path(grid, path, grid.shape[0])

    # Show Q-values for start state
    print(f"\nQ-values for start state (0):")
    for action in range(action_space_size):
        q_value = agent.get_q_value(0, action)
        action_name = ["Up", "Down", "Left", "Right"][action]
        print(f"  {action_name}: {q_value:.3f}")

    # Show final policy
    policy = agent.get_policy()
    print(f"\nFinal policy (first 10 states): {policy[:10]}")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
