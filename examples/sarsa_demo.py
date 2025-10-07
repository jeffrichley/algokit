#!/usr/bin/env python3
"""SARSA demonstration script with Q-Learning comparison.

This script demonstrates how to use the SARSA agent to solve
a simple grid world environment and compares it with Q-Learning.
"""

import numpy as np

from algokit.algorithms.reinforcement_learning.q_learning import QLearningAgent
from algokit.algorithms.reinforcement_learning.sarsa import SarsaAgent


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


def train_q_learning_agent(agent: QLearningAgent, grid: np.ndarray, episodes: int = 1000) -> list[float]:
    """Train the Q-Learning agent.

    Args:
        agent: Q-Learning agent
        grid: Grid world environment
        episodes: Number of training episodes

    Returns:
        List of episode rewards
    """
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

            # Update Q-values (Q-Learning)
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

        # Print progress every 200 episodes
        if (episode + 1) % 200 == 0:
            avg_reward = np.mean(episode_rewards[-200:])
            print(f"Q-Learning Episode {episode + 1}: Average reward = {avg_reward:.2f}, Epsilon = {agent.get_epsilon():.3f}")

    return episode_rewards


def train_sarsa_agent(agent: SarsaAgent, grid: np.ndarray, episodes: int = 1000) -> list[float]:
    """Train the SARSA agent.

    Args:
        agent: SARSA agent
        grid: Grid world environment
        episodes: Number of training episodes

    Returns:
        List of episode rewards
    """
    episode_rewards = []

    for episode in range(episodes):
        state = 0  # Start state
        total_reward = 0
        steps = 0
        max_steps = 100

        # Choose initial action
        valid_actions = get_valid_actions(state, grid)
        if not valid_actions:
            continue

        action = agent.get_action(state)
        while action not in valid_actions:
            action = agent.get_action(state)

        while steps < max_steps:
            # Take action
            next_state, reward, done = take_action(state, action, grid)

            # Choose next action (only if not done)
            if not done:
                valid_actions = get_valid_actions(next_state, grid)
                if not valid_actions:
                    break

                next_action = agent.get_action(next_state)
                while next_action not in valid_actions:
                    next_action = agent.get_action(next_state)
            else:
                next_action = 0  # Dummy action for terminal state

            # Update Q-values (SARSA)
            agent.update_q_value(state, action, reward, next_state, next_action, done)

            # Update state and tracking variables
            state = next_state
            action = next_action
            total_reward += reward
            steps += 1

            if done:
                break

        # Decay epsilon
        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        # Print progress every 200 episodes
        if (episode + 1) % 200 == 0:
            avg_reward = np.mean(episode_rewards[-200:])
            print(f"SARSA Episode {episode + 1}: Average reward = {avg_reward:.2f}, Epsilon = {agent.get_epsilon():.3f}")

    return episode_rewards


def test_agent(agent, grid: np.ndarray, agent_name: str) -> list[int]:
    """Test the trained agent.

    Args:
        agent: Trained agent (QLearningAgent or SarsaAgent)
        grid: Grid world environment
        agent_name: Name of the agent for display

    Returns:
        List of states in the path
    """
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


def print_grid_with_path(grid: np.ndarray, path: list[int], size: int, agent_name: str) -> None:
    """Print the grid with the agent's path.

    Args:
        grid: Grid world
        path: List of states in the path
        size: Grid size
        agent_name: Name of the agent
    """
    print(f"\n{agent_name} Grid World with Agent Path:")
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


def compare_policies(q_agent: QLearningAgent, sarsa_agent: SarsaAgent, grid: np.ndarray) -> None:
    """Compare the policies learned by Q-Learning and SARSA.

    Args:
        q_agent: Trained Q-Learning agent
        sarsa_agent: Trained SARSA agent
        grid: Grid world environment
    """
    print("\nPolicy Comparison:")
    print("=" * 50)

    size = grid.shape[0]
    action_names = ["Up", "Down", "Left", "Right"]

    print(f"{'State':<8} {'Q-Learning':<12} {'SARSA':<12} {'Difference':<12}")
    print("-" * 50)

    differences = 0
    for state in range(min(10, size * size)):  # Show first 10 states
        q_policy = q_agent.get_policy()[state]
        sarsa_policy = sarsa_agent.get_policy()[state]

        q_action_name = action_names[q_policy]
        sarsa_action_name = action_names[sarsa_policy]

        if q_policy != sarsa_policy:
            differences += 1
            diff_str = f"{q_action_name} vs {sarsa_action_name}"
        else:
            diff_str = "Same"

        print(f"{state:<8} {q_action_name:<12} {sarsa_action_name:<12} {diff_str:<12}")

    print(f"\nTotal policy differences in first 10 states: {differences}")


def main() -> None:
    """Main demonstration function."""
    print("SARSA vs Q-Learning Grid World Comparison")
    print("=" * 50)

    # Create environment
    grid, start_state, goal_state = create_grid_world(5)
    state_space_size = grid.size
    action_space_size = 4  # up, down, left, right

    print(f"Grid size: {grid.shape[0]}x{grid.shape[0]}")
    print(f"State space size: {state_space_size}")
    print(f"Action space size: {action_space_size}")
    print(f"Start state: {start_state}")
    print(f"Goal state: {goal_state}")

    # Create agents with same parameters for fair comparison

    # Q-Learning agent (old style - backwards compatible)
    q_agent = QLearningAgent(
        state_space_size=state_space_size,
        action_space_size=action_space_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        random_seed=42,
    )

    # SARSA agent - demonstrating new config-based approach (recommended)
    # from algokit.algorithms.reinforcement_learning.sarsa import SarsaConfig
    # config = SarsaConfig(
    #     state_space_size=state_space_size,
    #     action_space_size=action_space_size,
    #     learning_rate=0.1,
    #     discount_factor=0.95,
    #     epsilon_start=1.0,
    #     epsilon_decay=0.995,
    #     epsilon_end=0.01,
    #     random_seed=42,
    # )
    # sarsa_agent = SarsaAgent(config=config)

    # SARSA agent (old style - still works for backwards compatibility)
    sarsa_agent = SarsaAgent(
        state_space_size=state_space_size,
        action_space_size=action_space_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        random_seed=42,
    )

    print(f"\nQ-Learning Agent: {q_agent}")
    print(f"SARSA Agent: {sarsa_agent}")

    # Train both agents
    print("\nTraining Q-Learning agent...")
    q_rewards = train_q_learning_agent(q_agent, grid, episodes=1000)

    print("\nTraining SARSA agent...")
    sarsa_rewards = train_sarsa_agent(sarsa_agent, grid, episodes=1000)

    # Test both agents
    print("\nTesting trained agents...")
    q_path = test_agent(q_agent, grid, "Q-Learning")
    sarsa_path = test_agent(sarsa_agent, grid, "SARSA")

    # Display results
    print("\nQ-Learning Results:")
    print(f"  Final path length: {len(q_path)} steps")
    print(f"  Final average reward: {np.mean(q_rewards[-100:]):.2f}")

    print("\nSARSA Results:")
    print(f"  Final path length: {len(sarsa_path)} steps")
    print(f"  Final average reward: {np.mean(sarsa_rewards[-100:]):.2f}")

    print_grid_with_path(grid, q_path, grid.shape[0], "Q-Learning")
    print_grid_with_path(grid, sarsa_path, grid.shape[0], "SARSA")

    # Compare policies
    compare_policies(q_agent, sarsa_agent, grid)

    # Show final Q-values for start state
    print("\nQ-values for start state (0):")
    action_names = ["Up", "Down", "Left", "Right"]
    print(f"{'Action':<8} {'Q-Learning':<12} {'SARSA':<12}")
    print("-" * 32)
    for action in range(action_space_size):
        q_value = q_agent.get_q_value(0, action)
        sarsa_value = sarsa_agent.get_q_value(0, action)
        print(f"{action_names[action]:<8} {q_value:<12.3f} {sarsa_value:<12.3f}")

    print("\nComparison completed successfully!")


if __name__ == "__main__":
    main()
