#!/usr/bin/env python3
"""M* Multi-Robot Path Planning Demo.

This script demonstrates the M* algorithm for multi-robot path planning.
It shows how multiple robots can plan collision-free paths in a shared environment.
"""

import matplotlib.pyplot as plt
import networkx as nx

from algokit.algorithms.pathfinding.mstar import mstar_plan_paths
from algokit.core.helpers import create_grid_graph
from algokit.core.utils.distances import manhattan_distance


def print_ascii_visualization(
    graph: nx.Graph,
    paths: dict[str, list[tuple[int, int]]],
    title: str,
) -> None:
    """Print ASCII visualization of the graph and robot paths.

    Args:
        graph: NetworkX graph to visualize
        paths: Dictionary of robot paths
        title: Title for the visualization
    """
    print(f"\n{title} - ASCII Visualization")
    print("=" * 50)

    # Get the bounds of the graph
    nodes = list(graph.nodes())
    if not nodes:
        print("Empty graph")
        return

    max_x = max(node[0] for node in nodes)
    max_y = max(node[1] for node in nodes)
    min_x = min(node[0] for node in nodes)
    min_y = min(node[1] for node in nodes)

    # Create a grid representation
    grid = {}
    for y in range(max_y, min_y - 1, -1):  # Reverse y for display
        for x in range(min_x, max_x + 1):
            grid[(x, y)] = "."

    # Mark nodes that exist in the graph
    for node in nodes:
        x, y = node
        grid[(x, y)] = "O"

    # Mark robot paths
    colors = ["1", "2", "3", "4", "5"]
    for i, (_robot_id, path) in enumerate(paths.items()):
        color = colors[i % len(colors)]

        # Mark start position
        start = path[0]
        grid[start] = f"S{color}"

        # Mark goal position
        goal = path[-1]
        grid[goal] = f"G{color}"

        # Mark path
        for pos in path[1:-1]:
            if grid[pos] == "O":
                grid[pos] = color

    # Print the grid
    print("Legend: S1/G1=Robot1 Start/Goal, S2/G2=Robot2 Start/Goal, 1/2=Robot1/2 Path, O=Node, .=Empty")
    print()

    for y in range(max_y, min_y - 1, -1):
        row = ""
        for x in range(min_x, max_x + 1):
            row += grid.get((x, y), " ")
        print(f"{y:2d} |{row}|")

    # Print x-axis labels
    x_labels = "    "
    for x in range(min_x, max_x + 1):
        x_labels += str(x)
    print(x_labels)


def visualize_graph_and_paths(
    graph: nx.Graph,
    paths: dict[str, list[tuple[int, int]]],
    title: str,
    show_graph: bool = True,
) -> None:
    """Visualize the graph and robot paths.

    Args:
        graph: NetworkX graph to visualize
        paths: Dictionary of robot paths
        title: Title for the plot
        show_graph: Whether to show the graph (default: True)
    """
    if not show_graph:
        return

    # Always print ASCII visualization first
    print_ascii_visualization(graph, paths, title)

    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    except Exception as e:
        print(f"Could not create matplotlib visualization: {e}")
        print("ASCII visualization shown above.")
        return

    # Plot 1: Graph structure
    ax1.set_title(f"{title} - Graph Structure")
    ax1.set_aspect("equal")

    # Draw nodes with different colors for different areas
    pos = {node: node for node in graph.nodes()}

    # Color nodes based on position
    node_colors = []
    for node in graph.nodes():
        x, y = node
        if x < 3:  # Left room
            node_colors.append("lightblue")
        elif x > 5:  # Right room
            node_colors.append("lightgreen")
        else:  # Hallway
            node_colors.append("yellow")

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=300, ax=ax1)
    nx.draw_networkx_edges(graph, pos, edge_color="gray", ax=ax1)
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax1)

    # Add legend
    ax1.scatter([], [], c="lightblue", label="Left Room", s=100)
    ax1.scatter([], [], c="yellow", label="Hallway", s=100)
    ax1.scatter([], [], c="lightgreen", label="Right Room", s=100)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Robot paths over time
    ax2.set_title(f"{title} - Robot Paths")
    ax2.set_aspect("equal")

    # Draw the graph background
    nx.draw_networkx_nodes(graph, pos, node_color="lightgray", node_size=200, ax=ax2)
    nx.draw_networkx_edges(graph, pos, edge_color="lightgray", alpha=0.5, ax=ax2)

    # Draw robot paths with different colors
    colors = ["red", "blue", "green", "purple", "orange"]
    for i, (robot_id, path) in enumerate(paths.items()):
        color = colors[i % len(colors)]

        # Draw path as a line
        path_x = [pos[node][0] for node in path]
        path_y = [pos[node][1] for node in path]
        ax2.plot(path_x, path_y, color=color, linewidth=3, alpha=0.7,
                marker="o", markersize=6, label=f"{robot_id}")

        # Mark start and end positions
        start_pos = pos[path[0]]
        end_pos = pos[path[-1]]
        ax2.scatter(start_pos[0], start_pos[1], color=color, s=200,
                   marker="s", edgecolor="black", linewidth=2, label=f"{robot_id} Start")
        ax2.scatter(end_pos[0], end_pos[1], color=color, s=200,
                   marker="^", edgecolor="black", linewidth=2, label=f"{robot_id} Goal")

    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def demo_single_robot() -> None:
    """Demonstrate M* with a single robot (equivalent to A*)."""
    print("=== Single Robot Demo ===")

    # Create a 4x4 grid
    graph = create_grid_graph(4, 4)

    # Single robot planning
    start = (0, 0)
    goal = (3, 3)

    print(f"Planning path from {start} to {goal}")
    result = mstar_plan_paths(graph, {"robot1": start}, {"robot1": goal})

    if result:
        path = result["robot1"]
        print(f"Path found: {path}")
        print(f"Path length: {len(path)} steps")
    else:
        print("No path found!")


def demo_two_robots() -> None:
    """Demonstrate M* with two robots."""
    print("\n=== Two Robot Demo ===")

    # Create a 5x5 grid
    graph = create_grid_graph(5, 5)

    # Two robots with crossing paths
    starts = {"robot1": (0, 0), "robot2": (4, 4)}
    goals = {"robot1": (4, 0), "robot2": (0, 4)}

    print(f"Robot 1: {starts['robot1']} → {goals['robot1']}")
    print(f"Robot 2: {starts['robot2']} → {goals['robot2']}")

    paths = mstar_plan_paths(graph, starts, goals)

    if paths:
        print("\nPaths found:")
        for robot_id, path in paths.items():
            print(f"{robot_id}: {path}")
            print(f"  Length: {len(path)} steps")

        # Check for collisions
        print("\nCollision check:")
        path1 = paths["robot1"]
        path2 = paths["robot2"]

        max_length = max(len(path1), len(path2))
        collision_found = False

        for t in range(max_length):
            pos1 = path1[min(t, len(path1) - 1)]
            pos2 = path2[min(t, len(path2) - 1)]

            # Manhattan distance
            distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

            if distance < 1.0:
                print(f"  Collision at time {t}: {pos1} and {pos2}")
                collision_found = True

        if not collision_found:
            print("  No collisions detected!")

        # Visualize the simple two-robot scenario
        visualize_graph_and_paths(graph, paths, "Two Robot Scenario")
    else:
        print("No solution found!")


def demo_three_robots() -> None:
    """Demonstrate M* with three robots."""
    print("\n=== Three Robot Demo ===")

    # Create a 6x6 grid
    graph = create_grid_graph(6, 6)

    # Three robots with complex paths
    starts = {"robot1": (0, 0), "robot2": (5, 0), "robot3": (0, 5)}
    goals = {"robot1": (5, 5), "robot2": (0, 5), "robot3": (5, 0)}

    print(f"Robot 1: {starts['robot1']} → {goals['robot1']}")
    print(f"Robot 2: {starts['robot2']} → {goals['robot2']}")
    print(f"Robot 3: {starts['robot3']} → {goals['robot3']}")

    paths = mstar_plan_paths(graph, starts, goals)

    if paths:
        print("\nPaths found:")
        for robot_id, path in paths.items():
            print(f"{robot_id}: {path}")
            print(f"  Length: {len(path)} steps")

        # Visualize the three-robot scenario
        visualize_graph_and_paths(graph, paths, "Three Robot Scenario")
    else:
        print("No solution found!")


def demo_hallway_scenario() -> None:
    """Demonstrate M* with narrow hallway requiring coordination."""
    print("\n=== Hallway Scenario Demo ===")

    import networkx as nx

    # Create hallway structure: wide areas on sides, narrow corridor in middle
    graph = nx.Graph()

    # Layout: [0,0] to [2,2] = left room, [3,1] to [5,1] = single-file hallway, [6,0] to [8,2] = right room
    # Blocked areas: [3,0], [4,0], [5,0], [3,2], [4,2], [5,2]

    # Add nodes - exclude blocked areas
    for x in range(9):
        for y in range(3):
            # Block the top and bottom rows in the middle section
            if (y == 0 or y == 2) and 3 <= x <= 5:
                continue  # Skip blocked positions
            graph.add_node((x, y))

    # Add edges - full connectivity in rooms, single-file in hallway
    for x in range(9):
        for y in range(3):
            current = (x, y)

            # Skip if this position is blocked
            if (y == 0 or y == 2) and 3 <= x <= 5:
                continue

            # Horizontal connections
            if x < 8:
                right = (x + 1, y)
                # Only connect if the right position is not blocked
                if not ((y == 0 or y == 2) and 3 <= (x + 1) <= 5):
                    graph.add_edge(current, right, weight=1.0)

            # Vertical connections
            if y < 2:
                down = (x, y + 1)
                # Only connect if the down position is not blocked
                if not ((y + 1 == 0 or y + 1 == 2) and 3 <= x <= 5):
                    graph.add_edge(current, down, weight=1.0)

            # Diagonal connections in wide areas only (not in hallway)
            if x < 2 or x > 5:  # Wide areas only
                if x < 8 and y < 2:
                    diagonal = (x + 1, y + 1)
                    # Only connect if diagonal position is not blocked
                    if not ((y + 1 == 0 or y + 1 == 2) and 3 <= (x + 1) <= 5):
                        graph.add_edge(current, diagonal, weight=1.414)  # sqrt(2)
                if x < 8 and y > 0:
                    diagonal = (x + 1, y - 1)
                    # Only connect if diagonal position is not blocked
                    if not ((y - 1 == 0 or y - 1 == 2) and 3 <= (x + 1) <= 5):
                        graph.add_edge(current, diagonal, weight=1.414)

    # Robot scenario: one starts on left, one on right, need to swap positions
    starts = {"robot1": (1, 1), "robot2": (7, 1)}  # Start in wide areas
    goals = {"robot1": (7, 1), "robot2": (1, 1)}   # Swap positions through narrow hallway

    print("Single-File Hallway Layout:")
    print("L L L | | | | | R R R")
    print("L L L | H H H | R R R")
    print("L L L | | | | | R R R")
    print("L=Left Room, H=Single-File Hallway, R=Right Room, |=Blocked")
    print(f"Robot 1: {starts['robot1']} → {goals['robot1']}")
    print(f"Robot 2: {starts['robot2']} → {goals['robot2']}")

    paths = mstar_plan_paths(graph, starts, goals)

    if paths:
        print("\nPaths found:")
        for robot_id, path in paths.items():
            print(f"{robot_id}: {path}")
            print(f"  Length: {len(path)} steps")

        # Analyze hallway usage
        path1 = paths["robot1"]
        path2 = paths["robot2"]

        print("\nSingle-file hallway coordination analysis:")
        hallway_usage1 = [pos for pos in path1 if 3 <= pos[0] <= 5 and pos[1] == 1]
        hallway_usage2 = [pos for pos in path2 if 3 <= pos[0] <= 5 and pos[1] == 1]

        print(f"Robot 1 hallway positions: {hallway_usage1}")
        print(f"Robot 2 hallway positions: {hallway_usage2}")

        # Check for conflicts in the single-file hallway
        max_length = max(len(path1), len(path2))
        conflicts = []

        for t in range(max_length):
            pos1 = path1[min(t, len(path1) - 1)]
            pos2 = path2[min(t, len(path2) - 1)]

            # Check if both robots in single-file hallway at same time
            if (3 <= pos1[0] <= 5 and pos1[1] == 1 and
                3 <= pos2[0] <= 5 and pos2[1] == 1):
                distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                if distance < 1.0:
                    conflicts.append(f"Time {t}: {pos1} vs {pos2}")

        if conflicts:
            print(f"Conflicts found: {conflicts}")
        else:
            print("No conflicts detected - robots coordinated successfully!")

        # Visualize the graph and paths
        visualize_graph_and_paths(graph, paths, "Single-File Hallway Scenario")
    else:
        print("No solution found!")


def demo_collision_avoidance() -> None:
    """Demonstrate collision avoidance with different collision radii."""
    print("\n=== Collision Avoidance Demo ===")

    # Create a 4x4 grid
    graph = create_grid_graph(4, 4)

    # Two robots with potential collision
    starts = {"robot1": (0, 0), "robot2": (1, 0)}
    goals = {"robot1": (3, 0), "robot2": (2, 3)}

    print(f"Robot 1: {starts['robot1']} → {goals['robot1']}")
    print(f"Robot 2: {starts['robot2']} → {goals['robot2']}")

    # Test with different collision radii
    for radius in [0.5, 1.0, 2.0]:
        print(f"\nCollision radius: {radius}")
        paths = mstar_plan_paths(graph, starts, goals, collision_radius=radius)

        if paths:
            print("  Solution found!")
            # Check minimum distance between robots
            path1 = paths["robot1"]
            path2 = paths["robot2"]

            min_distance = float("inf")
            max_length = max(len(path1), len(path2))

            for t in range(max_length):
                pos1 = path1[min(t, len(path1) - 1)]
                pos2 = path2[min(t, len(path2) - 1)]

                distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                min_distance = min(min_distance, distance)

            print(f"  Minimum distance between robots: {min_distance}")
        else:
            print("  No solution found!")


def main() -> None:
    """Run all M* demonstrations."""
    import sys

    # Check if visualization should be shown
    show_visualization = "--show-graph" in sys.argv or "-v" in sys.argv

    if show_visualization:
        print("M* Multi-Robot Path Planning Algorithm Demo (with visualizations)")
    else:
        print("M* Multi-Robot Path Planning Algorithm Demo (text only)")
        print("Use --show-graph or -v to enable visualizations")

    print("=" * 50)

    # Temporarily override the visualization function if not showing graphs
    if not show_visualization:
        global visualize_graph_and_paths
        def no_visualization(*args, **kwargs):
            """No-op visualization function."""
            pass
        visualize_graph_and_paths = no_visualization

    demo_single_robot()
    demo_two_robots()
    demo_three_robots()
    demo_hallway_scenario()
    demo_collision_avoidance()

    print("\n" + "=" * 50)
    print("Demo completed!")

    if show_visualization:
        print("\nVisualization windows opened. Close them to exit.")


if __name__ == "__main__":
    main()
