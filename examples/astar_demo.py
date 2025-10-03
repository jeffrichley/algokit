#!/usr/bin/env python3
"""Demonstration of the A* pathfinding algorithm.

This script shows how to use the A* algorithm with different heuristics
to find optimal paths in grid graphs.
"""

import networkx as nx

from algokit.algorithms.pathfinding import astar_shortest_path
from algokit.core.utils.distances import (
    euclidean_distance,
    manhattan_distance,
    zero_heuristic,
)
from algokit.core.helpers import create_grid_graph


def main() -> None:
    """Demonstrate A* algorithm with different heuristics."""
    print("A* Pathfinding Algorithm Demonstration")
    print("=" * 50)

    # Create a 5x5 grid with some obstacles
    obstacles = {(1, 1), (2, 1), (3, 3), (1, 3)}
    graph = create_grid_graph(5, 5, blocked=obstacles)

    start = (0, 0)
    goal = (4, 4)

    print(f"Grid: 5x5 with obstacles at {obstacles}")
    print(f"Start: {start}, Goal: {goal}")
    print()

    # Test different heuristics
    heuristics = [
        ("Zero Heuristic (Dijkstra)", zero_heuristic),
        ("Manhattan Distance", manhattan_distance),
        ("Euclidean Distance", euclidean_distance),
    ]

    for name, heuristic in heuristics:
        print(f"Using {name}:")
        result = astar_shortest_path(graph, start, goal, heuristic)

        if result:
            path, cost = result
            print(f"  Path found: {path}")
            print(f"  Path length: {len(path)} nodes")
            print(f"  Total cost: {cost}")
            print(f"  Path visualization:")

            # Create a simple grid visualization
            grid = [["." for _ in range(5)] for _ in range(5)]

            # Mark obstacles
            for obs in obstacles:
                grid[obs[1]][obs[0]] = "#"

            # Mark path
            for node in path:
                if node not in obstacles:
                    grid[node[1]][node[0]] = "*"

            # Mark start and goal
            grid[start[1]][start[0]] = "S"
            grid[goal[1]][goal[0]] = "G"

            # Print grid (y-axis flipped for display)
            for row in reversed(grid):
                print(f"    {' '.join(row)}")
        else:
            print("  No path found!")

        print()

    # Compare with weighted graph
    print("Weighted Graph Example:")
    print("-" * 30)

    # Create a weighted graph where direct path is more expensive
    weighted_graph = nx.Graph()
    weighted_graph.add_edges_from([
        ("A", "B", {"weight": 1.0}),
        ("B", "C", {"weight": 1.0}),
        ("A", "C", {"weight": 5.0}),  # Direct path is expensive
    ])

    print("Graph: A --1-- B --1-- C")
    print("       |              |")
    print("       +-------5------+")
    print()

    result = astar_shortest_path(weighted_graph, "A", "C", zero_heuristic)
    if result:
        path, cost = result
        print(f"Optimal path: {path}")
        print(f"Total cost: {cost}")
        print("A* correctly chooses the longer path with lower total cost!")


if __name__ == "__main__":
    main()
