#!/usr/bin/env python3
"""Demonstration of the shared distance utilities.

This script shows how the distance functions are now shared across the codebase
and can be used in various contexts beyond just A* pathfinding.
"""

import networkx as nx

from algokit.algorithms.pathfinding import astar_shortest_path
from algokit.core.utils.distances import (
    chebyshev_distance,
    create_euclidean_heuristic,
    create_manhattan_heuristic,
    euclidean_distance,
    manhattan_distance,
    zero_heuristic,
)
from algokit.core.helpers import create_grid_graph


def demonstrate_distance_calculations() -> None:
    """Demonstrate various distance calculations."""
    print("Distance Calculation Examples")
    print("=" * 40)

    # Test points
    point1 = (0, 0)
    point2 = (3, 4)

    print(f"Points: {point1} to {point2}")
    print(f"Manhattan distance: {manhattan_distance(point1, point2)}")
    print(f"Euclidean distance: {euclidean_distance(point1, point2)}")
    print(f"Chebyshev distance: {chebyshev_distance(point1, point2)}")
    print()


def demonstrate_heuristic_factories() -> None:
    """Demonstrate heuristic factory functions."""
    print("Heuristic Factory Functions")
    print("=" * 30)

    goal = (5, 5)
    test_points = [(0, 0), (2, 3), (5, 5), (7, 8)]

    # Create heuristics for the goal
    manhattan_h = create_manhattan_heuristic(goal)
    euclidean_h = create_euclidean_heuristic(goal)

    print(f"Goal: {goal}")
    print("Point\t\tManhattan\tEuclidean")
    print("-" * 40)

    for point in test_points:
        manhattan_dist = manhattan_h(point)
        euclidean_dist = euclidean_h(point)
        print(f"{point}\t\t{manhattan_dist}\t\t{euclidean_dist:.2f}")

    print()


def demonstrate_astar_with_different_heuristics() -> None:
    """Demonstrate A* with different heuristics using shared utilities."""
    print("A* Algorithm with Shared Distance Utilities")
    print("=" * 45)

    # Create a grid with obstacles
    obstacles = {(1, 1), (2, 1), (3, 3)}
    graph = create_grid_graph(5, 5, blocked=obstacles)

    start = (0, 0)
    goal = (4, 4)

    print(f"Grid: 5x5 with obstacles at {obstacles}")
    print(f"Start: {start}, Goal: {goal}")
    print()

    # Test different heuristics
    heuristics = [
        ("Zero Heuristic", zero_heuristic),
        ("Manhattan Distance", manhattan_distance),
        ("Euclidean Distance", euclidean_distance),
    ]

    for name, heuristic in heuristics:
        print(f"Using {name}:")
        result = astar_shortest_path(graph, start, goal, heuristic)

        if result:
            path, cost = result
            print(f"  Path length: {len(path)} nodes")
            print(f"  Total cost: {cost}")
            print(f"  Path: {path[:3]}...{path[-3:] if len(path) > 6 else path[3:]}")
        else:
            print("  No path found!")
        print()


def demonstrate_distance_properties() -> None:
    """Demonstrate mathematical properties of distance functions."""
    print("Distance Function Properties")
    print("=" * 30)

    # Test symmetry
    point1 = (1, 2)
    point2 = (4, 6)

    print("Symmetry Test:")
    print(f"manhattan_distance({point1}, {point2}) = {manhattan_distance(point1, point2)}")
    print(f"manhattan_distance({point2}, {point1}) = {manhattan_distance(point2, point1)}")
    print(f"Symmetric: {manhattan_distance(point1, point2) == manhattan_distance(point2, point1)}")
    print()

    # Test triangle inequality
    a, b, c = (0, 0), (2, 3), (4, 6)
    print("Triangle Inequality Test:")
    ab = euclidean_distance(a, b)
    bc = euclidean_distance(b, c)
    ac = euclidean_distance(a, c)
    print(f"Distance A->B: {ab:.2f}")
    print(f"Distance B->C: {bc:.2f}")
    print(f"Distance A->C: {ac:.2f}")
    print(f"Triangle inequality holds: {ac <= ab + bc}")
    print()

    # Test distance relationships
    print("Distance Relationships:")
    manhattan = manhattan_distance(point1, point2)
    euclidean = euclidean_distance(point1, point2)
    chebyshev = chebyshev_distance(point1, point2)

    print(f"Manhattan: {manhattan}")
    print(f"Euclidean: {euclidean:.2f}")
    print(f"Chebyshev: {chebyshev}")
    print(f"Chebyshev ≤ Euclidean ≤ Manhattan: {chebyshev <= euclidean <= manhattan}")


def demonstrate_reusable_heuristics() -> None:
    """Demonstrate how heuristics can be reused across different searches."""
    print("Reusable Heuristics Example")
    print("=" * 30)

    # Create multiple goals and show how heuristics can be reused
    goals = [(2, 2), (4, 1), (1, 4)]
    test_point = (0, 0)

    print(f"Test point: {test_point}")
    print("Goal\t\tManhattan\tEuclidean")
    print("-" * 40)

    for goal in goals:
        manhattan_h = create_manhattan_heuristic(goal)
        euclidean_h = create_euclidean_heuristic(goal)

        manhattan_dist = manhattan_h(test_point)
        euclidean_dist = euclidean_h(test_point)

        print(f"{goal}\t\t{manhattan_dist}\t\t{euclidean_dist:.2f}")

    print()


def main() -> None:
    """Run all demonstrations."""
    print("Shared Distance Utilities Demonstration")
    print("=" * 50)
    print()

    demonstrate_distance_calculations()
    demonstrate_heuristic_factories()
    demonstrate_astar_with_different_heuristics()
    demonstrate_distance_properties()
    demonstrate_reusable_heuristics()

    print("Benefits of Shared Distance Utilities:")
    print("-" * 40)
    print("✓ DRY principle: No code duplication")
    print("✓ Consistent behavior across modules")
    print("✓ Easy to extend with new distance functions")
    print("✓ Centralized testing and maintenance")
    print("✓ Reusable across different algorithms")


if __name__ == "__main__":
    main()
