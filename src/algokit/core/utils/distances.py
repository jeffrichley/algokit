"""Distance calculation utilities for algorithms.

This module provides common distance calculation functions used across
various algorithm implementations, particularly for pathfinding and
graph traversal algorithms.
"""

import math
from collections.abc import Callable
from typing import TypeVar

Node = TypeVar("Node")


def manhattan_distance(node1: tuple[int, int], node2: tuple[int, int]) -> float:
    """Calculate Manhattan distance between two grid nodes.

    Manhattan distance (also known as L1 distance or taxicab distance) is the
    sum of the absolute differences of their coordinates. It's commonly used
    for grid-based pathfinding where movement is restricted to cardinal directions.

    Args:
        node1: First node as (x, y) coordinates
        node2: Second node as (x, y) coordinates

    Returns:
        Manhattan distance between the nodes

    Raises:
        TypeError: If nodes don't have numeric coordinates

    Example:
        >>> manhattan_distance((0, 0), (3, 4))
        7.0
        >>> manhattan_distance((1, 1), (1, 1))
        0.0
        >>> manhattan_distance((0, 0), (0, 5))
        5.0
    """
    return float(abs(node1[0] - node2[0]) + abs(node1[1] - node2[1]))


def euclidean_distance(node1: tuple[float, float], node2: tuple[float, float]) -> float:
    """Calculate Euclidean distance between two nodes.

    Euclidean distance (also known as L2 distance) is the straight-line distance
    between two points in Euclidean space. It's commonly used for continuous
    spaces or when diagonal movement is allowed.

    Args:
        node1: First node as (x, y) coordinates
        node2: Second node as (x, y) coordinates

    Returns:
        Euclidean distance between the nodes

    Raises:
        TypeError: If nodes don't have numeric coordinates

    Example:
        >>> euclidean_distance((0, 0), (3, 4))
        5.0
        >>> euclidean_distance((1, 1), (1, 1))
        0.0
        >>> euclidean_distance((0, 0), (1, 1))
        1.4142135623730951
    """
    return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)


def chebyshev_distance(node1: tuple[int, int], node2: tuple[int, int]) -> float:
    """Calculate Chebyshev distance between two grid nodes.

    Chebyshev distance (also known as L∞ distance or maximum metric) is the
    maximum of the absolute differences of their coordinates. It's commonly
    used for grid-based pathfinding where 8-directional movement is allowed.

    Args:
        node1: First node as (x, y) coordinates
        node2: Second node as (x, y) coordinates

    Returns:
        Chebyshev distance between the nodes

    Raises:
        TypeError: If nodes don't have numeric coordinates

    Example:
        >>> chebyshev_distance((0, 0), (3, 4))
        4.0
        >>> chebyshev_distance((1, 1), (1, 1))
        0.0
        >>> chebyshev_distance((0, 0), (1, 1))
        1.0
    """
    return float(max(abs(node1[0] - node2[0]), abs(node1[1] - node2[1])))


def zero_heuristic(node: Node, goal: Node) -> float:
    """Zero heuristic function for A* algorithm.

    This heuristic always returns 0, making the A* algorithm equivalent
    to Dijkstra's algorithm. It's useful when no domain-specific heuristic
    is available or when you want guaranteed optimality.

    Args:
        node: Current node (ignored)
        goal: Goal node (ignored)

    Returns:
        Always returns 0.0

    Example:
        >>> zero_heuristic("A", "B")
        0.0
        >>> zero_heuristic((0, 0), (3, 4))
        0.0
    """
    return 0.0


def create_manhattan_heuristic(
    goal: tuple[int, int],
) -> Callable[[tuple[int, int]], float]:
    """Create a Manhattan distance heuristic function for a specific goal.

    This factory function creates a heuristic function that calculates
    Manhattan distance to a fixed goal. Useful when the goal is known
    at creation time and you want to avoid passing it repeatedly.

    Args:
        goal: The target goal node as (x, y) coordinates

    Returns:
        A heuristic function that takes a node and returns Manhattan distance to goal

    Example:
        >>> heuristic = create_manhattan_heuristic((3, 4))
        >>> heuristic((0, 0))
        7.0
        >>> heuristic((3, 4))
        0.0
    """

    def heuristic(node: tuple[int, int]) -> float:
        return manhattan_distance(node, goal)

    return heuristic


def create_euclidean_heuristic(
    goal: tuple[float, float],
) -> Callable[[tuple[float, float]], float]:
    """Create a Euclidean distance heuristic function for a specific goal.

    This factory function creates a heuristic function that calculates
    Euclidean distance to a fixed goal. Useful when the goal is known
    at creation time and you want to avoid passing it repeatedly.

    Args:
        goal: The target goal node as (x, y) coordinates

    Returns:
        A heuristic function that takes a node and returns Euclidean distance to goal

    Example:
        >>> heuristic = create_euclidean_heuristic((3.0, 4.0))
        >>> heuristic((0.0, 0.0))
        5.0
        >>> heuristic((3.0, 4.0))
        0.0
    """

    def heuristic(node: tuple[float, float]) -> float:
        return euclidean_distance(node, goal)

    return heuristic
