"""Breadth-First Search (BFS) pathfinding algorithm.

This module contains the pure BFS algorithm implementation.
The algorithm is clean, readable, and easy to understand for junior developers.
"""

from collections import deque
from typing import TypeVar

import networkx as nx

Node = TypeVar("Node")


def bfs_shortest_path(
    graph: nx.Graph,
    start: Node,
    goal: Node,
) -> list[Node] | None:
    """Find shortest path using Breadth-First Search.

    BFS explores nodes level by level, guaranteeing the shortest path
    in terms of number of hops (unweighted edges).

    Args:
        graph: NetworkX graph to search
        start: Starting node
        goal: Target node

    Returns:
        List of nodes representing the shortest path, or None if no path exists

    Raises:
        ValueError: If start or goal nodes are not in the graph
        ValueError: If start and goal are the same node

    Example:
        >>> import networkx as nx
        >>> from algokit.core.helpers import create_grid_graph
        >>>
        >>> # Create a simple 3x3 grid
        >>> graph = create_grid_graph(3, 3)
        >>> path = bfs_shortest_path(graph, (0, 0), (2, 2))
        >>> print(f"Path length: {len(path) if path else 'No path'}")
    """
    # Validate inputs
    if start not in graph:
        raise ValueError(f"Start node {start} not found in graph")
    if goal not in graph:
        raise ValueError(f"Goal node {goal} not found in graph")
    if start == goal:
        raise ValueError("Start and goal nodes cannot be the same")

    # Initialize BFS data structures
    queue = deque([start])  # Queue of nodes to explore
    visited = {start}  # Set of visited nodes
    parent: dict[Node, Node | None] = {
        start: None
    }  # Parent pointers for path reconstruction

    # BFS main loop
    while queue:
        current = queue.popleft()

        # Check if we found the goal
        if current == goal:
            # Reconstruct path by following parent pointers
            path: list[Node] = []
            node: Node | None = current
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1]  # Reverse to get start -> goal

        # Explore all neighbors
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

    # No path found
    return None


def bfs_path_length[Node](
    graph: nx.Graph,
    start: Node,
    goal: Node,
) -> int | None:
    """Find the length of the shortest path using BFS.

    This is a lightweight version that only returns the path length,
    useful for distance calculations without needing the full path.

    Args:
        graph: NetworkX graph to search
        start: Starting node
        goal: Target node

    Returns:
        Length of shortest path, or None if no path exists

    Raises:
        ValueError: If start or goal nodes are not in the graph
        ValueError: If start and goal are the same node

    Example:
        >>> import networkx as nx
        >>> from algokit.core.helpers import create_grid_graph
        >>>
        >>> # Create a simple 3x3 grid
        >>> graph = create_grid_graph(3, 3)
        >>> length = bfs_path_length(graph, (0, 0), (2, 2))
        >>> print(f"Shortest path length: {length}")
    """
    # Validate inputs
    if start not in graph:
        raise ValueError(f"Start node {start} not found in graph")
    if goal not in graph:
        raise ValueError(f"Goal node {goal} not found in graph")
    if start == goal:
        raise ValueError("Start and goal nodes cannot be the same")

    # Initialize BFS data structures
    queue = deque([(start, 0)])  # (node, distance)
    visited = {start}

    # BFS main loop
    while queue:
        current, distance = queue.popleft()

        # Check if we found the goal
        if current == goal:
            return distance

        # Explore all neighbors
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

    # No path found
    return None


def bfs_all_reachable[Node](
    graph: nx.Graph,
    start: Node,
    max_distance: int | None = None,
) -> dict[Node, int]:
    """Find all nodes reachable from start within a given distance.

    This is useful for finding all nodes within a certain radius or
    for analyzing the connectivity of a graph from a specific node.

    Args:
        graph: NetworkX graph to search
        start: Starting node
        max_distance: Maximum distance to search (None for unlimited)

    Returns:
        Dictionary mapping each reachable node to its distance from start

    Raises:
        ValueError: If start node is not in the graph

    Example:
        >>> import networkx as nx
        >>> from algokit.core.helpers import create_grid_graph
        >>>
        >>> # Create a simple 3x3 grid
        >>> graph = create_grid_graph(3, 3)
        >>> reachable = bfs_all_reachable(graph, (1, 1), max_distance=2)
        >>> print(f"Nodes within distance 2: {len(reachable)}")
    """
    # Validate inputs
    if start not in graph:
        raise ValueError(f"Start node {start} not found in graph")

    # Initialize BFS data structures
    queue = deque([(start, 0)])  # (node, distance)
    visited = {start: 0}

    # BFS main loop
    while queue:
        current, distance = queue.popleft()

        # Check distance limit
        if max_distance is not None and distance >= max_distance:
            continue

        # Explore all neighbors
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                new_distance = distance + 1
                visited[neighbor] = new_distance
                queue.append((neighbor, new_distance))

    return visited
