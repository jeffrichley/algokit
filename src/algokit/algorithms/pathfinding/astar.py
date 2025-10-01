"""A* pathfinding algorithm.

This module contains the pure A* algorithm implementation.
The algorithm finds the shortest path in a weighted graph using a priority queue
and heuristic function to guide the search toward the goal.
"""

import heapq
from collections.abc import Callable
from typing import TypeVar

import networkx as nx

from algokit.core.utils.distances import (
    zero_heuristic,
)

Node = TypeVar("Node")


def astar_shortest_path(
    graph: nx.Graph,
    start: Node,
    goal: Node,
    heuristic: Callable[[Node, Node], float] | None = None,
) -> tuple[list[Node], float] | None:
    """Find shortest path using A* algorithm.

    A* is an informed search algorithm that uses a heuristic function to guide
    the search toward the goal, often finding paths faster than Dijkstra's algorithm.

    Args:
        graph: NetworkX graph to search (edges should have 'weight' attribute)
        start: Starting node
        goal: Target node
        heuristic: Heuristic function h(node, goal) -> float (default: zero heuristic)

    Returns:
        Tuple of (path, total_cost) if path exists, None otherwise

    Raises:
        ValueError: If start or goal nodes are not in the graph
        ValueError: If start and goal are the same node

    Example:
        >>> import networkx as nx
        >>> from algokit.core.helpers import create_grid_graph
        >>>
        >>> # Create a 3x3 grid
        >>> graph = create_grid_graph(3, 3)
        >>> path, cost = astar_shortest_path(graph, (0, 0), (2, 2), manhattan_distance)
        >>> print(f"Path found with cost: {cost}")
    """
    # Validate inputs
    if start not in graph:
        raise ValueError(f"Start node {start} not found in graph")
    if goal not in graph:
        raise ValueError(f"Goal node {goal} not found in graph")
    if start == goal:
        raise ValueError("Start and goal nodes cannot be the same")

    # Default heuristic is zero (makes A* equivalent to Dijkstra)
    if heuristic is None:
        heuristic = zero_heuristic

    # Initialize A* data structures
    g_score: dict[Node, float] = {start: 0.0}  # Actual cost from start
    f_score: dict[Node, float] = {start: heuristic(start, goal)}  # Estimated total cost
    previous: dict[Node, Node | None] = {start: None}

    # Priority queue: (f_score, node)
    pq: list[tuple[float, Node]] = [(f_score[start], start)]
    visited: set[Node] = set()

    # A* main loop
    while pq:
        current_f, current = heapq.heappop(pq)

        # Skip if we've already processed this node with a better f_score
        if current in visited:
            continue

        visited.add(current)

        # Check if we found the goal
        if current == goal:
            # Reconstruct path by following previous pointers
            path: list[Node] = []
            node: Node | None = current
            while node is not None:
                path.append(node)
                node = previous[node]
            return path[::-1], g_score[current]  # Reverse to get start -> goal

        # Explore all neighbors
        for neighbor in graph.neighbors(current):
            # Skip if already visited (standard A*)
            if neighbor in visited:
                continue

            # Get edge weight (default to 1.0 if no weight attribute)
            edge_data = graph.get_edge_data(current, neighbor)
            weight = edge_data.get("weight", 1.0) if edge_data else 1.0

            # Calculate tentative g_score
            tentative_g_score = g_score[current] + weight

            # Update if we found a better path
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                previous[neighbor] = current
                heapq.heappush(pq, (f_score[neighbor], neighbor))

    # No path found
    return None


def astar_shortest_distance(
    graph: nx.Graph,
    start: Node,
    goal: Node,
    heuristic: Callable[[Node, Node], float] | None = None,
) -> float | None:
    """Find the shortest distance using A* algorithm.

    This is a lightweight version that only returns the shortest distance,
    useful for distance calculations without needing the full path.

    Args:
        graph: NetworkX graph to search (edges should have 'weight' attribute)
        start: Starting node
        goal: Target node
        heuristic: Heuristic function h(node, goal) -> float (default: zero heuristic)

    Returns:
        Shortest distance if path exists, None otherwise

    Raises:
        ValueError: If start or goal nodes are not in the graph
        ValueError: If start and goal are the same node

    Example:
        >>> import networkx as nx
        >>> from algokit.core.helpers import create_grid_graph
        >>>
        >>> # Create a 3x3 grid
        >>> graph = create_grid_graph(3, 3)
        >>> distance = astar_shortest_distance(graph, (0, 0), (2, 2), manhattan_distance)
        >>> print(f"Shortest distance: {distance}")
    """
    # Validate inputs
    if start not in graph:
        raise ValueError(f"Start node {start} not found in graph")
    if goal not in graph:
        raise ValueError(f"Goal node {goal} not found in graph")
    if start == goal:
        raise ValueError("Start and goal nodes cannot be the same")

    # Default heuristic is zero (makes A* equivalent to Dijkstra)
    if heuristic is None:
        heuristic = zero_heuristic

    # Initialize A* data structures
    g_score: dict[Node, float] = {start: 0.0}
    f_score: dict[Node, float] = {start: heuristic(start, goal)}

    # Priority queue: (f_score, node)
    pq: list[tuple[float, Node]] = [(f_score[start], start)]
    visited: set[Node] = set()

    # A* main loop
    while pq:
        current_f, current = heapq.heappop(pq)

        # Skip if we've already processed this node with a better f_score
        if current in visited:
            continue

        visited.add(current)

        # Check if we found the goal
        if current == goal:
            return g_score[current]

        # Explore all neighbors
        for neighbor in graph.neighbors(current):
            # Skip if already visited (standard A*)
            if neighbor in visited:
                continue

            # Get edge weight (default to 1.0 if no weight attribute)
            edge_data = graph.get_edge_data(current, neighbor)
            weight = edge_data.get("weight", 1.0) if edge_data else 1.0

            # Calculate tentative g_score
            tentative_g_score = g_score[current] + weight

            # Update if we found a better path
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(pq, (f_score[neighbor], neighbor))

    # No path found
    return None


def astar_all_distances(
    graph: nx.Graph,
    start: Node,
    max_distance: float | None = None,
    heuristic: Callable[[Node, Node], float] | None = None,
) -> dict[Node, float]:
    """Find shortest distances to all reachable nodes using A* algorithm.

    This explores the entire connected component containing the start node,
    optionally limited by a maximum distance. Note: A* is typically used for
    point-to-point search, so this function uses the zero heuristic (making
    it equivalent to Dijkstra's algorithm).

    Args:
        graph: NetworkX graph to search (edges should have 'weight' attribute)
        start: Starting node
        max_distance: Maximum distance to search (None for unlimited)
        heuristic: Heuristic function (ignored for multi-goal search)

    Returns:
        Dictionary mapping each reachable node to its shortest distance from start

    Raises:
        ValueError: If start node is not in the graph

    Example:
        >>> import networkx as nx
        >>> from algokit.core.helpers import create_grid_graph
        >>>
        >>> # Create a 3x3 grid
        >>> graph = create_grid_graph(3, 3)
        >>> distances = astar_all_distances(graph, (1, 1), max_distance=2.0)
        >>> print(f"Nodes within distance 2: {len(distances)}")
    """
    # Validate inputs
    if start not in graph:
        raise ValueError(f"Start node {start} not found in graph")

    # For multi-goal search, A* reduces to Dijkstra (zero heuristic)
    # Initialize data structures
    distances: dict[Node, float] = {start: 0.0}
    pq: list[tuple[float, Node]] = [(0.0, start)]
    visited: set[Node] = set()

    # Main loop (equivalent to Dijkstra)
    while pq:
        current_distance, current = heapq.heappop(pq)

        # Skip if we've already processed this node
        if current in visited:
            continue

        visited.add(current)

        # Check distance limit
        if max_distance is not None and current_distance >= max_distance:
            continue

        # Explore all neighbors
        for neighbor in graph.neighbors(current):
            # Skip if already visited
            if neighbor in visited:
                continue

            # Get edge weight (default to 1.0 if no weight attribute)
            edge_data = graph.get_edge_data(current, neighbor)
            weight = edge_data.get("weight", 1.0) if edge_data else 1.0

            # Calculate new distance
            new_distance = current_distance + weight

            # Check if new distance exceeds max_distance limit
            if max_distance is not None and new_distance >= max_distance:
                continue

            # Update if we found a better path
            if neighbor not in distances or new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(pq, (new_distance, neighbor))

    return distances
