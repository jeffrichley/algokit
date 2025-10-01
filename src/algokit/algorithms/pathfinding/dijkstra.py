"""Dijkstra's shortest path algorithm.

This module contains the pure Dijkstra's algorithm implementation.
The algorithm finds the shortest path in a weighted graph using a priority queue.
"""

import heapq
from typing import TypeVar

import networkx as nx

Node = TypeVar("Node")


def dijkstra_shortest_path(
    graph: nx.Graph,
    start: Node,
    goal: Node,
) -> tuple[list[Node], float] | None:
    """Find shortest path using Dijkstra's algorithm.
    
    Dijkstra's algorithm finds the shortest path in a weighted graph by
    exploring nodes in order of their distance from the start node.
    
    Args:
        graph: NetworkX graph to search (edges should have 'weight' attribute)
        start: Starting node
        goal: Target node
        
    Returns:
        Tuple of (path, total_weight) if path exists, None otherwise
        
    Raises:
        ValueError: If start or goal nodes are not in the graph
        ValueError: If start and goal are the same node
        
    Example:
        >>> import networkx as nx
        >>> from algokit.core.helpers import create_weighted_grid_graph
        >>> 
        >>> # Create a weighted 3x3 grid
        >>> graph = create_weighted_grid_graph(3, 3)
        >>> path, weight = dijkstra_shortest_path(graph, (0, 0), (2, 2))
        >>> print(f"Shortest path weight: {weight}")
    """
    # Validate inputs
    if start not in graph:
        raise ValueError(f"Start node {start} not found in graph")
    if goal not in graph:
        raise ValueError(f"Goal node {goal} not found in graph")
    if start == goal:
        raise ValueError("Start and goal nodes cannot be the same")
    
    # Initialize Dijkstra's data structures
    distances: dict[Node, float] = {start: 0.0}
    previous: dict[Node, Node | None] = {start: None}
    pq: list[tuple[float, Node]] = [(0.0, start)]
    visited: set[Node] = set()
    
    # Dijkstra's main loop
    while pq:
        current_distance, current = heapq.heappop(pq)
        
        # Skip if we've already processed this node (standard Dijkstra)
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
            return path[::-1], current_distance  # Reverse to get start -> goal
        
        # Explore all neighbors
        for neighbor in graph.neighbors(current):
            # Skip if already visited (standard Dijkstra)
            if neighbor in visited:
                continue
                
            # Get edge weight (default to 1.0 if no weight attribute)
            edge_data = graph.get_edge_data(current, neighbor)
            weight = edge_data.get("weight", 1.0) if edge_data else 1.0
            
            # Calculate new distance
            new_distance = current_distance + weight
            
            # Update if we found a better path
            if neighbor not in distances or new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous[neighbor] = current
                heapq.heappush(pq, (new_distance, neighbor))
    
    # No path found
    return None


def dijkstra_shortest_distance(
    graph: nx.Graph,
    start: Node,
    goal: Node,
) -> float | None:
    """Find the shortest distance using Dijkstra's algorithm.
    
    This is a lightweight version that only returns the shortest distance,
    useful for distance calculations without needing the full path.
    
    Args:
        graph: NetworkX graph to search (edges should have 'weight' attribute)
        start: Starting node
        goal: Target node
        
    Returns:
        Shortest distance if path exists, None otherwise
        
    Raises:
        ValueError: If start or goal nodes are not in the graph
        ValueError: If start and goal are the same node
        
    Example:
        >>> import networkx as nx
        >>> from algokit.core.helpers import create_weighted_grid_graph
        >>> 
        >>> # Create a weighted 3x3 grid
        >>> graph = create_weighted_grid_graph(3, 3)
        >>> distance = dijkstra_shortest_distance(graph, (0, 0), (2, 2))
        >>> print(f"Shortest distance: {distance}")
    """
    # Validate inputs
    if start not in graph:
        raise ValueError(f"Start node {start} not found in graph")
    if goal not in graph:
        raise ValueError(f"Goal node {goal} not found in graph")
    if start == goal:
        raise ValueError("Start and goal nodes cannot be the same")
    
    # Initialize Dijkstra's data structures
    distances: dict[Node, float] = {start: 0.0}
    pq: list[tuple[float, Node]] = [(0.0, start)]
    visited: set[Node] = set()
    
    # Dijkstra's main loop
    while pq:
        current_distance, current = heapq.heappop(pq)
        
        # Skip if we've already processed this node (standard Dijkstra)
        if current in visited:
            continue
            
        visited.add(current)
        
        # Check if we found the goal
        if current == goal:
            return current_distance
        
        # Explore all neighbors
        for neighbor in graph.neighbors(current):
            # Skip if already visited (standard Dijkstra)
            if neighbor in visited:
                continue
                
            # Get edge weight (default to 1.0 if no weight attribute)
            edge_data = graph.get_edge_data(current, neighbor)
            weight = edge_data.get("weight", 1.0) if edge_data else 1.0
            
            # Calculate new distance
            new_distance = current_distance + weight
            
            # Update if we found a better path
            if neighbor not in distances or new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(pq, (new_distance, neighbor))
    
    # No path found
    return None


def dijkstra_all_distances[Node](
    graph: nx.Graph,
    start: Node,
    max_distance: float | None = None,
) -> dict[Node, float]:
    """Find shortest distances to all reachable nodes using Dijkstra's algorithm.
    
    This explores the entire connected component containing the start node,
    optionally limited by a maximum distance.
    
    Args:
        graph: NetworkX graph to search (edges should have 'weight' attribute)
        start: Starting node
        max_distance: Maximum distance to search (None for unlimited)
        
    Returns:
        Dictionary mapping each reachable node to its shortest distance from start
        
    Raises:
        ValueError: If start node is not in the graph
        
    Example:
        >>> import networkx as nx
        >>> from algokit.core.helpers import create_weighted_grid_graph
        >>> 
        >>> # Create a weighted 3x3 grid
        >>> graph = create_weighted_grid_graph(3, 3)
        >>> distances = dijkstra_all_distances(graph, (1, 1), max_distance=5.0)
        >>> print(f"Nodes within distance 5: {len(distances)}")
    """
    # Validate inputs
    if start not in graph:
        raise ValueError(f"Start node {start} not found in graph")
    
    # Initialize Dijkstra's data structures
    distances: dict[Node, float] = {start: 0.0}
    pq: list[tuple[float, Node]] = [(0.0, start)]
    visited: set[Node] = set()
    
    # Dijkstra's main loop
    while pq:
        current_distance, current = heapq.heappop(pq)
        
        # Skip if we've already processed this node (standard Dijkstra)
        if current in visited:
            continue
            
        visited.add(current)
        
        # Check distance limit - don't process nodes beyond max_distance
        if max_distance is not None and current_distance >= max_distance:
            continue
        
        # Explore all neighbors
        for neighbor in graph.neighbors(current):
            # Skip if already visited (standard Dijkstra)
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
