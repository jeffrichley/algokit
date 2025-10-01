"""Depth-First Search (DFS) pathfinding algorithm.

This module contains the pure DFS algorithm implementation.
The algorithm is clean, readable, and easy to understand for junior developers.
"""

import networkx as nx


def dfs_path[Node](
    graph: nx.Graph,
    start: Node,
    goal: Node,
) -> list[Node] | None:
    """Find a path using Depth-First Search.
    
    DFS explores nodes by going as deep as possible before backtracking.
    Unlike BFS, DFS does not guarantee the shortest path, but it can be
    more memory efficient for certain types of graphs.
    
    Args:
        graph: NetworkX graph to search
        start: Starting node
        goal: Target node
        
    Returns:
        List of nodes representing a path, or None if no path exists
        
    Raises:
        ValueError: If start or goal nodes are not in the graph
        ValueError: If start and goal are the same node
        
    Example:
        >>> import networkx as nx
        >>> from algokit.core.helpers import create_grid_graph
        >>> 
        >>> # Create a simple 3x3 grid
        >>> graph = create_grid_graph(3, 3)
        >>> path = dfs_path(graph, (0, 0), (2, 2))
        >>> print(f"Path found: {path is not None}")
    """
    # Validate inputs
    if start not in graph:
        raise ValueError(f"Start node {start} not found in graph")
    if goal not in graph:
        raise ValueError(f"Goal node {goal} not found in graph")
    if start == goal:
        raise ValueError("Start and goal nodes cannot be the same")
    
    # Initialize DFS data structures
    stack = [start]        # Stack of nodes to explore
    visited = {start}      # Set of visited nodes
    parent: dict[Node, Node | None] = {start: None} # Parent pointers for path reconstruction
    
    # DFS main loop
    while stack:
        current = stack.pop()
        
        # Check if we found the goal
        if current == goal:
            # Reconstruct path by following parent pointers
            path: list[Node] = []
            node: Node | None = current
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1]  # Reverse to get start -> goal
        
        # Explore all neighbors (add to stack in reverse order for consistent traversal)
        neighbors = list(graph.neighbors(current))
        neighbors.reverse()  # Process in consistent order
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                stack.append(neighbor)
    
    # No path found
    return None


def dfs_recursive_path[Node](
    graph: nx.Graph,
    start: Node,
    goal: Node,
) -> list[Node] | None:
    """Find a path using recursive Depth-First Search.
    
    This is a recursive implementation of DFS that uses the call stack
    instead of an explicit stack. It's more intuitive but may hit
    recursion limits on very deep graphs.
    
    Args:
        graph: NetworkX graph to search
        start: Starting node
        goal: Target node
        
    Returns:
        List of nodes representing a path, or None if no path exists
        
    Raises:
        ValueError: If start or goal nodes are not in the graph
        ValueError: If start and goal are the same node
        RecursionError: If the graph is too deep for recursion
        
    Example:
        >>> import networkx as nx
        >>> from algokit.core.helpers import create_grid_graph
        >>> 
        >>> # Create a simple 3x3 grid
        >>> graph = create_grid_graph(3, 3)
        >>> path = dfs_recursive_path(graph, (0, 0), (2, 2))
        >>> print(f"Path found: {path is not None}")
    """
    # Validate inputs
    if start not in graph:
        raise ValueError(f"Start node {start} not found in graph")
    if goal not in graph:
        raise ValueError(f"Goal node {goal} not found in graph")
    if start == goal:
        raise ValueError("Start and goal nodes cannot be the same")
    
    def _dfs_recursive(current: Node, visited: set[Node]) -> list[Node] | None:
        """Recursive DFS helper function."""
        # Check if we found the goal
        if current == goal:
            return [current]
        
        # Mark current node as visited
        visited.add(current)
        
        # Explore all neighbors
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                path = _dfs_recursive(neighbor, visited)
                if path is not None:
                    return [current] + path
        
        return None
    
    return _dfs_recursive(start, set())


def dfs_all_reachable[Node](
    graph: nx.Graph,
    start: Node,
    max_depth: int | None = None,
) -> dict[Node, int]:
    """Find all nodes reachable from start using DFS.
    
    This explores the entire connected component containing the start node,
    optionally limited by a maximum depth.
    
    Args:
        graph: NetworkX graph to search
        start: Starting node
        max_depth: Maximum depth to search (None for unlimited)
        
    Returns:
        Dictionary mapping each reachable node to its depth from start
        
    Raises:
        ValueError: If start node is not in the graph
        
    Example:
        >>> import networkx as nx
        >>> from algokit.core.helpers import create_grid_graph
        >>> 
        >>> # Create a simple 3x3 grid
        >>> graph = create_grid_graph(3, 3)
        >>> reachable = dfs_all_reachable(graph, (1, 1), max_depth=2)
        >>> print(f"Nodes within depth 2: {len(reachable)}")
    """
    # Validate inputs
    if start not in graph:
        raise ValueError(f"Start node {start} not found in graph")
    
    # Initialize DFS data structures
    stack = [(start, 0)]   # (node, depth)
    visited = {start: 0}
    
    # DFS main loop
    while stack:
        current, depth = stack.pop()
        
        # Check depth limit
        if max_depth is not None and depth >= max_depth:
            continue
        
        # Explore all neighbors
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                new_depth = depth + 1
                visited[neighbor] = new_depth
                stack.append((neighbor, new_depth))
    
    return visited


def dfs_connected_components[Node](graph: nx.Graph) -> list[set[Node]]:
    """Find all connected components in the graph using DFS.
    
    This uses DFS to identify all disconnected subgraphs in the graph.
    Each connected component is returned as a set of nodes.
    
    Args:
        graph: NetworkX graph to analyze
        
    Returns:
        List of sets, where each set contains nodes in one connected component
        
    Example:
        >>> import networkx as nx
        >>> 
        >>> # Create graph with multiple components
        >>> graph = nx.Graph()
        >>> graph.add_edges_from([(0, 1), (1, 2), (3, 4)])
        >>> components = dfs_connected_components(graph)
        >>> print(f"Number of components: {len(components)}")
    """
    visited = set()
    components = []
    
    for node in graph.nodes():
        if node not in visited:
            # Start DFS from this unvisited node
            component = set()
            stack = [node]
            
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.add(current)
                    
                    # Add all neighbors to stack
                    for neighbor in graph.neighbors(current):
                        if neighbor not in visited:
                            stack.append(neighbor)
            
            components.append(component)
    
    return components
