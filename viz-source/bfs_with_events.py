"""BFS with event tracking for visualization.

This module provides BFS implementations with event tracking for visualization.
It uses decorators to add functionality to the pure BFS algorithm without
polluting the core algorithm code.
"""

import networkx as nx

from decorators import with_event_tracking


# Create the decorated version for convenience
@with_event_tracking
def bfs_with_data_collection[Node](
    graph: nx.Graph,
    start: Node,
    goal: Node,
) -> list[Node] | None:
    """Find shortest path using BFS with event tracking for visualization.

    This is a decorated version of the pure BFS algorithm that collects
    events for post-processing visualization.

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
        >>> path, events = bfs_with_data_collection(graph, (0, 0), (2, 2))
        >>> print(f"Path found: {path is not None}")
        >>> print(f"Events collected: {len(events)}")
    """
    # The decorator completely replaces this function implementation
    # with its own BFS implementation that includes event tracking
    # This function body should never be executed
    pass
