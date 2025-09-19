"""Decorators for adding functionality to algorithms.

This module provides decorators that can add visualization tracking,
logging, or other functionality to pure algorithm implementations
without polluting the core algorithm code.
"""

from functools import wraps

from algokit.viz.adapters import EventType, SimpleTracker


def with_event_tracking(func):
    """Decorator to add event tracking to BFS algorithm.

    This decorator wraps the pure BFS algorithm to collect events for
    post-processing visualization while keeping the core algorithm clean.

    Args:
        func: The BFS function to wrap

    Returns:
        Wrapped function that returns (path, events) tuple

    Example:
        >>> @with_event_tracking
        ... def bfs_with_events(graph, start, goal):
        ...     return bfs_shortest_path(graph, start, goal)
        >>>
        >>> path, events = bfs_with_events(graph, (0, 0), (2, 2))
    """

    @wraps(func)
    def wrapper(graph, start, goal):
        # Validate inputs (same as pure function)
        if start not in graph:
            raise ValueError(f"Start node {start} not found in graph")
        if goal not in graph:
            raise ValueError(f"Goal node {goal} not found in graph")
        if start == goal:
            raise ValueError("Start and goal nodes cannot be the same")

        # Initialize tracker and BFS data structures
        tracker = SimpleTracker()

        from collections import deque

        with tracker.track(queue=deque([start]), visited={start}) as tracked:
            queue = tracked["queue"]
            visited = tracked["visited"]
            parent = {start: None}
            depth = {start: 0}  # Track depth of each node
            nodes_at_depth = {0: [start]}  # Track nodes at each depth level
            current_depth = 0
            nodes_processed_at_current_depth = 0

            # BFS main loop with event tracking
            while queue:
                # Check depth progression BEFORE dequeue to emit layer completion at right time
                next_node = queue._deque[0]  # Peek at next node without removing it
                next_node_depth = depth[next_node]

                if next_node_depth > current_depth:
                    # We're about to move to a new depth level - emit layer completion for previous depth
                    nodes_at_prev_depth = nodes_at_depth.get(current_depth, [])
                    tracker.emit(
                        EventType.LAYER_COMPLETE,
                        current_depth,
                        nodes_count=len(nodes_at_prev_depth),
                        nodes=nodes_at_prev_depth,
                    )
                    current_depth = next_node_depth
                    nodes_processed_at_current_depth = 0

                current = queue.popleft()  # Emits DEQUEUE event
                nodes_processed_at_current_depth += 1

                # Check if we found the goal
                if current == goal:
                    # Emit goal found event
                    tracker.emit(EventType.GOAL_FOUND, current)

                    # Reconstruct path with events
                    path = []
                    node = current
                    while node is not None:
                        path.append(node)
                        if parent[node] is not None:
                            tracker.emit(
                                EventType.PATH_RECONSTRUCT, node, parent=parent[node]
                            )
                        node = parent[node]

                    return path[::-1], tracker.events  # Reverse to get start -> goal

                # Explore neighbors
                for neighbor in graph.neighbors(current):
                    if neighbor not in visited:
                        parent[neighbor] = current
                        # Set depth for neighbor
                        neighbor_depth = depth[current] + 1
                        depth[neighbor] = neighbor_depth
                        # Track nodes at this depth
                        if neighbor_depth not in nodes_at_depth:
                            nodes_at_depth[neighbor_depth] = []
                        nodes_at_depth[neighbor_depth].append(neighbor)
                        # Emit ENQUEUE event manually with parent information and depth
                        tracker.emit(
                            EventType.ENQUEUE,
                            neighbor,
                            parent=current,
                            depth=neighbor_depth,
                        )
                        queue._deque.append(
                            neighbor
                        )  # Add to deque directly (no event)
                        visited.add(neighbor)  # Emits DISCOVER event second

        # No path found
        return None, tracker.events

    return wrapper


def with_logging(func):
    """Decorator to add logging to algorithm functions.

    This decorator adds logging functionality to track algorithm execution
    without modifying the core algorithm logic.

    Args:
        func: The algorithm function to wrap

    Returns:
        Wrapped function with logging
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Add logging here if needed
        # For now, just call the original function
        return func(*args, **kwargs)

    return wrapper


def with_timing(func):
    """Decorator to add timing information to algorithm functions.

    This decorator measures execution time of algorithm functions
    without modifying the core algorithm logic.

    Args:
        func: The algorithm function to wrap

    Returns:
        Wrapped function that returns (result, execution_time) tuple
    """
    import time

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time

    return wrapper
