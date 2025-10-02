"""Pathfinding algorithms module.

This module contains implementations of graph traversal and pathfinding algorithms
including BFS, DFS, Dijkstra, and A* search algorithms.
"""

from algokit.algorithms.pathfinding.astar import (
    astar_all_distances,
    astar_shortest_distance,
    astar_shortest_path,
)
from algokit.algorithms.pathfinding.bfs import (
    bfs_all_reachable,
    bfs_path_length,
    bfs_shortest_path,
)
from algokit.algorithms.pathfinding.bfs_with_events import bfs_with_data_collection
from algokit.algorithms.pathfinding.dfs import (
    dfs_all_reachable,
    dfs_connected_components,
    dfs_path,
    dfs_recursive_path,
)
from algokit.algorithms.pathfinding.dijkstra import (
    dijkstra_all_distances,
    dijkstra_shortest_distance,
    dijkstra_shortest_path,
)
from algokit.algorithms.pathfinding.mstar import mstar_plan_paths
from algokit.core.utils.distances import (
    euclidean_distance,
    manhattan_distance,
    zero_heuristic,
)

__all__ = [
    # A* algorithms
    "astar_shortest_path",
    "astar_shortest_distance",
    "astar_all_distances",
    "manhattan_distance",
    "euclidean_distance",
    "zero_heuristic",
    # BFS algorithms
    "bfs_shortest_path",
    "bfs_with_data_collection",
    "bfs_path_length",
    "bfs_all_reachable",
    # DFS algorithms
    "dfs_path",
    "dfs_recursive_path",
    "dfs_all_reachable",
    "dfs_connected_components",
    # Dijkstra algorithms
    "dijkstra_shortest_path",
    "dijkstra_shortest_distance",
    "dijkstra_all_distances",
    # M* algorithms
    "MStarPlanner",
    "mstar_plan_paths",
    "mstar_plan_single_robot",
]
