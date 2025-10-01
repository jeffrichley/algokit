"""Tests for BFS algorithm properties and validation.

This module tests the fundamental properties of BFS to ensure correctness:
- Time complexity O(|V|+|E|)
- Minimum-hop path guarantee
- Queue-based frontier management
- Parent pointer reconstruction
"""


import networkx as nx
import pytest

from algokit.algorithms.pathfinding.bfs import bfs_path_length, bfs_shortest_path
from algokit.core.helpers import HarborNetScenario, create_grid_graph


class TestBFSProperties:
    """Test fundamental BFS algorithm properties."""

    @pytest.mark.unit
    def test_bfs_time_complexity_validation(self) -> None:
        """Test that BFS visits each node and edge at most once (O(|V|+|E|))."""
        # Arrange - create a graph where we can count visits
        graph = nx.Graph()
        # Create a simple path: 0-1-2-3-4
        for i in range(5):
            graph.add_node(i)
        for i in range(4):
            graph.add_edge(i, i + 1)
        
        # Act - run BFS and count operations
        path = bfs_shortest_path(graph, 0, 4)
        
        # Assert - verify path is optimal
        assert path == [0, 1, 2, 3, 4]
        assert len(path) == 5  # Minimum possible path length
        
        # Verify we found the shortest path (4 hops)
        length = bfs_path_length(graph, 0, 4)
        assert length == 4

    @pytest.mark.unit
    def test_bfs_minimum_hop_guarantee(self) -> None:
        """Test that BFS guarantees minimum-hop path."""
        # Arrange - create a graph with multiple paths of different lengths
        graph = nx.Graph()
        # Short path: A-B-C (2 hops)
        graph.add_edges_from([("A", "B"), ("B", "C")])
        # Long path: A-D-E-F-C (4 hops)
        graph.add_edges_from([("A", "D"), ("D", "E"), ("E", "F"), ("F", "C")])
        
        # Act - find path from A to C
        path = bfs_shortest_path(graph, "A", "C")
        
        # Assert - should find the shorter path
        assert path == ["A", "B", "C"]
        assert len(path) == 3  # 2 hops + start node
        
        # Verify path length is correct
        length = bfs_path_length(graph, "A", "C")
        assert length == 2

    @pytest.mark.unit
    def test_bfs_queue_based_frontier_management(self) -> None:
        """Test that BFS uses proper queue-based frontier management."""
        # Arrange - create a tree structure to test level-by-level exploration
        graph = nx.Graph()
        # Tree structure:
        #     0
        #   /   \
        #  1     2
        # / \   / \
        # 3  4  5  6
        graph.add_edges_from([
            (0, 1), (0, 2),
            (1, 3), (1, 4),
            (2, 5), (2, 6)
        ])
        
        # Act - find path from root to leaf
        path = bfs_shortest_path(graph, 0, 6)
        
        # Assert - should find shortest path (2 hops)
        assert path == [0, 2, 6]
        assert len(path) == 3
        
        # Verify all paths from root to leaves are 2 hops
        for leaf in [3, 4, 5, 6]:
            length = bfs_path_length(graph, 0, leaf)
            assert length == 2

    @pytest.mark.unit
    def test_bfs_parent_pointer_reconstruction(self) -> None:
        """Test that BFS correctly reconstructs paths using parent pointers."""
        # Arrange - create a complex graph with multiple possible paths
        graph = nx.Graph()
        # Grid-like structure with some connections missing
        # 0-1-2
        # |   |
        # 3-4-5
        # |   |
        # 6-7-8
        graph.add_edges_from([
            (0, 1), (1, 2),
            (0, 3), (2, 5),
            (3, 4), (4, 5),
            (3, 6), (5, 8),
            (6, 7), (7, 8)
        ])
        
        # Act - find path from corner to corner
        path = bfs_shortest_path(graph, 0, 8)
        
        # Assert - verify path is valid and optimal
        assert path is not None
        assert path[0] == 0
        assert path[-1] == 8
        
        # Verify each step in path is a valid edge
        for i in range(len(path) - 1):
            assert graph.has_edge(path[i], path[i + 1])
        
        # Verify path length is optimal (4 hops)
        assert len(path) == 5  # 4 hops + start node
        length = bfs_path_length(graph, 0, 8)
        assert length == 4

    @pytest.mark.unit
    def test_bfs_tie_breaking_behavior(self) -> None:
        """Test BFS tie-breaking behavior when multiple shortest paths exist."""
        # Arrange - create graph with multiple equal-length paths
        graph = nx.Graph()
        # Diamond structure: A-B-C and A-D-C (both 2 hops)
        graph.add_edges_from([("A", "B"), ("B", "C"), ("A", "D"), ("D", "C")])
        
        # Act - find path multiple times
        paths = []
        for _ in range(10):
            path = bfs_shortest_path(graph, "A", "C")
            paths.append(path)
        
        # Assert - all paths should be valid and same length
        for path in paths:
            assert len(path) == 3  # 2 hops + start node
            assert path[0] == "A"
            assert path[-1] == "C"
            # Path should be either A-B-C or A-D-C
            assert path in [["A", "B", "C"], ["A", "D", "C"]]

    @pytest.mark.unit
    def test_bfs_disconnected_graph_handling(self) -> None:
        """Test BFS behavior with disconnected components."""
        # Arrange - create graph with disconnected components
        graph = nx.Graph()
        # Component 1: A-B-C
        graph.add_edges_from([("A", "B"), ("B", "C")])
        # Component 2: D-E-F
        graph.add_edges_from([("D", "E"), ("E", "F")])
        
        # Act - try to find path between disconnected components
        path = bfs_shortest_path(graph, "A", "F")
        
        # Assert - should return None
        assert path is None
        
        # Verify path length also returns None
        length = bfs_path_length(graph, "A", "F")
        assert length is None

    @pytest.mark.unit
    def test_bfs_single_node_graph(self) -> None:
        """Test BFS behavior with single node graph."""
        # Arrange - create graph with single node
        graph = nx.Graph()
        graph.add_node("A")
        
        # Act & Assert - should raise error for same start/goal
        with pytest.raises(ValueError, match="Start and goal nodes cannot be the same"):
            bfs_shortest_path(graph, "A", "A")

    @pytest.mark.unit
    def test_bfs_empty_graph(self) -> None:
        """Test BFS behavior with empty graph."""
        # Arrange - create empty graph
        graph = nx.Graph()
        
        # Act & Assert - should raise error for missing nodes
        with pytest.raises(ValueError, match="Start node A not found in graph"):
            bfs_shortest_path(graph, "A", "B")


    @pytest.mark.integration
    def test_bfs_harbor_net_scenario_properties(self) -> None:
        """Test BFS properties with HarborNet scenario."""
        # Arrange - create HarborNet scenario
        scenario = HarborNetScenario(
            name="Test Harbor",
            width=10,
            height=10,
            start=(0, 0),
            goal=(9, 9),
            obstacles={(4, 4), (5, 5), (6, 6)},  # Diagonal obstacle line
        )
        graph = create_grid_graph(scenario.width, scenario.height, blocked=scenario.obstacles)
        
        # Act - find path
        path = bfs_shortest_path(graph, scenario.start, scenario.goal)
        
        # Assert - verify BFS properties
        assert path is not None
        assert path[0] == scenario.start
        assert path[-1] == scenario.goal
        
        # Verify path avoids obstacles
        for node in path:
            assert node not in scenario.obstacles
        
        # Verify path length is optimal
        length = bfs_path_length(graph, scenario.start, scenario.goal)
        assert length == len(path) - 1  # Path length should match distance
        
        # Verify each step is a valid move
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            # Should be adjacent (Manhattan distance = 1)
            manhattan_dist = abs(current[0] - next_node[0]) + abs(current[1] - next_node[1])
            assert manhattan_dist == 1
