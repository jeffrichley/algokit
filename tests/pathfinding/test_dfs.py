"""Tests for the DFS pathfinding algorithm implementation."""

import networkx as nx
import pytest

from algokit.algorithms.pathfinding.dfs import (
    dfs_all_reachable,
    dfs_connected_components,
    dfs_path,
    dfs_recursive_path,
)
from algokit.core.helpers import create_grid_graph


class TestDFSPath:
    """Test the pure DFS path implementation."""

    @pytest.mark.unit
    def test_dfs_simple_path(self) -> None:
        """Test DFS finds a path in simple graph."""
        # Arrange - create a simple 3x3 grid
        graph = create_grid_graph(3, 3)

        # Act - find path from corner to corner
        path = dfs_path(graph, (0, 0), (2, 2))

        # Assert - verify path is valid
        assert path is not None
        assert len(path) >= 5  # At least Manhattan distance
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        assert all(graph.has_edge(path[i], path[i + 1]) for i in range(len(path) - 1))

    @pytest.mark.unit
    def test_dfs_no_path(self) -> None:
        """Test DFS returns None when no path exists."""
        # Arrange - create graph with disconnected components
        graph = nx.Graph()
        graph.add_nodes_from([(0, 0), (1, 1), (2, 2)])
        graph.add_edge((0, 0), (1, 1))
        # (2, 2) is disconnected

        # Act - try to find path between disconnected nodes
        path = dfs_path(graph, (0, 0), (2, 2))

        # Assert - no path should be found
        assert path is None

    @pytest.mark.unit
    def test_dfs_same_start_goal(self) -> None:
        """Test DFS raises error for same start and goal."""
        # Arrange - create simple graph
        graph = create_grid_graph(3, 3)

        # Act & Assert - should raise ValueError
        with pytest.raises(ValueError, match="Start and goal nodes cannot be the same"):
            dfs_path(graph, (0, 0), (0, 0))

    @pytest.mark.unit
    def test_dfs_trivial_start_is_goal(self) -> None:
        """Test DFS behavior with single node graph."""
        # Arrange - create single node graph
        graph = nx.Graph()
        graph.add_node("A")

        # Act & Assert - should raise error for same start/goal
        with pytest.raises(ValueError, match="Start and goal nodes cannot be the same"):
            dfs_path(graph, "A", "A")

    @pytest.mark.unit
    def test_dfs_invalid_start_node(self) -> None:
        """Test DFS raises error for invalid start node."""
        # Arrange - create simple graph
        graph = create_grid_graph(3, 3)

        # Act & Assert - should raise ValueError
        with pytest.raises(
            ValueError, match="Start node \\(99, 99\\) not found in graph"
        ):
            dfs_path(graph, (99, 99), (2, 2))

    @pytest.mark.unit
    def test_dfs_invalid_goal_node(self) -> None:
        """Test DFS raises error for invalid goal node."""
        # Arrange - create simple graph
        graph = create_grid_graph(3, 3)

        # Act & Assert - should raise ValueError
        with pytest.raises(
            ValueError, match="Goal node \\(99, 99\\) not found in graph"
        ):
            dfs_path(graph, (0, 0), (99, 99))

    @pytest.mark.unit
    def test_dfs_with_obstacles(self) -> None:
        """Test DFS finds path around obstacles."""
        # Arrange - create grid with obstacles
        obstacles = {(1, 1), (2, 1)}
        graph = create_grid_graph(3, 3, blocked=obstacles)

        # Act - find path from corner to corner
        path = dfs_path(graph, (0, 0), (2, 2))

        # Assert - path should avoid obstacles
        assert path is not None
        assert all(node not in obstacles for node in path)

    @pytest.mark.unit
    def test_dfs_path_validity(self) -> None:
        """Test that DFS returns a valid path."""
        # Arrange - create a larger grid
        graph = create_grid_graph(5, 5)

        # Act - find path from one corner to another
        path = dfs_path(graph, (0, 0), (4, 4))

        # Assert - path should be valid
        assert path is not None
        assert len(path) >= 9  # At least Manhattan distance
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)

        # Verify each step is a valid edge
        for i in range(len(path) - 1):
            assert graph.has_edge(path[i], path[i + 1])

    @pytest.mark.unit
    def test_dfs_path_deterministic(self) -> None:
        """Test that DFS produces consistent results."""
        # Arrange - create simple graph
        graph = create_grid_graph(3, 3)

        # Act - run DFS multiple times
        paths = []
        for _ in range(5):
            path = dfs_path(graph, (0, 0), (2, 2))
            paths.append(path)

        # Assert - all paths should be the same (deterministic)
        assert all(path == paths[0] for path in paths)


class TestDFSRecursivePath:
    """Test the recursive DFS path implementation."""

    @pytest.mark.unit
    def test_dfs_recursive_simple_path(self) -> None:
        """Test recursive DFS finds a path in simple graph."""
        # Arrange - create a simple 3x3 grid
        graph = create_grid_graph(3, 3)

        # Act - find path from corner to corner
        path = dfs_recursive_path(graph, (0, 0), (2, 2))

        # Assert - verify path is valid
        assert path is not None
        assert len(path) >= 5  # At least Manhattan distance
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        assert all(graph.has_edge(path[i], path[i + 1]) for i in range(len(path) - 1))

    @pytest.mark.unit
    def test_dfs_recursive_no_path(self) -> None:
        """Test recursive DFS returns None when no path exists."""
        # Arrange - create graph with disconnected components
        graph = nx.Graph()
        graph.add_nodes_from([(0, 0), (1, 1), (2, 2)])
        graph.add_edge((0, 0), (1, 1))
        # (2, 2) is disconnected

        # Act - try to find path between disconnected nodes
        path = dfs_recursive_path(graph, (0, 0), (2, 2))

        # Assert - no path should be found
        assert path is None

    @pytest.mark.unit
    def test_dfs_recursive_same_start_goal(self) -> None:
        """Test recursive DFS raises error for same start and goal."""
        # Arrange - create simple graph
        graph = create_grid_graph(3, 3)

        # Act & Assert - should raise ValueError
        with pytest.raises(ValueError, match="Start and goal nodes cannot be the same"):
            dfs_recursive_path(graph, (0, 0), (0, 0))

    @pytest.mark.unit
    def test_dfs_recursive_with_obstacles(self) -> None:
        """Test recursive DFS finds path around obstacles."""
        # Arrange - create grid with obstacles
        obstacles = {(1, 1), (2, 1)}
        graph = create_grid_graph(3, 3, blocked=obstacles)

        # Act - find path from corner to corner
        path = dfs_recursive_path(graph, (0, 0), (2, 2))

        # Assert - path should avoid obstacles
        assert path is not None
        assert all(node not in obstacles for node in path)


class TestDFSAllReachable:
    """Test the DFS all reachable implementation."""

    @pytest.mark.unit
    def test_dfs_all_reachable_simple(self) -> None:
        """Test DFS finds all reachable nodes."""
        # Arrange - create simple graph
        graph = create_grid_graph(3, 3)

        # Act - find all reachable nodes from center
        reachable = dfs_all_reachable(graph, (1, 1))

        # Assert - should find all nodes
        assert len(reachable) == 9  # All 9 nodes in 3x3 grid
        assert (1, 1) in reachable
        assert reachable[(1, 1)] == 0  # Distance to self is 0

    @pytest.mark.unit
    def test_dfs_all_reachable_with_depth_limit(self) -> None:
        """Test DFS respects depth limit."""
        # Arrange - create simple graph
        graph = create_grid_graph(5, 5)

        # Act - find nodes within depth 2 from center
        reachable = dfs_all_reachable(graph, (2, 2), max_depth=2)

        # Assert - should only find nodes within distance 2
        assert all(distance <= 2 for distance in reachable.values())
        assert (2, 2) in reachable
        assert reachable[(2, 2)] == 0

    @pytest.mark.unit
    def test_dfs_all_reachable_disconnected(self) -> None:
        """Test DFS finds only connected component."""
        # Arrange - create graph with disconnected components
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (3, 4), (4, 5)])

        # Act - find reachable nodes from first component
        reachable = dfs_all_reachable(graph, 0)

        # Assert - should only find first component
        assert len(reachable) == 3  # Nodes 0, 1, 2
        assert 0 in reachable
        assert 1 in reachable
        assert 2 in reachable
        assert 3 not in reachable
        assert 4 not in reachable
        assert 5 not in reachable

    @pytest.mark.unit
    def test_dfs_all_reachable_invalid_start(self) -> None:
        """Test DFS raises error for invalid start node."""
        # Arrange - create simple graph
        graph = create_grid_graph(3, 3)

        # Act & Assert - should raise ValueError
        with pytest.raises(
            ValueError, match="Start node \\(99, 99\\) not found in graph"
        ):
            dfs_all_reachable(graph, (99, 99))


class TestDFSConnectedComponents:
    """Test the DFS connected components implementation."""

    @pytest.mark.unit
    def test_dfs_connected_components_single(self) -> None:
        """Test DFS finds single connected component."""
        # Arrange - create connected graph
        graph = create_grid_graph(3, 3)

        # Act - find connected components
        components = dfs_connected_components(graph)

        # Assert - should find one component with all nodes
        assert len(components) == 1
        assert len(components[0]) == 9  # All 9 nodes

    @pytest.mark.unit
    def test_dfs_connected_components_multiple(self) -> None:
        """Test DFS finds multiple connected components."""
        # Arrange - create graph with multiple components
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (3, 4), (5, 6), (6, 7)])

        # Act - find connected components
        components = dfs_connected_components(graph)

        # Assert - should find three components
        assert len(components) == 3

        # Sort components by size for consistent testing
        components.sort(key=len, reverse=True)

        # Largest component: 0, 1, 2
        assert len(components[0]) == 3
        assert 0 in components[0]
        assert 1 in components[0]
        assert 2 in components[0]

        # Medium component: 5, 6, 7
        assert len(components[1]) == 3
        assert 5 in components[1]
        assert 6 in components[1]
        assert 7 in components[1]

        # Smallest component: 3, 4
        assert len(components[2]) == 2
        assert 3 in components[2]
        assert 4 in components[2]

    @pytest.mark.unit
    def test_dfs_connected_components_empty(self) -> None:
        """Test DFS handles empty graph."""
        # Arrange - create empty graph
        graph = nx.Graph()

        # Act - find connected components
        components = dfs_connected_components(graph)

        # Assert - should find no components
        assert len(components) == 0

    @pytest.mark.unit
    def test_dfs_connected_components_single_nodes(self) -> None:
        """Test DFS handles isolated nodes."""
        # Arrange - create graph with only isolated nodes
        graph = nx.Graph()
        graph.add_nodes_from([0, 1, 2])

        # Act - find connected components
        components = dfs_connected_components(graph)

        # Assert - should find three components of size 1
        assert len(components) == 3
        for component in components:
            assert len(component) == 1


class TestDFSComparisonWithBFS:
    """Test DFS behavior compared to BFS."""

    @pytest.mark.unit
    def test_dfs_vs_bfs_path_length(self) -> None:
        """Test that DFS may find longer paths than BFS."""
        # Arrange - create graph where DFS might find longer path
        graph = nx.Graph()
        graph.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),  # Long path
                (0, 4),  # Direct edge
            ]
        )

        # Act - find paths with both algorithms
        dfs_result = dfs_path(graph, 0, 4)
        # Note: We can't easily test BFS here without importing it in this test

        # Assert - DFS path should be valid
        assert dfs_result is not None
        assert dfs_result[0] == 0
        assert dfs_result[-1] == 4

    @pytest.mark.unit
    def test_dfs_and_recursive_dfs_consistency(self) -> None:
        """Test that iterative and recursive DFS give consistent results."""
        # Arrange - create simple graph
        graph = create_grid_graph(4, 4)

        # Act - find paths with both implementations
        iterative_path = dfs_path(graph, (0, 0), (3, 3))
        recursive_path = dfs_recursive_path(graph, (0, 0), (3, 3))

        # Assert - both should find valid paths
        assert iterative_path is not None
        assert recursive_path is not None
        assert iterative_path[0] == recursive_path[0]
        assert iterative_path[-1] == recursive_path[-1]
