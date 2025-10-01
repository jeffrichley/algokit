"""Tests for the BFS pathfinding algorithm implementation."""

import networkx as nx
import pytest

from algokit.algorithms.pathfinding.bfs import (
    bfs_all_reachable,
    bfs_path_length,
    bfs_shortest_path,
)
from algokit.algorithms.pathfinding.bfs_with_events import bfs_with_data_collection
from algokit.core.helpers import HarborNetScenario, create_grid_graph


class TestBFSShortestPath:
    """Test the pure BFS shortest path implementation."""

    @pytest.mark.unit
    def test_bfs_simple_path(self) -> None:
        """Test BFS finds shortest path in simple graph."""
        # Arrange - create a simple 3x3 grid
        graph = create_grid_graph(3, 3)

        # Act - find path from corner to corner
        path = bfs_shortest_path(graph, (0, 0), (2, 2))

        # Assert - verify path is correct
        assert path is not None
        assert len(path) == 5  # Manhattan distance + 1
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        assert all(graph.has_edge(path[i], path[i + 1]) for i in range(len(path) - 1))

    @pytest.mark.unit
    def test_bfs_no_path(self) -> None:
        """Test BFS returns None when no path exists."""
        # Arrange - create graph with disconnected components
        graph = nx.Graph()
        graph.add_nodes_from([(0, 0), (1, 1), (2, 2)])
        graph.add_edge((0, 0), (1, 1))
        # (2, 2) is disconnected

        # Act - try to find path between disconnected nodes
        path = bfs_shortest_path(graph, (0, 0), (2, 2))

        # Assert - no path should be found
        assert path is None

    @pytest.mark.unit
    def test_bfs_same_start_goal(self) -> None:
        """Test BFS raises error for same start and goal."""
        # Arrange - create simple graph
        graph = create_grid_graph(3, 3)

        # Act & Assert - should raise ValueError
        with pytest.raises(ValueError, match="Start and goal nodes cannot be the same"):
            bfs_shortest_path(graph, (0, 0), (0, 0))

    @pytest.mark.unit
    def test_bfs_trivial_start_is_goal(self) -> None:
        """Test BFS behavior with single node graph."""
        # Arrange - create single node graph
        graph = nx.Graph()
        graph.add_node("A")

        # Act & Assert - should raise error for same start/goal
        with pytest.raises(ValueError, match="Start and goal nodes cannot be the same"):
            bfs_shortest_path(graph, "A", "A")

    @pytest.mark.unit
    def test_bfs_invalid_start_node(self) -> None:
        """Test BFS raises error for invalid start node."""
        # Arrange - create simple graph
        graph = create_grid_graph(3, 3)

        # Act & Assert - should raise ValueError
        with pytest.raises(ValueError, match="Start node \\(99, 99\\) not found in graph"):
            bfs_shortest_path(graph, (99, 99), (2, 2))

    @pytest.mark.unit
    def test_bfs_invalid_goal_node(self) -> None:
        """Test BFS raises error for invalid goal node."""
        # Arrange - create simple graph
        graph = create_grid_graph(3, 3)

        # Act & Assert - should raise ValueError
        with pytest.raises(ValueError, match="Goal node \\(99, 99\\) not found in graph"):
            bfs_shortest_path(graph, (0, 0), (99, 99))

    @pytest.mark.unit
    def test_bfs_with_obstacles(self) -> None:
        """Test BFS finds path around obstacles."""
        # Arrange - create grid with obstacles
        obstacles = {(1, 1), (2, 1)}
        graph = create_grid_graph(3, 3, blocked=obstacles)

        # Act - find path from corner to corner
        path = bfs_shortest_path(graph, (0, 0), (2, 2))

        # Assert - path should avoid obstacles
        assert path is not None
        assert len(path) > 4  # Should be longer due to obstacles
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        assert not any(node in obstacles for node in path)


class TestBFSWithDataCollection:
    """Test the BFS implementation with event tracking."""

    @pytest.mark.unit
    def test_bfs_with_events_simple_path(self) -> None:
        """Test BFS with events finds path and collects data."""
        # Arrange - create a simple 3x3 grid
        graph = create_grid_graph(3, 3)

        # Act - find path with event tracking
        path, events = bfs_with_data_collection(graph, (0, 0), (2, 2))

        # Assert - verify path and events
        assert path is not None
        assert len(path) == 5
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        assert len(events) > 0

        # Check event types
        event_types = {event.type.value for event in events}
        assert "dequeue" in event_types
        assert "discover" in event_types
        assert "enqueue" in event_types
        assert "goal_found" in event_types
        assert "path_reconstruct" in event_types

    @pytest.mark.unit
    def test_bfs_with_events_no_path(self) -> None:
        """Test BFS with events returns None and events when no path exists."""
        # Arrange - create graph with disconnected components
        graph = nx.Graph()
        graph.add_nodes_from([(0, 0), (1, 1), (2, 2)])
        graph.add_edge((0, 0), (1, 1))

        # Act - try to find path between disconnected nodes
        path, events = bfs_with_data_collection(graph, (0, 0), (2, 2))

        # Assert - no path but events should be collected
        assert path is None
        assert len(events) > 0
        assert "goal_found" not in {event.type.value for event in events}

    @pytest.mark.unit
    def test_bfs_with_events_error_handling(self) -> None:
        """Test BFS with events raises errors for invalid inputs."""
        # Arrange - create simple graph
        graph = create_grid_graph(3, 3)

        # Act & Assert - should raise ValueError for same start/goal
        with pytest.raises(ValueError, match="Start and goal nodes cannot be the same"):
            bfs_with_data_collection(graph, (0, 0), (0, 0))


class TestBFSPathLength:
    """Test the BFS path length implementation."""

    @pytest.mark.unit
    def test_bfs_path_length_simple(self) -> None:
        """Test BFS path length calculation."""
        # Arrange - create a simple 3x3 grid
        graph = create_grid_graph(3, 3)

        # Act - calculate path length
        length = bfs_path_length(graph, (0, 0), (2, 2))

        # Assert - verify length is correct
        assert length == 4  # Manhattan distance

    @pytest.mark.unit
    def test_bfs_path_length_no_path(self) -> None:
        """Test BFS path length returns None when no path exists."""
        # Arrange - create graph with disconnected components
        graph = nx.Graph()
        graph.add_nodes_from([(0, 0), (1, 1), (2, 2)])
        graph.add_edge((0, 0), (1, 1))

        # Act - calculate path length between disconnected nodes
        length = bfs_path_length(graph, (0, 0), (2, 2))

        # Assert - should return None
        assert length is None

    @pytest.mark.unit
    def test_bfs_path_length_same_start_goal(self) -> None:
        """Test BFS path length raises error for same start and goal."""
        # Arrange - create simple graph
        graph = create_grid_graph(3, 3)

        # Act & Assert - should raise ValueError
        with pytest.raises(ValueError, match="Start and goal nodes cannot be the same"):
            bfs_path_length(graph, (0, 0), (0, 0))


class TestBFSAllReachable:
    """Test the BFS all reachable implementation."""

    @pytest.mark.unit
    def test_bfs_all_reachable_unlimited(self) -> None:
        """Test BFS finds all reachable nodes without distance limit."""
        # Arrange - create a simple 3x3 grid
        graph = create_grid_graph(3, 3)

        # Act - find all reachable nodes
        reachable = bfs_all_reachable(graph, (1, 1))

        # Assert - should find all nodes
        assert len(reachable) == 9  # All 9 nodes in 3x3 grid
        assert reachable[(1, 1)] == 0  # Start node has distance 0
        assert reachable[(0, 0)] == 2  # Corner nodes have distance 2
        assert reachable[(2, 2)] == 2

    @pytest.mark.unit
    def test_bfs_all_reachable_with_limit(self) -> None:
        """Test BFS finds nodes within distance limit."""
        # Arrange - create a simple 3x3 grid
        graph = create_grid_graph(3, 3)

        # Act - find nodes within distance 1
        reachable = bfs_all_reachable(graph, (1, 1), max_distance=1)

        # Assert - should find only immediate neighbors
        assert len(reachable) == 5  # Center + 4 neighbors
        assert reachable[(1, 1)] == 0
        assert reachable[(0, 1)] == 1
        assert reachable[(2, 1)] == 1
        assert reachable[(1, 0)] == 1
        assert reachable[(1, 2)] == 1

    @pytest.mark.unit
    def test_bfs_all_reachable_invalid_start(self) -> None:
        """Test BFS all reachable raises error for invalid start node."""
        # Arrange - create simple graph
        graph = create_grid_graph(3, 3)

        # Act & Assert - should raise ValueError
        with pytest.raises(ValueError, match="Start node \\(99, 99\\) not found in graph"):
            bfs_all_reachable(graph, (99, 99))


class TestBFSWithHarborNet:
    """Test BFS with HarborNet scenarios."""

    @pytest.mark.integration
    def test_bfs_harbor_net_scenario(self) -> None:
        """Test BFS with HarborNet scenario data."""
        # Arrange - create HarborNet scenario
        scenario = HarborNetScenario(
            name="Test Harbor",
            width=5,
            height=5,
            start=(0, 0),
            goal=(4, 4),
            obstacles={(1, 1), (2, 2), (3, 3)},
        )
        graph = create_grid_graph(scenario.width, scenario.height, blocked=scenario.obstacles)

        # Act - find path using BFS
        path = bfs_shortest_path(graph, scenario.start, scenario.goal)

        # Assert - path should avoid obstacles
        assert path is not None
        assert path[0] == scenario.start
        assert path[-1] == scenario.goal
        assert not any(node in scenario.obstacles for node in path)

    @pytest.mark.integration
    def test_bfs_harbor_net_with_events(self) -> None:
        """Test BFS with events using HarborNet scenario."""
        # Arrange - create HarborNet scenario
        scenario = HarborNetScenario(
            name="Test Harbor",
            width=4,
            height=4,
            start=(0, 0),
            goal=(3, 3),
            obstacles={(1, 1), (2, 2)},
        )
        graph = create_grid_graph(scenario.width, scenario.height, blocked=scenario.obstacles)

        # Act - find path with event tracking
        path, events = bfs_with_data_collection(graph, scenario.start, scenario.goal)

        # Assert - verify path and events
        assert path is not None
        assert len(events) > 0
        assert path[0] == scenario.start
        assert path[-1] == scenario.goal

        # Check that goal found event exists
        goal_events = [e for e in events if e.type.value == "goal_found"]
        assert len(goal_events) == 1
        assert goal_events[0].node == scenario.goal
