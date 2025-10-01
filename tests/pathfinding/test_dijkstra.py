"""Tests for the Dijkstra's algorithm implementation."""

import networkx as nx
import pytest

from algokit.algorithms.pathfinding.dijkstra import (
    dijkstra_all_distances,
    dijkstra_shortest_distance,
    dijkstra_shortest_path,
)


def create_weighted_grid_graph(width: int, height: int, default_weight: float = 1.0) -> nx.Graph:
    """Create a weighted grid graph for testing."""
    graph = nx.Graph()
    
    # Add nodes
    for x in range(width):
        for y in range(height):
            graph.add_node((x, y))
    
    # Add weighted edges
    for x in range(width):
        for y in range(height):
            # Right neighbor
            if x + 1 < width:
                graph.add_edge((x, y), (x + 1, y), weight=default_weight)
            # Bottom neighbor
            if y + 1 < height:
                graph.add_edge((x, y), (x, y + 1), weight=default_weight)
    
    return graph


class TestDijkstraShortestPath:
    """Test the pure Dijkstra shortest path implementation."""

    @pytest.mark.unit
    def test_dijkstra_simple_path(self) -> None:
        """Test Dijkstra finds shortest path in simple weighted graph."""
        # Arrange - create a simple weighted 3x3 grid
        graph = create_weighted_grid_graph(3, 3, default_weight=1.0)

        # Act - find path from corner to corner
        result = dijkstra_shortest_path(graph, (0, 0), (2, 2))

        # Assert - verify path is correct
        assert result is not None
        path, weight = result
        assert len(path) == 5  # Manhattan distance + 1
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        assert weight == 4.0  # Manhattan distance
        assert all(graph.has_edge(path[i], path[i + 1]) for i in range(len(path) - 1))

    @pytest.mark.unit
    def test_dijkstra_weighted_path(self) -> None:
        """Test Dijkstra finds optimal path in weighted graph."""
        # Arrange - create graph with different edge weights
        graph = nx.Graph()
        graph.add_edges_from([
            ("A", "B", {"weight": 1.0}),
            ("B", "C", {"weight": 1.0}),
            ("A", "C", {"weight": 5.0}),  # Direct path is more expensive
        ])

        # Act - find path from A to C
        result = dijkstra_shortest_path(graph, "A", "C")

        # Assert - should choose the longer but cheaper path
        assert result is not None
        path, weight = result
        assert path == ["A", "B", "C"]
        assert weight == 2.0

    @pytest.mark.unit
    def test_dijkstra_no_path(self) -> None:
        """Test Dijkstra returns None when no path exists."""
        # Arrange - create graph with disconnected components
        graph = nx.Graph()
        graph.add_nodes_from(["A", "B", "C"])
        graph.add_edge("A", "B", weight=1.0)
        # 'C' is disconnected

        # Act - try to find path between disconnected nodes
        result = dijkstra_shortest_path(graph, "A", "C")

        # Assert - no path should be found
        assert result is None

    @pytest.mark.unit
    def test_dijkstra_same_start_goal(self) -> None:
        """Test Dijkstra raises error for same start and goal."""
        # Arrange - create simple graph
        graph = create_weighted_grid_graph(3, 3)

        # Act & Assert - should raise ValueError
        with pytest.raises(ValueError, match="Start and goal nodes cannot be the same"):
            dijkstra_shortest_path(graph, (0, 0), (0, 0))

    @pytest.mark.unit
    def test_dijkstra_invalid_start_node(self) -> None:
        """Test Dijkstra raises error for invalid start node."""
        # Arrange - create simple graph
        graph = create_weighted_grid_graph(3, 3)

        # Act & Assert - should raise ValueError
        with pytest.raises(ValueError, match="Start node \\(99, 99\\) not found in graph"):
            dijkstra_shortest_path(graph, (99, 99), (2, 2))

    @pytest.mark.unit
    def test_dijkstra_invalid_goal_node(self) -> None:
        """Test Dijkstra raises error for invalid goal node."""
        # Arrange - create simple graph
        graph = create_weighted_grid_graph(3, 3)

        # Act & Assert - should raise ValueError
        with pytest.raises(ValueError, match="Goal node \\(99, 99\\) not found in graph"):
            dijkstra_shortest_path(graph, (0, 0), (99, 99))

    @pytest.mark.unit
    def test_dijkstra_with_obstacles(self) -> None:
        """Test Dijkstra finds path around obstacles."""
        # Arrange - create grid with obstacles (high-weight edges)
        graph = create_weighted_grid_graph(3, 3, default_weight=1.0)
        # Make some edges very expensive (effectively blocking)
        graph.add_edge((1, 1), (2, 1), weight=1000.0)
        graph.add_edge((2, 1), (2, 2), weight=1000.0)

        # Act - find path from corner to corner
        result = dijkstra_shortest_path(graph, (0, 0), (2, 2))

        # Assert - path should avoid expensive edges
        assert result is not None
        path, weight = result
        # Should find an alternative path
        assert weight < 1000.0

    @pytest.mark.unit
    def test_dijkstra_path_validity(self) -> None:
        """Test that Dijkstra returns a valid path."""
        # Arrange - create a larger weighted grid
        graph = create_weighted_grid_graph(5, 5, default_weight=2.0)

        # Act - find path from one corner to another
        result = dijkstra_shortest_path(graph, (0, 0), (4, 4))

        # Assert - path should be valid
        assert result is not None
        path, weight = result
        assert len(path) >= 9  # At least Manhattan distance
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)
        
        # Verify each step is a valid edge
        for i in range(len(path) - 1):
            assert graph.has_edge(path[i], path[i + 1])

    @pytest.mark.unit
    def test_dijkstra_deterministic(self) -> None:
        """Test that Dijkstra produces consistent results."""
        # Arrange - create simple graph
        graph = create_weighted_grid_graph(3, 3)

        # Act - run Dijkstra multiple times
        results = []
        for _ in range(5):
            result = dijkstra_shortest_path(graph, (0, 0), (2, 2))
            results.append(result)

        # Assert - all results should be the same (deterministic)
        assert all(result == results[0] for result in results)


class TestDijkstraShortestDistance:
    """Test the Dijkstra shortest distance implementation."""

    @pytest.mark.unit
    def test_dijkstra_shortest_distance_simple(self) -> None:
        """Test Dijkstra shortest distance in simple graph."""
        # Arrange - create simple weighted graph
        graph = create_weighted_grid_graph(3, 3, default_weight=1.0)

        # Act - find shortest distance
        distance = dijkstra_shortest_distance(graph, (0, 0), (2, 2))

        # Assert - should be Manhattan distance
        assert distance == 4.0

    @pytest.mark.unit
    def test_dijkstra_shortest_distance_weighted(self) -> None:
        """Test Dijkstra shortest distance in weighted graph."""
        # Arrange - create graph with different weights
        graph = nx.Graph()
        graph.add_edges_from([
            ("A", "B", {"weight": 2.0}),
            ("B", "C", {"weight": 3.0}),
            ("A", "C", {"weight": 10.0}),  # Direct path is expensive
        ])

        # Act - find shortest distance
        distance = dijkstra_shortest_distance(graph, "A", "C")

        # Assert - should choose the cheaper path
        assert distance == 5.0  # 2.0 + 3.0

    @pytest.mark.unit
    def test_dijkstra_shortest_distance_no_path(self) -> None:
        """Test Dijkstra shortest distance when no path exists."""
        # Arrange - create disconnected graph
        graph = nx.Graph()
        graph.add_nodes_from(["A", "B", "C"])
        graph.add_edge("A", "B", weight=1.0)
        # 'C' is disconnected

        # Act - try to find distance between disconnected nodes
        distance = dijkstra_shortest_distance(graph, "A", "C")

        # Assert - should return None
        assert distance is None

    @pytest.mark.unit
    def test_dijkstra_shortest_distance_same_start_goal(self) -> None:
        """Test Dijkstra shortest distance for same start and goal."""
        # Arrange - create simple graph
        graph = create_weighted_grid_graph(3, 3)

        # Act & Assert - should raise error
        with pytest.raises(ValueError, match="Start and goal nodes cannot be the same"):
            dijkstra_shortest_distance(graph, (0, 0), (0, 0))


class TestDijkstraAllDistances:
    """Test the Dijkstra all distances implementation."""

    @pytest.mark.unit
    def test_dijkstra_all_distances_simple(self) -> None:
        """Test Dijkstra finds all distances in simple graph."""
        # Arrange - create simple weighted graph
        graph = create_weighted_grid_graph(3, 3, default_weight=1.0)

        # Act - find all distances from center
        distances = dijkstra_all_distances(graph, (1, 1))

        # Assert - should find all nodes
        assert len(distances) == 9  # All 9 nodes in 3x3 grid
        assert (1, 1) in distances
        assert distances[(1, 1)] == 0.0  # Distance to self is 0

    @pytest.mark.unit
    def test_dijkstra_all_distances_with_limit(self) -> None:
        """Test Dijkstra respects distance limit."""
        # Arrange - create simple weighted graph
        graph = create_weighted_grid_graph(5, 5, default_weight=1.0)

        # Act - find nodes within distance 2 from center
        distances = dijkstra_all_distances(graph, (2, 2), max_distance=2.0)

        # Assert - should only find nodes within distance 2
        assert all(distance <= 2.0 for distance in distances.values())
        assert (2, 2) in distances
        assert distances[(2, 2)] == 0.0

    @pytest.mark.unit
    def test_dijkstra_all_distances_disconnected(self) -> None:
        """Test Dijkstra finds only connected component."""
        # Arrange - create graph with disconnected components
        graph = nx.Graph()
        graph.add_edges_from([
            ("A", "B", {"weight": 1.0}),
            ("B", "C", {"weight": 2.0}),
            ("D", "E", {"weight": 1.0}),
        ])

        # Act - find all distances from first component
        distances = dijkstra_all_distances(graph, "A")

        # Assert - should only find first component
        assert len(distances) == 3  # Nodes A, B, C
        assert "A" in distances
        assert "B" in distances
        assert "C" in distances
        assert "D" not in distances
        assert "E" not in distances

    @pytest.mark.unit
    def test_dijkstra_all_distances_invalid_start(self) -> None:
        """Test Dijkstra raises error for invalid start node."""
        # Arrange - create simple graph
        graph = create_weighted_grid_graph(3, 3)

        # Act & Assert - should raise error
        with pytest.raises(ValueError, match="Start node \\(99, 99\\) not found in graph"):
            dijkstra_all_distances(graph, (99, 99))


class TestDijkstraComparisonWithBFS:
    """Test Dijkstra behavior compared to BFS."""

    @pytest.mark.unit
    def test_dijkstra_vs_bfs_unweighted(self) -> None:
        """Test that Dijkstra gives same results as BFS for unweighted graphs."""
        # Arrange - create unweighted graph (all weights = 1.0)
        graph = create_weighted_grid_graph(3, 3, default_weight=1.0)

        # Act - find path with Dijkstra
        dijkstra_result = dijkstra_shortest_path(graph, (0, 0), (2, 2))
        
        # Import BFS for comparison
        from algokit.algorithms.pathfinding.bfs import bfs_shortest_path
        bfs_result = bfs_shortest_path(graph, (0, 0), (2, 2))

        # Assert - both should find paths of same length
        assert dijkstra_result is not None
        assert bfs_result is not None
        dijkstra_path, dijkstra_weight = dijkstra_result
        assert len(dijkstra_path) == len(bfs_result)
        assert dijkstra_weight == len(bfs_result) - 1  # Weight = path length - 1

    @pytest.mark.unit
    def test_dijkstra_optimal_for_weighted(self) -> None:
        """Test that Dijkstra finds optimal path in weighted graph."""
        # Arrange - create graph where shortest path by hops != shortest by weight
        graph = nx.Graph()
        graph.add_edges_from([
            ("A", "B", {"weight": 1.0}),
            ("B", "C", {"weight": 1.0}),
            ("C", "D", {"weight": 1.0}),
            ("A", "D", {"weight": 2.5}),  # Direct path has lower total weight
        ])

        # Act - find shortest path with Dijkstra
        result = dijkstra_shortest_path(graph, "A", "D")

        # Assert - should choose direct path (lower weight)
        assert result is not None
        path, weight = result
        assert path == ["A", "D"]  # Direct path
        assert weight == 2.5

    @pytest.mark.unit
    def test_dijkstra_handles_missing_weights(self) -> None:
        """Test that Dijkstra handles edges without weight attributes."""
        # Arrange - create graph without weight attributes
        graph = nx.Graph()
        graph.add_edges_from([
            ("A", "B"),
            ("B", "C"),
            ("A", "C"),
        ])

        # Act - find shortest path
        result = dijkstra_shortest_path(graph, "A", "C")

        # Assert - should work with default weight of 1.0
        assert result is not None
        path, weight = result
        assert len(path) == 2  # Direct path
        assert weight == 1.0  # Default weight
