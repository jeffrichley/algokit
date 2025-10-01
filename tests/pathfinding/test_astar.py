"""Tests for the A* pathfinding algorithm implementation."""

import networkx as nx
import pytest

from algokit.algorithms.pathfinding.astar import (
    astar_all_distances,
    astar_shortest_distance,
    astar_shortest_path,
)
from algokit.core.helpers import HarborNetScenario, create_grid_graph
from algokit.core.utils.distances import (
    euclidean_distance,
    manhattan_distance,
    zero_heuristic,
)


def create_weighted_grid_graph(
    width: int, height: int, default_weight: float = 1.0
) -> nx.Graph:
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


class TestAStarShortestPath:
    """Test the pure A* shortest path implementation."""

    @pytest.mark.unit
    def test_astar_simple_path_with_manhattan_heuristic(self) -> None:
        """Test A* finds shortest path in simple graph with Manhattan heuristic."""
        # Arrange - create a simple 3x3 grid
        graph = create_grid_graph(3, 3)

        # Act - find path from corner to corner using Manhattan heuristic
        result = astar_shortest_path(graph, (0, 0), (2, 2), manhattan_distance)

        # Assert - verify path is correct
        assert result is not None
        path, cost = result
        assert len(path) == 5  # Manhattan distance + 1
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        assert cost == 4.0  # Manhattan distance
        assert all(graph.has_edge(path[i], path[i + 1]) for i in range(len(path) - 1))

    @pytest.mark.unit
    def test_astar_simple_path_with_euclidean_heuristic(self) -> None:
        """Test A* finds shortest path in simple graph with Euclidean heuristic."""
        # Arrange - create a simple 3x3 grid
        graph = create_grid_graph(3, 3)

        # Act - find path from corner to corner using Euclidean heuristic
        result = astar_shortest_path(graph, (0, 0), (2, 2), euclidean_distance)

        # Assert - verify path is correct
        assert result is not None
        path, cost = result
        assert len(path) == 5  # Manhattan distance + 1
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        assert cost == 4.0  # Manhattan distance
        assert all(graph.has_edge(path[i], path[i + 1]) for i in range(len(path) - 1))

    @pytest.mark.unit
    def test_astar_with_zero_heuristic_equivalent_to_dijkstra(self) -> None:
        """Test A* with zero heuristic is equivalent to Dijkstra."""
        # Arrange - create weighted graph with different edge weights
        graph = nx.Graph()
        graph.add_edges_from(
            [
                ("A", "B", {"weight": 1.0}),
                ("B", "C", {"weight": 1.0}),
                ("A", "C", {"weight": 5.0}),  # Direct path is more expensive
            ]
        )

        # Act - find path with A* using zero heuristic
        def zero_heuristic_func(n: str, g: str) -> float:
            return 0.0

        result = astar_shortest_path(graph, "A", "C", zero_heuristic_func)

        # Assert - should choose the longer but cheaper path (like Dijkstra)
        assert result is not None
        path, weight = result
        assert path == ["A", "B", "C"]
        assert weight == 2.0

    @pytest.mark.unit
    def test_astar_weighted_path_with_heuristic(self) -> None:
        """Test A* finds optimal path in weighted graph with heuristic."""
        # Arrange - create graph with different edge weights
        graph = nx.Graph()
        graph.add_edges_from(
            [
                ("A", "B", {"weight": 1.0}),
                ("B", "C", {"weight": 1.0}),
                ("A", "C", {"weight": 5.0}),  # Direct path is more expensive
            ]
        )

        # Act - find path from A to C with simple heuristic
        def simple_heuristic_func(n: str, g: str) -> float:
            return 0.5  # Small heuristic value

        result = astar_shortest_path(graph, "A", "C", simple_heuristic_func)

        # Assert - should still choose the optimal path
        assert result is not None
        path, weight = result
        assert path == ["A", "B", "C"]
        assert weight == 2.0

    @pytest.mark.unit
    def test_astar_no_path(self) -> None:
        """Test A* returns None when no path exists."""
        # Arrange - create graph with disconnected components
        graph = nx.Graph()
        graph.add_nodes_from(["A", "B", "C"])
        graph.add_edge("A", "B", weight=1.0)
        # 'C' is disconnected

        # Act - try to find path between disconnected nodes
        result = astar_shortest_path(graph, "A", "C", zero_heuristic)

        # Assert - no path should be found
        assert result is None

    @pytest.mark.unit
    def test_astar_same_start_goal(self) -> None:
        """Test A* raises error for same start and goal."""
        # Arrange - create simple graph
        graph = create_grid_graph(3, 3)

        # Act & Assert - should raise ValueError
        with pytest.raises(ValueError, match="Start and goal nodes cannot be the same"):
            astar_shortest_path(graph, (0, 0), (0, 0), manhattan_distance)

    @pytest.mark.unit
    def test_astar_invalid_start_node(self) -> None:
        """Test A* raises error for invalid start node."""
        # Arrange - create simple graph
        graph = create_grid_graph(3, 3)

        # Act & Assert - should raise ValueError
        with pytest.raises(
            ValueError, match="Start node \\(99, 99\\) not found in graph"
        ):
            astar_shortest_path(graph, (99, 99), (2, 2), manhattan_distance)

    @pytest.mark.unit
    def test_astar_invalid_goal_node(self) -> None:
        """Test A* raises error for invalid goal node."""
        # Arrange - create simple graph
        graph = create_grid_graph(3, 3)

        # Act & Assert - should raise ValueError
        with pytest.raises(
            ValueError, match="Goal node \\(99, 99\\) not found in graph"
        ):
            astar_shortest_path(graph, (0, 0), (99, 99), manhattan_distance)

    @pytest.mark.unit
    def test_astar_with_obstacles(self) -> None:
        """Test A* finds path around obstacles."""
        # Arrange - create grid with obstacles
        obstacles = {(1, 1), (2, 1)}
        graph = create_grid_graph(3, 3, blocked=obstacles)

        # Act - find path from corner to corner
        result = astar_shortest_path(graph, (0, 0), (2, 2), manhattan_distance)

        # Assert - path should avoid obstacles
        assert result is not None
        path, cost = result
        assert len(path) > 4  # Should be longer due to obstacles
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        assert not any(node in obstacles for node in path)

    @pytest.mark.unit
    def test_astar_path_validity(self) -> None:
        """Test that A* returns a valid path."""
        # Arrange - create a larger grid
        graph = create_grid_graph(5, 5)

        # Act - find path from one corner to another
        result = astar_shortest_path(graph, (0, 0), (4, 4), manhattan_distance)

        # Assert - path should be valid
        assert result is not None
        path, cost = result
        assert len(path) >= 9  # At least Manhattan distance
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)

        # Verify each step is a valid edge
        for i in range(len(path) - 1):
            assert graph.has_edge(path[i], path[i + 1])

    @pytest.mark.unit
    def test_astar_deterministic(self) -> None:
        """Test that A* produces consistent results."""
        # Arrange - create simple graph
        graph = create_grid_graph(3, 3)

        # Act - run A* multiple times
        results = []
        for _ in range(5):
            result = astar_shortest_path(graph, (0, 0), (2, 2), manhattan_distance)
            results.append(result)

        # Assert - all results should be the same (deterministic)
        assert all(result == results[0] for result in results)

    @pytest.mark.unit
    def test_astar_heuristic_admissible(self) -> None:
        """Test that admissible heuristics produce optimal results."""
        # Arrange - create graph where heuristic helps guide search
        graph = create_grid_graph(4, 4)

        # Act - find path with Manhattan heuristic (admissible for grid)
        result = astar_shortest_path(graph, (0, 0), (3, 3), manhattan_distance)

        # Assert - should find optimal path
        assert result is not None
        path, cost = result
        assert cost == 6.0  # Manhattan distance
        assert len(path) == 7  # Path length

    @pytest.mark.unit
    def test_astar_handles_missing_weights(self) -> None:
        """Test that A* handles edges without weight attributes."""
        # Arrange - create graph without weight attributes
        graph = nx.Graph()
        graph.add_edges_from(
            [
                ("A", "B"),
                ("B", "C"),
                ("A", "C"),
            ]
        )

        # Act - find shortest path
        def zero_heuristic_func(n: str, g: str) -> float:
            return 0.0

        result = astar_shortest_path(graph, "A", "C", zero_heuristic_func)

        # Assert - should work with default weight of 1.0
        assert result is not None
        path, weight = result
        assert len(path) == 2  # Direct path
        assert weight == 1.0  # Default weight


class TestAStarShortestDistance:
    """Test the A* shortest distance implementation."""

    @pytest.mark.unit
    def test_astar_shortest_distance_simple(self) -> None:
        """Test A* shortest distance in simple graph."""
        # Arrange - create simple grid
        graph = create_grid_graph(3, 3)

        # Act - find shortest distance
        distance = astar_shortest_distance(graph, (0, 0), (2, 2), manhattan_distance)

        # Assert - should be Manhattan distance
        assert distance == 4.0

    @pytest.mark.unit
    def test_astar_shortest_distance_weighted(self) -> None:
        """Test A* shortest distance in weighted graph."""
        # Arrange - create graph with different weights
        graph = nx.Graph()
        graph.add_edges_from(
            [
                ("A", "B", {"weight": 2.0}),
                ("B", "C", {"weight": 3.0}),
                ("A", "C", {"weight": 10.0}),  # Direct path is expensive
            ]
        )

        # Act - find shortest distance
        def zero_heuristic_func(n: str, g: str) -> float:
            return 0.0

        distance = astar_shortest_distance(graph, "A", "C", zero_heuristic_func)

        # Assert - should choose the cheaper path
        assert distance == 5.0  # 2.0 + 3.0

    @pytest.mark.unit
    def test_astar_shortest_distance_no_path(self) -> None:
        """Test A* shortest distance when no path exists."""
        # Arrange - create disconnected graph
        graph = nx.Graph()
        graph.add_nodes_from(["A", "B", "C"])
        graph.add_edge("A", "B", weight=1.0)
        # 'C' is disconnected

        # Act - try to find distance between disconnected nodes
        distance = astar_shortest_distance(graph, "A", "C", zero_heuristic)

        # Assert - should return None
        assert distance is None

    @pytest.mark.unit
    def test_astar_shortest_distance_same_start_goal(self) -> None:
        """Test A* shortest distance for same start and goal."""
        # Arrange - create simple graph
        graph = create_grid_graph(3, 3)

        # Act & Assert - should raise error
        with pytest.raises(ValueError, match="Start and goal nodes cannot be the same"):
            astar_shortest_distance(graph, (0, 0), (0, 0), manhattan_distance)


class TestAStarAllDistances:
    """Test the A* all distances implementation."""

    @pytest.mark.unit
    def test_astar_all_distances_simple(self) -> None:
        """Test A* finds all distances in simple graph."""
        # Arrange - create simple grid
        graph = create_grid_graph(3, 3)

        # Act - find all distances from center
        distances = astar_all_distances(graph, (1, 1))

        # Assert - should find all nodes
        assert len(distances) == 9  # All 9 nodes in 3x3 grid
        assert (1, 1) in distances
        assert distances[(1, 1)] == 0.0  # Distance to self is 0

    @pytest.mark.unit
    def test_astar_all_distances_with_limit(self) -> None:
        """Test A* respects distance limit."""
        # Arrange - create simple grid
        graph = create_grid_graph(5, 5)

        # Act - find nodes within distance 2 from center
        distances = astar_all_distances(graph, (2, 2), max_distance=2.0)

        # Assert - should only find nodes within distance 2
        assert all(distance <= 2.0 for distance in distances.values())
        assert (2, 2) in distances
        assert distances[(2, 2)] == 0.0

    @pytest.mark.unit
    def test_astar_all_distances_disconnected(self) -> None:
        """Test A* finds only connected component."""
        # Arrange - create graph with disconnected components
        graph = nx.Graph()
        graph.add_edges_from(
            [
                ("A", "B", {"weight": 1.0}),
                ("B", "C", {"weight": 2.0}),
                ("D", "E", {"weight": 1.0}),
            ]
        )

        # Act - find all distances from first component
        distances = astar_all_distances(graph, "A")

        # Assert - should only find first component
        assert len(distances) == 3  # Nodes A, B, C
        assert "A" in distances
        assert "B" in distances
        assert "C" in distances
        assert "D" not in distances
        assert "E" not in distances

    @pytest.mark.unit
    def test_astar_all_distances_invalid_start(self) -> None:
        """Test A* raises error for invalid start node."""
        # Arrange - create simple graph
        graph = create_grid_graph(3, 3)

        # Act & Assert - should raise error
        with pytest.raises(
            ValueError, match="Start node \\(99, 99\\) not found in graph"
        ):
            astar_all_distances(graph, (99, 99))


class TestAStarComparisonWithOtherAlgorithms:
    """Test A* behavior compared to other algorithms."""

    @pytest.mark.unit
    def test_astar_vs_dijkstra_unweighted(self) -> None:
        """Test that A* with zero heuristic gives same results as Dijkstra."""
        # Arrange - create unweighted graph (all weights = 1.0)
        graph = create_weighted_grid_graph(3, 3, default_weight=1.0)

        # Act - find path with A* (zero heuristic)
        def zero_heuristic_func(n: tuple[int, int], g: tuple[int, int]) -> float:
            return 0.0

        astar_result = astar_shortest_path(graph, (0, 0), (2, 2), zero_heuristic_func)

        # Import Dijkstra for comparison
        from algokit.algorithms.pathfinding.dijkstra import dijkstra_shortest_path

        dijkstra_result = dijkstra_shortest_path(graph, (0, 0), (2, 2))

        # Assert - both should find same path and cost
        assert astar_result is not None
        assert dijkstra_result is not None
        astar_path, astar_cost = astar_result
        dijkstra_path, dijkstra_cost = dijkstra_result
        assert len(astar_path) == len(dijkstra_path)
        assert astar_cost == dijkstra_cost

    @pytest.mark.unit
    def test_astar_vs_bfs_unweighted(self) -> None:
        """Test that A* with zero heuristic gives same results as BFS for unweighted graphs."""
        # Arrange - create unweighted graph (all weights = 1.0)
        graph = create_weighted_grid_graph(3, 3, default_weight=1.0)

        # Act - find path with A* (zero heuristic)
        def zero_heuristic_func(n: tuple[int, int], g: tuple[int, int]) -> float:
            return 0.0

        astar_result = astar_shortest_path(graph, (0, 0), (2, 2), zero_heuristic_func)

        # Import BFS for comparison
        from algokit.algorithms.pathfinding.bfs import bfs_shortest_path

        bfs_result = bfs_shortest_path(graph, (0, 0), (2, 2))

        # Assert - both should find paths of same length
        assert astar_result is not None
        assert bfs_result is not None
        astar_path, astar_cost = astar_result
        assert len(astar_path) == len(bfs_result)
        assert astar_cost == len(bfs_result) - 1  # Cost = path length - 1

    @pytest.mark.unit
    def test_astar_optimal_for_weighted(self) -> None:
        """Test that A* finds optimal path in weighted graph."""
        # Arrange - create graph where shortest path by hops != shortest by weight
        graph = nx.Graph()
        graph.add_edges_from(
            [
                ("A", "B", {"weight": 1.0}),
                ("B", "C", {"weight": 1.0}),
                ("C", "D", {"weight": 1.0}),
                ("A", "D", {"weight": 2.5}),  # Direct path has lower total weight
            ]
        )

        # Act - find shortest path with A*
        def zero_heuristic_func(n: str, g: str) -> float:
            return 0.0

        result = astar_shortest_path(graph, "A", "D", zero_heuristic_func)

        # Assert - should choose direct path (lower weight)
        assert result is not None
        path, weight = result
        assert path == ["A", "D"]  # Direct path
        assert weight == 2.5


class TestAStarHeuristicFunctions:
    """Test A* heuristic functions."""

    @pytest.mark.unit
    def test_manhattan_distance_calculation(self) -> None:
        """Test Manhattan distance calculation."""
        # Arrange - test various coordinate pairs
        test_cases = [
            ((0, 0), (3, 4), 7.0),
            ((1, 1), (1, 1), 0.0),
            ((0, 0), (0, 5), 5.0),
            ((0, 0), (5, 0), 5.0),
        ]

        # Act & Assert - verify distance calculations
        for node1, node2, expected in test_cases:
            distance = manhattan_distance(node1, node2)
            assert distance == expected

    @pytest.mark.unit
    def test_euclidean_distance_calculation(self) -> None:
        """Test Euclidean distance calculation."""
        # Arrange - test various coordinate pairs
        test_cases = [
            ((0, 0), (3, 4), 5.0),
            ((1, 1), (1, 1), 0.0),
            ((0, 0), (0, 5), 5.0),
            ((0, 0), (5, 0), 5.0),
            ((0, 0), (1, 1), pytest.approx(1.414, abs=0.001)),
        ]

        # Act & Assert - verify distance calculations
        for node1, node2, expected in test_cases:
            distance = euclidean_distance(node1, node2)
            assert distance == expected

    @pytest.mark.unit
    def test_astar_with_different_heuristics(self) -> None:
        """Test A* with different heuristic functions."""
        # Arrange - create simple graph
        graph = create_grid_graph(3, 3)

        # Act - find path with different heuristics
        def zero_heuristic_func(n: tuple[int, int], g: tuple[int, int]) -> float:
            return 0.0

        manhattan_result = astar_shortest_path(
            graph, (0, 0), (2, 2), manhattan_distance
        )
        euclidean_result = astar_shortest_path(
            graph, (0, 0), (2, 2), euclidean_distance
        )
        zero_result = astar_shortest_path(graph, (0, 0), (2, 2), zero_heuristic_func)

        # Assert - all should find optimal path (since heuristics are admissible)
        assert manhattan_result is not None
        assert euclidean_result is not None
        assert zero_result is not None

        manhattan_path, manhattan_cost = manhattan_result
        euclidean_path, euclidean_cost = euclidean_result
        zero_path, zero_cost = zero_result

        # All should have same cost (optimal)
        assert manhattan_cost == euclidean_cost == zero_cost
        assert len(manhattan_path) == len(euclidean_path) == len(zero_path)


class TestAStarWithHarborNet:
    """Test A* with HarborNet scenarios."""

    @pytest.mark.integration
    def test_astar_harbor_net_scenario(self) -> None:
        """Test A* with HarborNet scenario data."""
        # Arrange - create HarborNet scenario
        scenario = HarborNetScenario(
            name="Test Harbor",
            width=5,
            height=5,
            start=(0, 0),
            goal=(4, 4),
            obstacles={(1, 1), (2, 2), (3, 3)},
        )
        graph = create_grid_graph(
            scenario.width, scenario.height, blocked=scenario.obstacles
        )

        # Act - find path using A*
        result = astar_shortest_path(
            graph, scenario.start, scenario.goal, manhattan_distance
        )

        # Assert - path should avoid obstacles
        assert result is not None
        path, cost = result
        assert path[0] == scenario.start
        assert path[-1] == scenario.goal
        assert not any(node in scenario.obstacles for node in path)

    @pytest.mark.integration
    def test_astar_harbor_net_with_different_heuristics(self) -> None:
        """Test A* with HarborNet scenario using different heuristics."""
        # Arrange - create HarborNet scenario
        scenario = HarborNetScenario(
            name="Test Harbor",
            width=4,
            height=4,
            start=(0, 0),
            goal=(3, 3),
            obstacles={(1, 1), (2, 2)},
        )
        graph = create_grid_graph(
            scenario.width, scenario.height, blocked=scenario.obstacles
        )

        # Act - find path with different heuristics
        manhattan_result = astar_shortest_path(
            graph, scenario.start, scenario.goal, manhattan_distance
        )
        euclidean_result = astar_shortest_path(
            graph, scenario.start, scenario.goal, euclidean_distance
        )

        # Assert - both should find valid paths
        assert manhattan_result is not None
        assert euclidean_result is not None

        manhattan_path, manhattan_cost = manhattan_result
        euclidean_path, euclidean_cost = euclidean_result

        assert manhattan_path[0] == scenario.start
        assert manhattan_path[-1] == scenario.goal
        assert euclidean_path[0] == scenario.start
        assert euclidean_path[-1] == scenario.goal

        # Both should find optimal cost (heuristics are admissible)
        assert manhattan_cost == euclidean_cost
