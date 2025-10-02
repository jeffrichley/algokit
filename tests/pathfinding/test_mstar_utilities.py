"""Focused tests for M* utility functions to improve coverage.

This module contains tests for utility functions in the M* algorithm
to improve test coverage without complex scenarios.
"""

import math

import networkx as nx
import pytest

from algokit.algorithms.pathfinding.mstar import (
    astar_single,
    edge_weight,
    euclid,
    has_edge_swap,
    has_vertex_collision,
    pad_to_length,
    reconstruct_path,
    single_source_to_goal_costs,
)


class TestMStarUtilityFunctions:
    """Test utility functions in M* algorithm."""

    @pytest.mark.unit
    def test_edge_weight_with_weight(self) -> None:
        """Test edge_weight function with weighted edges."""
        # Arrange - create a graph with weighted edges
        graph = nx.Graph()
        graph.add_edge("A", "B", weight=3.0)
        graph.add_edge("B", "C", weight=2.5)

        # Act - get edge weights from the graph
        weight_ab = edge_weight(graph, "A", "B")
        weight_bc = edge_weight(graph, "B", "C")

        # Assert - verify correct weights are returned
        assert weight_ab == 3.0
        assert weight_bc == 2.5

    @pytest.mark.unit
    def test_edge_weight_without_weight(self) -> None:
        """Test edge_weight function with unweighted edges."""
        # Arrange - create a graph with unweighted edges
        graph = nx.Graph()
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        # Act - get edge weights from the graph
        weight_ab = edge_weight(graph, "A", "B")
        weight_bc = edge_weight(graph, "B", "C")

        # Assert - verify default weight of 1.0 is returned
        assert weight_ab == 1.0
        assert weight_bc == 1.0

    @pytest.mark.unit
    def test_edge_weight_self_loop(self) -> None:
        """Test edge_weight function with self-loops (waiting)."""
        # Arrange - create a graph with a single node
        graph = nx.Graph()
        graph.add_node("A")

        # Act - get weight for self-loop
        weight = edge_weight(graph, "A", "A")

        # Assert - verify self-loop has weight 1.0
        assert weight == 1.0

    @pytest.mark.unit
    def test_edge_weight_missing_edge(self) -> None:
        """Test edge_weight function with missing edge."""
        # Arrange - create a graph with disconnected nodes
        graph = nx.Graph()
        graph.add_node("A")
        graph.add_node("B")

        # Act & Assert - verify KeyError is raised for missing edge
        with pytest.raises(KeyError):
            edge_weight(graph, "A", "B")

    @pytest.mark.unit
    def test_euclid_with_numeric_tuples(self) -> None:
        """Test euclid function with numeric tuples."""
        # Arrange - prepare numeric coordinate tuples
        point1 = (0, 0)
        point2 = (3, 4)
        point3 = (1, 1)

        # Act - calculate distances between points
        dist1 = euclid(point1, point2)
        dist2 = euclid(point1, point1)
        dist3 = euclid(point1, point3)

        # Assert - verify correct Euclidean distances
        assert dist1 == 5.0
        assert dist2 == 0.0
        assert dist3 == math.sqrt(2)

    @pytest.mark.unit
    def test_euclid_with_non_numeric(self) -> None:
        """Test euclid function with non-numeric data."""
        # Arrange - prepare non-numeric data
        non_numeric1 = "A"
        non_numeric2 = "B"
        mixed_data = (1, 2)

        # Act - attempt to calculate distances
        result1 = euclid(non_numeric1, non_numeric2)
        result2 = euclid(mixed_data, non_numeric1)

        # Assert - verify None is returned for non-numeric data
        assert result1 is None
        assert result2 is None

    @pytest.mark.unit
    def test_euclid_with_different_lengths(self) -> None:
        """Test euclid function with tuples of different lengths."""
        # Arrange - prepare tuples of different lengths
        tuple1 = (1, 2)
        tuple2 = (3, 4, 5)

        # Act - attempt to calculate distance
        result = euclid(tuple1, tuple2)

        # Assert - verify distance is calculated using available dimensions
        assert result is not None
        assert abs(result - math.sqrt(8)) < 1e-10  # sqrt((3-1)^2 + (4-2)^2)

    @pytest.mark.unit
    def test_has_vertex_collision_same_position(self) -> None:
        """Test has_vertex_collision with same position."""
        # Arrange - prepare same position coordinates
        pos1 = (1, 1)
        pos2 = (1, 1)
        radius = 0.5

        # Act - check for collision at same position
        result = has_vertex_collision(pos1, pos2, radius)

        # Assert - verify collision is detected at same position
        assert result is True

    @pytest.mark.unit
    def test_has_vertex_collision_within_radius(self) -> None:
        """Test has_vertex_collision within collision radius."""
        # Arrange - prepare positions within and outside radius
        pos1 = (0, 0)
        pos2 = (1, 0)
        large_radius = 1.5
        small_radius = 0.5

        # Act - check for collisions with different radii
        result_large = has_vertex_collision(pos1, pos2, large_radius)
        result_small = has_vertex_collision(pos1, pos2, small_radius)

        # Assert - verify collision detection based on radius
        assert result_large is True
        assert result_small is False

    @pytest.mark.unit
    def test_has_vertex_collision_non_numeric(self) -> None:
        """Test has_vertex_collision with non-numeric positions."""
        # Arrange - prepare non-numeric positions
        pos1 = "A"
        pos2 = "B"
        radius = 1.0

        # Act - check for collision with non-numeric positions
        result = has_vertex_collision(pos1, pos2, radius)

        # Assert - verify no collision detected for non-numeric data
        assert result is False

    @pytest.mark.unit
    def test_has_edge_swap_true(self) -> None:
        """Test has_edge_swap with actual edge swap."""
        # Arrange - prepare positions that create an edge swap
        prev1 = "A"
        next1 = "B"
        prev2 = "B"
        next2 = "A"

        # Act - check for edge swap
        result = has_edge_swap(prev1, next1, prev2, next2)

        # Assert - verify edge swap is detected
        assert result is True

    @pytest.mark.unit
    def test_has_edge_swap_false(self) -> None:
        """Test has_edge_swap without edge swap."""
        # Arrange - prepare positions that don't create edge swap
        prev1 = "A"
        next1 = "B"
        prev2 = "A"
        next2 = "C"

        # Act - check for edge swap
        result = has_edge_swap(prev1, next1, prev2, next2)

        # Assert - verify no edge swap is detected
        assert result is False

    @pytest.mark.unit
    def test_pad_to_length_already_long_enough(self) -> None:
        """Test pad_to_length when path is already long enough."""
        # Arrange - prepare a path that's already long enough
        path = [1, 2, 3, 4, 5]
        target_length = 3

        # Act - pad the path to target length
        result = pad_to_length(path, target_length)

        # Assert - verify original path is returned unchanged
        assert result == [1, 2, 3, 4, 5]

    @pytest.mark.unit
    def test_pad_to_length_needs_padding(self) -> None:
        """Test pad_to_length when path needs padding."""
        # Arrange - prepare a short path that needs padding
        path = [1, 2, 3]
        target_length = 5

        # Act - pad the path to target length
        result = pad_to_length(path, target_length)

        # Assert - verify path is padded with last element
        assert result == [1, 2, 3, 3, 3]

    @pytest.mark.unit
    def test_pad_to_length_empty_path(self) -> None:
        """Test pad_to_length with empty path."""
        # Arrange - prepare an empty path
        path = []
        target_length = 3

        # Act - attempt to pad empty path
        result = pad_to_length(path, target_length)

        # Assert - verify empty path is returned unchanged
        assert result == []

    @pytest.mark.unit
    def test_reconstruct_path(self) -> None:
        """Test reconstruct_path function."""
        # Arrange - prepare path reconstruction data
        came_from = {"B": "A", "C": "B"}
        start = "A"
        goal = "C"

        # Act - reconstruct path from start to goal
        path = reconstruct_path(came_from, start, goal)

        # Assert - verify correct path is reconstructed
        assert path == ["A", "B", "C"]

    @pytest.mark.unit
    def test_reconstruct_path_same_start_goal(self) -> None:
        """Test reconstruct_path with same start and goal."""
        # Arrange - prepare data with same start and goal
        came_from = {}
        start = "A"
        goal = "A"

        # Act - reconstruct path with same start and goal
        path = reconstruct_path(came_from, start, goal)

        # Assert - verify single node path is returned
        assert path == ["A"]

    @pytest.mark.unit
    def test_single_source_to_goal_costs(self) -> None:
        """Test single_source_to_goal_costs function."""
        # Arrange - create a weighted graph
        graph = nx.Graph()
        graph.add_edge("A", "B", weight=2.0)
        graph.add_edge("B", "C", weight=3.0)
        graph.add_edge("A", "C", weight=6.0)
        goal = "C"

        # Act - calculate costs from all nodes to goal
        costs = single_source_to_goal_costs(graph, goal)

        # Assert - verify correct costs are calculated
        assert costs["C"] == 0.0
        assert costs["B"] == 3.0
        assert costs["A"] == 5.0  # A->B->C = 2+3 = 5

    @pytest.mark.unit
    def test_astar_single_same_start_goal(self) -> None:
        """Test astar_single with same start and goal."""
        # Arrange - create a simple graph with single node
        graph = nx.Graph()
        graph.add_node("A")
        h_costs = {"A": 0.0}

        # Act - run A* with same start and goal
        result = astar_single(graph, "A", "A", h_costs)

        # Assert - verify single node path is returned
        assert result == ["A"]

    @pytest.mark.unit
    def test_astar_single_no_path(self) -> None:
        """Test astar_single when no path exists."""
        # Arrange - create disconnected graph
        graph = nx.Graph()
        graph.add_node("A")
        graph.add_node("B")
        h_costs = {"A": 0.0, "B": 0.0}

        # Act - attempt to find path between disconnected nodes
        result = astar_single(graph, "A", "B", h_costs)

        # Assert - verify no path is found
        assert result is None
