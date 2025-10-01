"""Tests for the distance calculation utilities."""

import pytest

from algokit.core.utils.distances import (
    chebyshev_distance,
    create_euclidean_heuristic,
    create_manhattan_heuristic,
    euclidean_distance,
    manhattan_distance,
    zero_heuristic,
)


class TestManhattanDistance:
    """Test Manhattan distance calculation."""

    @pytest.mark.unit
    def test_manhattan_distance_basic(self) -> None:
        """Test basic Manhattan distance calculations."""
        # Arrange - test various coordinate pairs
        test_cases = [
            ((0, 0), (3, 4), 7.0),
            ((1, 1), (1, 1), 0.0),
            ((0, 0), (0, 5), 5.0),
            ((0, 0), (5, 0), 5.0),
            ((-1, -1), (1, 1), 4.0),
        ]

        # Act & Assert - verify distance calculations
        for node1, node2, expected in test_cases:
            distance = manhattan_distance(node1, node2)
            assert distance == expected

    @pytest.mark.unit
    def test_manhattan_distance_symmetry(self) -> None:
        """Test that Manhattan distance is symmetric."""
        # Arrange - test symmetric property
        node1 = (2, 3)
        node2 = (5, 7)

        # Act - calculate distance both ways
        distance1 = manhattan_distance(node1, node2)
        distance2 = manhattan_distance(node2, node1)

        # Assert - distances should be equal
        assert distance1 == distance2

    @pytest.mark.unit
    def test_manhattan_distance_triangle_inequality(self) -> None:
        """Test Manhattan distance triangle inequality."""
        # Arrange - three points
        a = (0, 0)
        b = (3, 4)
        c = (6, 8)

        # Act - calculate distances
        ab = manhattan_distance(a, b)
        bc = manhattan_distance(b, c)
        ac = manhattan_distance(a, c)

        # Assert - triangle inequality should hold
        assert ac <= ab + bc


class TestEuclideanDistance:
    """Test Euclidean distance calculation."""

    @pytest.mark.unit
    def test_euclidean_distance_basic(self) -> None:
        """Test basic Euclidean distance calculations."""
        # Arrange - test various coordinate pairs
        test_cases = [
            ((0, 0), (3, 4), 5.0),
            ((1, 1), (1, 1), 0.0),
            ((0, 0), (0, 5), 5.0),
            ((0, 0), (5, 0), 5.0),
            ((-1, -1), (1, 1), pytest.approx(2.828, abs=0.001)),
        ]

        # Act & Assert - verify distance calculations
        for node1, node2, expected in test_cases:
            distance = euclidean_distance(node1, node2)
            assert distance == expected

    @pytest.mark.unit
    def test_euclidean_distance_symmetry(self) -> None:
        """Test that Euclidean distance is symmetric."""
        # Arrange - test symmetric property
        node1 = (2.0, 3.0)
        node2 = (5.0, 7.0)

        # Act - calculate distance both ways
        distance1 = euclidean_distance(node1, node2)
        distance2 = euclidean_distance(node2, node1)

        # Assert - distances should be equal
        assert distance1 == distance2

    @pytest.mark.unit
    def test_euclidean_distance_triangle_inequality(self) -> None:
        """Test Euclidean distance triangle inequality."""
        # Arrange - three points
        a = (0.0, 0.0)
        b = (3.0, 4.0)
        c = (6.0, 8.0)

        # Act - calculate distances
        ab = euclidean_distance(a, b)
        bc = euclidean_distance(b, c)
        ac = euclidean_distance(a, c)

        # Assert - triangle inequality should hold
        assert ac <= ab + bc


class TestChebyshevDistance:
    """Test Chebyshev distance calculation."""

    @pytest.mark.unit
    def test_chebyshev_distance_basic(self) -> None:
        """Test basic Chebyshev distance calculations."""
        # Arrange - test various coordinate pairs
        test_cases = [
            ((0, 0), (3, 4), 4.0),  # max(3, 4) = 4
            ((1, 1), (1, 1), 0.0),
            ((0, 0), (0, 5), 5.0),  # max(0, 5) = 5
            ((0, 0), (5, 0), 5.0),  # max(5, 0) = 5
            ((2, 3), (5, 7), 4.0),  # max(3, 4) = 4
        ]

        # Act & Assert - verify distance calculations
        for node1, node2, expected in test_cases:
            distance = chebyshev_distance(node1, node2)
            assert distance == expected

    @pytest.mark.unit
    def test_chebyshev_distance_symmetry(self) -> None:
        """Test that Chebyshev distance is symmetric."""
        # Arrange - test symmetric property
        node1 = (2, 3)
        node2 = (5, 7)

        # Act - calculate distance both ways
        distance1 = chebyshev_distance(node1, node2)
        distance2 = chebyshev_distance(node2, node1)

        # Assert - distances should be equal
        assert distance1 == distance2


class TestZeroHeuristic:
    """Test zero heuristic function."""

    @pytest.mark.unit
    def test_zero_heuristic_always_zero(self) -> None:
        """Test that zero heuristic always returns zero."""
        # Arrange - various node types
        test_cases = [
            ("A", "B"),
            ((0, 0), (3, 4)),
            (1, 2),
            ([1, 2], [3, 4]),
        ]

        # Act & Assert - all should return 0.0
        for node, goal in test_cases:
            result = zero_heuristic(node, goal)
            assert result == 0.0


class TestHeuristicFactories:
    """Test heuristic factory functions."""

    @pytest.mark.unit
    def test_create_manhattan_heuristic(self) -> None:
        """Test Manhattan heuristic factory function."""
        # Arrange - create heuristic for specific goal
        goal = (3, 4)
        heuristic = create_manhattan_heuristic(goal)

        # Act & Assert - test various nodes
        assert heuristic((0, 0)) == 7.0  # Manhattan distance to (3, 4)
        assert heuristic((3, 4)) == 0.0  # Distance to self
        assert heuristic((1, 1)) == 5.0  # Manhattan distance to (3, 4)

    @pytest.mark.unit
    def test_create_euclidean_heuristic(self) -> None:
        """Test Euclidean heuristic factory function."""
        # Arrange - create heuristic for specific goal
        goal = (3.0, 4.0)
        heuristic = create_euclidean_heuristic(goal)

        # Act & Assert - test various nodes
        assert heuristic((0.0, 0.0)) == 5.0  # Euclidean distance to (3, 4)
        assert heuristic((3.0, 4.0)) == 0.0  # Distance to self
        assert heuristic((1.0, 1.0)) == pytest.approx(3.606, abs=0.001)

    @pytest.mark.unit
    def test_heuristic_factory_consistency(self) -> None:
        """Test that factory heuristics are consistent with direct calculations."""
        # Arrange - test points
        goal = (3, 4)
        test_nodes = [(0, 0), (1, 1), (2, 2), (3, 4)]

        # Act - create heuristics
        manhattan_heuristic = create_manhattan_heuristic(goal)
        euclidean_heuristic = create_euclidean_heuristic(goal)

        # Assert - factory results should match direct calculations
        for node in test_nodes:
            assert manhattan_heuristic(node) == manhattan_distance(node, goal)
            assert euclidean_heuristic(node) == euclidean_distance(node, goal)


class TestDistanceComparison:
    """Test comparison between different distance metrics."""

    @pytest.mark.unit
    def test_distance_relationships(self) -> None:
        """Test relationships between different distance metrics."""
        # Arrange - test points
        node1 = (0, 0)
        node2 = (3, 4)

        # Act - calculate all distances
        manhattan = manhattan_distance(node1, node2)
        euclidean = euclidean_distance(node1, node2)
        chebyshev = chebyshev_distance(node1, node2)

        # Assert - known relationships
        assert manhattan == 7.0
        assert euclidean == 5.0
        assert chebyshev == 4.0

        # In general: Chebyshev <= Euclidean <= Manhattan
        assert chebyshev <= euclidean <= manhattan

    @pytest.mark.unit
    def test_distance_equality_for_same_points(self) -> None:
        """Test that all distances are equal when points are the same."""
        # Arrange - same point
        point = (2, 3)

        # Act - calculate all distances
        manhattan = manhattan_distance(point, point)
        euclidean = euclidean_distance(point, point)
        chebyshev = chebyshev_distance(point, point)

        # Assert - all should be zero
        assert manhattan == 0.0
        assert euclidean == 0.0
        assert chebyshev == 0.0
