"""Property-based tests for Dijkstra's algorithm.

These tests verify that Dijkstra's algorithm maintains its expected properties across
various graph structures and input conditions.
"""

import networkx as nx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from algokit.algorithms.pathfinding.dijkstra import (
    dijkstra_all_distances,
    dijkstra_shortest_distance,
    dijkstra_shortest_path,
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


class TestDijkstraProperties:
    """Test Dijkstra maintains expected properties."""

    @pytest.mark.unit
    @given(
        width=st.integers(min_value=2, max_value=8),
        height=st.integers(min_value=2, max_value=8),
        weight=st.floats(min_value=0.1, max_value=10.0),
    )
    def test_dijkstra_path_validity_property(
        self, width: int, height: int, weight: float
    ) -> None:
        """Test that Dijkstra always returns valid paths when they exist."""
        # Arrange - create weighted grid graph
        graph = create_weighted_grid_graph(width, height, default_weight=weight)
        start = (0, 0)
        goal = (width - 1, height - 1)

        # Act - find path with Dijkstra
        result = dijkstra_shortest_path(graph, start, goal)

        # Assert - path should be valid
        assert result is not None
        path, total_weight = result
        assert path[0] == start
        assert path[-1] == goal

        # Verify each step is a valid edge
        for i in range(len(path) - 1):
            assert graph.has_edge(path[i], path[i + 1])

    @pytest.mark.unit
    @given(
        width=st.integers(min_value=2, max_value=6),
        height=st.integers(min_value=2, max_value=6),
        weight=st.floats(min_value=0.1, max_value=5.0),
    )
    def test_dijkstra_distance_validity_property(
        self, width: int, height: int, weight: float
    ) -> None:
        """Test that Dijkstra always returns valid distances when paths exist."""
        # Arrange - create weighted grid graph
        graph = create_weighted_grid_graph(width, height, default_weight=weight)
        start = (0, 0)
        goal = (width - 1, height - 1)
        manhattan_distance = (width - 1) + (height - 1)

        # Act - find distance with Dijkstra
        distance = dijkstra_shortest_distance(graph, start, goal)

        # Assert - distance should be valid and match expected Manhattan distance
        assert distance is not None
        assert (
            abs(distance - manhattan_distance * weight) < 1e-10
        )  # Allow for floating-point precision
        assert distance >= 0  # Distance should be non-negative

    @pytest.mark.unit
    @given(
        num_nodes=st.integers(min_value=3, max_value=15),
        num_edges=st.integers(min_value=2, max_value=30),
        weight_range=st.tuples(
            st.floats(min_value=0.1, max_value=5.0),
            st.floats(min_value=0.1, max_value=5.0),
        ),
    )
    def test_dijkstra_termination_property(
        self, num_nodes: int, num_edges: int, weight_range: tuple[float, float]
    ) -> None:
        """Test that Dijkstra always terminates on connected graphs."""
        # Arrange - create random connected graph
        min_weight, max_weight = weight_range
        graph = nx.erdos_renyi_graph(
            num_nodes, min(0.4, num_edges / (num_nodes * (num_nodes - 1) / 2))
        )

        # Ensure graph is connected
        if not nx.is_connected(graph):
            components = list(nx.connected_components(graph))
            for i in range(len(components) - 1):
                graph.add_edge(list(components[i])[0], list(components[i + 1])[0])

        # Add random weights to edges
        for edge in graph.edges():
            weight = (
                min_weight + (max_weight - min_weight) * (hash(str(edge)) % 100) / 100
            )
            graph[edge[0]][edge[1]]["weight"] = weight

        # Choose start and goal nodes
        nodes = list(graph.nodes())
        start = nodes[0]
        goal = nodes[-1]

        # Act - find path with Dijkstra
        result = dijkstra_shortest_path(graph, start, goal)

        # Assert - should always find a path on connected graph
        assert result is not None
        path, weight = result
        assert len(path) <= num_nodes  # Path can't be longer than number of nodes
        assert weight >= 0  # Weight should be non-negative

    @pytest.mark.unit
    @given(
        width=st.integers(min_value=3, max_value=6),
        height=st.integers(min_value=3, max_value=6),
        weight=st.floats(min_value=0.1, max_value=2.0),
        max_distance=st.floats(min_value=1.0, max_value=5.0),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_dijkstra_all_distances_limit_property(
        self, width: int, height: int, weight: float, max_distance: float
    ) -> None:
        """Test that Dijkstra all distances respects distance limits."""
        # Arrange - create weighted grid graph
        graph = create_weighted_grid_graph(width, height, default_weight=weight)
        start = (width // 2, height // 2)  # Start from center

        # Act - find all distances with limit
        distances = dijkstra_all_distances(graph, start, max_distance=max_distance)

        # Assert - all distances should be within limit
        assert all(distance <= max_distance for distance in distances.values())
        assert start in distances
        assert distances[start] == 0.0

    @pytest.mark.unit
    @given(
        width=st.integers(min_value=2, max_value=6),
        height=st.integers(min_value=2, max_value=6),
        weight=st.floats(min_value=0.1, max_value=2.0),
    )
    def test_dijkstra_optimality_property(
        self, width: int, height: int, weight: float
    ) -> None:
        """Test that Dijkstra finds optimal (shortest) paths."""
        # Arrange - create weighted grid graph
        graph = create_weighted_grid_graph(width, height, default_weight=weight)
        start = (0, 0)
        goal = (width - 1, height - 1)
        manhattan_distance = (width - 1) + (height - 1)

        # Act - find shortest path
        result = dijkstra_shortest_path(graph, start, goal)

        # Assert - path should be optimal
        assert result is not None
        path, total_weight = result
        assert (
            abs(total_weight - manhattan_distance * weight) < 1e-10
        )  # Allow for floating-point precision
        assert (
            len(path) >= manhattan_distance + 1
        )  # Path length should be at least Manhattan distance + 1
        assert len(path) <= width * height  # Path can't be longer than total nodes

    @pytest.mark.unit
    @given(
        num_components=st.integers(min_value=1, max_value=4),
        nodes_per_component=st.integers(min_value=2, max_value=6),
        weight=st.floats(min_value=0.1, max_value=2.0),
    )
    def test_dijkstra_connected_component_property(
        self, num_components: int, nodes_per_component: int, weight: float
    ) -> None:
        """Test that Dijkstra correctly identifies connected components."""
        # Arrange - create graph with known components
        graph = nx.Graph()
        node_id = 0

        for _component in range(num_components):
            # Create a connected component (path)
            component_nodes = list(range(node_id, node_id + nodes_per_component))
            for i in range(len(component_nodes) - 1):
                graph.add_edge(
                    component_nodes[i], component_nodes[i + 1], weight=weight
                )
            node_id += nodes_per_component

        # Choose start node from first component
        start = 0

        # Act - find all distances from start
        distances = dijkstra_all_distances(graph, start)

        # Assert - should only find nodes in connected component
        assert len(distances) == nodes_per_component
        assert all(node in distances for node in range(nodes_per_component))

    @pytest.mark.unit
    @given(
        width=st.integers(min_value=2, max_value=6),
        height=st.integers(min_value=2, max_value=6),
        weight=st.floats(min_value=0.1, max_value=2.0),
    )
    def test_dijkstra_monotonicity_property(
        self, width: int, height: int, weight: float
    ) -> None:
        """Test that Dijkstra distances are monotonic (non-decreasing)."""
        # Arrange - create weighted grid graph with uniform weights
        graph = create_weighted_grid_graph(width, height, default_weight=weight)
        start = (0, 0)
        goal = (width - 1, height - 1)

        # Act - find distances to all nodes
        all_distances = dijkstra_all_distances(graph, start)

        # Assert - distances should be non-negative and monotonic
        assert all(distance >= 0 for distance in all_distances.values())
        assert start in all_distances
        assert all_distances[start] == 0.0  # Distance to self should be 0

        # For grid graphs, the distance to goal should be Manhattan distance * weight
        expected_distance = ((width - 1) + (height - 1)) * weight
        assert abs(all_distances[goal] - expected_distance) < 1e-10

    @pytest.mark.unit
    @given(
        width=st.integers(min_value=3, max_value=8),
        height=st.integers(min_value=3, max_value=8),
        weight=st.floats(min_value=0.1, max_value=2.0),
    )
    def test_dijkstra_deterministic_property(
        self, width: int, height: int, weight: float
    ) -> None:
        """Test that Dijkstra produces deterministic results."""
        # Arrange - create weighted grid graph
        graph = create_weighted_grid_graph(width, height, default_weight=weight)
        start = (0, 0)
        goal = (width - 1, height - 1)

        # Act - run Dijkstra multiple times
        results = []
        for _ in range(3):
            result = dijkstra_shortest_path(graph, start, goal)
            results.append(result)

        # Assert - all results should be identical (deterministic)
        assert all(result == results[0] for result in results)

    @pytest.mark.unit
    @given(
        num_nodes=st.integers(min_value=3, max_value=10),
        weight=st.floats(min_value=0.1, max_value=2.0),
    )
    def test_dijkstra_completeness_property(
        self, num_nodes: int, weight: float
    ) -> None:
        """Test that Dijkstra finds all nodes in connected component."""
        # Arrange - create complete graph (fully connected)
        graph = nx.complete_graph(num_nodes)

        # Add weights to all edges
        for edge in graph.edges():
            graph[edge[0]][edge[1]]["weight"] = weight

        start = 0

        # Act - find all distances from start
        distances = dijkstra_all_distances(graph, start)

        # Assert - should find all nodes in connected component
        assert len(distances) == num_nodes
        assert all(node in distances for node in graph.nodes())

    @pytest.mark.unit
    def test_dijkstra_handles_empty_graph_property(self) -> None:
        """Test that Dijkstra handles edge cases gracefully."""
        # Arrange - create empty graph
        empty_graph = nx.Graph()

        # Act & Assert - should raise appropriate errors
        with pytest.raises(ValueError):
            dijkstra_shortest_path(empty_graph, "start", "goal")

        with pytest.raises(ValueError):
            dijkstra_shortest_distance(empty_graph, "start", "goal")

        with pytest.raises(ValueError):
            dijkstra_all_distances(empty_graph, "start")

    @pytest.mark.unit
    @given(
        width=st.integers(min_value=2, max_value=6),
        height=st.integers(min_value=2, max_value=6),
        weight=st.floats(min_value=0.1, max_value=2.0),
    )
    def test_dijkstra_path_consistency_property(
        self, width: int, height: int, weight: float
    ) -> None:
        """Test that path and distance results are consistent."""
        # Arrange - create weighted grid graph
        graph = create_weighted_grid_graph(width, height, default_weight=weight)
        start = (0, 0)
        goal = (width - 1, height - 1)

        # Act - find both path and distance
        path_result = dijkstra_shortest_path(graph, start, goal)
        distance_result = dijkstra_shortest_distance(graph, start, goal)

        # Assert - results should be consistent
        assert path_result is not None
        assert distance_result is not None

        path, path_weight = path_result
        assert path_weight == distance_result

    @pytest.mark.unit
    @given(
        width=st.integers(min_value=2, max_value=6),
        height=st.integers(min_value=2, max_value=6),
        weight=st.floats(min_value=0.1, max_value=2.0),
    )
    def test_dijkstra_non_negative_weights_property(
        self, width: int, height: int, weight: float
    ) -> None:
        """Test that Dijkstra works with non-negative weights."""
        # Arrange - create weighted grid graph (weight is already non-negative)
        graph = create_weighted_grid_graph(width, height, default_weight=weight)
        start = (0, 0)
        goal = (width - 1, height - 1)

        # Act - find shortest path
        result = dijkstra_shortest_path(graph, start, goal)

        # Assert - should work correctly with non-negative weights
        assert result is not None
        path, total_weight = result
        assert total_weight >= 0  # Weight should be non-negative
        assert len(path) >= 2  # Should have at least start and goal
