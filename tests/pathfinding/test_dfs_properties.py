"""Property-based tests for DFS pathfinding algorithm.

These tests verify that DFS maintains its expected properties across
various graph structures and input conditions.
"""

import networkx as nx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from algokit.algorithms.pathfinding.dfs import (
    dfs_all_reachable,
    dfs_connected_components,
    dfs_path,
    dfs_recursive_path,
)
from algokit.core.helpers import create_grid_graph


class TestDFSProperties:
    """Test DFS maintains expected properties."""

    @pytest.mark.unit
    @given(
        width=st.integers(min_value=2, max_value=10),
        height=st.integers(min_value=2, max_value=10),
    )
    def test_dfs_path_validity_property(self, width: int, height: int) -> None:
        """Test that DFS always returns valid paths when they exist."""
        # Arrange - create grid graph
        graph = create_grid_graph(width, height)
        start = (0, 0)
        goal = (width - 1, height - 1)

        # Act - find path with DFS
        path = dfs_path(graph, start, goal)

        # Assert - path should be valid
        assert path is not None
        assert path[0] == start
        assert path[-1] == goal

        # Verify each step is a valid edge
        for i in range(len(path) - 1):
            assert graph.has_edge(path[i], path[i + 1])

    @pytest.mark.unit
    @given(
        width=st.integers(min_value=2, max_value=8),
        height=st.integers(min_value=2, max_value=8),
    )
    def test_dfs_recursive_path_validity_property(
        self, width: int, height: int
    ) -> None:
        """Test that recursive DFS always returns valid paths when they exist."""
        # Arrange - create grid graph
        graph = create_grid_graph(width, height)
        start = (0, 0)
        goal = (width - 1, height - 1)

        # Act - find path with recursive DFS
        path = dfs_recursive_path(graph, start, goal)

        # Assert - path should be valid
        assert path is not None
        assert path[0] == start
        assert path[-1] == goal

        # Verify each step is a valid edge
        for i in range(len(path) - 1):
            assert graph.has_edge(path[i], path[i + 1])

    @pytest.mark.unit
    @given(
        num_nodes=st.integers(min_value=3, max_value=20),
        num_edges=st.integers(min_value=2, max_value=50),
    )
    def test_dfs_path_termination_property(
        self, num_nodes: int, num_edges: int
    ) -> None:
        """Test that DFS always terminates on connected graphs."""
        # Arrange - create random connected graph
        graph = nx.erdos_renyi_graph(
            num_nodes, min(0.3, num_edges / (num_nodes * (num_nodes - 1) / 2))
        )

        # Ensure graph is connected
        if not nx.is_connected(graph):
            # Add edges to make it connected
            components = list(nx.connected_components(graph))
            for i in range(len(components) - 1):
                graph.add_edge(list(components[i])[0], list(components[i + 1])[0])

        # Choose start and goal nodes
        nodes = list(graph.nodes())
        start = nodes[0]
        goal = nodes[-1]

        # Act - find path with DFS
        path = dfs_path(graph, start, goal)

        # Assert - should always find a path on connected graph
        assert path is not None
        assert len(path) <= num_nodes  # Path can't be longer than number of nodes

    @pytest.mark.unit
    @given(
        width=st.integers(min_value=3, max_value=10),
        height=st.integers(min_value=3, max_value=10),
        max_depth=st.integers(min_value=1, max_value=5),
    )
    def test_dfs_all_reachable_depth_property(
        self, width: int, height: int, max_depth: int
    ) -> None:
        """Test that DFS all reachable respects depth limits."""
        # Arrange - create grid graph
        graph = create_grid_graph(width, height)
        start = (width // 2, height // 2)  # Start from center

        # Act - find reachable nodes with depth limit
        reachable = dfs_all_reachable(graph, start, max_depth=max_depth)

        # Assert - all distances should be within limit
        assert all(distance <= max_depth for distance in reachable.values())
        assert start in reachable
        assert reachable[start] == 0

    @pytest.mark.unit
    @given(
        num_components=st.integers(min_value=1, max_value=5),
        nodes_per_component=st.integers(min_value=2, max_value=8),
    )
    def test_dfs_connected_components_property(
        self, num_components: int, nodes_per_component: int
    ) -> None:
        """Test that DFS correctly identifies connected components."""
        # Arrange - create graph with known components
        graph = nx.Graph()
        node_id = 0

        for _component in range(num_components):
            # Create a connected component (path)
            component_nodes = list(range(node_id, node_id + nodes_per_component))
            for i in range(len(component_nodes) - 1):
                graph.add_edge(component_nodes[i], component_nodes[i + 1])
            node_id += nodes_per_component

        # Act - find connected components
        components = dfs_connected_components(graph)

        # Assert - should find correct number of components
        assert len(components) == num_components

        # Each component should have correct size
        component_sizes = [len(comp) for comp in components]
        assert all(size == nodes_per_component for size in component_sizes)

        # All nodes should be in exactly one component
        all_nodes = set()
        for component in components:
            all_nodes.update(component)
        assert len(all_nodes) == num_components * nodes_per_component

    @pytest.mark.unit
    @given(
        width=st.integers(min_value=2, max_value=6),
        height=st.integers(min_value=2, max_value=6),
    )
    def test_dfs_path_length_property(self, width: int, height: int) -> None:
        """Test that DFS path length is reasonable."""
        # Arrange - create grid graph
        graph = create_grid_graph(width, height)
        start = (0, 0)
        goal = (width - 1, height - 1)
        manhattan_distance = (width - 1) + (height - 1)

        # Act - find path with DFS
        path = dfs_path(graph, start, goal)

        # Assert - path length should be at least Manhattan distance
        assert path is not None
        assert (
            len(path) >= manhattan_distance + 1
        )  # +1 because path includes both start and goal

        # Path length should be at most total nodes (in worst case)
        assert len(path) <= width * height

    @pytest.mark.unit
    @given(
        graph_size=st.integers(min_value=3, max_value=15),
    )
    def test_dfs_visits_all_nodes_property(self, graph_size: int) -> None:
        """Test that DFS visits all nodes in connected component."""
        # Arrange - create complete graph (fully connected)
        graph = nx.complete_graph(graph_size)
        start = 0

        # Act - find all reachable nodes
        reachable = dfs_all_reachable(graph, start)

        # Assert - should visit all nodes in connected component
        assert len(reachable) == graph_size
        assert all(node in reachable for node in graph.nodes())

    @pytest.mark.unit
    @given(
        width=st.integers(min_value=3, max_value=8),
        height=st.integers(min_value=3, max_value=8),
    )
    def test_dfs_deterministic_property(self, width: int, height: int) -> None:
        """Test that DFS produces deterministic results."""
        # Arrange - create grid graph
        graph = create_grid_graph(width, height)
        start = (0, 0)
        goal = (width - 1, height - 1)

        # Act - run DFS multiple times
        paths = []
        for _ in range(3):
            path = dfs_path(graph, start, goal)
            paths.append(path)

        # Assert - all paths should be identical (deterministic)
        assert all(path == paths[0] for path in paths)

    @pytest.mark.unit
    def test_dfs_handles_empty_graph_property(self) -> None:
        """Test that DFS handles edge cases gracefully."""
        # Arrange - create empty graph
        empty_graph = nx.Graph()

        # Act & Assert - should raise appropriate errors
        with pytest.raises(ValueError):
            dfs_path(empty_graph, "start", "goal")

        # Connected components should return empty list
        components = dfs_connected_components(empty_graph)
        assert len(components) == 0

    @pytest.mark.unit
    @given(
        width=st.integers(min_value=2, max_value=6),
        height=st.integers(min_value=2, max_value=6),
    )
    def test_dfs_recursive_vs_iterative_consistency_property(
        self, width: int, height: int
    ) -> None:
        """Test that iterative and recursive DFS give consistent results."""
        # Arrange - create grid graph
        graph = create_grid_graph(width, height)
        start = (0, 0)
        goal = (width - 1, height - 1)

        # Act - find paths with both implementations
        iterative_path = dfs_path(graph, start, goal)
        recursive_path = dfs_recursive_path(graph, start, goal)

        # Assert - both should find valid paths with same endpoints
        assert iterative_path is not None
        assert recursive_path is not None
        assert iterative_path[0] == recursive_path[0]
        assert iterative_path[-1] == recursive_path[-1]

        # Both paths should be valid
        for path in [iterative_path, recursive_path]:
            for i in range(len(path) - 1):
                assert graph.has_edge(path[i], path[i + 1])

    @pytest.mark.unit
    @given(
        num_nodes=st.integers(min_value=3, max_value=10),
    )
    def test_dfs_path_no_cycles_property(self, num_nodes: int) -> None:
        """Test that DFS paths don't contain cycles."""
        # Arrange - create tree graph (no cycles by definition)
        graph = nx.balanced_tree(2, num_nodes // 2)  # Binary tree
        nodes = list(graph.nodes())
        if len(nodes) < 2:
            return  # Skip if not enough nodes

        start = nodes[0]
        goal = nodes[-1]

        # Act - find path with DFS
        path = dfs_path(graph, start, goal)

        # Assert - path should not contain cycles (no repeated nodes)
        assert path is not None
        assert len(path) == len(set(path))  # No duplicates

    @pytest.mark.unit
    @given(
        width=st.integers(min_value=3, max_value=8),
        height=st.integers(min_value=3, max_value=8),
    )
    def test_dfs_all_reachable_completeness_property(
        self, width: int, height: int
    ) -> None:
        """Test that DFS all reachable finds all nodes in connected component."""
        # Arrange - create grid graph
        graph = create_grid_graph(width, height)
        start = (width // 2, height // 2)  # Start from center

        # Act - find all reachable nodes
        reachable = dfs_all_reachable(graph, start)

        # Assert - should find all nodes in grid (since it's connected)
        assert len(reachable) == width * height
        assert all(node in reachable for node in graph.nodes())
