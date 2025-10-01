"""Property-based tests for BFS algorithm correctness."""

import random

import networkx as nx
import pytest

from algokit.core.helpers import create_grid_graph
from algokit.algorithms.pathfinding.bfs import bfs_path_length, bfs_shortest_path


class TestBFSPropertyBased:
    """Property-based tests to validate BFS algorithm properties."""

    @pytest.mark.unit
    def test_path_validity_property(self) -> None:
        """Test that all returned paths are valid (connected and reachable)."""
        # Arrange - test multiple random graphs
        random.seed(42)  # For reproducible tests
        
        for _ in range(10):
            # Create random grid size
            width = random.randint(3, 8)
            height = random.randint(3, 8)
            graph = create_grid_graph(width, height)
            
            # Add random obstacles
            num_obstacles = random.randint(0, min(width * height // 4, 5))
            obstacles = set()
            for _ in range(num_obstacles):
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                obstacles.add((x, y))
                if (x, y) in graph:
                    graph.remove_node((x, y))
            
            # Test multiple start/goal pairs
            for _ in range(5):
                start = (random.randint(0, width - 1), random.randint(0, height - 1))
                goal = (random.randint(0, width - 1), random.randint(0, height - 1))
                
                # Skip if start/goal are obstacles or same
                if start in obstacles or goal in obstacles or start == goal:
                    continue
                
                # Act - find path
                path = bfs_shortest_path(graph, start, goal)
                
                # Assert - path validity properties
                if path is not None:
                    # Path should start and end correctly
                    assert path[0] == start
                    assert path[-1] == goal
                    
                    # All nodes in path should be connected
                    for i in range(len(path) - 1):
                        assert graph.has_edge(path[i], path[i + 1])
                    
                    # Path should not contain obstacles
                    assert not any(node in obstacles for node in path)

    @pytest.mark.unit
    def test_optimality_property(self) -> None:
        """Test that returned paths have minimum hop count."""
        # Arrange - test various graph structures
        test_cases = [
            # Simple linear path
            ([(0, 0), (1, 0), (2, 0), (3, 0)], (0, 0), (3, 0), 3),
            # Grid with obstacles
            ([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)], (0, 0), (2, 1), 3),
            # Diamond structure
            ([(0, 0), (1, 0), (2, 0), (1, 1), (0, 2), (1, 2), (2, 2)], (0, 0), (2, 2), 4),
        ]
        
        for edges, start, goal, expected_hops in test_cases:
            # Create graph
            graph = nx.Graph()
            graph.add_edges_from(edges)
            # Ensure all nodes are in the graph
            graph.add_nodes_from([start, goal])
            
            # Act - find path
            path = bfs_shortest_path(graph, start, goal)
            
            # Assert - path should be optimal
            if path is not None:
                actual_hops = len(path) - 1  # Subtract 1 for start node
                assert actual_hops == expected_hops, f"Expected {expected_hops} hops, got {actual_hops}"

    @pytest.mark.unit
    def test_completeness_property(self) -> None:
        """Test that BFS finds a path if one exists."""
        # Arrange - test various scenarios
        test_cases = [
            # Connected graph - should find path
            ([((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (2, 1))], (0, 0), (2, 1), True),
            # Disconnected graph - should not find path
            ([((0, 0), (1, 0)), ((2, 2), (3, 2))], (0, 0), (2, 2), False),
            # Single node - should not find path (start == goal)
            ([((0, 0), (0, 0))], (0, 0), (0, 0), False),  # This should raise ValueError
        ]
        
        for edges, start, goal, should_find_path in test_cases:
            # Arrange - create graph
            graph = nx.Graph()
            graph.add_edges_from(edges)
            graph.add_nodes_from([start, goal])
            
            # Act - attempt to find path
            if start == goal:
                # Should raise ValueError for same start/goal
                with pytest.raises(ValueError):
                    bfs_shortest_path(graph, start, goal)
            else:
                path = bfs_shortest_path(graph, start, goal)
                
                # Assert - verify completeness property
                if should_find_path:
                    assert path is not None, f"Should find path from {start} to {goal}"
                else:
                    assert path is None, f"Should not find path from {start} to {goal}"

    @pytest.mark.unit
    def test_determinism_property(self) -> None:
        """Test that same input produces same output."""
        # Arrange - create test graph
        graph = create_grid_graph(4, 4)
        start, goal = (0, 0), (3, 3)
        
        # Act - run BFS multiple times
        paths = []
        for _ in range(10):
            path = bfs_shortest_path(graph, start, goal)
            paths.append(path)
        
        # Assert - all paths should be identical
        assert all(path == paths[0] for path in paths)
        assert paths[0] is not None

    @pytest.mark.unit
    def test_bfs_vs_manual_shortest_path(self) -> None:
        """Test BFS against manual shortest path calculation."""
        # Arrange - create various graph structures
        test_graphs = [
            # Simple grid
            create_grid_graph(3, 3),
            # Grid with obstacles
            create_grid_graph(4, 4, blocked={(1, 1), (2, 2)}),
            # Custom graph
            nx.Graph([(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]),
        ]
        
        for graph in test_graphs:
            nodes = list(graph.nodes())
            if len(nodes) < 2:
                continue
                
            # Test multiple start/goal pairs
            for _ in range(5):
                start = random.choice(nodes)
                goal = random.choice([n for n in nodes if n != start])
                
                # Act - find path with BFS
                bfs_path = bfs_shortest_path(graph, start, goal)
                
                # Calculate manual shortest path using NetworkX
                try:
                    nx_path = nx.shortest_path(graph, start, goal)
                except nx.NetworkXNoPath:
                    nx_path = None
                
                # Assert - BFS should match NetworkX shortest path length
                if bfs_path is None:
                    assert nx_path is None, f"BFS found no path but NetworkX found {nx_path}"
                else:
                    assert nx_path is not None, f"BFS found path {bfs_path} but NetworkX found none"
                    assert len(bfs_path) == len(nx_path), f"BFS path length {len(bfs_path)} != NetworkX path length {len(nx_path)}"

    @pytest.mark.unit
    def test_path_length_consistency(self) -> None:
        """Test that path length function matches actual path length."""
        # Arrange - create test graph
        graph = create_grid_graph(5, 5)
        
        # Test multiple start/goal pairs
        for _ in range(10):
            start = (random.randint(0, 4), random.randint(0, 4))
            goal = (random.randint(0, 4), random.randint(0, 4))
            
            if start == goal:
                continue
            
            # Act - get path and length
            path = bfs_shortest_path(graph, start, goal)
            length = bfs_path_length(graph, start, goal)
            
            # Assert - consistency between path and length
            if path is None:
                assert length is None
            else:
                expected_length = len(path) - 1  # Subtract 1 for start node
                assert length == expected_length, f"Length {length} != expected {expected_length}"

    @pytest.mark.unit
    def test_bfs_exploration_order(self) -> None:
        """Test that BFS explores nodes in breadth-first order."""
        # Arrange - create a tree-like structure
        graph = nx.Graph()
        # Create a tree: 0 -> [1, 2], 1 -> [3, 4], 2 -> [5, 6]
        graph.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
        
        # Act - find path from root to leaf
        path = bfs_shortest_path(graph, 0, 6)
        
        # Assert - path should go through level 2 (node 2), not level 1 (nodes 3,4)
        assert path is not None
        assert 2 in path  # Should go through level 2
        assert 3 not in path  # Should not go through level 1 nodes
        assert 4 not in path

    @pytest.mark.unit
    def test_large_graph_properties(self) -> None:
        """Test BFS properties on larger graphs."""
        # Arrange - create larger graph
        graph = create_grid_graph(20, 20)
        
        # Act - test multiple paths
        test_cases = [
            ((0, 0), (19, 19)),  # Corner to corner
            ((10, 10), (15, 15)),  # Center to center
            ((0, 10), (19, 10)),  # Left to right
        ]
        
        for start, goal in test_cases:
            # Act - find path
            path = bfs_shortest_path(graph, start, goal)
            
            # Assert - path should be optimal (Manhattan distance)
            if path is not None:
                manhattan_dist = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
                actual_hops = len(path) - 1
                assert actual_hops == manhattan_dist, f"Expected {manhattan_dist} hops, got {actual_hops}"
