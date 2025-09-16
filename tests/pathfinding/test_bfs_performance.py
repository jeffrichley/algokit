"""Performance tests for BFS algorithm.

This module validates the O(|V|+|E|) time complexity and memory usage
of the BFS implementation across different graph sizes and densities.
"""

import time

import networkx as nx
import pytest

from algokit.core.helpers import create_grid_graph
from algokit.pathfinding.bfs import bfs_path_length, bfs_shortest_path


class TestBFSPerformance:
    """Test BFS performance characteristics."""

    @pytest.mark.performance
    def test_bfs_time_complexity_linear_vertices(self) -> None:
        """Test that BFS time scales linearly with number of vertices."""
        # Arrange - create graphs of increasing size
        sizes = [100, 200, 400, 800]
        times = []
        
        for size in sizes:
            # Create a simple path graph (linear structure)
            graph = nx.Graph()
            for i in range(size):
                graph.add_node(i)
            for i in range(size - 1):
                graph.add_edge(i, i + 1)
            
            # Act - measure execution time
            start_time = time.time()
            path = bfs_shortest_path(graph, 0, size - 1)
            end_time = time.time()
            
            execution_time = end_time - start_time
            times.append(execution_time)
            
            # Assert - path should be found
            assert path is not None
            assert len(path) == size
        
        # Assert - times should scale roughly linearly
        # (allowing for some variance due to system load)
        for i in range(1, len(times)):
            ratio = times[i] / times[0]
            expected_ratio = sizes[i] / sizes[0]
            # Allow 50% variance for system load
            assert ratio <= expected_ratio * 1.5, f"Time scaling not linear: {ratio} vs {expected_ratio}"

    @pytest.mark.performance
    def test_bfs_time_complexity_linear_edges(self) -> None:
        """Test that BFS time scales linearly with number of edges."""
        # Arrange - create graphs with increasing edge density
        sizes = [50, 100, 150, 200]
        times = []
        
        for size in sizes:
            # Create a grid graph (quadratic edge growth)
            graph = create_grid_graph(size, size)
            
            # Act - measure execution time
            start_time = time.time()
            path = bfs_shortest_path(graph, (0, 0), (size - 1, size - 1))
            end_time = time.time()
            
            execution_time = end_time - start_time
            times.append(execution_time)
            
            # Assert - path should be found
            assert path is not None
            assert path[0] == (0, 0)
            assert path[-1] == (size - 1, size - 1)
        
        # Assert - times should scale roughly with graph size
        # (grid graphs have O(n²) edges, so we expect some growth)
        for i in range(1, len(times)):
            ratio = times[i] / times[0]
            expected_ratio = (sizes[i] / sizes[0]) ** 2  # Quadratic growth
            # Allow 100% variance for system load and complexity
            assert ratio <= expected_ratio * 2.0, f"Time scaling not as expected: {ratio} vs {expected_ratio}"

    @pytest.mark.performance
    def test_bfs_memory_usage_scaling(self) -> None:
        """Test that BFS memory usage scales appropriately."""
        # Arrange - create graphs of increasing size
        sizes = [50, 100, 150]
        
        for size in sizes:
            # Create a grid graph
            graph = create_grid_graph(size, size)
            
            # Act - run BFS and measure memory usage (approximate)
            import os

            import psutil
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss
            
            path = bfs_shortest_path(graph, (0, 0), (size - 1, size - 1))
            
            memory_after = process.memory_info().rss
            memory_used = memory_after - memory_before
            
            # Assert - path should be found
            assert path is not None
            
            # Assert - memory usage should be reasonable (less than 10MB per test)
            assert memory_used < 10 * 1024 * 1024, f"Memory usage too high: {memory_used / 1024 / 1024:.2f}MB"

    @pytest.mark.performance
    def test_bfs_large_graph_performance(self) -> None:
        """Test BFS performance on large graphs."""
        # Arrange - create a large graph (1000x1000 grid)
        graph = create_grid_graph(100, 100)  # 10,000 nodes
        
        # Act - measure execution time
        start_time = time.time()
        path = bfs_shortest_path(graph, (0, 0), (99, 99))
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Assert - should complete in reasonable time (< 1 second)
        assert execution_time < 1.0, f"BFS too slow on large graph: {execution_time:.3f}s"
        
        # Assert - path should be found and optimal
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (99, 99)
        
        # Verify path length is optimal (Manhattan distance)
        length = bfs_path_length(graph, (0, 0), (99, 99))
        assert length == 198  # 99 + 99

    @pytest.mark.performance
    def test_bfs_sparse_vs_dense_performance(self) -> None:
        """Test BFS performance on sparse vs dense graphs."""
        # Arrange - create sparse and dense graphs of same size
        size = 100
        
        # Sparse graph (tree structure)
        sparse_graph = nx.Graph()
        for i in range(size):
            sparse_graph.add_node(i)
        # Create a binary tree
        for i in range(size // 2):
            if 2 * i + 1 < size:
                sparse_graph.add_edge(i, 2 * i + 1)
            if 2 * i + 2 < size:
                sparse_graph.add_edge(i, 2 * i + 2)
        
        # Dense graph (grid)
        dense_graph = create_grid_graph(10, 10)  # 100 nodes
        
        # Act - measure execution time on both
        start_time = time.time()
        sparse_path = bfs_shortest_path(sparse_graph, 0, size - 1)
        sparse_time = time.time() - start_time
        
        start_time = time.time()
        dense_path = bfs_shortest_path(dense_graph, (0, 0), (9, 9))
        dense_time = time.time() - start_time
        
        # Assert - both should complete quickly
        assert sparse_time < 0.1, f"Sparse graph BFS too slow: {sparse_time:.3f}s"
        assert dense_time < 0.1, f"Dense graph BFS too slow: {dense_time:.3f}s"
        
        # Assert - paths should be found
        assert sparse_path is not None
        assert dense_path is not None

    @pytest.mark.performance
    def test_bfs_path_length_vs_full_path_performance(self) -> None:
        """Test that path length calculation is faster than full path."""
        # Arrange - create a large graph
        graph = create_grid_graph(50, 50)  # 2,500 nodes
        
        # Act - measure both operations
        start_time = time.time()
        path = bfs_shortest_path(graph, (0, 0), (49, 49))
        full_path_time = time.time() - start_time
        
        start_time = time.time()
        length = bfs_path_length(graph, (0, 0), (49, 49))
        length_time = time.time() - start_time
        
        # Assert - both should complete quickly
        assert full_path_time < 0.5, f"Full path BFS too slow: {full_path_time:.3f}s"
        assert length_time < 0.5, f"Path length BFS too slow: {length_time:.3f}s"
        
        # Assert - results should be consistent
        assert path is not None
        assert length is not None
        assert len(path) - 1 == length  # Path length should match distance
        
        # Path length should be faster (no path reconstruction)
        # But allow for variance due to system load
        assert length_time <= full_path_time * 1.5, "Path length should be faster than full path"

    @pytest.mark.performance
    def test_bfs_no_path_performance(self) -> None:
        """Test BFS performance when no path exists."""
        # Arrange - create disconnected graph
        graph = nx.Graph()
        # Component 1: 0-1-2
        graph.add_edges_from([(0, 1), (1, 2)])
        # Component 2: 3-4-5
        graph.add_edges_from([(3, 4), (4, 5)])
        
        # Act - measure execution time for impossible path
        start_time = time.time()
        path = bfs_shortest_path(graph, 0, 5)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Assert - should complete quickly even when no path exists
        assert execution_time < 0.1, f"BFS too slow on disconnected graph: {execution_time:.3f}s"
        
        # Assert - should return None
        assert path is None
        
        # Test path length version too
        start_time = time.time()
        length = bfs_path_length(graph, 0, 5)
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time < 0.1, f"Path length BFS too slow on disconnected graph: {execution_time:.3f}s"
        assert length is None
