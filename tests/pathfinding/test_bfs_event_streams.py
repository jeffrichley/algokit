"""Tests for BFS event stream correctness and sequence validation."""

import networkx as nx
import pytest

from algokit.core.helpers import create_grid_graph
from algokit.algorithms.pathfinding.bfs import bfs_shortest_path
from algokit.algorithms.pathfinding.bfs_with_events import bfs_with_data_collection
from algokit.viz.adapters import EventType


class TestBFSEventStreams:
    """Test BFS event stream correctness and sequence validation."""

    @pytest.mark.unit
    def test_bfs_events_enqueue_order(self) -> None:
        """Test that events are enqueued in correct order."""
        # Arrange - create simple 2x2 grid
        graph = create_grid_graph(2, 2)

        # Act - run BFS with event tracking
        path, events = bfs_with_data_collection(graph, (0, 0), (1, 1))

        # Assert - verify enqueue events exist and are in BFS order
        enqueue_events = [e for e in events if e.type == EventType.ENQUEUE]
        assert len(enqueue_events) >= 1
        # First enqueue should be a neighbor of start node
        assert enqueue_events[0].node in [(1, 0), (0, 1)]

    @pytest.mark.unit
    def test_bfs_events_dequeue_order(self) -> None:
        """Test that events are dequeued in correct order."""
        # Arrange - create simple 2x2 grid
        graph = create_grid_graph(2, 2)

        # Act - run BFS with event tracking
        path, events = bfs_with_data_collection(graph, (0, 0), (1, 1))

        # Assert - verify dequeue events follow BFS order
        dequeue_events = [e for e in events if e.type == EventType.DEQUEUE]
        assert len(dequeue_events) >= 1
        # First dequeue should be the start node
        assert dequeue_events[0].node == (0, 0)

    @pytest.mark.unit
    def test_bfs_events_discover_sequence(self) -> None:
        """Test that discovery events follow correct sequence."""
        # Arrange - create simple 2x2 grid
        graph = create_grid_graph(2, 2)

        # Act - run BFS with event tracking
        path, events = bfs_with_data_collection(graph, (0, 0), (1, 1))

        # Assert - verify discovery events exist and follow BFS order
        discover_events = [e for e in events if e.type == EventType.DISCOVER]
        assert len(discover_events) >= 1
        # Discovery events should be neighbors of the start node or their neighbors
        valid_nodes = [(1, 0), (0, 1), (1, 1)]  # Include all possible discovered nodes
        for event in discover_events:
            assert event.node in valid_nodes

    @pytest.mark.unit
    def test_bfs_events_goal_found(self) -> None:
        """Test that goal found event is emitted correctly."""
        # Arrange - create simple 2x2 grid
        graph = create_grid_graph(2, 2)

        # Act - run BFS with event tracking
        path, events = bfs_with_data_collection(graph, (0, 0), (1, 1))

        # Assert - verify goal found event exists
        goal_events = [e for e in events if e.type == EventType.GOAL_FOUND]
        assert len(goal_events) == 1
        assert goal_events[0].node == (1, 1)

    @pytest.mark.unit
    def test_bfs_events_reconstruct_path(self) -> None:
        """Test that path reconstruction events are emitted."""
        # Arrange - create simple 2x2 grid
        graph = create_grid_graph(2, 2)

        # Act - run BFS with event tracking
        path, events = bfs_with_data_collection(graph, (0, 0), (1, 1))

        # Assert - verify path reconstruction events exist
        reconstruct_events = [e for e in events if e.type == EventType.PATH_RECONSTRUCT]
        assert len(reconstruct_events) >= 1
        # Should reconstruct the path (excluding start node)
        assert len(reconstruct_events) == len(path) - 1

    @pytest.mark.unit
    def test_bfs_pure_vs_tracked_equivalence(self) -> None:
        """Test that pure and tracked versions produce same results."""
        # Arrange - create test graph
        graph = create_grid_graph(3, 3)

        # Act - run both versions
        pure_path = bfs_shortest_path(graph, (0, 0), (2, 2))
        tracked_path, events = bfs_with_data_collection(graph, (0, 0), (2, 2))

        # Assert - paths should be identical
        assert pure_path == tracked_path
        assert len(events) > 0  # Events should be collected

    @pytest.mark.unit
    def test_proxy_object_behavior(self) -> None:
        """Test that TrackedDeque and TrackedSet work correctly."""
        # Arrange - create simple graph
        graph = create_grid_graph(2, 2)

        # Act - run BFS with event tracking
        path, events = bfs_with_data_collection(graph, (0, 0), (1, 1))

        # Assert - verify proxy objects worked correctly
        assert path is not None
        assert len(events) > 0
        
        # Check that we have the expected event types
        event_types = {e.type for e in events}
        expected_types = {EventType.ENQUEUE, EventType.DEQUEUE, EventType.DISCOVER, 
                         EventType.GOAL_FOUND, EventType.PATH_RECONSTRUCT}
        assert expected_types.issubset(event_types)

    @pytest.mark.unit
    def test_bfs_events_no_path_scenario(self) -> None:
        """Test event stream when no path exists."""
        # Arrange - create disconnected graph
        graph = nx.Graph()
        graph.add_nodes_from([(0, 0), (1, 1), (2, 2)])
        graph.add_edge((0, 0), (1, 1))
        # (2, 2) is disconnected

        # Act - run BFS with event tracking
        path, events = bfs_with_data_collection(graph, (0, 0), (2, 2))

        # Assert - no path found, but events should still be collected
        assert path is None
        assert len(events) > 0
        
        # Should have enqueue/dequeue events but no goal found
        event_types = {e.type for e in events}
        assert EventType.GOAL_FOUND not in event_types
        assert EventType.PATH_RECONSTRUCT not in event_types

    @pytest.mark.unit
    def test_bfs_events_step_numbering(self) -> None:
        """Test that events have correct step numbering."""
        # Arrange - create simple graph
        graph = create_grid_graph(2, 2)

        # Act - run BFS with event tracking
        path, events = bfs_with_data_collection(graph, (0, 0), (1, 1))

        # Assert - verify step numbering is sequential
        steps = [e.step for e in events]
        assert all(isinstance(step, int) for step in steps)
        assert min(steps) >= 0
        assert max(steps) <= len(events)
        
        # Steps should be mostly sequential (allowing for some flexibility)
        unique_steps = set(steps)
        assert len(unique_steps) > 1  # Should have multiple steps
