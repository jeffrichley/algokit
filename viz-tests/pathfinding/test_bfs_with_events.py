"""Tests for BFS with event tracking functionality.

This module tests the BFS implementation with event tracking that is used
for visualization. These tests are separate from the main test suite since
they depend on viz-source code.
"""

import networkx as nx
import pytest

# Import from viz-source
import sys
from pathlib import Path

# Add viz-source to path so we can import from it
viz_source_path = Path(__file__).parent.parent.parent / "viz-source"
sys.path.insert(0, str(viz_source_path))

from bfs_with_events import bfs_with_data_collection
from algokit.core.helpers import HarborNetScenario, create_grid_graph


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
        graph = create_grid_graph(
            scenario.width, scenario.height, blocked=scenario.obstacles
        )

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
