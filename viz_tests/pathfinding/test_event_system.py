"""Tests for the event system used in pathfinding algorithms."""

from collections import deque
import sys
from pathlib import Path

import pytest

# Add viz-source to path so we can import from it
viz_source_path = Path(__file__).parent.parent.parent / "viz-source"
sys.path.insert(0, str(viz_source_path))

from viz.adapters import (
    EventType,
    SimpleTracker,
    process_events_for_visualization,
)


class TestEventSystem:
    """Test the event system for algorithm visualization."""

    @pytest.mark.unit
    def test_simple_tracker_creation(self) -> None:
        """Test that SimpleTracker can be created and used."""
        # Arrange - create a new tracker instance
        tracker = SimpleTracker()

        # Act - emit a discover event
        tracker.emit(EventType.DISCOVER, "test_node")

        # Assert - verify event was recorded correctly
        assert len(tracker.events) == 1
        assert tracker.events[0].type == EventType.DISCOVER
        assert tracker.events[0].node == "test_node"

    @pytest.mark.unit
    def test_tracked_deque_operations(self) -> None:
        """Test that TrackedDeque emits correct events."""
        # Arrange - create tracker and initialize with test data
        tracker = SimpleTracker()

        # Act - perform deque operations that should emit events
        with tracker.track(queue=deque(["A"])) as tracked:
            queue = tracked["queue"]

            # Dequeue should emit DEQUEUE event
            queue.popleft()

            # Append should emit ENQUEUE event
            queue.append("B")

        # Assert - verify correct events were emitted
        assert len(tracker.events) == 2
        assert tracker.events[0].type == EventType.DEQUEUE
        assert tracker.events[0].node == "A"
        assert tracker.events[1].type == EventType.ENQUEUE
        assert tracker.events[1].node == "B"

    @pytest.mark.unit
    def test_tracked_set_operations(self) -> None:
        """Test that TrackedSet emits correct events."""
        # Arrange - create tracker and initialize with test data
        tracker = SimpleTracker()

        # Act - perform set operations that should emit events
        with tracker.track(visited={"A"}) as tracked:
            visited = tracked["visited"]

            # Adding new item should emit DISCOVER event
            visited.add("B")

            # Adding existing item should not emit event
            visited.add("A")

        # Assert - verify only new items emit events
        assert len(tracker.events) == 1
        assert tracker.events[0].type == EventType.DISCOVER
        assert tracker.events[0].node == "B"

    @pytest.mark.unit
    def test_bfs_like_workflow(self) -> None:
        """Test a BFS-like workflow with event tracking."""
        # Arrange - create tracker and simulate BFS operations
        tracker = SimpleTracker()

        # Act - simulate BFS operations with event tracking
        with tracker.track(queue=deque(["start"]), visited={"start"}) as tracked:
            queue = tracked["queue"]
            visited = tracked["visited"]

            # Dequeue start node
            queue.popleft()

            # Discover and enqueue neighbors
            for neighbor in ["A", "B"]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

            # Emit goal found
            tracker.emit(EventType.GOAL_FOUND, "B")

            # Emit path reconstruction
            tracker.emit(EventType.PATH_RECONSTRUCT, "start")
            tracker.emit(EventType.PATH_RECONSTRUCT, "B")

        # Assert - verify all expected events were emitted
        assert (
            len(tracker.events) == 8
        )  # 1 dequeue + 2 discover + 2 enqueue + 1 goal + 2 path
        assert tracker.events[0].type == EventType.DEQUEUE
        assert tracker.events[1].type == EventType.DISCOVER
        assert tracker.events[2].type == EventType.ENQUEUE
        assert tracker.events[3].type == EventType.DISCOVER
        assert tracker.events[4].type == EventType.ENQUEUE
        assert tracker.events[5].type == EventType.GOAL_FOUND
        assert tracker.events[6].type == EventType.PATH_RECONSTRUCT
        assert tracker.events[7].type == EventType.PATH_RECONSTRUCT

    @pytest.mark.unit
    def test_visualization_data_processing(self) -> None:
        """Test that events are processed correctly for visualization."""
        # Arrange - create tracker and emit test events
        tracker = SimpleTracker()

        # Act - create events and process for visualization
        tracker.emit(EventType.ENQUEUE, "A")
        tracker.emit(EventType.DEQUEUE, "A")
        tracker.emit(EventType.DISCOVER, "B")
        tracker.emit(EventType.GOAL_FOUND, "B")
        tracker.emit(EventType.PATH_RECONSTRUCT, "A")
        tracker.emit(EventType.PATH_RECONSTRUCT, "B")

        viz_data = process_events_for_visualization(tracker.events)

        # Assert - verify visualization data is correct
        assert len(viz_data["steps"]) == 6
        assert viz_data["goal_found_step"] == 3
        assert viz_data["path"] == ["A", "B"]
        assert len(viz_data["frontier_history"]) == 6
        assert len(viz_data["visited_history"]) == 6

    @pytest.mark.unit
    def test_event_step_numbering(self) -> None:
        """Test that events are numbered correctly."""
        # Arrange - create tracker for step numbering test
        tracker = SimpleTracker()

        # Act - emit multiple events to test step numbering
        tracker.emit(EventType.DISCOVER, "A")
        tracker.emit(EventType.DISCOVER, "B")
        tracker.emit(EventType.DISCOVER, "C")

        # Assert - verify events are numbered sequentially
        assert tracker.events[0].step == 0
        assert tracker.events[1].step == 1
        assert tracker.events[2].step == 2

    @pytest.mark.unit
    def test_event_with_parent_data(self) -> None:
        """Test that events can include parent information."""
        # Arrange - create tracker for parent data test
        tracker = SimpleTracker()

        # Act - emit events with parent information
        tracker.emit(EventType.DISCOVER, "B", parent="A")
        tracker.emit(EventType.PATH_RECONSTRUCT, "B", parent="A")

        # Assert - verify parent information is preserved
        assert tracker.events[0].parent == "A"
        assert tracker.events[1].parent == "A"

    @pytest.mark.unit
    def test_event_with_additional_data(self) -> None:
        """Test that events can include additional data."""
        # Arrange - create tracker for additional data test
        tracker = SimpleTracker()

        # Act - emit event with additional data
        tracker.emit(EventType.DISCOVER, "B", depth=2, cost=1.5)

        # Assert - verify additional data is preserved
        assert tracker.events[0].data["depth"] == 2
        assert tracker.events[0].data["cost"] == 1.5
