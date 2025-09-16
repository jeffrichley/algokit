"""Event system for algorithm visualization data collection.

This module provides a simple event system for collecting algorithm execution
data for post-processing visualization without polluting the core algorithm logic.
"""

from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(Enum):
    """Core algorithm events for visualization."""
    
    ENQUEUE = "enqueue"
    DEQUEUE = "dequeue"
    DISCOVER = "discover"
    GOAL_FOUND = "goal_found"
    PATH_RECONSTRUCT = "path_reconstruct"


@dataclass(frozen=True)
class SearchEvent:
    """Simple event for post-processing visualization.
    
    Attributes:
        type: Type of algorithm event
        node: Node involved in the event
        parent: Parent node (for path reconstruction)
        step: Algorithm step number
        data: Additional event data
    """
    
    type: EventType
    node: Any
    parent: Any | None = None
    step: int = 0
    data: dict[str, Any] = field(default_factory=dict)


class SimpleTracker:
    """Lightweight tracker that just collects events for post-processing."""
    
    def __init__(self) -> None:
        """Initialize the event tracker."""
        self.events: list[SearchEvent] = []
        self.step = 0
    
    def emit(self, event_type: EventType, node: Any, parent: Any | None = None, **data: Any) -> None:
        """Emit a simple event.
        
        Args:
            event_type: Type of event to emit
            node: Node involved in the event
            parent: Parent node (optional)
            **data: Additional event data
        """
        self.events.append(SearchEvent(
            type=event_type,
            node=node,
            parent=parent,
            step=self.step,
            data=data,
        ))
        self.step += 1
    
    @contextmanager
    def track(self, **structures: Any) -> Generator[dict[str, Any], None, None]:
        """Simple context manager for data collection.
        
        Args:
            **structures: Data structures to track (deque, set, list, etc.)
            
        Yields:
            Dictionary of tracked data structures
        """
        tracked_objects = {}
        
        for name, structure in structures.items():
            if isinstance(structure, deque):
                tracked_obj = self._create_tracked_deque()
                # Copy initial data
                for item in structure:
                    tracked_obj._deque.append(item)
                tracked_objects[name] = tracked_obj
            elif isinstance(structure, set):
                tracked_obj = self._create_tracked_set()
                # Copy initial data
                for item in structure:
                    tracked_obj._set.add(item)
                tracked_objects[name] = tracked_obj
            elif isinstance(structure, list):
                tracked_obj = self._create_tracked_list()
                # Copy initial data
                for item in structure:
                    tracked_obj._list.append(item)
                tracked_objects[name] = tracked_obj
        
        try:
            yield tracked_objects
        finally:
            pass  # Just collect events, no cleanup needed
    
    def _create_tracked_deque(self) -> Any:
        """Create a tracked deque that emits events."""
        class TrackedDeque:
            def __init__(self, tracker: SimpleTracker) -> None:
                self._deque = deque()
                self._tracker = tracker
            
            def popleft(self) -> Any:
                """Pop left element and emit dequeue event."""
                result = self._deque.popleft()
                self._tracker.emit(EventType.DEQUEUE, result)
                return result
            
            def append(self, item: Any) -> None:
                """Append item and emit enqueue event."""
                self._deque.append(item)
                self._tracker.emit(EventType.ENQUEUE, item)
            
            def appendleft(self, item: Any) -> None:
                """Append left item and emit enqueue event."""
                self._deque.appendleft(item)
                self._tracker.emit(EventType.ENQUEUE, item)
            
            def __len__(self) -> int:
                """Get deque length."""
                return len(self._deque)
            
            def __bool__(self) -> bool:
                """Check if deque is non-empty."""
                return bool(self._deque)
            
            def __iter__(self) -> Any:
                """Iterate over deque elements."""
                return iter(self._deque)
        
        return TrackedDeque(self)
    
    def _create_tracked_set(self) -> Any:
        """Create a tracked set that emits events."""
        class TrackedSet:
            def __init__(self, tracker: SimpleTracker) -> None:
                self._set = set()
                self._tracker = tracker
            
            def add(self, item: Any) -> None:
                """Add item and emit discover event if new."""
                if item not in self._set:
                    self._set.add(item)
                    self._tracker.emit(EventType.DISCOVER, item)
            
            def __contains__(self, item: Any) -> bool:
                """Check if item is in set."""
                return item in self._set
            
            def __len__(self) -> int:
                """Get set length."""
                return len(self._set)
        
        return TrackedSet(self)
    
    def _create_tracked_list(self) -> Any:
        """Create a tracked list that emits events."""
        class TrackedList:
            def __init__(self, tracker: SimpleTracker) -> None:
                self._list: list[Any] = []
                self._tracker = tracker
            
            def append(self, item: Any) -> None:
                """Append item and emit event."""
                self._list.append(item)
                self._tracker.emit(EventType.ENQUEUE, item)
            
            def pop(self) -> Any:
                """Pop item and emit event."""
                result = self._list.pop()
                self._tracker.emit(EventType.DEQUEUE, result)
                return result
            
            def __len__(self) -> int:
                """Get list length."""
                return len(self._list)
            
            def __bool__(self) -> bool:
                """Check if list is non-empty."""
                return bool(self._list)
        
        return TrackedList(self)


def process_events_for_visualization(events: list[SearchEvent]) -> dict[str, Any]:
    """Process events to build visualization data.
    
    Args:
        events: List of search events from algorithm execution
        
    Returns:
        Dictionary containing visualization data
    """
    visualization_data = {
        "steps": [],
        "frontier_history": [],
        "visited_history": [],
        "path": None,
        "goal_found_step": None,
    }
    
    current_frontier: list[Any] = []
    current_visited: set[Any] = set()
    
    for event in events:
        if event.type == EventType.ENQUEUE:
            current_frontier.append(event.node)
        elif event.type == EventType.DEQUEUE:
            if event.node in current_frontier:
                current_frontier.remove(event.node)
        elif event.type == EventType.DISCOVER:
            current_visited.add(event.node)
        elif event.type == EventType.GOAL_FOUND:
            visualization_data["goal_found_step"] = event.step
        elif event.type == EventType.PATH_RECONSTRUCT:
            if visualization_data["path"] is None:
                visualization_data["path"] = []
            visualization_data["path"].append(event.node)
        
        # Store state at each step
        visualization_data["steps"].append({
            "step": event.step,
            "event_type": event.type.value,
            "node": event.node,
            "parent": event.parent,
            "frontier": current_frontier.copy(),
            "visited": current_visited.copy(),
        })
        
        # Store frontier and visited history
        visualization_data["frontier_history"].append(current_frontier.copy())
        visualization_data["visited_history"].append(current_visited.copy())
    
    return visualization_data


def create_algorithm_tracker(algorithm_name: str) -> SimpleTracker:
    """Create a tracker for a specific algorithm.
    
    Args:
        algorithm_name: Name of the algorithm (for debugging/logging)
        
    Returns:
        SimpleTracker instance
    """
    return SimpleTracker()
