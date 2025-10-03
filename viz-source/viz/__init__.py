"""Visualization framework module.

This module provides reusable visualization components for algorithm animations
and data collection systems for post-processing visualization.
"""

from algokit.viz.adapters import (
    EventType,
    SearchEvent,
    SimpleTracker,
    create_algorithm_tracker,
    process_events_for_visualization,
)

__all__ = [
    "SearchEvent",
    "EventType",
    "SimpleTracker",
    "process_events_for_visualization",
    "create_algorithm_tracker",
]
