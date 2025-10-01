"""Utility functions and decorators for algorithm implementations.

This module provides utility functions, decorators, and infrastructure
that support algorithm implementations without polluting the main algorithm code.
"""

from algokit.core.utils.decorators import (
    with_event_tracking,
    with_logging,
    with_timing,
)
from algokit.core.utils.distances import (
    chebyshev_distance,
    create_euclidean_heuristic,
    create_manhattan_heuristic,
    euclidean_distance,
    manhattan_distance,
    zero_heuristic,
)

__all__ = [
    # Decorators
    "with_event_tracking",
    "with_logging",
    "with_timing",
    # Distance functions
    "manhattan_distance",
    "euclidean_distance",
    "chebyshev_distance",
    "zero_heuristic",
    "create_manhattan_heuristic",
    "create_euclidean_heuristic",
]
