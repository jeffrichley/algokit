"""Hierarchical Reinforcement Learning algorithms package.

This package contains implementations of hierarchical reinforcement learning
algorithms that enable temporal abstraction and multi-level decision making.

Algorithms included:
- Options Framework: Temporal abstraction using options (skills/sub-policies)
- Feudal RL: Hierarchical structure with managers and workers
- HIRO: Data-efficient hierarchical reinforcement learning
"""

from algokit.algorithms.hierarchical_rl.feudal_rl import FeudalAgent
from algokit.algorithms.hierarchical_rl.hiro import HIROAgent
from algokit.algorithms.hierarchical_rl.options_framework import OptionsAgent

__all__ = [
    "OptionsAgent",
    "FeudalAgent",
    "HIROAgent",
]
