"""Algorithm Kit - A python implementation of control and learning algorithms."""

__version__ = "0.11.1"
__author__ = "Jeff Richley"
__email__ = "jeffrichley@gmail.com"


# Core helpers and utilities
from algokit.algorithms.dynamic_programming import fibonacci
from algokit.core.helpers import (
    HarborNetScenario,
    create_grid_graph,
    load_harbor_scenario,
)

# CLI imports
try:
    from algokit.cli import app  # noqa: F401

    _cli_available = True
except ImportError:
    _cli_available = False

__all__ = [
    "fibonacci",
    # Core helpers
    "HarborNetScenario",
    "create_grid_graph",
    "load_harbor_scenario",
]

if _cli_available:
    __all__.append("app")
