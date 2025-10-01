"""Algorithm Kit - A python implementation of control and learning algorithms."""

__version__ = "0.10.0"
__author__ = "Jeff Richley"
__email__ = "jeffrichley@gmail.com"


# Core helpers and utilities
from algokit.core.helpers import (
    HarborNetScenario,
    create_grid_graph,
    get_graph_info,
    load_graph_from_json,
    load_harbor_scenario,
    save_graph_to_json,
    save_harbor_scenario,
    validate_graph,
)
from algokit.algorithms.dynamic_programming import fibonacci

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
    "save_harbor_scenario",
    "load_graph_from_json",
    "save_graph_to_json",
    "validate_graph",
    "get_graph_info",
]

if _cli_available:
    __all__.append("app")
