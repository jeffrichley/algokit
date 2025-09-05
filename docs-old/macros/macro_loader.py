"""
Macro loader for MkDocs Macros plugin.

This module provides the on_startup hook to register our custom macro functions
with the MkDocs Macros plugin.
"""

from typing import Any, Dict

from .navigation import (
    generate_navigation_grid,
    generate_family_overview,
    generate_algorithm_card,
    generate_progress_summary,
    generate_learning_paths,
    generate_complexity_badge,
    generate_status_indicator,
)

from .utils import (
    format_complexity,
    truncate_text,
    generate_slug,
    get_algorithm_metadata,
    get_family_metadata,
    validate_algorithm_key,
    validate_family_key,
    get_related_content,
    format_progress_bar,
    get_status_emoji,
    sanitize_html,
)

from .main import (
    algorithm_page,
    list_all_algorithms,
    list_family_algorithms,
)

from .hooks import (
    on_files,
    on_page_markdown,
    on_post_build,
)


def on_startup(plugin: Any, config: Dict[str, Any]) -> None:
    """Register our custom macro functions with MkDocs Macros.

    This function is called by MkDocs Macros during startup to register
    our custom functions that can be used in markdown files.

    Args:
        plugin: The MkDocs Macros plugin instance
        config: The MkDocs configuration
    """
    # Register navigation functions
    plugin.macros.update(
        {
            "generate_navigation_grid": generate_navigation_grid,
            "generate_family_overview": generate_family_overview,
            "generate_algorithm_card": generate_algorithm_card,
            "generate_progress_summary": generate_progress_summary,
            "generate_learning_paths": generate_learning_paths,
            "generate_complexity_badge": generate_complexity_badge,
            "generate_status_indicator": generate_status_indicator,
        }
    )

    # Register utility functions
    plugin.macros.update(
        {
            "format_complexity": format_complexity,
            "truncate_text": truncate_text,
            "generate_slug": generate_slug,
            "get_algorithm_metadata": get_algorithm_metadata,
            "get_family_metadata": get_family_metadata,
            "validate_algorithm_key": validate_algorithm_key,
            "validate_family_key": validate_family_key,
            "get_related_content": get_related_content,
            "format_progress_bar": format_progress_bar,
            "get_status_emoji": get_status_emoji,
            "sanitize_html": sanitize_html,
        }
    )

    # Register dynamic page generation functions
    plugin.macros.update(
        {
            "algorithm_page": algorithm_page,
            "list_all_algorithms": list_all_algorithms,
            "list_family_algorithms": list_family_algorithms,
        }
    )

    print("✅ AlgoKit macros registered successfully!")
