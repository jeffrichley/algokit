"""Utility functions for AlgoKit documentation macros.

This module provides helper functions and utilities used by the navigation
and content generation macros.
"""

import re
from typing import Any

from .data_loader import get_algorithm, get_family


def format_complexity(complexity: str) -> str:
    """Return complexity notation (assumes LaTeX is already formatted).

    Args:
        complexity: LaTeX-formatted complexity string (e.g., "$O(n^2)$")

    Returns:
        The complexity string as-is (should already be LaTeX)
    """
    if not complexity:
        return "Unknown"

    # Complexity should already be LaTeX-formatted in the YAML
    # No mapping needed - MathJax will render it directly
    return complexity


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to a maximum length with optional suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
        suffix: Suffix to add when truncating

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def generate_slug(text: str) -> str:
    """Generate a URL-friendly slug from text.

    Args:
        text: Text to convert to slug

    Returns:
        URL-friendly slug
    """
    # Convert to lowercase and replace spaces with hyphens
    slug = text.lower().replace(" ", "-")

    # Remove special characters except hyphens
    slug = re.sub(r"[^a-z0-9\-]", "", slug)

    # Remove multiple consecutive hyphens
    slug = re.sub(r"-+", "-", slug)

    # Remove leading and trailing hyphens
    slug = slug.strip("-")

    return slug


def get_algorithm_metadata(algorithm_key: str) -> dict[str, Any]:
    """Get comprehensive metadata for an algorithm.

    Args:
        algorithm_key: Key identifying the algorithm

    Returns:
        Dictionary containing algorithm metadata
    """
    algorithm = get_algorithm(algorithm_key)
    if not algorithm:
        return {}

    family_key = algorithm.get("family", "")
    family_data = get_family(family_key)

    return {
        "key": algorithm_key,
        "name": algorithm.get("name", ""),
        "description": algorithm.get("description", ""),
        "overview": algorithm.get("overview", ""),
        "status": algorithm.get("status", "unknown"),
        "family": {
            "key": family_key,
            "name": (
                family_data.get("name", family_key.title())
                if family_data
                else family_key.title()
            ),
            "description": family_data.get("description", "") if family_data else "",
        },
        "complexity": algorithm.get("complexity", {}),
        "tags": algorithm.get("tags", []),
        "applications": algorithm.get("applications", []),
        "related_algorithms": algorithm.get("related_algorithms", []),
        "implementation": algorithm.get("implementation", {}),
        "mathematical_formulation": algorithm.get("mathematical_formulation", {}),
    }


def get_family_metadata(family_key: str) -> dict[str, Any]:
    """Get comprehensive metadata for an algorithm family.

    Args:
        family_key: Key identifying the family

    Returns:
        Dictionary containing family metadata
    """
    family_data = get_family(family_key)
    if not family_data:
        return {}

    from .data_loader import get_algorithms_by_family, get_progress

    algorithms = get_algorithms_by_family(family_key)
    progress = get_progress()
    family_progress = progress.get("by_family", {}).get(family_key, {})

    return {
        "key": family_key,
        "name": family_data.get("name", ""),
        "description": family_data.get("description", ""),
        "overview": family_data.get("overview", ""),
        "status": family_data.get("status", "unknown"),
        "key_characteristics": family_data.get("key_characteristics", []),
        "common_applications": family_data.get("common_applications", []),
        "related_families": family_data.get("related_families", []),
        "algorithms_count": len(algorithms),
        "progress": family_progress,
        "algorithms": algorithms,
    }


def validate_algorithm_key(algorithm_key: str) -> bool:
    """Validate that an algorithm key exists.

    Args:
        algorithm_key: Key to validate

    Returns:
        True if the algorithm exists, False otherwise
    """
    return get_algorithm(algorithm_key) is not None


def validate_family_key(family_key: str) -> bool:
    """Validate that a family key exists.

    Args:
        family_key: Key to validate

    Returns:
        True if the family exists, False otherwise
    """
    return get_family(family_key) is not None


def get_related_content(
    algorithm_key: str, max_items: int = 5
) -> dict[str, list[dict[str, Any]]]:
    """Get related content for an algorithm.

    Args:
        algorithm_key: Key identifying the algorithm
        max_items: Maximum number of items to return per category

    Returns:
        Dictionary containing related content by category
    """
    algorithm = get_algorithm(algorithm_key)
    if not algorithm:
        return {}

    family_key = algorithm.get("family", "")
    related_algorithms = algorithm.get("related_algorithms", [])

    # Get algorithms in the same family
    from .data_loader import get_algorithms_by_family

    family_algorithms = get_algorithms_by_family(family_key)

    # Get related families
    family_data = get_family(family_key)
    related_families = family_data.get("related_families", []) if family_data else []

    return {
        "same_family": [
            {
                "key": next(k for k, v in get_algorithm().items() if v == algo),
                "name": algo.get("name", ""),
                "description": algo.get("description", ""),
                "status": algo.get("status", "unknown"),
            }
            for algo in family_algorithms[:max_items]
            if algo != algorithm
        ],
        "related_algorithms": [
            {
                "key": related_key,
                "name": (
                    get_algorithm(related_key).get("name", "")
                    if get_algorithm(related_key)
                    else ""
                ),
                "description": (
                    get_algorithm(related_key).get("description", "")
                    if get_algorithm(related_key)
                    else ""
                ),
                "status": (
                    get_algorithm(related_key).get("status", "unknown")
                    if get_algorithm(related_key)
                    else "unknown"
                ),
            }
            for related_key in related_algorithms[:max_items]
        ],
        "related_families": [
            {
                "key": related_family,
                "name": (
                    get_family(related_family).get("name", "")
                    if get_family(related_family)
                    else ""
                ),
                "description": (
                    get_family(related_family).get("description", "")
                    if get_family(related_family)
                    else ""
                ),
            }
            for related_family in related_families[:max_items]
        ],
    }


def format_progress_bar(percentage: float, bar_length: int = 20) -> str:
    """Format a progress bar for display.

    Args:
        percentage: Progress percentage (0-100)
        bar_length: Length of the progress bar

    Returns:
        Formatted progress bar string
    """
    filled = int((percentage / 100) * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    return f"`{bar}` {percentage:.1f}%"


def get_status_emoji(status: str) -> str:
    """Get appropriate emoji for a status.

    Args:
        status: Status string

    Returns:
        Status emoji
    """
    status_emojis = {
        "complete": "✅",
        "planned": "⏳",
        "in-progress": "🔄",
        "deprecated": "❌",
        "testing": "🧪",
        "review": "👀",
    }

    return status_emojis.get(status, "❓")


def sanitize_html(text: str) -> str:
    """Sanitize text for safe HTML output.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text safe for HTML
    """
    if not text:
        return ""

    # Basic HTML entity escaping
    html_entities = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
    }

    for char, entity in html_entities.items():
        text = text.replace(char, entity)

    return text
