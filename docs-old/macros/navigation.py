"""
Navigation macros for AlgoKit documentation.

This module provides functions that generate dynamic navigation content
from the algorithms.yaml data structure, including navigation grids,
family overviews, and related content links.
"""

from typing import Any
from .data_loader import (
    get_families,
    get_family,
    get_algorithms,
    get_algorithm,
    get_algorithms_by_family,
    get_progress,
    get_relationships,
    get_macro_config,
)


def generate_navigation_grid(
    current_algorithm: str | None = None,
    current_family: str | None = None,
    max_related: int = 5,
) -> str:
    """Generate a comprehensive navigation grid for algorithm pages.

    Args:
        current_algorithm: Key of the current algorithm (if any)
        current_family: Key of the current family (if any)
        max_related: Maximum number of related algorithms to show

    Returns:
        HTML string for the navigation grid
    """
    config = get_macro_config()
    families = get_families()
    algorithms = get_algorithms()
    progress = get_progress()

    # Build the navigation grid
    html_parts = ['!!! grid "Related Content"']

    # Family Overview section
    if current_family:
        family_data = families.get(current_family, {})
        html_parts.append('    !!! grid-item "Family Overview"')
        html_parts.append(
            f'        - **[{family_data.get("name", current_family.title())}](../../families/{current_family}.md)** - {family_data.get("description", "Family overview")}'
        )

        # Show family progress
        family_progress = progress.get("by_family", {}).get(current_family, {})
        if family_progress:
            implemented = family_progress.get("implemented", 0)
            total = family_progress.get("total", 0)
            percentage = family_progress.get("percentage", 0)
            html_parts.append(
                f"        - **Progress**: {implemented}/{total} algorithms ({percentage:.1f}%)"
            )

    # Related Algorithms section
    if current_family:
        family_algorithms = get_algorithms_by_family(current_family)
        if family_algorithms:
            html_parts.append('    !!! grid-item "Related Algorithms"')
            for algo in family_algorithms[:max_related]:
                algo_key = next(k for k, v in algorithms.items() if v == algo)
                status_map = {
                    "complete": "✅",
                    "planned": "⏳",
                    "in-progress": "🔄",
                    "deprecated": "❌",
                }
                status_icon = status_map.get(algo.get("status", "planned"), "❓")
                html_parts.append(
                    f'        - {status_icon} **[{algo["name"]}](./{algo_key}.md)** - {algo.get("description", "Algorithm description")}'
                )

    # Algorithm Families section
    html_parts.append('    !!! grid-item "Algorithm Families"')
    for family_key, family_data in families.items():
        if family_key != current_family:  # Don't show current family again
            status_map = {
                "complete": "✅",
                "planned": "⏳",
                "in-progress": "🔄",
                "deprecated": "❌",
            }
            status_icon = status_map.get(family_data.get("status", "planned"), "❓")
            html_parts.append(
                f'        - {status_icon} **[{family_data["name"]}](../../families/{family_key}.md)** - {family_data.get("description", "Family description")}'
            )

    # Documentation section
    html_parts.append('    !!! grid-item "Documentation"')
    html_parts.append(
        "        - **[API Reference](../../api.md)** - Complete algorithm implementations"
    )
    html_parts.append(
        "        - **[Contributing Guide](../../contributing.md)** - How to add new algorithms"
    )
    html_parts.append("        - **[Home](../../index.md)** - Main documentation index")

    return "\n".join(html_parts)


def generate_family_overview(family_key: str) -> str:
    """Generate a comprehensive family overview section.

    Args:
        family_key: Key identifying the algorithm family

    Returns:
        HTML string for the family overview
    """
    family_data = get_family(family_key)
    if not family_data:
        return f"Family '{family_key}' not found."

    algorithms = get_algorithms_by_family(family_key)
    progress = get_progress()
    family_progress = progress.get("by_family", {}).get(family_key, {})

    html_parts = []

    # Family header
    html_parts.append(f'# {family_data["name"]}')
    html_parts.append("")
    html_parts.append(f'{family_data["overview"]}')
    html_parts.append("")

    # Key characteristics
    if family_data.get("key_characteristics"):
        html_parts.append("## Key Characteristics")
        html_parts.append("")
        for characteristic in family_data["key_characteristics"]:
            html_parts.append(f"- **{characteristic}**")
        html_parts.append("")

    # Common applications
    if family_data.get("common_applications"):
        html_parts.append("## Common Applications")
        html_parts.append("")
        for application in family_data["common_applications"]:
            html_parts.append(f"- {application}")
        html_parts.append("")

    # Implementation status
    if family_progress:
        implemented = family_progress.get("implemented", 0)
        total = family_progress.get("total", 0)
        percentage = family_progress.get("percentage", 0)

        html_parts.append("## Implementation Status")
        html_parts.append("")
        html_parts.append(
            f"**Progress**: {implemented}/{total} algorithms ({percentage:.1f}%)"
        )
        html_parts.append("")

        # Progress bar
        bar_length = 20
        filled = int((percentage / 100) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        html_parts.append(f"`{bar}` {percentage:.1f}%")
        html_parts.append("")

    # Algorithms in this family
    if algorithms:
        html_parts.append("## Algorithms in This Family")
        html_parts.append("")

        # Group by status
        complete_algorithms = [a for a in algorithms if a.get("status") == "complete"]
        planned_algorithms = [a for a in algorithms if a.get("status") == "planned"]

        if complete_algorithms:
            html_parts.append("### ✅ Complete Implementations")
            html_parts.append("")
            for algo in complete_algorithms:
                algo_key = next(k for k, v in get_algorithms().items() if v == algo)
                html_parts.append(
                    f'- **[{algo["name"]}](../algorithms/{family_key}/{algo_key}.md)** - {algo.get("description", "Algorithm description")}'
                )
            html_parts.append("")

        if planned_algorithms:
            html_parts.append("### ⏳ Planned Implementations")
            html_parts.append("")
            for algo in planned_algorithms:
                html_parts.append(
                    f'- **{algo["name"]}** - {algo.get("description", "Algorithm description")}'
                )
            html_parts.append("")

    # Related families
    if family_data.get("related_families"):
        html_parts.append("## Related Algorithm Families")
        html_parts.append("")
        for related_family in family_data["related_families"]:
            related_data = get_family(related_family)
            if related_data:
                html_parts.append(
                    f'- **[{related_data["name"]}](../{related_family}.md)** - {related_data.get("description", "Related family")}'
                )
        html_parts.append("")

    return "\n".join(html_parts)


def generate_algorithm_card(algorithm_key: str) -> str:
    """Generate a detailed algorithm information card.

    Args:
        algorithm_key: Key identifying the algorithm

    Returns:
        HTML string for the algorithm card
    """
    algorithm = get_algorithm(algorithm_key)
    if not algorithm:
        return f"Algorithm '{algorithm_key}' not found."

    family_key = algorithm.get("family", "")
    family_data = get_family(family_key)

    html_parts = []

    # Algorithm header
    status = algorithm.get("status", "planned")
    status_map = {
        "complete": "✅",
        "planned": "⏳",
        "in-progress": "🔄",
        "deprecated": "❌",
    }
    status_icon = status_map.get(status, "❓")
    html_parts.append(f'!!! info "{status_icon} {algorithm["name"]}"')
    html_parts.append(
        f'    **Family:** [{family_data["name"] if family_data else family_key.title()}](../../families/{family_key}.md)'
    )
    html_parts.append(f'    **Status:** {algorithm.get("status", "unknown").title()}')
    html_parts.append("")

    # Description
    html_parts.append(
        f'{algorithm.get("overview", algorithm.get("description", "No description available."))}'
    )
    html_parts.append("")

    # Complexity
    if algorithm.get("complexity"):
        complexity = algorithm["complexity"]
        html_parts.append("**Complexity:**")
        html_parts.append(f'- **Time:** {complexity.get("time", "Unknown")}')
        html_parts.append(f'- **Space:** {complexity.get("space", "Unknown")}')
        if complexity.get("notes"):
            html_parts.append(f'- **Notes:** {complexity["notes"]}')
        html_parts.append("")

    # Implementation approaches
    if algorithm.get("implementation", {}).get("approaches"):
        html_parts.append("**Implementation Approaches:**")
        for approach in algorithm["implementation"]["approaches"]:
            html_parts.append(f"- {approach}")
        html_parts.append("")

    # Applications
    if algorithm.get("applications"):
        html_parts.append("**Applications:**")
        for application in algorithm["applications"]:
            html_parts.append(f"- {application}")
        html_parts.append("")

    # Related algorithms
    if algorithm.get("related_algorithms"):
        html_parts.append("**Related Algorithms:**")
        for related in algorithm["related_algorithms"]:
            related_data = get_algorithm(related)
            if related_data:
                html_parts.append(f'- [{related_data["name"]}](./{related}.md)')
        html_parts.append("")

    return "\n".join(html_parts)


def generate_progress_summary() -> str:
    """Generate a comprehensive progress summary.

    Returns:
        HTML string for the progress summary
    """
    progress = get_progress()
    overall = progress.get("overall", {})
    by_family = progress.get("by_family", {})

    html_parts = []

    # Overall progress
    html_parts.append("## Implementation Progress")
    html_parts.append("")
    html_parts.append(
        f'**Overall Progress**: {overall.get("implemented", 0)}/{overall.get("total", 0)} algorithms ({overall.get("percentage", 0):.1f}%)'
    )
    html_parts.append("")

    # Progress bar
    percentage = overall.get("percentage", 0)
    bar_length = 30
    filled = int((percentage / 100) * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    html_parts.append(f"`{bar}` {percentage:.1f}%")
    html_parts.append("")

    # Family breakdown
    html_parts.append("### Progress by Family")
    html_parts.append("")

    for family_key, family_progress in by_family.items():
        family_data = get_family(family_key)
        if family_data:
            implemented = family_progress.get("implemented", 0)
            total = family_progress.get("total", 0)
            family_percentage = family_progress.get("percentage", 0)

            # Family progress bar
            bar_length = 20
            filled = int((family_percentage / 100) * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)

            status_icon = (
                "✅" if family_percentage == 100 else "⏳"
            )  # Family progress uses different logic
            html_parts.append(
                f'{status_icon} **{family_data["name"]}**: {implemented}/{total} ({family_percentage:.1f}%)'
            )
            html_parts.append(f"`{bar}`")
            html_parts.append("")

    return "\n".join(html_parts)


def generate_learning_paths() -> str:
    """Generate learning path recommendations.

    Returns:
        HTML string for learning path recommendations
    """
    relationships = get_relationships()
    learning_paths = relationships.get("learning_paths", {})

    html_parts = []

    html_parts.append("## Learning Paths")
    html_parts.append("")
    html_parts.append("Choose a learning path based on your experience level:")
    html_parts.append("")

    for level, algorithms in learning_paths.items():
        html_parts.append(f"### {level.title()}")
        html_parts.append("")
        for algo_key in algorithms:
            algorithm = get_algorithm(algo_key)
            if algorithm:
                status_map = {
                    "complete": "✅",
                    "planned": "⏳",
                    "in-progress": "🔄",
                    "deprecated": "❌",
                }
                status_icon = status_map.get(algorithm.get("status", "planned"), "❓")
                family_key = algorithm.get("family", "")
                html_parts.append(
                    f'- {status_icon} **[{algorithm["name"]}](../algorithms/{family_key}/{algo_key}.md)** - {algorithm.get("description", "Algorithm description")}'
                )
        html_parts.append("")

    return "\n".join(html_parts)


def generate_complexity_badge(complexity: str) -> str:
    """Generate a complexity badge for display.

    Args:
        complexity: LaTeX-formatted complexity string (e.g., "$O(n)$", "$O(n^2)$")

    Returns:
        HTML string for the complexity badge with LaTeX rendering
    """
    # Since complexity is already LaTeX-formatted, just wrap it in a badge
    # MathJax will render the LaTeX automatically
    return f'<span class="badge badge--blue">{complexity}</span>'


def generate_status_indicator(status: str) -> str:
    """Generate a status indicator badge.

    Args:
        status: Status string ("complete", "planned", "in-progress")

    Returns:
        HTML string for the status indicator
    """
    status_map = {
        "complete": ("✅ Complete", "green"),
        "planned": ("⏳ Planned", "orange"),
        "in-progress": ("🔄 In Progress", "blue"),
        "deprecated": ("❌ Deprecated", "red"),
    }

    text, color = status_map.get(status, ("❓ Unknown", "gray"))
    return f'<span class="badge badge--{color}">{text}</span>'
