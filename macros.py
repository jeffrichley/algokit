"""
MkDocs Macros Module for Algorithm Documentation

This module provides macros and filters for rendering algorithm documentation
from YAML data using Jinja2 templates.
"""

import yaml
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from typing import Dict, List, Any, Optional


def load_family_data(family_id: str) -> dict:
    """Load family data from YAML file."""
    data_dir = Path(__file__).parent / "mkdocs_plugins" / "data"
    family_file = data_dir / family_id / "family.yaml"

    if not family_file.exists():
        return {}

    with open(family_file, "r") as f:
        return yaml.safe_load(f) or {}


def load_algorithm_data(family_id: str) -> list:
    """Load algorithm data for a family."""
    data_dir = Path(__file__).parent / "mkdocs_plugins" / "data"
    algorithms_dir = data_dir / family_id / "algorithms"

    if not algorithms_dir.exists():
        return []

    algorithms = []
    for algo_file in algorithms_dir.glob("*.yaml"):
        with open(algo_file, "r") as f:
            algo_data = yaml.safe_load(f)
            if algo_data:
                # Add slug from filename if not present
                if "slug" not in algo_data:
                    algo_data["slug"] = algo_file.stem
                algorithms.append(algo_data)

    return algorithms


def discover_families() -> List[str]:
    """Discover all algorithm families by scanning the data directory."""
    data_dir = Path(__file__).parent / "mkdocs_plugins" / "data"

    if not data_dir.exists():
        return []

    families = []
    for item in data_dir.iterdir():
        if item.is_dir() and (item / "family.yaml").exists():
            families.append(item.name)

    return sorted(families)


def load_all_families() -> List[Dict[str, Any]]:
    """Load all family data with algorithms and status information."""
    families = []

    for family_id in discover_families():
        family_data = load_family_data(family_id)
        if not family_data:
            continue

        # Load algorithms for this family
        algorithms = load_algorithm_data(family_id)

        # Determine status and completion based on algorithms
        status = "planned"  # default
        available_algorithms = []
        all_algorithms = []
        completion_percentage = 0

        if algorithms:
            # Categorize algorithms by status
            complete_algorithms = [
                algo
                for algo in algorithms
                if algo.get("status", {}).get("current") == "complete"
            ]
            in_progress_algorithms = [
                algo
                for algo in algorithms
                if algo.get("status", {}).get("current") == "in-progress"
            ]
            planned_algorithms = [
                algo
                for algo in algorithms
                if algo.get("status", {}).get("current") == "planned"
            ]

            # Calculate completion percentage
            total_algorithms = len(algorithms)
            complete_count = len(complete_algorithms)
            completion_percentage = (
                round((complete_count / total_algorithms) * 100)
                if total_algorithms > 0
                else 0
            )

            # Determine overall family status
            if complete_count > 0:
                status = "complete"
                available_algorithms = complete_algorithms
            elif len(in_progress_algorithms) > 0:
                status = "in-progress"
            else:
                status = "planned"

            # Create list of all algorithms with status indicators
            all_algorithms = []
            for algo in algorithms:
                algo_name = algo.get("name", algo.get("slug", ""))
                algo_status = algo.get("status", {}).get("current", "planned")

                if algo_status == "complete":
                    all_algorithms.append(f"**{algo_name}** ✓")
                elif algo_status == "in-progress":
                    all_algorithms.append(f"*{algo_name}* 🚧")
                else:
                    all_algorithms.append(algo_name)

        # Add computed fields
        family_data["available_algorithms"] = available_algorithms
        family_data["all_algorithms"] = algorithms
        family_data["all_algorithms_list"] = (
            ", ".join(all_algorithms) if all_algorithms else "Coming Soon"
        )
        family_data["completion_percentage"] = completion_percentage
        family_data["status"] = status
        family_data["status_badge"] = (
            ":material-code-tags: Code"
            if status == "complete"
            else ":material-progress-clock: Coming Soon"
        )

        families.append(family_data)

    return families


def render_index_page() -> str:
    """Render the index page using Jinja2 template."""
    # Load all families data
    families = load_all_families()

    # Set up Jinja2 environment
    template_dir = Path(__file__).parent / "mkdocs_plugins" / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,  # We want markdown, not HTML
    )

    # Load and render template
    try:
        template = env.get_template("index_page.md")
        return template.render(families=families)
    except Exception as e:
        return f"**Error:** Template rendering error: {str(e)}"


def render_family_page(family_id: str) -> str:
    """Render a family page using YAML data and Jinja2 templates."""
    family_data = load_family_data(family_id)
    if not family_data:
        return f"**Error:** Family '{family_id}' not found."

    algorithms = load_algorithm_data(family_id)

    # Set up Jinja2 environment
    template_dir = Path(__file__).parent / "mkdocs_plugins" / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=False)

    # Load and render template
    try:
        template = env.get_template("family_page.md")
        return template.render(family=family_data, algorithms=algorithms)
    except Exception as e:
        return f"**Error:** Template rendering error: {str(e)}"


def render_algorithm_page(family_id: str, algorithm_slug: str) -> str:
    """Render an algorithm page using YAML data and Jinja2 templates."""
    family_data = load_family_data(family_id)
    if not family_data:
        return f"**Error:** Family '{family_id}' not found."

    # Find the specific algorithm
    algorithms = load_algorithm_data(family_id)
    algorithm_data = None

    for algo in algorithms:
        if algo.get("slug") == algorithm_slug:
            algorithm_data = algo
            break

    if not algorithm_data:
        return f"**Error:** Algorithm '{algorithm_slug}' not found in family '{family_id}'."

    # Set up Jinja2 environment
    template_dir = Path(__file__).parent / "mkdocs_plugins" / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=False)

    # Load and render template
    try:
        template = env.get_template("algorithm_page.md")
        return template.render(
            algo=algorithm_data, family=family_data, algorithms=algorithms
        )
    except Exception as e:
        import traceback

        return f"**Error:** Template rendering error: {str(e)}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"


def define_env(env):
    """
    Define the environment for MkDocs Macros.

    This function is called by mkdocs-macros to set up the Jinja2 environment
    with custom variables, macros, and filters.
    """

    # Add a simple test macro to verify the system works
    @env.macro
    def test_macro():
        """Test macro to verify macros are working."""
        return "🎉 **MkDocs Macros is working!** This message was generated by a Python function."

    # Add the index page macro
    @env.macro
    def render_index() -> str:
        """
        Render the main index page using YAML data and Jinja2 templates.

        Returns:
            Rendered markdown content for the index page
        """
        return render_index_page()

    @env.macro
    def algorithm_card(algorithm_key: str) -> str:
        """
        Generate an algorithm card with key information.

        Args:
            algorithm_key: The algorithm identifier

        Returns:
            Rendered algorithm card markdown
        """
        # Find the algorithm data
        algorithm_data = None
        family_data = None

        for family_id in discover_families():
            algorithms = load_algorithm_data(family_id)
            for algo in algorithms:
                if algo.get("slug") == algorithm_key:
                    algorithm_data = algo
                    family_data = load_family_data(family_id)
                    break
            if algorithm_data:
                break

        if not algorithm_data:
            return f"**Error:** Algorithm '{algorithm_key}' not found."

        # Generate the card
        algo_name = algorithm_data.get("name", algorithm_key)
        algo_summary = algorithm_data.get("summary", "")
        family_name = family_data.get("name", "Unknown") if family_data else "Unknown"

        # Determine status
        status = algorithm_data.get("status", {}).get("current", "planned")
        if status == "complete":
            status_text = "✅ Complete"
        elif status == "in-progress":
            status_text = "🚧 In Progress"
        else:
            status_text = "📋 Planned"

        card_content = f"""
<div class="algorithm-card" style="border: 2px solid var(--md-default-fg-color--light); border-radius: 8px; padding: 16px; margin: 16px 0; background-color: var(--md-default-bg-color--lightest);">

**{algo_name}**

{algo_summary}

**Family:** {family_name}  
**Status:** {status_text}

</div>
"""
        return card_content

    @env.macro
    def nav_grid(
        current_algorithm: str = None, current_family: str = None, max_related: int = 5
    ) -> str:
        """
        Generate a navigation grid showing related algorithms.

        Args:
            current_algorithm: Current algorithm slug
            current_family: Current family slug
            max_related: Maximum number of related algorithms to show

        Returns:
            Rendered navigation grid markdown
        """
        if not current_family:
            return "<!-- No family specified for navigation -->"

        # Load family data and algorithms
        family_data = load_family_data(current_family)
        if not family_data:
            return f"<!-- Family '{current_family}' not found -->"

        algorithms = load_algorithm_data(current_family)

        # Filter out current algorithm and limit results
        related_algorithms = [
            algo for algo in algorithms if algo.get("slug") != current_algorithm
        ][:max_related]

        if not related_algorithms:
            return f"<!-- No related algorithms found in family '{current_family}' -->"

        # Generate navigation grid
        family_name = family_data.get("name", current_family)
        nav_content = f"""
<div class="nav-grid" style="border: 1px solid var(--md-default-fg-color--light); border-radius: 8px; padding: 16px; margin: 16px 0; background-color: var(--md-default-bg-color--lightest);">

**Related Algorithms in {family_name}:**

"""

        for algo in related_algorithms:
            algo_name = algo.get("name", algo.get("slug", ""))
            algo_slug = algo.get("slug", "")
            algo_summary = algo.get("summary", "")

            nav_content += f"- [{algo_name}]({algo_slug}.md) - {algo_summary}\n"

        nav_content += "\n</div>"
        return nav_content

    @env.macro
    def render_family_page(family_id: str) -> str:
        """
        Render a family page using YAML data and Jinja2 templates.

        Args:
            family_id: The family identifier

        Returns:
            Rendered markdown content for the family page
        """
        family_data = load_family_data(family_id)
        if not family_data:
            return f"**Error:** Family '{family_id}' not found."

        algorithms = load_algorithm_data(family_id)

        # Set up Jinja2 environment
        template_dir = Path(__file__).parent / "mkdocs_plugins" / "templates"
        env_jinja = Environment(
            loader=FileSystemLoader(str(template_dir)), autoescape=False
        )

        # Load and render template
        try:
            template = env_jinja.get_template("family_page.md")
            return template.render(family=family_data, algorithms=algorithms)
        except Exception as e:
            return f"**Error:** Template rendering error: {str(e)}"

    @env.macro
    def render_algorithm_page(family_id: str, algorithm_slug: str) -> str:
        """
        Render an algorithm page using YAML data and Jinja2 templates.

        Args:
            family_id: The family identifier
            algorithm_slug: The algorithm slug

        Returns:
            Rendered markdown content for the algorithm page
        """
        family_data = load_family_data(family_id)
        if not family_data:
            return f"**Error:** Family '{family_id}' not found."

        # Find the specific algorithm
        algorithms = load_algorithm_data(family_id)
        algorithm_data = None

        for algo in algorithms:
            if algo.get("slug") == algorithm_slug:
                algorithm_data = algo
                break

        if not algorithm_data:
            return f"**Error:** Algorithm '{algorithm_slug}' not found in family '{family_id}'."

        # Set up Jinja2 environment
        template_dir = Path(__file__).parent / "mkdocs_plugins" / "templates"
        env_jinja = Environment(
            loader=FileSystemLoader(str(template_dir)), autoescape=False
        )

        # Load and render template
        try:
            template = env_jinja.get_template("algorithm_page.md")
            return template.render(algo=algorithm_data, family=family_data)
        except Exception as e:
            return f"**Error:** Template rendering error: {str(e)}"

    # Add a simple test variable
    env.variables["current_date"] = "January 4, 2025"
    env.variables["project_name"] = "AlgoKit"
    env.variables["project_description"] = (
        "A python implementation of control and learning algorithms"
    )
