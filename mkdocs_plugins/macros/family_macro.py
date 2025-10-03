"""Family page macro for MkDocs Macros.

This module provides the family_page() macro function that renders
family overview pages using Jinja2 templates and YAML data.
"""

import urllib.parse
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader


def load_family_data(family_id: str) -> dict:
    """Load family data from YAML file."""
    data_dir = Path(__file__).parent.parent / "data"
    family_file = data_dir / family_id / "family.yaml"

    if not family_file.exists():
        return {}

    with open(family_file, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_algorithm_data(family_id: str) -> list:
    """Load algorithm data for a family."""
    data_dir = Path(__file__).parent.parent / "data"
    algorithms_dir = data_dir / family_id / "algorithms"

    if not algorithms_dir.exists():
        return []

    algorithms = []
    for algo_file in algorithms_dir.glob("*.yaml"):
        with open(algo_file, encoding="utf-8") as f:
            algo_data = yaml.safe_load(f)
            if algo_data:
                # Add slug from filename if not present
                if "slug" not in algo_data:
                    algo_data["slug"] = algo_file.stem

                # Filter out hidden algorithms
                if not algo_data.get("hidden", False):
                    algorithms.append(algo_data)

    return algorithms


def load_shared_data() -> dict:
    """Load shared data (tags, references)."""
    data_dir = Path(__file__).parent.parent / "data"
    shared_dir = data_dir / "shared"

    shared_data = {}

    # Load tags
    tags_file = shared_dir / "tags.yaml"
    if tags_file.exists():
        with open(tags_file, encoding="utf-8") as f:
            shared_data["tags"] = yaml.safe_load(f) or {}

    # Load references
    refs_file = shared_dir / "refs.bib"
    if refs_file.exists():
        # For now, just store the file path
        # In a full implementation, you'd parse the .bib file
        shared_data["refs_file"] = str(refs_file)

    return shared_data


def render_family_page(family_id: str) -> str:
    """Render a family page using Jinja2 template."""
    # Load data
    family_data = load_family_data(family_id)
    if not family_data:
        return f"**Error:** Family '{family_id}' not found."

    algorithms = load_algorithm_data(family_id)
    shared_data = load_shared_data()

    # Set up Jinja2 environment
    template_dir = Path(__file__).parent.parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,  # We want markdown, not HTML
    )

    # Load and render template
    try:
        template = env.get_template("family_page.md")
        return template.render(
            family=family_data, algorithms=algorithms, shared=shared_data
        )
    except Exception as e:
        return f"**Error:** Template rendering error: {str(e)}"


def chatgpt_widget(
    algorithm: str,
    section: str = None,
    context: str = None,
    button_text: str = None,
    style: str = "primary"
) -> str:
    """Generate a complete ChatGPT button widget.

    Args:
        algorithm: The algorithm name (e.g., 'Fibonacci', 'Dynamic Programming')
        section: Optional section context (e.g., 'implementation', 'complexity')
        context: Optional additional context or prompt
        button_text: Custom button text (defaults to "🤖 Ask ChatGPT about {algorithm}")
        style: Button style - 'primary', 'secondary', or 'accent'

    Returns:
        HTML string for the ChatGPT button widget
    """
    # Build context parts
    parts = [f"Algorithm: {algorithm}"]
    if section:
        parts.append(f"Section: {section}")
    if context:
        parts.append(context)

    # Generate URL
    query = " | ".join(parts)
    url = f"https://chat.openai.com/?q={urllib.parse.quote(query)}"

    # Default button text
    if not button_text:
        button_text = f"🤖 Ask ChatGPT about {algorithm}"

    # Generate HTML with proper Material Design styling
    return f"""
    <a href="{url}"
       target="_blank"
       rel="noopener noreferrer"
       class="md-button md-button--{style}"
       style="margin: 8px 4px; text-decoration: none;">
        {button_text}
    </a>
    """


def family_page(family_id: str) -> str:
    """MkDocs macro function to render a family page.

    Args:
        family_id: The ID of the family to render (e.g., 'dp', 'control')

    Returns:
        Rendered HTML content for the family page
    """
    return render_family_page(family_id)
