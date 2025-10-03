"""MkDocs Macros Module for Algorithm Documentation

This module provides macros and filters for rendering algorithm documentation
from YAML data using Jinja2 templates.
"""

import re
import urllib.parse
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader


def load_bibtex_references(bib_file: Path) -> dict:
    """Load BibTeX references from a .bib file and return as dict."""
    references = {}

    if not bib_file.exists():
        print(f"DEBUG: BibTeX file not found: {bib_file}")
        return references

    with open(bib_file, encoding="utf-8") as f:
        content = f.read()

    # Parse BibTeX entries - split by @ entries
    entries = re.split(r"@(\w+)\s*\{", content)[1:]  # Skip first empty element

    for i in range(0, len(entries), 2):
        if i + 1 >= len(entries):
            break

        entry_type = entries[i].strip()
        entry_content = entries[i + 1]

        # Find the key (first part before comma)
        key_match = re.match(r"([^,]+),", entry_content)
        if not key_match:
            continue

        key = key_match.group(1).strip()

        # Parse fields - look for field = {value} patterns
        field_dict = {}
        field_pattern = r"(\w+)\s*=\s*\{([^}]+)\}"
        field_matches = re.findall(field_pattern, entry_content, re.DOTALL)

        for field_name, field_value in field_matches:
            # Clean up the field value
            cleaned_value = re.sub(r"\s+", " ", field_value.strip())
            field_dict[field_name.strip()] = cleaned_value

        references[key] = {"type": entry_type, "key": key, **field_dict}
    return references


def load_family_data(family_id: str) -> dict:
    """Load family data from YAML file."""
    data_dir = Path(__file__).parent / "mkdocs_plugins" / "data"
    family_file = data_dir / family_id / "family.yaml"

    if not family_file.exists():
        return {}

    with open(family_file, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_algorithm_data(family_id: str) -> list:
    """Load algorithm data for a family."""
    data_dir = Path(__file__).parent / "mkdocs_plugins" / "data"
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


def load_all_algorithm_data(family_id: str) -> list:
    """Load all algorithm data for a family (including hidden algorithms)."""
    data_dir = Path(__file__).parent / "mkdocs_plugins" / "data"
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

                # Include all algorithms (including hidden ones)
                algorithms.append(algo_data)

    return algorithms


def discover_families() -> list[str]:
    """Discover all algorithm families by scanning the data directory."""
    data_dir = Path(__file__).parent / "mkdocs_plugins" / "data"

    if not data_dir.exists():
        return []

    families = []
    for item in data_dir.iterdir():
        if item.is_dir() and (item / "family.yaml").exists():
            families.append(item.name)

    return sorted(families)


def load_all_families() -> list[dict[str, Any]]:
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

            # Determine overall family status based on completion
            if complete_count == total_algorithms and total_algorithms > 0:
                # All algorithms are complete
                status = "complete"
                available_algorithms = complete_algorithms
            elif complete_count > 0 or len(in_progress_algorithms) > 0:
                # Some algorithms are complete or in progress
                status = "in-progress"
                available_algorithms = complete_algorithms + in_progress_algorithms
            else:
                # No algorithms are complete or in progress
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
    all_algorithms = load_all_algorithm_data(family_id)

    # Load shared data for references and tags
    data_dir = Path(__file__).parent / "mkdocs_plugins" / "data"
    shared_dir = data_dir / "shared"

    # Load tags
    tags_file = shared_dir / "tags.yaml"
    tags_dict = {}
    if tags_file.exists():
        with open(tags_file, encoding="utf-8") as f:
            tags_data = yaml.safe_load(f) or {}
            # Convert tags list to dict for easy lookup
            tags_dict = {tag["id"]: tag for tag in tags_data.get("tags", [])}

    # Load references
    refs_file = shared_dir / "refs.bib"
    references_dict = {}
    if refs_file.exists():
        references_dict = load_bibtex_references(refs_file)

    # Process family references - convert bib_key list to full reference objects
    processed_references = []
    family_references = family_data.get("references", [])
    for ref_item in family_references:
        if isinstance(ref_item, dict) and "bib_key" in ref_item:
            bib_key = ref_item["bib_key"]
            if bib_key in references_dict:
                processed_references.append(references_dict[bib_key])
        elif isinstance(ref_item, str):
            # Handle case where references is just a list of bib_key strings
            if ref_item in references_dict:
                processed_references.append(references_dict[ref_item])

    # Process family tags - convert tag ID list to full tag objects
    processed_tags = []
    family_tags = family_data.get("tags", [])
    for tag_item in family_tags:
        if isinstance(tag_item, str):
            if tag_item in tags_dict:
                processed_tags.append(tags_dict[tag_item])

    # Set up Jinja2 environment
    template_dir = Path(__file__).parent / "mkdocs_plugins" / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=False)

    # Load and render template
    try:
        template = env.get_template("family_page.md")
        return template.render(
            family=family_data,
            algorithms=algorithms,
            references=processed_references,
            tags=processed_tags,
        )
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
        # Read template content directly to avoid MkDocs macros processing it
        template_file = template_dir / "algorithm_page.md"
        with open(template_file, encoding="utf-8") as f:
            template_content = f.read()

        # Create a reference to the global chatgpt_widget macro function
        def chatgpt_widget(algorithm: str, section: str = None, context: str = None, button_text: str = None, style: str = "primary") -> str:
            """Generate a complete ChatGPT button widget."""
            # Build context parts - include algorithm name for better context
            parts = [f"Algorithm: {algorithm}"]
            if section:
                parts.append(f"Section: {section}")
            if context:
                parts.append(context)

            # Generate URL with properly structured query
            query = " | ".join(parts)
            url = f"https://chat.openai.com/?q={urllib.parse.quote(query)}"

            # Default button text
            if not button_text:
                button_text = f"🤖 Ask ChatGPT about {algorithm}"

            # Generate markdown link that will be styled by Material theme
            return f'<a href="{url}" target="_blank" rel="noopener noreferrer" class="md-button md-button--{style}">{button_text}</a>'

        # Create template from string content
        template = env.from_string(template_content)
        return template.render(
            algo=algorithm_data,
            family=family_data,
            algorithms=algorithms,
            chatgpt_widget=chatgpt_widget
        )
    except Exception as e:
        import traceback

        return f"**Error:** Template rendering error: {str(e)}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"


def define_env(env):
    """Define the environment for MkDocs Macros.

    This function is called by mkdocs-macros to set up the Jinja2 environment
    with custom variables, macros, and filters.
    """

    # Add a simple test macro to verify the system works
    @env.macro
    def test_macro():
        """Test macro to verify macros are working."""
        return "🎉 **MkDocs Macros is working!** This message was generated by a Python function."

    @env.macro
    def render_families_grid() -> str:
        """Render a grid of algorithm family cards for the families landing page.

        Returns:
            Rendered HTML grid of family cards
        """
        try:
            families = load_all_families()
        except Exception as e:
            return f"**Error loading families: {e}**"

        if not families:
            return "**No algorithm families found.**"

        # Create family cards
        cards_html = ""
        for family in families:
            family_id = family.get("id", "")
            family_name = family.get("name", "Unknown Family")
            family_slug = family.get("slug", family_id)
            family_summary = family.get("summary", "No description available.")
            family_description = family.get("description", "")

            # Get algorithm count and status
            algorithms = family.get("all_algorithms", [])
            algorithm_count = len(algorithms)

            # Determine status and badge
            status = family.get("status", "planned")
            completion_percentage = family.get("completion_percentage", 0)

            if status == "complete":
                status_badge = f"✅ Complete ({completion_percentage}%)"
                status_class = "status-complete"
            elif status == "in-progress":
                status_badge = f"🚧 In Progress ({completion_percentage}%)"
                status_class = "status-in-progress"
            else:
                status_badge = "📋 Planned"
                status_class = "status-planned"

            # Create the card HTML with clickable link
            card_html = f"""
<a href="../{family_slug}/" class="family-card" style="
    border: 2px solid var(--md-default-fg-color--light);
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
    background-color: var(--md-default-bg-color--lightest);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: pointer;
    text-decoration: none;
    color: inherit;
    display: block;
">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px;">
        <h3 style="margin: 0; color: var(--md-primary-fg-color);">{family_name}</h3>
        <span class="{status_class}" style="
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.8em;
            font-weight: bold;
            background-color: var(--md-default-bg-color--dark);
        ">{status_badge}</span>
    </div>

    <p style="margin: 0 0 16px 0; font-style: italic; color: var(--md-default-fg-color--light);">
        {family_summary}
    </p>

    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 16px;">
        <span style="color: var(--md-default-fg-color--light); font-size: 0.9em;">
            {algorithm_count} algorithm{'s' if algorithm_count != 1 else ''}
        </span>
        <span style="color: var(--md-primary-fg-color); font-weight: bold;">
            Explore →
        </span>
    </div>
</a>
"""
            cards_html += card_html

        # Wrap cards in a grid container
        grid_html = f"""
<div class="families-grid" style="
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 24px;
    margin: 24px 0;
">
    {cards_html}
</div>

<style>
.family-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}}

.status-complete {{
    background-color: #4caf50 !important;
    color: white !important;
}}

.status-in-progress {{
    background-color: #ff9800 !important;
    color: white !important;
}}

.status-planned {{
    background-color: #9e9e9e !important;
    color: white !important;
}}

@media (max-width: 768px) {{
    .families-grid {{
        grid-template-columns: 1fr;
    }}
}}
</style>
"""

        return grid_html

    # Add the index page macro
    @env.macro
    def render_index() -> str:
        """Render the main index page using YAML data and Jinja2 templates.

        Returns:
            Rendered markdown content for the index page
        """
        return render_index_page()

    @env.macro
    def algorithm_card(algorithm_key: str) -> str:
        """Generate an algorithm card with key information.

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
        """Generate a navigation grid showing related algorithms.

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
        """Render a family page using YAML data and Jinja2 templates.

        Args:
            family_id: The family identifier

        Returns:
            Rendered markdown content for the family page
        """
        family_data = load_family_data(family_id)
        if not family_data:
            return f"**Error:** Family '{family_id}' not found."

        algorithms = load_algorithm_data(family_id)
        all_algorithms = load_all_algorithm_data(family_id)

        # Load shared data for references and tags
        data_dir = Path(__file__).parent / "mkdocs_plugins" / "data"
        shared_dir = data_dir / "shared"

        # Load tags
        tags_file = shared_dir / "tags.yaml"
        tags_dict = {}
        if tags_file.exists():
            with open(tags_file, encoding="utf-8") as f:
                tags_data = yaml.safe_load(f) or {}
                # Convert tags list to dict for easy lookup
                tags_dict = {tag["id"]: tag for tag in tags_data.get("tags", [])}

        # Load references
        refs_file = shared_dir / "refs.bib"
        references_dict = {}
        if refs_file.exists():
            references_dict = load_bibtex_references(refs_file)

        # Process family references - convert bib_key list to full reference objects
        processed_references = []
        family_references = family_data.get("references", [])
        for ref_item in family_references:
            if isinstance(ref_item, dict) and "bib_key" in ref_item:
                bib_key = ref_item["bib_key"]
                if bib_key in references_dict:
                    processed_references.append(references_dict[bib_key])
            elif isinstance(ref_item, str):
                # Handle case where references is just a list of bib_key strings
                if ref_item in references_dict:
                    processed_references.append(references_dict[ref_item])

        # Process family tags - convert tag ID list to full tag objects
        processed_tags = []
        family_tags = family_data.get("tags", [])
        for tag_item in family_tags:
            if isinstance(tag_item, str):
                if tag_item in tags_dict:
                    processed_tags.append(tags_dict[tag_item])

        # Add all algorithms to family data for template access
        family_data["all_algorithms"] = all_algorithms

        # Set up Jinja2 environment
        template_dir = Path(__file__).parent / "mkdocs_plugins" / "templates"
        env_jinja = Environment(
            loader=FileSystemLoader(str(template_dir)), autoescape=False
        )

    # Load and render template
    try:
        template = env_jinja.get_template("family_page.md")
        return template.render(
            family=family_data,
            algorithms=algorithms,
            references=processed_references,
            tags=processed_tags,
        )
    except Exception as e:
        return f"**Error:** Template rendering error: {str(e)}"

    @env.macro
    def render_algorithm_page(family_id: str, algorithm_slug: str) -> str:
        """Render an algorithm page using YAML data and Jinja2 templates.

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

        # Load shared data for references
        data_dir = Path(__file__).parent / "mkdocs_plugins" / "data"
        shared_dir = data_dir / "shared"

        # Load references
        refs_file = shared_dir / "refs.bib"
        references_dict = {}
        if refs_file.exists():
            references_dict = load_bibtex_references(refs_file)

        # Process algorithm references - convert bib_key list to full reference objects
        processed_references = []
        algorithm_references = algorithm_data.get("references", [])
        for ref_category in algorithm_references:
            if "items" in ref_category:
                processed_items = []
                for item in ref_category["items"]:
                    if isinstance(item, dict) and "bib_key" in item:
                        bib_key = item["bib_key"]
                        if bib_key in references_dict:
                            processed_items.append(references_dict[bib_key])
                    elif isinstance(item, str):
                        if item in references_dict:
                            processed_items.append(references_dict[item])
                    else:
                        processed_items.append(item)
                ref_category["items"] = processed_items
            processed_references.append(ref_category)

        # Load and render template
        try:
            print(f"DEBUG: Loading template from: {template_dir}")
            print(f"DEBUG: Template file exists: {(template_dir / 'algorithm_page.md').exists()}")
            template = env_jinja.get_template("algorithm_page.md")

            # Create a reference to the global chatgpt_widget macro function
            def chatgpt_widget(algorithm: str, section: str = None, context: str = None, button_text: str = None, style: str = "primary") -> str:
                """Generate a complete ChatGPT button widget."""
                # Build context parts - include algorithm name for better context
                parts = [f"Algorithm: {algorithm}"]
                if section:
                    parts.append(f"Section: {section}")
                if context:
                    parts.append(context)

                # Generate URL with properly structured query
                query = " | ".join(parts)
                url = f"https://chat.openai.com/?q={urllib.parse.quote(query)}"

                # Default button text
                if not button_text:
                    button_text = f"🤖 Ask ChatGPT about {algorithm}"

                # Generate markdown link that will be styled by Material theme
                return f'<a href="{url}" target="_blank" rel="noopener noreferrer" class="md-button md-button--{style}">{button_text}</a>'

            return template.render(
                algo=algorithm_data,
                family=family_data,
                algorithms=algorithms,
                chatgpt_widget=chatgpt_widget,
                env={
                    "get": lambda key, default=None: {
                        "AMAZON_AFFILIATE_ID": "your-affiliate-id"
                    }.get(key, default)
                },
            )
        except Exception as e:
            return f"**Error:** Template rendering error: {str(e)}"

    @env.macro
    def isbn_link(isbn: str, text: str = None, tag: str = "mathybits-20") -> str:
        """Generate an Amazon affiliate link from an ISBN.

        Args:
            isbn: The ISBN number of the book (e.g., '0-201-89683-4' or '9780134685991').
            text: Optional display text. Defaults to the ISBN itself.
            tag: Amazon Associates tag. Defaults to 'mathybits-20'.

        Returns:
            HTML anchor tag that opens the affiliate link in a new tab.
        """
        # Strip hyphens and spaces from ISBN for Amazon /dp/ format
        clean_isbn = isbn.replace("-", "").replace(" ", "")
        url = f"https://www.amazon.com/dp/{clean_isbn}/?tag={tag}"
        label = text or isbn
        return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{label}</a>'

    @env.macro
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
        # Build context parts - include algorithm name for better context
        parts = [f"Algorithm: {algorithm}"]
        if section:
            parts.append(f"Section: {section}")
        if context:
            parts.append(context)

        # Generate URL with properly structured query
        query = " | ".join(parts)
        url = f"https://chat.openai.com/?q={urllib.parse.quote(query)}"

        # Default button text
        if not button_text:
            button_text = f"🤖 Ask ChatGPT about {algorithm}"

        # Generate markdown link that will be styled by Material theme
        return f'<a href="{url}" target="_blank" rel="noopener noreferrer" class="md-button md-button--{style}">{button_text}</a>'

    # Add a simple test variable
    env.variables["current_date"] = "January 4, 2025"
    env.variables["project_name"] = "AlgoKit"
    env.variables["project_description"] = (
        "A python implementation of control and learning algorithms"
    )
