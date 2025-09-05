"""
Generate family pages for algorithm documentation.
"""

import mkdocs_gen_files
import yaml
import re
from pathlib import Path
from jinja2 import Environment, FileSystemLoader


def load_bibtex_references(bib_file: Path) -> dict:
    """Load BibTeX references from a .bib file and return as dict."""
    references = {}

    if not bib_file.exists():
        print(f"DEBUG: BibTeX file not found: {bib_file}")
        return references

    with open(bib_file, "r", encoding="utf-8") as f:
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


def generate_family_pages():
    """Generate family overview pages from YAML data."""
    print("DEBUG: gen_families.py is running!")
    # Load shared data
    data_dir = Path(__file__).parent / "data"
    shared_dir = data_dir / "shared"

    # Load tags
    tags_file = shared_dir / "tags.yaml"
    tags = {}
    if tags_file.exists():
        with open(tags_file, "r", encoding="utf-8") as f:
            tags_data = yaml.safe_load(f) or {}
            # Convert tags list to dict for easy lookup
            tags = {tag["id"]: tag for tag in tags_data.get("tags", [])}

    # Load references
    refs_file = shared_dir / "refs.bib"
    references = {}
    if refs_file.exists():
        references = load_bibtex_references(refs_file)

    # Discover family directories
    families = []
    for family_dir in data_dir.iterdir():
        if family_dir.is_dir() and family_dir.name != "shared":
            family_file = family_dir / "family.yaml"
            if family_file.exists():
                with open(family_file, "r", encoding="utf-8") as f:
                    family_data = yaml.safe_load(f)
                    if family_data:
                        family_data["_source_file"] = str(family_file)
                        families.append((family_dir.name, family_data))

    # Generate family pages
    for family_id, family in families:
        family_name = family.get("name", family_id.title())
        family_description = family.get("description", "")
        family_overview = family.get("overview", "")
        family_slug = family.get(
            "slug", family_id
        )  # Use slug if available, fallback to id
        family_summary = family.get(
            "summary",
            (
                family_description[:100] + "..."
                if len(family_description) > 100
                else family_description
            ),
        )

        # Generate family page (using slug for cleaner URLs)
        family_path = f"{family_slug}/index.md"

        # Load algorithms for this family with management controls
        algorithms_dir = Path(__file__).parent / "data" / family_id / "algorithms"
        algorithms = []
        if algorithms_dir.exists():
            for algo_file in algorithms_dir.glob("*.yaml"):
                with open(algo_file, "r", encoding="utf-8") as af:
                    algo_data = yaml.safe_load(af)
                    if algo_data:
                        algorithms.append(algo_data)

        # Apply algorithm management settings
        algorithms = apply_algorithm_management(algorithms, family)

        # Use the macro system for template rendering
        try:
            # Import the macro system
            import sys

            sys.path.append(str(Path(__file__).parent.parent))
            from macros import render_family_page

            # Render the page using the macro function directly
            rendered_content = render_family_page(family_id)

            # Add meta description to the page
            meta_description = f"description: {family_summary}"

            with mkdocs_gen_files.open(family_path, "w") as f:
                print(f"DEBUG: Generating family page for {family_path}")
                f.write(f"---\n{meta_description}\n---\n\n")
                f.write(rendered_content)

        except Exception as e:
            print(f"ERROR: Failed to render template for {family_id}: {e}")
            # Fallback to simple content
            with mkdocs_gen_files.open(family_path, "w") as f:
                f.write(f"# {family_name} Algorithms\n\n{family_description}\n")

        # Set edit path to the family YAML file
        mkdocs_gen_files.set_edit_path(family_path, family.get("_source_file", ""))


def apply_algorithm_management(algorithms: list, family: dict) -> list:
    """Apply algorithm management settings from family configuration."""
    if not algorithms:
        return algorithms

    # Get algorithm management settings
    algo_settings = family.get("algorithms", {})
    order_mode = algo_settings.get("order_mode", "by_algo_order")
    include_list = algo_settings.get("include", [])
    exclude_list = algo_settings.get("exclude", [])

    # Filter algorithms based on include/exclude lists
    filtered_algorithms = []
    for algo in algorithms:
        algo_slug = algo.get("slug", algo.get("name", "").lower().replace(" ", "-"))

        # Check include list (if not empty, only include specified algorithms)
        if include_list and algo_slug not in include_list:
            continue

        # Check exclude list (exclude specified algorithms)
        if algo_slug in exclude_list:
            continue

        filtered_algorithms.append(algo)

    # Sort algorithms based on order_mode
    if order_mode == "by_name":
        filtered_algorithms.sort(key=lambda x: x.get("name", "").lower())
    elif order_mode == "by_slug":
        filtered_algorithms.sort(
            key=lambda x: x.get("slug", x.get("name", "").lower().replace(" ", "-"))
        )
    elif order_mode == "by_complexity":
        # Sort by time complexity (simple heuristic)
        def complexity_key(algo):
            complexity = algo.get("complexity", {})
            analysis = complexity.get("analysis", [])
            if analysis:
                time_comp = analysis[0].get("time", "O(n)")
                # Simple complexity ordering
                if "O(1)" in time_comp:
                    return 0
                elif "O(log" in time_comp:
                    return 1
                elif "O(n)" in time_comp:
                    return 2
                elif "O(n log" in time_comp:
                    return 3
                elif "O(n²)" in time_comp or "O(n^2)" in time_comp:
                    return 4
                elif "O(n³)" in time_comp or "O(n^3)" in time_comp:
                    return 5
                else:
                    return 6
            return 7  # Unknown complexity goes last

        filtered_algorithms.sort(key=complexity_key)
    elif order_mode == "by_algo_order":
        # Sort by algo_order field if it exists, otherwise by name
        def algo_order_key(algo):
            order = algo.get("algo_order", 999)  # Default high number for no order
            name = algo.get("name", "").lower()
            return (order, name)

        filtered_algorithms.sort(key=algo_order_key)
    else:
        # Default: keep original order
        pass

    return filtered_algorithms
