"""
Generate algorithm pages for algorithm documentation.
"""

import mkdocs_gen_files
import yaml
from pathlib import Path


def generate_algorithm_pages():
    """Generate algorithm detail pages from YAML data using templates."""
    print("DEBUG: gen_algorithms.py is running!")

    # Import the macro system to use template rendering
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from macros import load_family_data, load_algorithm_data, discover_families

    # Discover and generate algorithm pages
    for family_id in discover_families():
        family_data = load_family_data(family_id)
        if not family_data:
            continue

        algorithms = load_algorithm_data(family_id)

        for algo in algorithms:
            algo_slug = algo.get("slug", "")
            if not algo_slug:
                continue

            # Get family slug for consistent URLs
            family_slug = family_data.get("slug", family_id)

            # Generate algorithm page using template
            algo_path = f"{family_slug}/{algo_slug}.md"

            with mkdocs_gen_files.open(algo_path, "w") as f:
                print(f"DEBUG: Generating algorithm page for {algo_path}")

                # Use the render_algorithm_page function directly
                from macros import render_algorithm_page

                # Render the page using the function
                rendered_content = render_algorithm_page(family_id, algo_slug)
                f.write(rendered_content)

            # Set edit path to the algorithm YAML file
            mkdocs_gen_files.set_edit_path(algo_path, algo.get("_source_file", ""))
