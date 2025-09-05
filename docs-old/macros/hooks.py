"""MkDocs hooks for dynamic algorithm page generation.

This module implements the hooks that generate algorithm pages dynamically
during the MkDocs build process without requiring static .md files.
"""

from typing import Any

from mkdocs.structure.files import File
from mkdocs.structure.pages import Page

from .page_generator import generate_algorithm_page_content, load_algorithms_data


def on_files(files: list[File], config: dict[str, Any]) -> list[File]:
    """Add virtual algorithm pages to the MkDocs file collection.

    This hook runs during the file discovery phase and injects virtual
    algorithm pages into the build process.

    Args:
        files: List of files discovered by MkDocs
        config: MkDocs configuration

    Returns:
        Updated list of files including virtual algorithm pages
    """
    print("🚀 Adding virtual algorithm pages to MkDocs...")

    # Load algorithm data
    data = load_algorithms_data()
    algorithms = data.get("algorithms", {})
    families = data.get("families", {})

    # Create virtual files for each algorithm
    for algorithm_key, algorithm_data in algorithms.items():
        family_key = algorithm_data.get("family", "unknown")

        # Create the file path
        file_path = f"algorithms/{family_key}/{algorithm_key}.md"

        # Create a virtual file object
        virtual_file = File(
            path=file_path,
            src_dir=config["docs_dir"],
            dest_dir=config["site_dir"],
            use_directory_urls=config.get("use_directory_urls", True),
        )

        # Add to files collection
        files.append(virtual_file)
        print(f"✅ Added virtual file: {file_path}")

    print(f"🎉 Added {len(algorithms)} virtual algorithm pages!")
    return files


def on_page_markdown(
    markdown: str, page: Page, config: dict[str, Any], files: list[File]
) -> str:
    """Generate algorithm page content when MkDocs processes the page.

    This hook runs for each page during the markdown processing phase.
    If the page is an algorithm page, it generates the content dynamically.

    Args:
        markdown: Original markdown content (empty for virtual pages)
        page: The page being processed
        config: MkDocs configuration
        files: List of all files

    Returns:
        Generated markdown content for algorithm pages, or original content for others
    """
    # Check if this is an algorithm page
    if not page.file.src_path.startswith("algorithms/"):
        return markdown

    # Extract algorithm key from file path
    # Path format: algorithms/family/algorithm.md
    path_parts = page.file.src_path.split("/")
    if len(path_parts) != 3 or not path_parts[2].endswith(".md"):
        return markdown

    family_key = path_parts[1]
    algorithm_key = path_parts[2][:-3]  # Remove .md extension

    print(f"🔄 Generating content for algorithm: {algorithm_key}")

    # Load algorithm data
    data = load_algorithms_data()
    algorithms = data.get("algorithms", {})
    families = data.get("families", {})

    # Check if algorithm exists
    if algorithm_key not in algorithms:
        print(f"❌ Algorithm '{algorithm_key}' not found in algorithms.yaml")
        return f"# Algorithm Not Found\n\nAlgorithm '{algorithm_key}' not found in algorithms.yaml"

    # Get algorithm and family data
    algorithm_data = algorithms[algorithm_key]
    family_data = families.get(family_key, {})

    # Generate the page content
    try:
        content = generate_algorithm_page_content(
            algorithm_key, algorithm_data, family_data
        )
        print(f"✅ Generated content for {algorithm_key}")
        return content
    except Exception as e:
        print(f"❌ Error generating content for {algorithm_key}: {e}")
        return f"# Error Generating Page\n\nError generating content for algorithm '{algorithm_key}': {e}"


def on_post_build(config: dict[str, Any]) -> None:
    """Post-build hook for cleanup or additional processing.

    Args:
        config: MkDocs configuration
    """
    print("🎉 Dynamic algorithm page generation completed!")
