"""
Custom MkDocs plugin for dynamic algorithm page generation.

This plugin creates virtual algorithm pages during the MkDocs build process.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

from mkdocs.plugins import BasePlugin
from mkdocs.structure.files import File
from mkdocs.structure.pages import Page

from .page_generator import load_algorithms_data, generate_algorithm_page_content


class DynamicAlgorithmPlugin(BasePlugin):
    """MkDocs plugin that generates algorithm pages dynamically from YAML data."""

    def __init__(self):
        self.algorithms_data = None
        self.virtual_files = []

    def on_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load algorithm data when config is processed."""
        print("🚀 Loading algorithm data for dynamic generation...")
        self.algorithms_data = load_algorithms_data()
        return config

    def on_files(self, files: List[File], config: Dict[str, Any]) -> List[File]:
        """Add virtual algorithm pages to the MkDocs file collection."""
        if not self.algorithms_data:
            return files

        print("🚀 Adding virtual algorithm pages to MkDocs...")

        algorithms = self.algorithms_data.get("algorithms", {})
        families = self.algorithms_data.get("families", {})

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
            self.virtual_files.append(virtual_file)
            print(f"✅ Added virtual file: {file_path}")

        print(f"🎉 Added {len(algorithms)} virtual algorithm pages!")
        return files

    def on_page_markdown(
        self, markdown: str, page: Page, config: Dict[str, Any], files: List[File]
    ) -> str:
        """Generate algorithm page content when MkDocs processes the page."""
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

        if not self.algorithms_data:
            return f"# Algorithm Not Found\n\nAlgorithm data not loaded."

        algorithms = self.algorithms_data.get("algorithms", {})
        families = self.algorithms_data.get("families", {})

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

    def on_post_build(self, config: Dict[str, Any]) -> None:
        """Post-build hook for cleanup or additional processing."""
        print("🎉 Dynamic algorithm page generation completed!")
