"""
MkDocs Gen Files Script for Algorithm Documentation

This script generates virtual pages for algorithm families and algorithms
from YAML data using mkdocs-gen-files. It implements comprehensive discovery
logic that scans data/families/*/family.yaml and algorithms/*.yaml files,
applying family-level filtering and ordering rules.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import mkdocs_gen_files


class DataDiscovery:
    """Handles discovery and loading of algorithm data from YAML files."""

    def __init__(self, data_dir: str = "mkdocs_plugins/data"):
        """Initialize the data discovery system.

        Args:
            data_dir: Path to the data directory containing families and shared data
        """
        self.data_dir = Path(data_dir)
        # The actual structure is data/family_id/ not data/families/family_id/
        self.families_dir = self.data_dir
        self.shared_dir = self.data_dir / "shared"

        # Cache for loaded data
        self._families_cache: Dict[str, Dict[str, Any]] = {}
        self._algorithms_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._shared_cache: Dict[str, Any] = {}

    def load_shared_data(self) -> Dict[str, Any]:
        """Load shared data (tags, references) from the shared directory.

        Returns:
            Dictionary containing shared data with keys: tags, refs
        """
        if self._shared_cache:
            return self._shared_cache

        shared_data = {}

        # Load tags
        tags_file = self.shared_dir / "tags.yaml"
        if tags_file.exists():
            with open(tags_file, "r", encoding="utf-8") as f:
                shared_data["tags"] = yaml.safe_load(f)
        else:
            shared_data["tags"] = {"tags": []}

        # Load references (BibTeX format)
        refs_file = self.shared_dir / "refs.bib"
        if refs_file.exists():
            with open(refs_file, "r", encoding="utf-8") as f:
                shared_data["refs"] = f.read()
        else:
            shared_data["refs"] = ""

        self._shared_cache = shared_data
        return shared_data

    def discover_families(self) -> List[str]:
        """Discover all algorithm families by scanning the families directory.

        Returns:
            List of family IDs (directory names)
        """
        if not self.families_dir.exists():
            return []

        families = []
        for item in self.families_dir.iterdir():
            if item.is_dir() and (item / "family.yaml").exists():
                families.append(item.name)

        return sorted(families)

    def load_family(self, family_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific family's metadata from its family.yaml file.

        Args:
            family_id: The family identifier (directory name)

        Returns:
            Family metadata dictionary or None if not found
        """
        if family_id in self._families_cache:
            return self._families_cache[family_id]

        family_file = self.families_dir / family_id / "family.yaml"
        if not family_file.exists():
            return None

        try:
            with open(family_file, "r", encoding="utf-8") as f:
                family_data = yaml.safe_load(f)
                family_data["_source_file"] = str(family_file)
                self._families_cache[family_id] = family_data
                return family_data
        except (yaml.YAMLError, IOError) as e:
            print(f"ERROR: Failed to load family {family_id}: {e}")
            return None

    def discover_algorithms(self, family_id: str) -> List[str]:
        """Discover all algorithms in a specific family.

        Args:
            family_id: The family identifier

        Returns:
            List of algorithm slugs (YAML filenames without extension)
        """
        algorithms_dir = self.families_dir / family_id / "algorithms"
        if not algorithms_dir.exists():
            return []

        algorithms = []
        for item in algorithms_dir.iterdir():
            if item.is_file() and item.suffix == ".yaml":
                algorithms.append(item.stem)

        return sorted(algorithms)

    def load_algorithm(
        self, family_id: str, algorithm_slug: str
    ) -> Optional[Dict[str, Any]]:
        """Load a specific algorithm's metadata from its YAML file.

        Args:
            family_id: The family identifier
            algorithm_slug: The algorithm slug (filename without extension)

        Returns:
            Algorithm metadata dictionary or None if not found
        """
        cache_key = f"{family_id}/{algorithm_slug}"
        if cache_key in self._algorithms_cache:
            return self._algorithms_cache[cache_key]

        algorithm_file = (
            self.families_dir / family_id / "algorithms" / f"{algorithm_slug}.yaml"
        )
        if not algorithm_file.exists():
            return None

        try:
            with open(algorithm_file, "r", encoding="utf-8") as f:
                algorithm_data = yaml.safe_load(f)
                algorithm_data["_source_file"] = str(algorithm_file)
                algorithm_data["_family_id"] = family_id
                self._algorithms_cache[cache_key] = algorithm_data
                return algorithm_data
        except (yaml.YAMLError, IOError) as e:
            print(f"ERROR: Failed to load algorithm {family_id}/{algorithm_slug}: {e}")
            return None

    def load_family_algorithms(self, family_id: str) -> List[Dict[str, Any]]:
        """Load all algorithms for a specific family with filtering and ordering.

        Args:
            family_id: The family identifier

        Returns:
            List of algorithm metadata dictionaries, filtered and ordered
        """
        if family_id in self._algorithms_cache:
            return self._algorithms_cache[family_id]

        family_data = self.load_family(family_id)
        if not family_data:
            return []

        # Get algorithm slugs
        algorithm_slugs = self.discover_algorithms(family_id)

        # Load all algorithms
        algorithms = []
        for slug in algorithm_slugs:
            algorithm_data = self.load_algorithm(family_id, slug)
            if algorithm_data:
                algorithms.append(algorithm_data)

        # Apply family-level filtering and ordering
        algorithms = self._apply_family_filtering(family_data, algorithms)
        algorithms = self._apply_family_ordering(family_data, algorithms)

        self._algorithms_cache[family_id] = algorithms
        return algorithms

    def _apply_family_filtering(
        self, family_data: Dict[str, Any], algorithms: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply family-level filtering rules to algorithms.

        Args:
            family_data: Family metadata containing filtering rules
            algorithms: List of algorithm metadata

        Returns:
            Filtered list of algorithms
        """
        if "algorithms" not in family_data:
            return algorithms

        algo_config = family_data["algorithms"]

        # Apply include filter
        if "include" in algo_config and algo_config["include"]:
            include_slugs = set(algo_config["include"])
            algorithms = [
                algo for algo in algorithms if algo.get("slug") in include_slugs
            ]

        # Apply exclude filter
        if "exclude" in algo_config and algo_config["exclude"]:
            exclude_slugs = set(algo_config["exclude"])
            algorithms = [
                algo for algo in algorithms if algo.get("slug") not in exclude_slugs
            ]

        return algorithms

    def _apply_family_ordering(
        self, family_data: Dict[str, Any], algorithms: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply family-level ordering rules to algorithms.

        Args:
            family_data: Family metadata containing ordering rules
            algorithms: List of algorithm metadata

        Returns:
            Ordered list of algorithms
        """
        if "algorithms" not in family_data:
            return algorithms

        algo_config = family_data["algorithms"]
        order_mode = algo_config.get("order_mode", "by_algo_order")

        if order_mode == "by_algo_order":
            # Sort by the 'order' field in each algorithm
            return sorted(algorithms, key=lambda x: x.get("order", 999))
        elif order_mode == "by_name":
            # Sort alphabetically by name
            return sorted(algorithms, key=lambda x: x.get("name", ""))
        elif order_mode == "by_slug":
            # Sort alphabetically by slug
            return sorted(algorithms, key=lambda x: x.get("slug", ""))
        elif order_mode == "by_complexity":
            # Sort by complexity (would need to parse complexity strings)
            # For now, fall back to by_algo_order
            return sorted(algorithms, key=lambda x: x.get("order", 999))
        else:
            # Default to by_algo_order
            return sorted(algorithms, key=lambda x: x.get("order", 999))

    def get_all_data(self) -> Dict[str, Any]:
        """Get all discovered data in a structured format.

        Returns:
            Dictionary containing all families, algorithms, and shared data
        """
        shared_data = self.load_shared_data()
        families = self.discover_families()

        result = {"shared": shared_data, "families": {}, "algorithms": {}}

        for family_id in families:
            family_data = self.load_family(family_id)
            if family_data:
                result["families"][family_id] = family_data

                # Load algorithms for this family
                algorithms = self.load_family_algorithms(family_id)
                result["algorithms"][family_id] = algorithms

        return result


def generate_family_pages(discovery: DataDiscovery) -> None:
    """Generate virtual pages for algorithm families.

    Args:
        discovery: DataDiscovery instance with loaded data
    """
    families = discovery.discover_families()

    for family_id in families:
        family_data = discovery.load_family(family_id)
        if not family_data:
            continue

        # Generate family page (using family_id as directory name to match data structure)
        family_path = f"{family_id}/index.md"

        with mkdocs_gen_files.open(family_path, "w") as f:
            f.write(f"# {family_data.get('name', family_id)}\n\n")
            f.write(f"{family_data.get('summary', '')}\n\n")
            f.write(f"{family_data.get('description', '')}\n\n")

            # Add algorithms list
            algorithms = discovery.load_family_algorithms(family_id)
            if algorithms:
                f.write("## Algorithms\n\n")
                for algo in algorithms:
                    algo_slug = algo.get("slug", "")
                    algo_name = algo.get("name", algo_slug)
                    f.write(f"- [{algo_name}]({algo_slug}.md)\n")
                f.write("\n")

        # Set edit path to the family.yaml file
        mkdocs_gen_files.set_edit_path(family_path, family_data.get("_source_file", ""))


def generate_algorithm_pages(discovery: DataDiscovery) -> None:
    """Generate virtual pages for individual algorithms.

    Args:
        discovery: DataDiscovery instance with loaded data
    """
    families = discovery.discover_families()

    for family_id in families:
        algorithms = discovery.load_family_algorithms(family_id)

        for algo in algorithms:
            algo_slug = algo.get("slug", "")
            if not algo_slug:
                continue

            # Generate algorithm page (in the same directory as the family)
            algo_path = f"{family_id}/{algo_slug}.md"

            with mkdocs_gen_files.open(algo_path, "w") as f:
                f.write(f"# {algo.get('name', algo_slug)}\n\n")
                f.write(f"{algo.get('summary', '')}\n\n")
                f.write(f"{algo.get('description', '')}\n\n")

                # Add family information
                family_data = discovery.load_family(family_id)
                if family_data:
                    family_name = family_data.get("name", family_id)
                    f.write(f"**Family:** [{family_name}](index.md)\n\n")

                # Add implementations if available
                if "implementations" in algo:
                    f.write("## Implementations\n\n")
                    for impl in algo["implementations"]:
                        impl_name = impl.get("name", "Unknown")
                        impl_type = impl.get("type", "unknown")
                        f.write(f"### {impl_name}\n\n")
                        f.write(f"**Type:** {impl_type}\n\n")

                        if "description" in impl:
                            f.write(f"{impl['description']}\n\n")

                        if "complexity" in impl:
                            complexity = impl["complexity"]
                            f.write("**Complexity:**\n")
                            if "time" in complexity:
                                f.write(f"- Time: {complexity['time']}\n")
                            if "space" in complexity:
                                f.write(f"- Space: {complexity['space']}\n")
                            f.write("\n")

                        if "code" in impl:
                            f.write("```python\n")
                            f.write(impl["code"])
                            f.write("\n```\n\n")

            # Set edit path to the algorithm YAML file
            mkdocs_gen_files.set_edit_path(algo_path, algo.get("_source_file", ""))


def main():
    """Generate virtual pages for algorithm documentation."""
    print("DEBUG: gen_pages.py is running!")

    # Import and call the other modules
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent))
    from gen_families import generate_family_pages
    from gen_algorithms import generate_algorithm_pages

    # Generate family pages
    try:
        generate_family_pages()
        print("DEBUG: Generated family pages")
    except Exception as e:
        print(f"ERROR: Failed to generate family pages: {e}")

    # Generate algorithm pages
    try:
        generate_algorithm_pages()
        print("DEBUG: Generated algorithm pages")
    except Exception as e:
        print(f"ERROR: Failed to generate algorithm pages: {e}")

    print("DEBUG: gen_pages.py completed!")


# This is the entry point for mkdocs-gen-files
main()
