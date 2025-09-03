"""Data loader module for AlgoKit algorithms metadata.

This module provides functions to load, validate, and access the algorithms.yaml
data structure that serves as the single source of truth for all algorithm information.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases for better code readability
AlgorithmData = dict[str, Any]
FamilyData = dict[str, Any]
ProgressData = dict[str, Any]


class AlgorithmsDataLoader:
    """Loads and manages algorithms metadata from YAML files.

    This class provides a clean interface for accessing algorithm data,
    with built-in caching and error handling.
    """

    def __init__(self, data_file: str | Path | None = None) -> None:
        """Initialize the data loader.

        Args:
            data_file: Path to the algorithms.yaml file. If None, uses default location.
        """
        if data_file is None:
            # Default to algorithms.yaml in the project root
            self.data_file = Path(__file__).parent.parent / "algorithms.yaml"
        else:
            self.data_file = Path(data_file)

        self._data: dict[str, Any] | None = None
        self._cache_valid = False

    def load_data(self) -> dict[str, Any]:
        """Load and cache the algorithms data.

        Returns:
            The complete algorithms data structure.

        Raises:
            FileNotFoundError: If the algorithms.yaml file doesn't exist.
            yaml.YAMLError: If the YAML file is malformed.
        """
        if self._cache_valid and self._data is not None:
            return self._data

        try:
            with open(self.data_file, encoding="utf-8") as file:
                self._data = yaml.safe_load(file)
                self._cache_valid = True
                logger.info(
                    f"Successfully loaded algorithms data from {self.data_file}"
                )
                return self._data
        except FileNotFoundError:
            logger.error(f"Algorithms data file not found: {self.data_file}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {self.data_file}: {e}")
            raise

    def reload_data(self) -> dict[str, Any]:
        """Force reload of the algorithms data.

        Returns:
            The freshly loaded algorithms data structure.
        """
        self._cache_valid = False
        return self.load_data()

    def get_families(self) -> dict[str, FamilyData]:
        """Get all algorithm families.

        Returns:
            Dictionary mapping family keys to family data.
        """
        data = self.load_data()
        return data.get("families", {})

    def get_family(self, family_key: str) -> FamilyData | None:
        """Get a specific algorithm family by key.

        Args:
            family_key: The key identifying the family (e.g., 'dynamic-programming').

        Returns:
            Family data if found, None otherwise.
        """
        families = self.get_families()
        return families.get(family_key)

    def get_algorithms(self) -> dict[str, AlgorithmData]:
        """Get all algorithms.

        Returns:
            Dictionary mapping algorithm keys to algorithm data.
        """
        data = self.load_data()
        return data.get("algorithms", {})

    def get_algorithm(self, algorithm_key: str) -> AlgorithmData | None:
        """Get a specific algorithm by key.

        Args:
            algorithm_key: The key identifying the algorithm (e.g., 'fibonacci').

        Returns:
            Algorithm data if found, None otherwise.
        """
        algorithms = self.get_algorithms()
        return algorithms.get(algorithm_key)

    def get_algorithms_by_family(self, family_key: str) -> list[AlgorithmData]:
        """Get all algorithms belonging to a specific family.

        Args:
            family_key: The key identifying the family.

        Returns:
            List of algorithm data for the specified family.
        """
        algorithms = self.get_algorithms()
        return [
            algo for algo in algorithms.values() if algo.get("family") == family_key
        ]

    def get_progress(self) -> ProgressData:
        """Get implementation progress data.

        Returns:
            Progress tracking information.
        """
        data = self.load_data()
        return data.get("progress", {})

    def get_relationships(self) -> dict[str, Any]:
        """Get algorithm relationships and dependencies.

        Returns:
            Relationship data including family hierarchy and learning paths.
        """
        data = self.load_data()
        return data.get("relationships", {})

    def get_metadata(self) -> dict[str, Any]:
        """Get global metadata and configuration.

        Returns:
            Global metadata including version and configuration.
        """
        data = self.load_data()
        return data.get("metadata", {})

    def get_macro_config(self) -> dict[str, Any]:
        """Get configuration for macro generation.

        Returns:
            Configuration settings for navigation and content generation.
        """
        data = self.load_data()
        return data.get("macro_config", {})

    def validate_data_structure(self) -> bool:
        """Validate the loaded data structure.

        Returns:
            True if the data structure is valid, False otherwise.
        """
        try:
            data = self.load_data()

            # Check required top-level keys
            required_keys = ["families", "algorithms", "progress", "metadata"]
            for key in required_keys:
                if key not in data:
                    logger.error(f"Missing required key: {key}")
                    return False

            # Check that all algorithms reference valid families
            families = set(data["families"].keys())
            algorithms = data["algorithms"]

            for algo_key, algo_data in algorithms.items():
                family = algo_data.get("family")
                if family not in families:
                    logger.error(
                        f"Algorithm {algo_key} references unknown family: {family}"
                    )
                    return False

            logger.info("Data structure validation passed")
            return True

        except Exception as e:
            logger.error(f"Data structure validation failed: {e}")
            return False


# Global instance for easy access
_data_loader = AlgorithmsDataLoader()


def get_families() -> dict[str, FamilyData]:
    """Get all algorithm families (convenience function).

    Returns:
        Dictionary mapping family keys to family data.
    """
    return _data_loader.get_families()


def get_family(family_key: str) -> FamilyData | None:
    """Get a specific algorithm family (convenience function).

    Args:
        family_key: The key identifying the family.

    Returns:
        Family data if found, None otherwise.
    """
    return _data_loader.get_family(family_key)


def get_algorithms() -> dict[str, AlgorithmData]:
    """Get all algorithms (convenience function).

    Returns:
        Dictionary mapping algorithm keys to algorithm data.
    """
    return _data_loader.get_algorithms()


def get_algorithm(algorithm_key: str) -> AlgorithmData | None:
    """Get a specific algorithm (convenience function).

    Args:
        algorithm_key: The key identifying the algorithm.

    Returns:
        Algorithm data if found, None otherwise.
    """
    return _data_loader.get_algorithm(algorithm_key)


def get_algorithms_by_family(family_key: str) -> list[AlgorithmData]:
    """Get all algorithms belonging to a specific family (convenience function).

    Args:
        family_key: The key identifying the family.

    Returns:
        List of algorithm data for the specified family.
    """
    return _data_loader.get_algorithms_by_family(family_key)


def get_progress() -> ProgressData:
    """Get implementation progress (convenience function).

    Returns:
        Progress tracking information.
    """
    return _data_loader.get_progress()


def get_relationships() -> dict[str, Any]:
    """Get algorithm relationships (convenience function).

    Returns:
        Relationship data including family hierarchy and learning paths.
    """
    return _data_loader.get_relationships()


def get_metadata() -> dict[str, Any]:
    """Get global metadata (convenience function).

    Returns:
        Global metadata including version and configuration.
    """
    return _data_loader.get_metadata()


def get_macro_config() -> dict[str, Any]:
    """Get macro configuration (convenience function).

    Returns:
        Configuration settings for navigation and content generation.
    """
    return _data_loader.get_macro_config()
