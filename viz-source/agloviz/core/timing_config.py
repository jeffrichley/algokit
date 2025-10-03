"""Timing configuration loader for BFS scene animations."""

from pathlib import Path
from typing import Any

import yaml


class BfsTimingConfig:
    """Manages timing configuration for BFS scene animations."""

    def __init__(self, config_file: str | None = None):
        """Initialize timing configuration.

        Args:
            config_file: Path to timing configuration file. If None, uses default.
        """
        if config_file is None:
            # Default config file location
            project_root = Path(__file__).parent.parent.parent.parent
            config_file = str(
                project_root
                / "data"
                / "examples"
                / "scenarios"
                / "bfs_timing_config.yaml"
            )

        self.config_file = Path(config_file)
        self._config: dict[str, Any] = {}
        self._load_config()

        # Current speed mode
        self.current_mode = "cinematic"

    def _load_config(self) -> None:
        """Load timing configuration from YAML file."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Timing config file not found: {self.config_file}")

        with open(self.config_file) as f:
            self._config = yaml.safe_load(f)

    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._load_config()

    def get_timing(
        self, category: str, animation: str, mode: str | None = None
    ) -> float:
        """Get adjusted timing for an animation.

        Args:
            category: Timing category (base_timings, wait_times)
            animation: Animation name
            mode: Speed mode (cinematic, development, quick_demo). If None, uses current mode.

        Returns:
            Adjusted timing in seconds
        """
        if mode is None:
            mode = self.current_mode

        # Get base timing
        base_time = self._config.get(category, {}).get(animation, 1.0)

        # Get speed multiplier
        multiplier = (
            self._config.get("speed_multipliers", {})
            .get(mode, {})
            .get(self._get_timing_category(category), 1.0)
        )

        return base_time / multiplier

    def _get_timing_category(self, category: str) -> str:
        """Map timing category to speed multiplier category."""
        category_mapping = {
            "base_timings": "setup",  # Most base timings are setup-related
            "wait_times": "waits",
        }
        return category_mapping.get(category, "setup")

    def get_wait_time(self, wait_name: str, mode: str | None = None) -> float:
        """Get wait time for a specific wait."""
        return self.get_timing("wait_times", wait_name, mode)

    def get_animation_time(self, animation_name: str, mode: str | None = None) -> float:
        """Get animation time for a specific animation."""
        return self.get_timing("base_timings", animation_name, mode)

    def set_mode(self, mode: str) -> None:
        """Set the current speed mode.

        Args:
            mode: Speed mode ('cinematic', 'development', 'quick_demo')
        """
        if mode not in self._config.get("speed_multipliers", {}):
            raise ValueError(
                f"Invalid speed mode: {mode}. Available: {list(self._config.get('speed_multipliers', {}).keys())}"
            )
        self.current_mode = mode

    def get_event_limit(self) -> int:
        """Get maximum number of events to display."""
        return self._config.get("event_limits", {}).get("max_events_displayed", 999)

    def get_path_reconstruction_config(self) -> dict[str, Any]:
        """Get path reconstruction configuration."""
        return self._config.get("path_reconstruction", {})

    def get_available_modes(self) -> list[str]:
        """Get list of available speed modes."""
        return list(self._config.get("speed_multipliers", {}).keys())

    def print_current_settings(self) -> None:
        """Print current timing settings."""
        print("🎬 BFS Timing Configuration:")
        print(f"   Mode: {self.current_mode}")
        print(f"   Max Events: {self.get_event_limit()}")
        print(f"   Config File: {self.config_file}")
        print(f"   Available Modes: {', '.join(self.get_available_modes())}")


# Global timing config instance
_timing_config: BfsTimingConfig | None = None


def get_timing_config(config_file: str | None = None) -> BfsTimingConfig:
    """Get global timing configuration instance.

    Args:
        config_file: Path to timing configuration file. If None, uses default.

    Returns:
        BfsTimingConfig instance
    """
    global _timing_config
    if _timing_config is None:
        _timing_config = BfsTimingConfig(config_file)
    return _timing_config


def reload_timing_config() -> None:
    """Reload the global timing configuration."""
    global _timing_config
    if _timing_config is not None:
        _timing_config.reload_config()
