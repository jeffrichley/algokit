"""Configuration utilities for AGLoViz CLI.

This module provides configuration functions and utilities for the AGLoViz CLI.
"""

from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


def get_scenarios_directory() -> Path:
    """Get the default scenarios directory.
    
    Returns:
        Path to the scenarios directory
    """
    # Look for scenarios in the data/examples/scenarios directory
    # relative to the project root
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    scenarios_dir = project_root / "data" / "examples" / "scenarios"
    
    return scenarios_dir


def validate_scenario_file(scenario_file: str) -> Path:
    """Validate and return a scenario file path.
    
    Args:
        scenario_file: Path to scenario file
        
    Returns:
        Validated Path object
        
    Raises:
        FileNotFoundError: If scenario file doesn't exist
        ValueError: If scenario file is invalid
    """
    scenario_path = Path(scenario_file)
    
    # Check if file exists
    if not scenario_path.exists():
        # Try relative to scenarios directory
        scenarios_dir = get_scenarios_directory()
        alt_path = scenarios_dir / scenario_path.name
        
        if alt_path.exists():
            scenario_path = alt_path
        else:
            raise FileNotFoundError(f"Scenario file not found: {scenario_file}")
    
    # Check if it's a file
    if not scenario_path.is_file():
        raise ValueError(f"Path is not a file: {scenario_file}")
    
    # Check file extension
    if scenario_path.suffix.lower() not in ['.yaml', '.yml']:
        raise ValueError(f"Scenario file must be YAML (.yaml or .yml): {scenario_file}")
    
    return scenario_path


def get_output_path(
    algorithm: str,
    scenario_file: Path,
    output_format: str,
    output_dir: Optional[Path] = None,
) -> str:
    """Generate output file path for rendered visualization.
    
    Args:
        algorithm: Algorithm name
        scenario_file: Scenario file path
        output_format: Output format (mp4, gif, images)
        output_dir: Output directory (optional)
        
    Returns:
        Generated output file path
    """
    # Determine output directory
    if output_dir is None:
        output_dir = Path.cwd() / "output"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    scenario_name = scenario_file.stem
    timestamp = _get_timestamp()
    
    if output_format == "images":
        # For images, create a directory
        output_path = output_dir / f"{algorithm}_{scenario_name}_{timestamp}"
        output_path.mkdir(exist_ok=True)
        return str(output_path)
    else:
        # For single files
        extension = _get_file_extension(output_format)
        filename = f"{algorithm}_{scenario_name}_{timestamp}.{extension}"
        return str(output_dir / filename)


def _get_timestamp() -> str:
    """Get a timestamp string for unique filenames.
    
    Returns:
        Timestamp string in YYYYMMDD_HHMMSS format
    """
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _get_file_extension(output_format: str) -> str:
    """Get file extension for output format.
    
    Args:
        output_format: Output format
        
    Returns:
        File extension
    """
    extension_map = {
        "mp4": "mp4",
        "gif": "gif",
        "images": "png",
    }
    
    return extension_map.get(output_format.lower(), "mp4")


def get_default_quality() -> str:
    """Get default quality setting.
    
    Returns:
        Default quality level
    """
    return "medium"


def get_default_format() -> str:
    """Get default output format.
    
    Returns:
        Default output format
    """
    return "mp4"


def validate_quality(quality: str) -> bool:
    """Validate quality setting.
    
    Args:
        quality: Quality level to validate
        
    Returns:
        True if quality is valid
    """
    valid_qualities = ["low", "medium", "high"]
    return quality.lower() in valid_qualities


def validate_output_format(output_format: str) -> bool:
    """Validate output format.
    
    Args:
        output_format: Output format to validate
        
    Returns:
        True if format is valid
    """
    valid_formats = ["mp4", "gif", "images"]
    return output_format.lower() in valid_formats


def get_quality_description(quality: str) -> str:
    """Get description for quality level.
    
    Args:
        quality: Quality level
        
    Returns:
        Description of quality level
    """
    descriptions = {
        "low": "Fast rendering, lower resolution (480p)",
        "medium": "Balanced rendering, medium resolution (720p)",
        "high": "Slow rendering, high resolution (1080p)",
    }
    
    return descriptions.get(quality.lower(), "Unknown quality level")


def get_format_description(output_format: str) -> str:
    """Get description for output format.
    
    Args:
        output_format: Output format
        
    Returns:
        Description of output format
    """
    descriptions = {
        "mp4": "MP4 video file (recommended)",
        "gif": "Animated GIF file",
        "images": "PNG image sequence",
    }
    
    return descriptions.get(output_format.lower(), "Unknown output format")


def create_output_directory(output_path: str) -> Path:
    """Create output directory if it doesn't exist.
    
    Args:
        output_path: Output file path
        
    Returns:
        Path to output directory
    """
    output_file = Path(output_path)
    output_dir = output_file.parent
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def get_config_file_path() -> Path:
    """Get path to AGLoViz configuration file.
    
    Returns:
        Path to configuration file
    """
    # Use user's home directory for config
    home_dir = Path.home()
    config_dir = home_dir / ".agloviz"
    config_dir.mkdir(exist_ok=True)
    
    return config_dir / "config.yaml"


def load_config() -> dict:
    """Load AGLoViz configuration.
    
    Returns:
        Configuration dictionary
    """
    config_file = get_config_file_path()
    
    if not config_file.exists():
        return _get_default_config()
    
    try:
        import yaml
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load config file: {e}[/yellow]")
        return _get_default_config()


def save_config(config: dict) -> None:
    """Save AGLoViz configuration.
    
    Args:
        config: Configuration dictionary to save
    """
    config_file = get_config_file_path()
    
    try:
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        console.print(f"[green]Configuration saved to: {config_file}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving config: {e}[/red]")


def _get_default_config() -> dict:
    """Get default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "default_quality": "medium",
        "default_format": "mp4",
        "output_directory": str(Path.cwd() / "output"),
        "auto_open": False,
        "verbose": False,
        "scenarios_directory": str(get_scenarios_directory()),
    }
