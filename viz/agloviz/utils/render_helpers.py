"""Rendering helper functions for AGLoViz CLI.

This module provides utility functions for rendering algorithm visualizations
using Manim with different output formats and quality settings.
"""

import subprocess
import sys
from pathlib import Path
from typing import Type, Any, Optional

from rich.console import Console

console = Console()


def render_scene(
    scene_class: Type[Any],
    scenario_file: Path,
    output_format: str = "mp4",
    quality: str = "medium",
    output_file: Optional[str] = None,
) -> None:
    """Render a scene using Manim.
    
    Args:
        scene_class: Manim scene class to render
        scenario_file: Path to scenario file
        output_format: Output format (mp4, gif, images)
        quality: Quality setting (low, medium, high)
        output_file: Output file path (optional)
    """
    try:
        # Get Manim command arguments
        cmd_args = _build_manim_command(
            scene_class, scenario_file, output_format, quality, output_file
        )
        
        # Run Manim command with environment variable for scenario file
        console.print(f"[dim]Running: {' '.join(cmd_args)}[/dim]")
        
        # Set environment variable for scenario file
        import os
        env = os.environ.copy()
        env['AGLOVIZ_SCENARIO_FILE'] = str(scenario_file)
        
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            check=True,
            env=env
        )
        
        if result.stdout:
            console.print(f"[dim]Manim output: {result.stdout}[/dim]")
        
        if result.stderr:
            console.print(f"[yellow]Manim warnings: {result.stderr}[/yellow]")
            
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Manim rendering failed: {e}[/red]")
        if e.stdout:
            console.print(f"[dim]STDOUT: {e.stdout}[/dim]")
        if e.stderr:
            console.print(f"[dim]STDERR: {e.stderr}[/dim]")
        raise
    except Exception as e:
        console.print(f"[red]Error during rendering: {e}[/red]")
        raise


def _build_manim_command(
    scene_class: Type[Any],
    scenario_file: Path,
    output_format: str,
    quality: str,
    output_file: Optional[str],
) -> list[str]:
    """Build Manim command arguments.
    
    Args:
        scene_class: Manim scene class to render
        scenario_file: Path to scenario file
        output_format: Output format
        quality: Quality setting
        output_file: Output file path
        
    Returns:
        List of command arguments
    """
    cmd = ["manim", "render"]
    
    # Add quality flags
    quality_flags = _get_quality_flags(quality)
    cmd.extend(quality_flags)
    
    # Add format flags
    format_flags = _get_format_flags(output_format)
    cmd.extend(format_flags)
    
    # Add output file if specified
    if output_file:
        cmd.extend(["-o", output_file])
    
    # Add scene class file path - get the actual file path from the module
    import inspect
    scene_file = inspect.getfile(scene_class)
    cmd.append(scene_file)
    
    # Add scene class name
    cmd.append(scene_class.__name__)
    
    return cmd


def _get_quality_flags(quality: str) -> list[str]:
    """Get Manim quality flags for the specified quality level.
    
    Args:
        quality: Quality level (low, medium, high)
        
    Returns:
        List of quality flags
    """
    quality_map = {
        "low": ["-ql"],      # low quality
        "medium": ["-qm"],   # medium quality  
        "high": ["-qh"],     # high quality
    }
    
    flags = quality_map.get(quality.lower(), quality_map["medium"])
    
    # Add additional quality-specific settings
    if quality.lower() == "low":
        flags.append("--preview")  # Auto-open preview
    elif quality.lower() == "high":
        flags.append("--disable_caching")  # Disable caching for high quality
    
    return flags


def _get_format_flags(output_format: str) -> list[str]:
    """Get Manim format flags for the specified output format.
    
    Args:
        output_format: Output format (mp4, gif, images)
        
    Returns:
        List of format flags
    """
    format_map = {
        "mp4": [],
        "gif": ["--format", "gif"],
        "images": ["--format", "png"],
    }
    
    return format_map.get(output_format.lower(), [])


def validate_manim_installation() -> bool:
    """Validate that Manim is properly installed.
    
    Returns:
        True if Manim is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["manim", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        console.print(f"[green]Manim version: {result.stdout.strip()}[/green]")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("[red]Error: Manim is not installed or not available in PATH[/red]")
        console.print("[yellow]Please install Manim: pip install manim[/yellow]")
        return False


def get_supported_formats() -> list[str]:
    """Get list of supported output formats.
    
    Returns:
        List of supported output formats
    """
    return ["mp4", "gif", "images"]


def get_supported_qualities() -> list[str]:
    """Get list of supported quality levels.
    
    Returns:
        List of supported quality levels
    """
    return ["low", "medium", "high"]


def estimate_render_time(
    scene_class: Type[Any],
    quality: str,
    output_format: str,
) -> str:
    """Estimate render time for a scene.
    
    Args:
        scene_class: Scene class to estimate
        quality: Quality level
        output_format: Output format
        
    Returns:
        Estimated render time as string
    """
    # Base time estimates (very rough)
    base_times = {
        "low": 30,      # 30 seconds
        "medium": 120,  # 2 minutes  
        "high": 300,    # 5 minutes
    }
    
    base_time = base_times.get(quality.lower(), base_times["medium"])
    
    # Adjust for format
    format_multipliers = {
        "mp4": 1.0,
        "gif": 1.5,     # GIFs take longer
        "images": 2.0,  # Images take much longer
    }
    
    multiplier = format_multipliers.get(output_format.lower(), 1.0)
    estimated_seconds = int(base_time * multiplier)
    
    if estimated_seconds < 60:
        return f"{estimated_seconds} seconds"
    elif estimated_seconds < 3600:
        minutes = estimated_seconds // 60
        return f"{minutes} minutes"
    else:
        hours = estimated_seconds // 3600
        return f"{hours} hours"
