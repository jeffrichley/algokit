"""Render command for AGLoViz CLI.

This module provides commands for rendering algorithm visualizations.
"""

import typer
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from agloviz.utils.scene_registry import SceneRegistry
from agloviz.utils.render_helpers import render_scene
from agloviz.utils.config import get_output_path, validate_scenario_file

console = Console()

app = typer.Typer(name="render")


@app.command("bfs")
def render_bfs(
    scenario_file: str = typer.Option(
        ..., "--scenario", "-s", help="Path to HarborNet scenario file"
    ),
    output_format: str = typer.Option(
        "mp4", "--format", "-f", help="Output format: mp4, gif, images"
    ),
    quality: str = typer.Option(
        "medium", "--quality", "-q", help="Quality: low, medium, high"
    ),
    output_file: str = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """Render BFS algorithm visualization."""
    try:
        # Validate scenario file
        scenario_path = validate_scenario_file(scenario_file)
        
        # Get output path
        if not output_file:
            output_file = get_output_path("bfs", scenario_path, output_format)
        
        # Show render info
        info_panel = Panel(
            f"[bold blue]Rendering BFS Visualization[/bold blue]\n"
            f"Scenario: [cyan]{scenario_path.name}[/cyan]\n"
            f"Format: [green]{output_format}[/green]\n"
            f"Quality: [yellow]{quality}[/yellow]\n"
            f"Output: [magenta]{output_file}[/magenta]",
            title="Render Configuration",
            border_style="blue",
        )
        console.print(info_panel)
        
        # Get scene class and render
        registry = SceneRegistry()
        scene_class = registry.get_scene_class("bfs")
        
        if not scene_class:
            console.print("[red]Error: BFS scene class not found.[/red]")
            raise typer.Exit(1)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Rendering BFS visualization...", total=None)
            render_scene(scene_class, scenario_path, output_format, quality, output_file)
            progress.update(task, description="[green]Rendering complete![/green]")
        
        console.print(f"[bold green]✓ BFS visualization rendered to: {output_file}[/bold green]")
        
    except Exception as e:
        console.print(f"[red]Error rendering BFS visualization: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from e




@app.command("algorithm")
def render_algorithm(
    algorithm: str = typer.Argument(..., help="Algorithm name (bfs, dfs, astar, etc.)"),
    scenario_file: str = typer.Option(
        ..., "--scenario", "-s", help="Path to HarborNet scenario file"
    ),
    output_format: str = typer.Option(
        "mp4", "--format", "-f", help="Output format: mp4, gif, images"
    ),
    quality: str = typer.Option(
        "medium", "--quality", "-q", help="Quality: low, medium, high"
    ),
    output_file: str = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """Render any algorithm visualization."""
    try:
        # Validate scenario file
        scenario_path = validate_scenario_file(scenario_file)
        
        # Get output path
        if not output_file:
            output_file = get_output_path(algorithm, scenario_path, output_format)
        
        # Show render info
        info_panel = Panel(
            f"[bold blue]Rendering {algorithm.upper()} Visualization[/bold blue]\n"
            f"Scenario: [cyan]{scenario_path.name}[/cyan]\n"
            f"Format: [green]{output_format}[/green]\n"
            f"Quality: [yellow]{quality}[/yellow]\n"
            f"Output: [magenta]{output_file}[/magenta]",
            title="Render Configuration",
            border_style="blue",
        )
        console.print(info_panel)
        
        # Get scene class and render
        registry = SceneRegistry()
        scene_class = registry.get_scene_class(algorithm)
        
        if not scene_class:
            console.print(f"[red]Error: Algorithm '{algorithm}' not found.[/red]")
            console.print("[yellow]Use 'agloviz list algorithms' to see available algorithms.[/yellow]")
            raise typer.Exit(1)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Rendering {algorithm} visualization...", total=None)
            render_scene(scene_class, scenario_path, output_format, quality, output_file)
            progress.update(task, description="[green]Rendering complete![/green]")
        
        console.print(f"[bold green]✓ {algorithm.upper()} visualization rendered to: {output_file}[/bold green]")
        
    except Exception as e:
        console.print(f"[red]Error rendering {algorithm} visualization: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from e
