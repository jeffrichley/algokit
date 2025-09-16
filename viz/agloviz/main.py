"""Main AGLoViz CLI application.

This module provides the main entry point for the AGLoViz CLI, including
command registration, global configuration management, and error handling.
"""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.traceback import install

from agloviz import __version__

# Install rich traceback for better error display
install(show_locals=True)

# Create console for rich output
console = Console()

# Create main Typer application
app = typer.Typer(
    name="agloviz",
    help="AGLoViz - Algorithm Visualization CLI",
    add_completion=True,  # Enable command completion
    rich_markup_mode="rich",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def version_callback(value: bool) -> None:
    """Show version information and exit.

    Args:
        value: If True, show version and exit.
    """
    if value:
        version_info = Panel(
            f"[bold blue]AGLoViz CLI[/bold blue]\n"
            f"Version: [bold]{__version__}[/bold]\n"
            f"Python: {sys.version.split()[0]}\n"
            f"Platform: {sys.platform}",
            title="Version Information",
            border_style="blue",
        )
        console.print(version_info)
        raise typer.Exit()


@app.callback()
def main(
    _version: bool | None = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version information and exit.",
        callback=version_callback,
        is_eager=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Enable verbose output.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress non-essential output.",
    ),
) -> None:
    """AGLoViz - A standalone CLI tool for rendering algorithm visualizations.

    This CLI provides access to all algorithm visualizations using Manim:

    [bold blue]Available Commands:[/bold blue]
    • [bold]render[/bold] - Render algorithm visualizations (MP4, GIF, images)
    • [bold]list[/bold] - List available algorithms and scenarios
    • [bold]validate[/bold] - Validate scenario files

    [bold green]Getting Started:[/bold green]
    • Use [bold]'agloviz --help'[/bold] to see all available commands
    • Use [bold]'agloviz list algorithms'[/bold] to see available algorithms
    • Use [bold]'agloviz render bfs --scenario path/to/scenario.yaml'[/bold] to render

    [bold yellow]Examples:[/bold yellow]
    • [bold]agloviz render bfs --scenario harbor_storm.yaml[/bold] - Render BFS visualization
    • [bold]agloviz render dfs --format gif --quality high[/bold] - Render DFS as high-quality GIF
    • [bold]agloviz list algorithms[/bold] - Show all available algorithms
    • [bold]agloviz list scenarios[/bold] - Show available HarborNet scenarios
    • [bold]agloviz validate my_scenario.yaml[/bold] - Validate scenario file
    """
    # Set logging level based on verbosity options
    if verbose and quiet:
        console.print(
            "[yellow]Warning: Both --verbose and --quiet specified. Using verbose mode.[/yellow]"
        )
        quiet = False

    if verbose:
        console.print("[blue]Verbose mode enabled[/blue]")
    elif quiet:
        console.print("[dim]Quiet mode enabled[/dim]")


def handle_exception(
    exc_type: type[BaseException], exc_value: BaseException, _exc_traceback: object
) -> None:
    """Handle uncaught exceptions with rich formatting.

    Args:
        exc_type: Exception type.
        exc_value: Exception value.
        exc_traceback: Exception traceback.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(1)
    elif issubclass(exc_type, typer.Exit):
        # Let typer handle its own exits
        raise exc_value
    else:
        error_panel = Panel(
            f"[red]An unexpected error occurred:[/red]\n"
            f"[bold red]{exc_type.__name__}:[/bold red] {exc_value}",
            title="Error",
            border_style="red",
        )
        console.print(error_panel)

        # Show helpful message
        console.print("\n[yellow]For help, try:[/yellow]")
        console.print("• [bold]agloviz --help[/bold] - Show all available commands")
        console.print("• [bold]agloviz list algorithms[/bold] - Show available algorithms")
        console.print(
            "• [bold]agloviz --verbose[/bold] - Enable verbose output for debugging"
        )

        sys.exit(1)


# Set up global exception handler
sys.excepthook = handle_exception

# ============================================================================
# COMMAND REGISTRATION
# ============================================================================

# Import and register command modules
try:
    from agloviz.commands import render
    from agloviz.commands import list as list_cmds
    from agloviz.commands import validate

    # Register command groups
    app.add_typer(
        render.app,
        name="render",
        help="Render algorithm visualizations (MP4, GIF, images)",
    )
    app.add_typer(
        list_cmds.app,
        name="list",
        help="List available algorithms and scenarios",
    )
    app.add_typer(
        validate.app,
        name="validate",
        help="Validate scenario files",
    )

except ImportError as e:
    console.print(f"[red]Failed to import command modules: {e}[/red]")
    console.print("[yellow]Some commands may not be available.[/yellow]")


if __name__ == "__main__":
    app()
