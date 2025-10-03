"""Main CLI application for AlgoKit.

This module provides the main entry point for the AlgoKit CLI, including
command registration, global configuration management, error handling, and
the foundation for all algorithm family commands.
"""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install

from algokit import __version__

# from algokit.cli.commands import sarsa_app  # Removed SARSA implementation
from algokit.cli.models.config import Config, LogLevel
from algokit.cli.utils.logging import get_logger

# Install rich traceback for better error display
install(show_locals=True)

# Create console for rich output
console = Console()
logger = get_logger(__name__)

# Global configuration instance
_global_config: Config | None = None

# Create main Typer application
app = typer.Typer(
    name="algokit",
    help="AlgoKit CLI - Train, replay, and manage algorithms across all families",
    add_completion=True,  # Enable command completion
    rich_markup_mode="rich",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def get_global_config() -> Config:
    """Get the global configuration instance.

    Returns:
        Global configuration instance.
    """
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def load_config_file(config_path: str) -> Config:
    """Load configuration from a file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Loaded configuration instance.

    Raises:
        typer.Exit: If configuration file cannot be loaded.
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            console.print(f"[red]Configuration file not found: {config_path}[/red]")
            raise typer.Exit(1)

        if not config_file.is_file():
            console.print(f"[red]Configuration path is not a file: {config_path}[/red]")
            raise typer.Exit(1)

        config = Config.from_yaml_file(config_path)
        console.print(f"[green]Loaded configuration from: {config_path}[/green]")
        return config

    except Exception as e:
        console.print(f"[red]Failed to load configuration file: {e}[/red]")
        logger.error(f"Configuration loading error: {e}")
        raise typer.Exit(1) from e


def version_callback(value: bool) -> None:
    """Show version information and exit.

    Args:
        value: If True, show version and exit.
    """
    if value:
        version_info = Panel(
            f"[bold blue]AlgoKit CLI[/bold blue]\n"
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
    config_file: str | None = typer.Option(
        None,
        "--config-file",
        "-c",
        help="Path to configuration file.",
        exists=True,
        readable=True,
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
    """AlgoKit CLI - A comprehensive command-line interface for training, replaying, and managing algorithms.

    This CLI provides access to all algorithm families in the AlgoKit ecosystem:

    [bold blue]Available Algorithm Families:[/bold blue]
    • [bold]Reinforcement Learning (RL)[/bold] - Q-Learning, SARSA, DQN, Policy Gradient, Actor-Critic, PPO
    • [bold]Dynamic Movement Primitives (DMPs)[/bold] - Basic, Constrained, Hierarchical, and specialized variants
    • [bold]Control Systems[/bold] - PID, Adaptive, H∞, Robust, Sliding Mode Control
    • [bold]Model Predictive Control (MPC)[/bold] - Linear, Nonlinear, Robust, Distributed variants
    • [bold]Planning[/bold] - A*, RRT, PRM, D*, and advanced pathfinding algorithms
    • [bold]Gaussian Process[/bold] - Regression, Classification, Sparse, Multi-output variants
    • [bold]Hierarchical Reinforcement Learning[/bold] - Feudal Networks, Option-Critic, and hierarchical methods
    • [bold]Dynamic Programming[/bold] - Classic optimization algorithms and problem-solving methods
    • [bold]Real-time Control[/bold] - Real-time variants of control algorithms

    [bold green]Getting Started:[/bold green]
    • Use [bold]'algokit --help'[/bold] to see all available commands
    • Use [bold]'algokit list-families'[/bold] to see available algorithm families
    • Use [bold]'algokit list-algorithms'[/bold] to see available algorithms

    [bold yellow]Examples:[/bold yellow]
    • [bold]algokit list-families[/bold] - Show all algorithm families
    • [bold]algokit list-algorithms[/bold] - Show all available algorithms
    • [bold]algokit info[/bold] - Show system information
    """
    # Load global configuration
    global _global_config

    if config_file:
        try:
            _global_config = load_config_file(config_file)
            logger.info(f"Loaded configuration from: {config_file}")
        except Exception as e:
            console.print(f"[red]Failed to load configuration: {e}[/red]")
            raise typer.Exit(1) from e
    else:
        _global_config = get_global_config()
        logger.info("Using default configuration")

    # Set logging level based on verbosity options
    if verbose and quiet:
        console.print(
            "[yellow]Warning: Both --verbose and --quiet specified. Using verbose mode.[/yellow]"
        )
        quiet = False

    if verbose:
        _global_config.global_.log_level = LogLevel.DEBUG
        logger.info("Verbose mode enabled")
    elif quiet:
        _global_config.global_.log_level = LogLevel.WARNING
        logger.info("Quiet mode enabled")

    # Log CLI startup
    logger.info(f"AlgoKit CLI v{__version__} started")
    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"Platform: {sys.platform}")


def show_command_discovery() -> None:
    """Show available commands and families in a formatted table."""
    table = Table(
        title="Available Commands", show_header=True, header_style="bold blue"
    )
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Status", style="green")

    # No algorithms currently available
    table.add_row("list-families", "List all algorithm families", "✅ Available")
    table.add_row("list-algorithms", "List all available algorithms", "✅ Available")
    table.add_row("info", "Show system information", "✅ Available")
    table.add_row("status", "Show system status", "✅ Available")
    table.add_row("config", "Manage CLI configuration", "✅ Available")

    # Add placeholder commands for future algorithms
    table.add_row("q-learning train", "Train Q-Learning algorithm", "🚧 Coming Soon")
    table.add_row("dqn train", "Train Deep Q-Network", "🚧 Coming Soon")
    table.add_row("fibonacci", "Run Fibonacci algorithm", "🚧 Coming Soon")

    console.print(table)


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
        logger.info("Operation cancelled by user")
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

        # Log the full exception for debugging
        logger.error(
            f"Uncaught exception: {exc_type.__name__}: {exc_value}", exc_info=True
        )

        # Show helpful message
        console.print("\n[yellow]For help, try:[/yellow]")
        console.print("• [bold]algokit --help[/bold] - Show all available commands")
        console.print(
            "• [bold]algokit list-families[/bold] - Show available algorithm families"
        )
        console.print(
            "• [bold]algokit --verbose[/bold] - Enable verbose output for debugging"
        )

        sys.exit(1)


# Set up global exception handler
sys.excepthook = handle_exception

# ============================================================================
# COMMAND REGISTRATION
# ============================================================================

# Algorithm command groups will be added here as they are implemented
# app.add_typer(
#     sarsa_app,
#     name="sarsa",
#     help="SARSA (State-Action-Reward-State-Action) reinforcement learning algorithm commands",
# )

# Render command moved to viz-source - no longer part of main CLI

# ============================================================================
# GLOBAL COMMANDS
# ============================================================================


@app.command("list-families")
def list_families() -> None:
    """List all available algorithm families with their status and algorithm counts."""
    console.print("[bold blue]Algorithm Families[/bold blue]")
    console.print()

    families_table = Table(show_header=True, header_style="bold blue")
    families_table.add_column("Family", style="cyan", no_wrap=True)
    families_table.add_column("Description", style="white")
    families_table.add_column("Algorithms", style="green")
    families_table.add_column("Status", style="yellow")

    families_table.add_row(
        "Reinforcement Learning (RL)",
        "Q-Learning, DQN, Policy Gradient, Actor-Critic, PPO",
        "5",
        "📋 Planned",
    )
    families_table.add_row(
        "Dynamic Movement Primitives (DMPs)",
        "Basic, Constrained, Hierarchical, and specialized variants",
        "14",
        "📋 Planned",
    )
    families_table.add_row(
        "Control Systems",
        "PID, Adaptive, H∞, Robust, Sliding Mode Control",
        "5",
        "📋 Planned",
    )
    families_table.add_row(
        "Model Predictive Control (MPC)",
        "Linear, Nonlinear, Robust, Distributed variants",
        "8",
        "📋 Planned",
    )
    families_table.add_row(
        "Planning",
        "A*, RRT, PRM, D*, and advanced pathfinding algorithms",
        "8",
        "📋 Planned",
    )
    families_table.add_row(
        "Gaussian Process",
        "Regression, Classification, Sparse, Multi-output variants",
        "6",
        "📋 Planned",
    )
    families_table.add_row(
        "Hierarchical Reinforcement Learning",
        "Feudal Networks, Option-Critic, and hierarchical methods",
        "6",
        "📋 Planned",
    )
    families_table.add_row(
        "Dynamic Programming",
        "Classic optimization algorithms and problem-solving methods",
        "6",
        "📋 Planned",
    )
    families_table.add_row(
        "Real-time Control",
        "Real-time variants of control algorithms",
        "5",
        "📋 Planned",
    )

    console.print(families_table)
    console.print()
    console.print(
        "[yellow]Note:[/yellow] No algorithms are currently available for training."
    )
    console.print(
        "[bold green]Use 'algokit list-algorithms' to see planned algorithms.[/bold green]"
    )


@app.command("list-algorithms")
def list_algorithms() -> None:
    """List all available algorithms across all families."""
    console.print("[bold blue]Available Algorithms[/bold blue]")
    console.print()

    algorithms_table = Table(show_header=True, header_style="bold blue")
    algorithms_table.add_column("Algorithm", style="cyan", no_wrap=True)
    algorithms_table.add_column("Family", style="blue")
    algorithms_table.add_column("Description", style="white")
    algorithms_table.add_column("Status", style="green")

    # Currently available algorithms - none implemented yet

    # Coming soon algorithms
    algorithms_table.add_row(
        "Q-Learning", "RL", "Q-value based reinforcement learning", "🚧 Coming Soon"
    )
    algorithms_table.add_row(
        "DQN", "RL", "Deep Q-Network with neural networks", "🚧 Coming Soon"
    )
    algorithms_table.add_row(
        "Policy Gradient", "RL", "Direct policy optimization", "🚧 Coming Soon"
    )
    algorithms_table.add_row(
        "Actor-Critic", "RL", "Actor-Critic method with dual networks", "🚧 Coming Soon"
    )
    algorithms_table.add_row(
        "PPO", "RL", "Proximal Policy Optimization", "🚧 Coming Soon"
    )

    console.print(algorithms_table)
    console.print()
    console.print(
        "[bold green]Use 'algokit list-families' to see available algorithm families.[/bold green]"
    )


@app.command()
def info() -> None:
    """Show detailed information about algorithms or families."""
    console.print("[bold blue]AlgoKit CLI Information[/bold blue]")
    console.print()

    info_panel = Panel(
        f"[bold]Version:[/bold] {__version__}\n"
        f"[bold]Python:[/bold] {sys.version.split()[0]}\n"
        f"[bold]Platform:[/bold] {sys.platform}\n"
        f"[bold]Available Algorithms:[/bold] 0\n"
        f"[bold]Total Algorithm Families:[/bold] 9\n"
        f"[bold]Total Algorithms Planned:[/bold] 64+\n\n"
        f"[bold green]Currently Available:[/bold green]\n"
        f"• [bold]None[/bold] - No algorithms currently implemented\n\n"
        f"[bold yellow]Coming Soon:[/bold yellow]\n"
        f"• Complete RL family (Q-Learning, DQN, Policy Gradient, Actor-Critic, PPO)\n"
        f"• DMPs family (14 algorithms)\n"
        f"• Control Systems family (5 algorithms)\n"
        f"• And 6 more families...",
        title="System Information",
        border_style="blue",
    )
    console.print(info_panel)


@app.command()
def status() -> None:
    """Show system status and configuration."""
    config = get_global_config()

    console.print("[bold blue]System Status[/bold blue]")
    console.print()

    # System information
    status_table = Table(show_header=True, header_style="bold blue")
    status_table.add_column("Component", style="cyan", no_wrap=True)
    status_table.add_column("Status", style="green")
    status_table.add_column("Details", style="white")

    status_table.add_row("CLI Application", "✅ Running", f"Version {__version__}")
    status_table.add_row("Configuration", "✅ Loaded", "Default configuration active")
    status_table.add_row(
        "Algorithm Implementations", "❌ None", "No algorithms currently implemented"
    )
    status_table.add_row("Output Directory", "✅ Ready", str(config.global_.output_dir))
    status_table.add_row("Logging", "✅ Active", f"Level: {config.global_.log_level}")
    status_table.add_row(
        "Python Environment", "✅ Compatible", f"Python {sys.version.split()[0]}"
    )

    console.print(status_table)
    console.print()
    console.print(
        "[bold yellow]System is ready but no algorithms are currently implemented.[/bold yellow]"
    )


@app.command()
def config() -> None:
    """Manage CLI configuration."""
    console.print("[bold blue]Configuration Management[/bold blue]")
    console.print()

    config_panel = Panel(
        "[bold yellow]Configuration commands will be implemented in future tasks.[/bold yellow]\n\n"
        "[bold]Planned features:[/bold]\n"
        "• [bold]config show[/bold] - Display current configuration\n"
        "• [bold]config set <key> <value>[/bold] - Set configuration values\n"
        "• [bold]config get <key>[/bold] - Get configuration values\n"
        "• [bold]config reset[/bold] - Reset to defaults\n"
        "• [bold]config export[/bold] - Export configuration to file\n"
        "• [bold]config import[/bold] - Import configuration from file\n\n"
        "[bold green]Current configuration:[/bold green]\n"
        "• Using default configuration\n"
        "• Output directory: output/\n"
        "• Log level: info",
        title="Configuration Management",
        border_style="yellow",
    )
    console.print(config_panel)


if __name__ == "__main__":
    app()
