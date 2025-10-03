"""CLI for rendering algorithm visualizations."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Protocol

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project to path for imports
# We're running from project root, so src/ should be available
project_root = Path.cwd()  # Current working directory (project root)
sys.path.insert(0, str(project_root / "src"))

try:
    from algokit.core.helpers.graph_utils import load_harbor_scenario
except ImportError as e:
    print(f"Warning: Could not import visualization modules: {e}")
    print("Make sure you're running from the project root directory.")


# ---------- Types ----------
class Renderer(Protocol):
    """Protocol for algorithm renderers."""

    def __call__(
        self, *, scenario: str | None, quality: str, timing: str, output: str | None
    ) -> None:
        """Render an algorithm visualization."""
        ...


# ---------- Typer App ----------
app = typer.Typer(
    name="render",
    help="🎬 Render algorithm visualizations with Manim",
    rich_markup_mode="rich",
)
console = Console()

# ---------- Algorithm Registry ----------
ALGORITHMS: dict[str, Renderer] = {}


def register_algorithm(name: str) -> Callable[[Renderer], Renderer]:
    """Register an algorithm renderer.

    Args:
        name: Algorithm name (e.g., 'bfs', 'astar')

    Returns:
        Decorator function
    """
    name = name.strip().lower()

    def _decorator(fn: Renderer) -> Renderer:
        ALGORITHMS[name] = fn
        return fn

    return _decorator


def list_algorithms() -> str:
    """List all registered algorithms."""
    return ", ".join(sorted(ALGORITHMS.keys())) or "(none registered)"


def dispatch_algorithm(
    algorithm: str, scenario: str | None, quality: str, timing: str, output: str | None
) -> None:
    """Dispatch to the appropriate algorithm renderer.

    Args:
        algorithm: Algorithm name
        scenario: Scenario file path or None
        quality: Video quality setting
        timing: Animation timing mode
        output: Output directory or None

    Raises:
        typer.Exit: If algorithm is unknown
    """
    key = (algorithm or "").strip().lower()
    renderer = ALGORITHMS.get(key)
    if not renderer:
        available = list_algorithms()
        console.print(
            f"❌ Unknown algorithm '{algorithm}'. Available: {available}", style="red"
        )
        raise typer.Exit(code=2)
    renderer(scenario=scenario, quality=quality, timing=timing, output=output)


# ---------- Presets ----------
PRESETS: dict[str, dict[str, str]] = {
    "none": {},
    "quick": {"quality": "low", "timing": "development"},
    "demo": {"quality": "high", "timing": "cinematic"},
}


def apply_preset(
    preset: str, quality: str | None, timing: str | None
) -> tuple[str, str]:
    """Apply preset and allow CLI flags to override.

    Args:
        preset: Preset name (none, quick, demo)
        quality: CLI quality flag or None
        timing: CLI timing flag or None

    Returns:
        Tuple of (resolved_quality, resolved_timing)
    """
    base = PRESETS.get((preset or "none").lower(), {})
    # CLI flags override preset-derived defaults
    resolved_quality = quality or base.get("quality", "medium")
    resolved_timing = timing or base.get("timing", "cinematic")
    return resolved_quality, resolved_timing


def resolve_scenario(scenario: str | None) -> str | None:
    """Resolve scenario from CLI arg or environment variable.

    Args:
        scenario: CLI scenario argument or None

    Returns:
        Resolved scenario path or None
    """
    return scenario or os.environ.get("AGLOVIZ_SCENARIO_FILE")


def list_scenarios() -> list[tuple[str, Path]]:
    """List all available scenario files."""
    scenarios_dir = project_root / "data" / "examples" / "scenarios"
    scenarios = []

    if scenarios_dir.exists():
        for scenario_file in scenarios_dir.glob("*.yaml"):
            # Skip config files and documentation
            if scenario_file.name not in ["bfs_timing_config.yaml"]:
                try:
                    scenario = load_harbor_scenario(str(scenario_file))
                    scenarios.append((scenario.name, scenario_file))
                except Exception:
                    # If loading fails, just use the filename
                    scenarios.append((scenario_file.stem, scenario_file))

    return scenarios


@app.command()
def scenarios(
    show_details: Annotated[
        bool, typer.Option("--details", "-d", help="Show detailed scenario information")
    ] = False,
) -> None:
    """📋 List all available scenarios."""
    console.print(Panel("Available Scenarios", style="bold blue"))

    scenarios_list = list_scenarios()

    if not scenarios_list:
        console.print("❌ No scenarios found in data/examples/scenarios/", style="red")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("File", style="green")

    if show_details:
        table.add_column("Description", style="yellow")
        table.add_column("Grid Size", style="blue")

    for name, file_path in scenarios_list:
        if show_details:
            try:
                scenario = load_harbor_scenario(str(file_path))
                description = scenario.description or "No description"
                grid_size = f"{scenario.width}x{scenario.height}"
                table.add_row(name, file_path.name, description, grid_size)
            except Exception:
                table.add_row(name, file_path.name, "Error loading details", "Unknown")
        else:
            table.add_row(name, file_path.name)

    console.print(table)

    console.print(
        "\n💡 Use [bold cyan]--details[/bold cyan] to see more information about each scenario"
    )


@app.command()
def algorithms() -> None:
    """🧮 List all available algorithms."""
    console.print(Panel("Available Algorithms", style="bold blue"))

    algorithms_list = list_algorithms()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Algorithm", style="cyan", no_wrap=True)
    table.add_column("Description", style="yellow")

    algorithm_descriptions = {
        "bfs": "Breadth-First Search pathfinding visualization",
        "breadth-first-search": "Full BFS algorithm with depth rings and path reconstruction",
    }

    for alg in algorithms_list:
        description = algorithm_descriptions.get(alg, "Algorithm visualization")
        table.add_row(alg, description)

    console.print(table)


# ---------- BFS Command (Direct) ----------
@app.command(name="bfs")
def render_bfs_command(
    preset: Annotated[
        str,
        typer.Option(
            "--preset",
            help="Preset: none, quick (low+development), demo (high+cinematic)",
        ),
    ] = "none",
    scenario: Annotated[
        str | None,
        typer.Option(
            "--scenario",
            "-s",
            help="Scenario name or file path (or env AGLOVIZ_SCENARIO_FILE)",
        ),
    ] = None,
    quality: Annotated[
        str | None,
        typer.Option("--quality", "-q", help="Video quality: low, medium, high, ultra"),
    ] = None,
    timing: Annotated[
        str | None,
        typer.Option(
            "--timing",
            "-t",
            help="Animation timing: cinematic, development, quick_demo",
        ),
    ] = None,
    max_events: Annotated[
        int | None,
        typer.Option(
            "--max-events", "-e", help="Maximum BFS events to display (default: all)"
        ),
    ] = None,
    output: Annotated[
        str | None, typer.Option("--output", "-o", help="Output directory for videos")
    ] = None,
    preview: Annotated[
        bool,
        typer.Option("--preview", "-p", help="Preview mode (low quality, fast timing)"),
    ] = False,
    show_progress: Annotated[
        bool, typer.Option("--progress", help="Show rendering progress")
    ] = True,
) -> None:
    """🎬 Render BFS (Breadth-First Search) algorithm visualization.

    Examples:

    # Render BFS with default scenario
    algokit render bfs

    # Render BFS with specific scenario
    algokit render bfs --scenario "Tiny Test"

    # Quick preset (low quality, fast timing)
    algokit render bfs --preset quick

    # Demo preset (high quality, cinematic timing)
    algokit render bfs --preset demo

    # Custom quality and timing (overrides preset)
    algokit render bfs --preset demo --timing development --quality low

    # Using environment variable for scenario
    AGLOVIZ_SCENARIO_FILE=data/examples/tiny.yaml algokit render bfs
    """
    # Handle preview mode (shortcut for quick preset)
    if preview:
        preset = "quick"

    # Apply preset and allow CLI flags to override
    resolved_quality, resolved_timing = apply_preset(preset, quality, timing)
    resolved_scenario = resolve_scenario(scenario)

    # Call BFS renderer directly
    render_bfs(
        scenario=resolved_scenario,
        quality=resolved_quality,
        timing=resolved_timing,
        output=output,
    )


# ---------- Algorithm Renderers ----------


def _render_bfs_implementation(
    *, scenario: str | None, quality: str, timing: str, output: str | None
) -> None:
    """Implementation of BFS rendering using Manim."""
    console.print("🎬 Starting BFS render...", style="bold green")

    # Handle scenario selection
    scenario_file = None
    if scenario:
        # Check if it's a file path
        if Path(scenario).exists():
            scenario_file = Path(scenario)
        else:
            # Look for scenario by name
            scenarios_list = list_scenarios()
            for name, file_path in scenarios_list:
                if name.lower() == scenario.lower():
                    scenario_file = file_path
                    break

            if not scenario_file:
                console.print(f"❌ Scenario not found: {scenario}", style="red")
                console.print(
                    "Use 'algokit render scenarios' to see available scenarios"
                )
                raise typer.Exit(1)
    else:
        console.print("📋 Using default scenario", style="yellow")

    # Set up output directory
    if output:
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        os.environ["MANIM_OUTPUT_DIR"] = str(output_path)

    # Set up environment for scenario and timing
    if scenario_file:
        os.environ["AGLOVIZ_SCENARIO_FILE"] = str(scenario_file)
        console.print(f"📁 Using scenario: {scenario_file.name}", style="green")

    # Set timing mode environment variable
    os.environ["AGLOVIZ_TIMING_MODE"] = timing

    # Quality settings
    quality_settings = {
        "low": "-pql",  # Preview, low quality, low fps
        "medium": "-pqm",  # Preview, medium quality, medium fps
        "high": "-pqh",  # Preview, high quality, high fps
        "ultra": "-pqk",  # Preview, 4K quality, high fps
    }

    if quality not in quality_settings:
        console.print(f"❌ Invalid quality: {quality}", style="red")
        console.print(f"Available qualities: {', '.join(quality_settings.keys())}")
        raise typer.Exit(1)

    # Show configuration
    console.print("\n📋 BFS Render Configuration:", style="bold blue")
    config_table = Table(show_header=False, box=None)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")

    config_table.add_row("Algorithm", "BFS")
    config_table.add_row("Quality", quality)
    config_table.add_row("Timing", timing)
    if scenario_file:
        config_table.add_row("Scenario", scenario_file.name)
    if output:
        config_table.add_row("Output", str(output))

    console.print(config_table)
    console.print()

    # Build manim command
    scene_file = "viz-source/agloviz/scenes/planning/bfs/render_bfs_scene.py"
    manim_args = [
        "uv",
        "run",
        "manim",
        quality_settings[quality],
        scene_file,
        "RenderBfsScene",
    ]

    console.print(f"🎬 Running: {' '.join(manim_args)}", style="green")
    console.print("🎯 Starting render...\n", style="bold green")

    # Execute manim command
    try:
        import subprocess

        result = subprocess.run(manim_args, check=True, cwd=project_root)

        if result.returncode == 0:
            console.print("✅ BFS render completed successfully!", style="bold green")

            # Show output location
            media_dir = project_root / "media"
            if media_dir.exists():
                console.print(f"📁 Output saved to: {media_dir}", style="blue")
        else:
            console.print("❌ BFS render failed", style="bold red")
            raise typer.Exit(result.returncode)

    except subprocess.CalledProcessError as e:
        console.print(
            f"❌ BFS render failed with exit code {e.returncode}", style="bold red"
        )
        raise typer.Exit(e.returncode) from e
    except FileNotFoundError as e:
        console.print(
            "❌ Manim not found. Install with: pip install manim", style="bold red"
        )
        raise typer.Exit(1) from e


# Register BFS algorithm
@register_algorithm("bfs")
@register_algorithm("breadth-first-search")  # Alternative name
def render_bfs(
    *, scenario: str | None, quality: str, timing: str, output: str | None
) -> None:
    """Render BFS (Breadth-First Search) algorithm visualization."""
    _render_bfs_implementation(
        scenario=scenario, quality=quality, timing=timing, output=output
    )


# ---------- Updated quick and demo commands ----------
@app.command()
def quick(
    algorithm: Annotated[
        str | None, typer.Argument(help="Algorithm to render")
    ] = "bfs",
    scenario: Annotated[
        str | None, typer.Option("--scenario", "-s", help="Scenario name")
    ] = None,
) -> None:
    """⚡ Quick render for testing (low quality, fast timing)."""
    console.print("⚡ Quick render mode", style="bold yellow")

    # Call BFS renderer with quick preset
    if algorithm == "bfs":
        render_bfs_command(
            preset="quick",
            scenario=scenario,
            quality=None,  # Will be set by preset
            timing=None,  # Will be set by preset
            max_events=None,
            output=None,
            preview=False,
            show_progress=False,
        )
    else:
        console.print(
            f"❌ Quick render only supports BFS currently. Requested: {algorithm}",
            style="red",
        )
        raise typer.Exit(1)


@app.command()
def demo(
    algorithm: Annotated[
        str | None, typer.Argument(help="Algorithm to render")
    ] = "bfs",
) -> None:
    """🎭 Render a demo video (high quality, cinematic timing)."""
    console.print("🎭 Demo render mode", style="bold blue")

    # Call BFS renderer with demo preset
    if algorithm == "bfs":
        render_bfs_command(
            preset="demo",
            scenario=None,  # Use default
            quality=None,  # Will be set by preset
            timing=None,  # Will be set by preset
            max_events=None,
            output=None,
            preview=False,
            show_progress=True,
        )
    else:
        console.print(
            f"❌ Demo render only supports BFS currently. Requested: {algorithm}",
            style="red",
        )
        raise typer.Exit(1)


# ---------- Legacy alias for backward compatibility ----------
@app.command(name="render", hidden=True)
def _legacy_render_alias(
    algorithm: Annotated[str, typer.Argument(help="Algorithm to render")],
    preset: Annotated[str, typer.Option("--preset")] = "none",
    scenario: Annotated[str | None, typer.Option("--scenario", "-s")] = None,
    quality: Annotated[str | None, typer.Option("--quality", "-q")] = None,
    timing: Annotated[str | None, typer.Option("--timing", "-t")] = None,
    max_events: Annotated[int | None, typer.Option("--max-events", "-e")] = None,
    output: Annotated[str | None, typer.Option("--output", "-o")] = None,
    preview: Annotated[bool, typer.Option("--preview", "-p")] = False,
    show_progress: Annotated[bool, typer.Option("--progress")] = True,
) -> None:
    """Legacy alias: algokit render render <algorithm> (hidden from help)."""
    # Forward to BFS command (only algorithm supported currently)
    if algorithm.lower() in ["bfs", "breadth-first-search"]:
        return render_bfs_command(
            preset=preset,
            scenario=scenario,
            quality=quality,
            timing=timing,
            max_events=max_events,
            output=output,
            preview=preview,
            show_progress=show_progress,
        )
    else:
        console.print(
            f"❌ Legacy render only supports BFS. Requested: {algorithm}", style="red"
        )
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
