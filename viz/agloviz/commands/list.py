"""List command for AGLoViz CLI.

This module provides commands for listing available algorithms and scenarios.
"""

import sys
import yaml
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add the main project to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from algokit.core.helpers import load_harbor_scenario, HarborNetScenario
    ALGOKIT_AVAILABLE = True
except ImportError:
    ALGOKIT_AVAILABLE = False
    print("Warning: Algokit modules not available. Using fallback implementations.")

from agloviz.utils.scene_registry import SceneRegistry
from agloviz.utils.config import get_scenarios_directory

console = Console()

app = typer.Typer(name="list")


def get_cache_directory() -> Path:
    """Get the path to the AGLoViz cache directory."""
    cache_dir = project_root / ".algokit" / "agloviz"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_file(cache_name: str) -> Path:
    """Get the path to a specific cache file."""
    return get_cache_directory() / f"{cache_name}.json"


def is_file_newer_than_cache(file_path: Path, cache_file: Path) -> bool:
    """Check if a file is newer than its cache file."""
    if not cache_file.exists():
        return True
    
    try:
        file_mtime = os.path.getmtime(file_path)
        cache_mtime = os.path.getmtime(cache_file)
        return file_mtime > cache_mtime
    except OSError:
        return True


def save_cache(cache_name: str, data: Any) -> None:
    """Save data to cache file."""
    cache_file = get_cache_file(cache_name)
    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not save cache {cache_name}: {e}[/yellow]")


def load_cache(cache_name: str) -> Any:
    """Load data from cache file."""
    cache_file = get_cache_file(cache_name)
    try:
        with open(cache_file, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def get_data_directory() -> Path:
    """Get the path to the MkDocs data directory."""
    return project_root / "mkdocs_plugins" / "data"


def load_family_data(family_id: str) -> Dict[str, Any]:
    """Load family data from YAML file."""
    family_file = get_data_directory() / family_id / "family.yaml"
    if not family_file.exists():
        return {}
    
    with open(family_file, 'r') as f:
        return yaml.safe_load(f)


def load_algorithm_data(family_id: str, algorithm_slug: str) -> Dict[str, Any]:
    """Load algorithm data from YAML file."""
    algorithm_file = get_data_directory() / family_id / "algorithms" / f"{algorithm_slug}.yaml"
    if not algorithm_file.exists():
        return {}
    
    with open(algorithm_file, 'r') as f:
        return yaml.safe_load(f)


def get_visible_families() -> List[Tuple[str, Dict[str, Any]]]:
    """Get all families that have at least one visible algorithm."""
    # Check cache first
    cache_file = get_cache_file("visible_families")
    data_dir = get_data_directory()
    
    # Check if we need to rebuild cache
    needs_rebuild = False
    if not cache_file.exists():
        needs_rebuild = True
    else:
        # Check if any family.yaml or algorithm files are newer than cache
        for family_dir in data_dir.iterdir():
            if not family_dir.is_dir() or family_dir.name.startswith('.'):
                continue
            
            family_file = family_dir / "family.yaml"
            if family_file.exists() and is_file_newer_than_cache(family_file, cache_file):
                needs_rebuild = True
                break
            
            algorithms_dir = family_dir / "algorithms"
            if algorithms_dir.exists():
                for algo_file in algorithms_dir.glob("*.yaml"):
                    if is_file_newer_than_cache(algo_file, cache_file):
                        needs_rebuild = True
                        break
                if needs_rebuild:
                    break
    
    # Load from cache if not rebuilding
    if not needs_rebuild:
        cached_data = load_cache("visible_families")
        if cached_data:
            return [(item[0], item[1]) for item in cached_data]
    
    # Rebuild cache - only store minimal data needed for CLI
    families = []
    for family_dir in data_dir.iterdir():
        if not family_dir.is_dir() or family_dir.name.startswith('.'):
            continue
        
        family_id = family_dir.name
        family_data = load_family_data(family_id)
        
        # Check if family has any visible algorithms
        algorithms_dir = family_dir / "algorithms"
        if algorithms_dir.exists():
            for algo_file in algorithms_dir.glob("*.yaml"):
                algo_data = yaml.safe_load(open(algo_file, 'r'))
                if not algo_data.get('hidden', False):  # Algorithm is visible
                    # Only cache essential fields for CLI display
                    minimal_family_data = {
                        'name': family_data.get('name', family_id.title()),
                        'summary': family_data.get('summary', family_data.get('description', 'No description available'))
                    }
                    families.append((family_id, minimal_family_data))
                    break
    
    # Save to cache
    save_cache("visible_families", families)
    return families


def get_visible_algorithms() -> List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]]:
    """Get all visible algorithms with their family info."""
    # Check cache first
    cache_file = get_cache_file("visible_algorithms")
    data_dir = get_data_directory()
    
    # Check if we need to rebuild cache
    needs_rebuild = False
    if not cache_file.exists():
        needs_rebuild = True
    else:
        # Check if any family.yaml or algorithm files are newer than cache
        for family_dir in data_dir.iterdir():
            if not family_dir.is_dir() or family_dir.name.startswith('.'):
                continue
            
            family_file = family_dir / "family.yaml"
            if family_file.exists() and is_file_newer_than_cache(family_file, cache_file):
                needs_rebuild = True
                break
            
            algorithms_dir = family_dir / "algorithms"
            if algorithms_dir.exists():
                for algo_file in algorithms_dir.glob("*.yaml"):
                    if is_file_newer_than_cache(algo_file, cache_file):
                        needs_rebuild = True
                        break
                if needs_rebuild:
                    break
    
    # Load from cache if not rebuilding
    if not needs_rebuild:
        cached_data = load_cache("visible_algorithms")
        if cached_data:
            return [(item[0], item[1], item[2], item[3]) for item in cached_data]
    
    # Rebuild cache - only store minimal data needed for CLI
    algorithms = []
    for family_dir in data_dir.iterdir():
        if not family_dir.is_dir() or family_dir.name.startswith('.'):
            continue
        
        family_id = family_dir.name
        family_data = load_family_data(family_id)
        
        algorithms_dir = family_dir / "algorithms"
        if algorithms_dir.exists():
            for algo_file in algorithms_dir.glob("*.yaml"):
                algo_data = yaml.safe_load(open(algo_file, 'r'))
                if not algo_data.get('hidden', False):  # Algorithm is visible
                    # Only cache essential fields for CLI display
                    minimal_family_data = {
                        'name': family_data.get('name', family_id.title()),
                        'summary': family_data.get('summary', family_data.get('description', 'No description available'))
                    }
                    minimal_algo_data = {
                        'name': algo_data.get('name', algo_file.stem.replace('-', ' ').title()),
                        'summary': algo_data.get('summary', 'No description available')
                    }
                    algorithms.append((family_id, algo_file.stem, minimal_family_data, minimal_algo_data))
    
    # Save to cache
    save_cache("visible_algorithms", algorithms)
    return algorithms


@app.command("algorithms")
def list_algorithms() -> None:
    """List all available algorithms."""
    try:
        # Get visible algorithms from data files
        visible_algorithms = get_visible_algorithms()
        registry = SceneRegistry()
        available_scenes = registry.list_algorithms()
        
        if not visible_algorithms:
            console.print("[yellow]No algorithms available.[/yellow]")
            return
        
        # Create algorithms table
        table = Table(
            title=f"Available Algorithms ({len(visible_algorithms)} total)",
            show_header=True,
            header_style="bold blue"
        )
        table.add_column("Algorithm", style="cyan", no_wrap=True)
        table.add_column("Family", style="blue")
        table.add_column("Description", style="white")
        table.add_column("Visualization", style="green")
        
        for family_id, algo_slug, family_data, algo_data in visible_algorithms:
            # Get algorithm name and description
            algo_name = algo_data.get('name', algo_slug.replace('-', ' ').title())
            algo_summary = algo_data.get('summary', 'No description available')
            family_name = family_data.get('name', family_id.title())
            
            # Check if visualization scene exists
            scene_name = algo_slug.replace('-', '_')  # Convert kebab-case to snake_case
            if scene_name in available_scenes:
                viz_status = "✅ Available"
            else:
                viz_status = "🚧 Coming Soon"
            
            table.add_row(
                algo_name,
                family_name,
                algo_summary[:60] + "..." if len(algo_summary) > 60 else algo_summary,
                viz_status
            )
        
        console.print(table)
        
        # Show usage example for available visualizations
        available_algorithms = [algo for algo in available_scenes if algo.replace('_', '-') in [a[1] for a in visible_algorithms]]
        if available_algorithms:
            first_algo = available_algorithms[0]
            usage_panel = Panel(
                f"[bold green]Usage Example:[/bold green]\n"
                f"[cyan]agloviz render {first_algo} --scenario path/to/scenario.yaml[/cyan]",
                title="Quick Start",
                border_style="green",
            )
            console.print(usage_panel)
        
    except Exception as e:
        console.print(f"[red]Error listing algorithms: {e}[/red]")
        raise typer.Exit(1) from e


@app.command("scenarios")
def list_scenarios() -> None:
    """List all available HarborNet scenarios."""
    try:
        scenarios_dir = get_scenarios_directory()
        
        if not scenarios_dir.exists():
            console.print("[yellow]Scenarios directory not found.[/yellow]")
            return
        
        # Find all YAML scenario files
        scenario_files = list(scenarios_dir.glob("*.yaml")) + list(scenarios_dir.glob("*.yml"))
        
        if not scenario_files:
            console.print("[yellow]No scenario files found.[/yellow]")
            return
        
        # Create scenarios table
        table = Table(
            title=f"Available Scenarios ({len(scenario_files)} found)",
            show_header=True,
            header_style="bold blue"
        )
        table.add_column("File", style="cyan", no_wrap=True)
        table.add_column("Name", style="white")
        table.add_column("Size", style="green")
        table.add_column("Description", style="dim")
        
        # Add scenario information
        for scenario_file in sorted(scenario_files):
            try:
                # Try to load scenario info
                import yaml
                with open(scenario_file, 'r') as f:
                    scenario_data = yaml.safe_load(f)
                
                name = scenario_data.get('name', 'Unknown')
                description = scenario_data.get('description', 'No description')
                if len(description) > 50:
                    description = description[:47] + "..."
                
                file_size = scenario_file.stat().st_size
                size_str = f"{file_size} bytes"
                
                table.add_row(
                    scenario_file.name,
                    name,
                    size_str,
                    description
                )
                
            except Exception:
                # If we can't parse the file, just show basic info
                file_size = scenario_file.stat().st_size
                table.add_row(
                    scenario_file.name,
                    "Unknown",
                    f"{file_size} bytes",
                    "Could not parse file"
                )
        
        console.print(table)
        
        # Show usage example
        if scenario_files:
            example_file = scenario_files[0].name
            usage_panel = Panel(
                f"[bold green]Usage Example:[/bold green]\n"
                f"[cyan]agloviz render bfs --scenario {scenarios_dir / example_file}[/cyan]",
                title="Quick Start",
                border_style="green",
            )
            console.print(usage_panel)
        
    except Exception as e:
        console.print(f"[red]Error listing scenarios: {e}[/red]")
        raise typer.Exit(1) from e


@app.command("families")
def list_families() -> None:
    """List all algorithm families."""
    try:
        # Get visible families from data files
        visible_families = get_visible_families()
        registry = SceneRegistry()
        available_scenes = registry.list_algorithms()
        
        if not visible_families:
            console.print("[yellow]No families available.[/yellow]")
            return
        
        # Create families table
        table = Table(
            title=f"Algorithm Families ({len(visible_families)} total)",
            show_header=True,
            header_style="bold blue"
        )
        table.add_column("Family", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        table.add_column("Algorithms", style="green")
        table.add_column("Visualizations", style="yellow")
        
        total_algorithms = 0
        total_visualizations = 0
        
        for family_id, family_data in visible_families:
            # Get family info
            family_name = family_data.get('name', family_id.title())
            family_summary = family_data.get('summary', family_data.get('description', 'No description available'))
            if len(family_summary) > 100:
                family_summary = family_summary[:97] + "..."
            
            # Count algorithms in this family
            family_algorithms = get_visible_algorithms()
            family_algo_count = sum(1 for fam_id, _, _, _ in family_algorithms if fam_id == family_id)
            total_algorithms += family_algo_count
            
            # Count available visualizations for this family
            family_viz_count = 0
            for fam_id, algo_slug, _, _ in family_algorithms:
                if fam_id == family_id:
                    scene_name = algo_slug.replace('-', '_')
                    if scene_name in available_scenes:
                        family_viz_count += 1
                        total_visualizations += 1
            
            # Status based on visualization availability
            if family_viz_count > 0:
                viz_status = f"✅ {family_viz_count} available"
            else:
                viz_status = "🚧 Coming Soon"
            
            table.add_row(
                family_name,
                family_summary,
                f"{family_algo_count} algorithms",
                viz_status
            )
        
        console.print(table)
        
        # Show summary
        summary_panel = Panel(
            f"[bold green]Summary:[/bold green]\n"
            f"Total Families: [cyan]{len(visible_families)}[/cyan]\n"
            f"Total Algorithms: [blue]{total_algorithms}[/blue]\n"
            f"Available Visualizations: [green]{total_visualizations}[/green]\n"
            f"Coming Soon: [yellow]{total_algorithms - total_visualizations}[/yellow]",
            title="Family Overview",
            border_style="blue",
        )
        console.print(summary_panel)
        
    except Exception as e:
        console.print(f"[red]Error listing families: {e}[/red]")
        raise typer.Exit(1) from e


@app.command("clear-cache")
def clear_cache() -> None:
    """Clear the AGLoViz cache to force rebuilding."""
    try:
        cache_dir = get_cache_directory()
        if not cache_dir.exists():
            console.print("[yellow]No cache directory found.[/yellow]")
            return
        
        # Remove all cache files
        cache_files = list(cache_dir.glob("*.json"))
        if not cache_files:
            console.print("[yellow]No cache files found.[/yellow]")
            return
        
        for cache_file in cache_files:
            cache_file.unlink()
        
        console.print(f"[green]Cleared {len(cache_files)} cache files.[/green]")
        console.print("[blue]Next command will rebuild cache from YAML files.[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error clearing cache: {e}[/red]")
        raise typer.Exit(1) from e
