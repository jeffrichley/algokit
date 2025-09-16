"""Validate command for AGLoViz CLI.

This module provides commands for validating scenario files and configurations.
"""

import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agloviz.utils.config import validate_scenario_file

console = Console()

app = typer.Typer(name="validate")


@app.command("scenario")
def validate_scenario(
    scenario_file: str = typer.Argument(..., help="Path to scenario file to validate"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """Validate a HarborNet scenario file."""
    try:
        scenario_path = Path(scenario_file)
        
        if not scenario_path.exists():
            console.print(f"[red]Error: Scenario file not found: {scenario_file}[/red]")
            raise typer.Exit(1)
        
        if not scenario_path.is_file():
            console.print(f"[red]Error: Path is not a file: {scenario_file}[/red]")
            raise typer.Exit(1)
        
        # Validate the scenario file
        validated_path = validate_scenario_file(scenario_file)
        
        # Load and validate scenario content
        import yaml
        with open(validated_path, 'r') as f:
            scenario_data = yaml.safe_load(f)
        
        # Create validation results
        validation_table = Table(
            title="Scenario Validation Results",
            show_header=True,
            header_style="bold blue"
        )
        validation_table.add_column("Field", style="cyan", no_wrap=True)
        validation_table.add_column("Status", style="green")
        validation_table.add_column("Value", style="white")
        
        # Validate required fields
        required_fields = ['name', 'description', 'width', 'height', 'start', 'goal']
        all_valid = True
        
        for field in required_fields:
            if field in scenario_data:
                value = scenario_data[field]
                if verbose:
                    validation_table.add_row(field, "✅ Present", str(value))
                else:
                    validation_table.add_row(field, "✅ Valid", "✓")
            else:
                validation_table.add_row(field, "❌ Missing", "Not found")
                all_valid = False
        
        # Validate optional fields
        optional_fields = ['narrative', 'obstacles']
        for field in optional_fields:
            if field in scenario_data:
                value = scenario_data[field]
                if verbose:
                    validation_table.add_row(field, "✅ Present", str(value))
                else:
                    validation_table.add_row(field, "✅ Valid", "✓")
            else:
                validation_table.add_row(field, "⚠️ Optional", "Not specified")
        
        # Validate data types and values
        if 'width' in scenario_data and 'height' in scenario_data:
            width = scenario_data['width']
            height = scenario_data['height']
            if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
                validation_table.add_row("Grid Size", "✅ Valid", f"{width}x{height}")
            else:
                validation_table.add_row("Grid Size", "❌ Invalid", f"Width: {width}, Height: {height}")
                all_valid = False
        
        if 'start' in scenario_data and 'goal' in scenario_data:
            start = scenario_data['start']
            goal = scenario_data['goal']
            if (isinstance(start, list) and len(start) == 2 and 
                isinstance(goal, list) and len(goal) == 2):
                validation_table.add_row("Start/Goal", "✅ Valid", f"Start: {start}, Goal: {goal}")
            else:
                validation_table.add_row("Start/Goal", "❌ Invalid", f"Start: {start}, Goal: {goal}")
                all_valid = False
        
        if 'obstacles' in scenario_data:
            obstacles = scenario_data['obstacles']
            if isinstance(obstacles, list):
                validation_table.add_row("Obstacles", "✅ Valid", f"{len(obstacles)} obstacles")
            else:
                validation_table.add_row("Obstacles", "❌ Invalid", f"Expected list, got {type(obstacles)}")
                all_valid = False
        
        console.print(validation_table)
        
        # Show validation result
        if all_valid:
            result_panel = Panel(
                f"[bold green]✓ Scenario file is valid![/bold green]\n"
                f"File: [cyan]{scenario_path.name}[/cyan]\n"
                f"Size: [green]{scenario_path.stat().st_size} bytes[/green]\n"
                f"Ready for visualization rendering.",
                title="Validation Successful",
                border_style="green",
            )
            console.print(result_panel)
        else:
            error_panel = Panel(
                f"[bold red]✗ Scenario file has validation errors.[/bold red]\n"
                f"File: [cyan]{scenario_path.name}[/cyan]\n"
                f"Please fix the errors above and try again.",
                title="Validation Failed",
                border_style="red",
            )
            console.print(error_panel)
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[red]Error validating scenario: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from e


@app.command("all")
def validate_all_scenarios(
    scenarios_dir: str = typer.Option(
        None, "--dir", "-d", help="Directory containing scenario files"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """Validate all scenario files in a directory."""
    try:
        # Determine scenarios directory
        if scenarios_dir:
            scenarios_path = Path(scenarios_dir)
        else:
            # Use default scenarios directory
            from agloviz.utils.config import get_scenarios_directory
            scenarios_path = get_scenarios_directory()
        
        if not scenarios_path.exists():
            console.print(f"[red]Error: Scenarios directory not found: {scenarios_path}[/red]")
            raise typer.Exit(1)
        
        # Find all YAML files
        scenario_files = list(scenarios_path.glob("*.yaml")) + list(scenarios_path.glob("*.yml"))
        
        if not scenario_files:
            console.print(f"[yellow]No scenario files found in: {scenarios_path}[/yellow]")
            return
        
        console.print(f"[blue]Validating {len(scenario_files)} scenario files...[/blue]")
        
        # Create validation summary table
        summary_table = Table(
            title="Validation Summary",
            show_header=True,
            header_style="bold blue"
        )
        summary_table.add_column("File", style="cyan", no_wrap=True)
        summary_table.add_column("Status", style="white")
        summary_table.add_column("Issues", style="yellow")
        
        valid_count = 0
        invalid_count = 0
        
        # Validate each file
        for scenario_file in sorted(scenario_files):
            try:
                # Validate the file
                validated_path = validate_scenario_file(str(scenario_file))
                
                # Load and check basic structure
                import yaml
                with open(validated_path, 'r') as f:
                    scenario_data = yaml.safe_load(f)
                
                # Check required fields
                required_fields = ['name', 'description', 'width', 'height', 'start', 'goal']
                missing_fields = [field for field in required_fields if field not in scenario_data]
                
                if missing_fields:
                    summary_table.add_row(
                        scenario_file.name,
                        "❌ Invalid",
                        f"Missing: {', '.join(missing_fields)}"
                    )
                    invalid_count += 1
                else:
                    summary_table.add_row(
                        scenario_file.name,
                        "✅ Valid",
                        "No issues"
                    )
                    valid_count += 1
                    
            except Exception as e:
                summary_table.add_row(
                    scenario_file.name,
                    "❌ Error",
                    str(e)
                )
                invalid_count += 1
        
        console.print(summary_table)
        
        # Show final results
        if invalid_count == 0:
            result_panel = Panel(
                f"[bold green]✓ All {valid_count} scenario files are valid![/bold green]\n"
                f"Directory: [cyan]{scenarios_path}[/cyan]\n"
                f"All files ready for visualization rendering.",
                title="Validation Complete",
                border_style="green",
            )
        else:
            result_panel = Panel(
                f"[bold yellow]Validation Summary:[/bold yellow]\n"
                f"✅ Valid: [green]{valid_count}[/green]\n"
                f"❌ Invalid: [red]{invalid_count}[/red]\n"
                f"Total: [cyan]{len(scenario_files)}[/cyan]\n"
                f"Directory: [cyan]{scenarios_path}[/cyan]",
                title="Validation Complete",
                border_style="yellow",
            )
        
        console.print(result_panel)
        
        if invalid_count > 0:
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[red]Error validating scenarios: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from e
