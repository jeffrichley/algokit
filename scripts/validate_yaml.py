#!/usr/bin/env python3
"""
AlgoKit YAML Schema Validation CLI

A beautiful CLI tool for validating YAML files against AlgoKit schemas.
Built with Typer and Rich for an excellent user experience.
"""

import yaml
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from cerberus import Validator

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.text import Text
from rich import box
from rich.prompt import Confirm

# Initialize Rich console and Typer app
console = Console()
app = typer.Typer(
    name="validate-yaml",
    help="🔍 Validate AlgoKit YAML files against their schemas",
    rich_markup_mode="rich",
    no_args_is_help=True,
)


class AlgoKitYAMLValidator:
    """YAML validator for AlgoKit project files with Rich output."""
    
    def __init__(self):
        self.family_schema = self._create_family_schema()
        self.algorithm_schema = self._create_algorithm_schema()
    
    def _create_family_schema(self) -> Dict[str, Any]:
        """Create schema for family.yaml files."""
        return {
            # Required core fields
            'id': {'type': 'string', 'required': True},
            'name': {'type': 'string', 'required': True},
            'slug': {'type': 'string', 'required': True},
            'summary': {'type': 'string', 'required': True},
            'description': {'type': 'string', 'required': True},
            
            # Key characteristics
            'key_characteristics': {
                'type': 'list',
                'schema': {
                    'type': 'dict',
                    'schema': {
                        'name': {'type': 'string'},
                        'description': {'type': 'string'},
                        'importance': {'type': 'string'}
                    }
                }
            },
            
            # Common applications
            'common_applications': {
                'type': 'list',
                'schema': {
                    'type': 'dict',
                    'schema': {
                        'category': {'type': 'string'},
                        'examples': {'type': 'list', 'schema': {'type': 'string'}}
                    }
                }
            },
            
            # Concepts
            'concepts': {
                'type': 'list',
                'schema': {
                    'type': 'dict',
                    'schema': {
                        'name': {'type': 'string'},
                        'description': {'type': 'string'},
                        'type': {'type': 'string'}
                    }
                }
            },
            
            # Algorithm management
            'algorithms': {
                'type': 'dict',
                'schema': {
                    'order_mode': {'type': 'string'},
                    'include': {'type': 'list', 'schema': {'type': 'string'}},
                    'exclude': {'type': 'list', 'schema': {'type': 'string'}},
                    'comparison': {
                        'type': 'dict',
                        'schema': {
                            'enabled': {'type': 'boolean'},
                            'metrics': {'type': 'list', 'schema': {'type': 'string'}}
                        }
                    }
                }
            },
            
            # Related families
            'related_families': {
                'type': 'list',
                'schema': {
                    'type': 'dict',
                    'schema': {
                        'id': {'type': 'string'},
                        'relationship': {'type': 'string'},
                        'description': {'type': 'string'}
                    }
                }
            },
            
            # Complexity information
            'complexity': {
                'type': 'dict',
                'schema': {
                    'typical_time': {'type': 'string'},
                    'typical_space': {'type': 'string'},
                    'notes': {'type': 'string'}
                }
            },
            
            # Domain sections
            'domain_sections': {
                'type': 'list',
                'schema': {
                    'type': 'dict',
                    'schema': {
                        'name': {'type': 'string'},
                        'content': {'type': 'string'}
                    }
                }
            },
            
            # References
            'references': {
                'type': 'list',
                'schema': {
                    'type': 'dict',
                    'schema': {
                        'bib_key': {'type': 'string'}
                    }
                }
            },
            
            # Tags
            'tags': {
                'type': 'list',
                'schema': {'type': 'string'}
            },
            
            # Template options
            'template_options': {
                'type': 'dict',
                'schema': {
                    'show_comparison_table': {'type': 'boolean'},
                    'show_complexity_analysis': {'type': 'boolean'},
                    'show_implementation_status': {'type': 'boolean'},
                    'show_related_families': {'type': 'boolean'},
                    'show_references': {'type': 'boolean'},
                    'custom_sections': {'type': 'boolean'}
                }
            },
            
            # Metadata
            'meta': {
                'type': 'dict',
                'schema': {
                    'created': {'type': 'string'},
                    'version': {'type': 'string'},
                    'author': {'type': 'string'}
                }
            }
        }
    
    def _create_algorithm_schema(self) -> Dict[str, Any]:
        """Create schema for algorithm.yaml files."""
        return {
            # Required core fields
            'slug': {'type': 'string', 'required': True},
            'name': {'type': 'string', 'required': True},
            'family_id': {'type': 'string', 'required': True},
            'summary': {'type': 'string', 'required': True},
            'description': {'type': 'string', 'required': True},
            
            # Algorithm-specific fields
            'complexity': {
                'type': 'dict',
                'schema': {
                    'analysis': {
                        'type': 'list',
                        'schema': {
                            'type': 'dict',
                            'schema': {
                                'approach': {'type': 'string'},
                                'time': {'type': 'string'},
                                'space': {'type': 'string'},
                                'notes': {'type': 'string'}
                            }
                        }
                    }
                }
            },
            
            # Implementation details
            'implementations': {
                'type': 'list',
                'schema': {
                    'type': 'dict',
                    'schema': {
                        'type': {'type': 'string'},
                        'name': {'type': 'string'},
                        'description': {'type': 'string'},
                        'complexity': {
                            'type': 'dict',
                            'schema': {
                                'time': {'type': 'string'},
                                'space': {'type': 'string'}
                            }
                        },
                        'code': {'type': 'string'},
                        'advantages': {'type': 'list', 'schema': {'type': 'string'}},
                        'disadvantages': {'type': 'list', 'schema': {'type': 'string'}}
                    }
                }
            },
            
            # Problem formulation
            'formulation': {
                'type': 'dict',
                'schema': {
                    'recurrence_relation': {'type': 'string'},
                    'problem_definition': {'type': 'string'},
                    'mathematical_properties': {
                        'type': 'list',
                        'schema': {
                            'type': 'dict',
                            'schema': {
                                'name': {'type': 'string'},
                                'formula': {'type': 'string'},
                                'description': {'type': 'string'}
                            }
                        }
                    }
                }
            },
            
            # Properties and characteristics
            'properties': {
                'type': 'list',
                'schema': {
                    'type': 'dict',
                    'schema': {
                        'name': {'type': 'string'},
                        'description': {'type': 'string'},
                        'importance': {'type': 'string'}
                    }
                }
            },
            
            # Applications
            'applications': {
                'type': 'list',
                'schema': {
                    'type': 'dict',
                    'schema': {
                        'category': {'type': 'string'},
                        'examples': {'type': 'list', 'schema': {'type': 'string'}}
                    }
                }
            },
            
            # Educational value
            'educational_value': {
                'type': 'list',
                'schema': {'type': 'string'}
            },
            
            # Status
            'status': {
                'type': 'dict',
                'schema': {
                    'current': {'type': 'string'},
                    'implementation_quality': {'type': 'string'},
                    'test_coverage': {'type': 'string'},
                    'documentation_quality': {'type': 'string'},
                    'source_files': {
                        'type': 'list',
                        'schema': {
                            'type': 'dict',
                            'schema': {
                                'path': {'type': 'string'},
                                'description': {'type': 'string'}
                            }
                        }
                    }
                }
            },
            
            # References
            'references': {
                'type': 'list',
                'schema': {
                    'type': 'dict',
                    'schema': {
                        'category': {'type': 'string'},
                        'items': {
                            'type': 'list',
                            'schema': {
                                'type': 'dict',
                                'schema': {
                                    'author': {'type': 'string'},
                                    'year': {'type': 'string'},
                                    'title': {'type': 'string'},
                                    'publisher': {'type': 'string'},
                                    'note': {'type': 'string'},
                                    'url': {'type': 'string'}
                                }
                            }
                        }
                    }
                }
            },
            
            # Related algorithms
            'related_algorithms': {
                'type': 'list',
                'schema': {
                    'type': 'dict',
                    'schema': {
                        'slug': {'type': 'string'},
                        'relationship': {'type': 'string'},
                        'description': {'type': 'string'}
                    }
                }
            },
            
            # Tags
            'tags': {'type': 'list', 'schema': {'type': 'string'}}
        }
    
    def validate_yaml_syntax(self, file_path: str) -> tuple[bool, Optional[str]]:
        """Validate basic YAML syntax."""
        try:
            with open(file_path, 'r') as f:
                yaml.safe_load(f)
            return True, None
        except yaml.YAMLError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)
    
    def validate_family_yaml(self, file_path: str, verbose: bool = False) -> tuple[bool, Optional[Dict]]:
        """Validate a family.yaml file."""
        try:
            # Check YAML syntax first
            syntax_ok, syntax_error = self.validate_yaml_syntax(file_path)
            if not syntax_ok:
                return False, {"syntax_error": syntax_error}
            
            # Load and validate against schema
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            
            validator = Validator(self.family_schema, allow_unknown=True)
            
            if validator.validate(data):
                return True, validator.document
            else:
                return False, {"validation_errors": validator.errors}
                
        except Exception as e:
            return False, {"error": str(e)}
    
    def validate_algorithm_yaml(self, file_path: str, verbose: bool = False) -> tuple[bool, Optional[Dict]]:
        """Validate an algorithm.yaml file."""
        try:
            # Check YAML syntax first
            syntax_ok, syntax_error = self.validate_yaml_syntax(file_path)
            if not syntax_ok:
                return False, {"syntax_error": syntax_error}
            
            # Load and validate against schema
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            
            validator = Validator(self.algorithm_schema, allow_unknown=True)
            
            if validator.validate(data):
                return True, validator.document
            else:
                return False, {"validation_errors": validator.errors}
                
        except Exception as e:
            return False, {"error": str(e)}
    
    def find_family_files(self) -> List[Path]:
        """Find all family.yaml files."""
        data_dir = Path("mkdocs_plugins/data")
        if not data_dir.exists():
            return []
        
        family_files = []
        for family_dir in data_dir.iterdir():
            if family_dir.is_dir():
                family_file = family_dir / "family.yaml"
                if family_file.exists():
                    family_files.append(family_file)
        
        return family_files
    
    def find_algorithm_files(self) -> List[Path]:
        """Find all algorithm.yaml files."""
        data_dir = Path("mkdocs_plugins/data")
        if not data_dir.exists():
            return []
        
        algorithm_files = []
        for family_dir in data_dir.iterdir():
            if family_dir.is_dir():
                algorithms_dir = family_dir / "algorithms"
                if algorithms_dir.exists():
                    for algorithm_file in algorithms_dir.glob("*.yaml"):
                        algorithm_files.append(algorithm_file)
        
        return algorithm_files


def display_validation_result(file_path: str, is_valid: bool, data: Optional[Dict], file_type: str = "family") -> None:
    """Display validation result with Rich formatting."""
    file_name = Path(file_path).name
    
    if is_valid:
        # Success display
        console.print(f"✅ [green]{file_name}[/green] is valid!")
        
        if data and file_type == "family":
            # Create info table
            table = Table(show_header=False, box=box.ROUNDED, padding=(0, 1))
            table.add_column("Field", style="cyan", width=20)
            table.add_column("Value", style="white")
            
            table.add_row("📊 Family", data.get('name', 'Unknown'))
            table.add_row("🆔 ID", data.get('id', 'Unknown'))
            table.add_row("🔧 Characteristics", str(len(data.get('key_characteristics', []))))
            table.add_row("🧮 Algorithms", "All" if not data.get('algorithms', {}).get('include') else str(len(data.get('algorithms', {}).get('include', []))))
            table.add_row("🏷️ Tags", str(len(data.get('tags', []))))
            
            console.print(table)
    
    else:
        # Error display
        console.print(f"❌ [red]{file_name}[/red] validation failed!")
        
        if data:
            if "syntax_error" in data:
                console.print(f"   [red]YAML Syntax Error:[/red] {data['syntax_error']}")
            elif "validation_errors" in data:
                console.print("   [red]Schema Validation Errors:[/red]")
                for field, errors in data["validation_errors"].items():
                    console.print(f"     [yellow]{field}:[/yellow] {errors}")
            elif "error" in data:
                console.print(f"   [red]Error:[/red] {data['error']}")


def display_summary_table(results: List[tuple], file_type: str) -> None:
    """Display a summary table of validation results."""
    if not results:
        return
    
    table = Table(title=f"{file_type.title()} Files Validation Summary", box=box.ROUNDED)
    table.add_column("File", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="white")
    
    for file_path, is_valid, data in results:
        file_name = Path(file_path).name
        status = "✅ Valid" if is_valid else "❌ Invalid"
        status_style = "green" if is_valid else "red"
        
        if is_valid and data and file_type == "family":
            details = f"{data.get('name', 'Unknown')} ({data.get('id', 'Unknown')})"
        elif not is_valid and data:
            if "syntax_error" in data:
                details = "YAML syntax error"
            elif "validation_errors" in data:
                details = f"{len(data['validation_errors'])} validation errors"
            else:
                details = "Unknown error"
        else:
            details = ""
        
        table.add_row(file_name, f"[{status_style}]{status}[/{status_style}]", details)
    
    console.print(table)


@app.command()
def validate(
    file: Optional[str] = typer.Argument(None, help="Specific file to validate"),
    families: bool = typer.Option(False, "--families", "-f", help="Validate only family files"),
    algorithms: bool = typer.Option(False, "--algorithms", "-a", help="Validate only algorithm files"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    summary: bool = typer.Option(True, "--summary/--no-summary", help="Show summary table"),
) -> None:
    """
    🔍 Validate AlgoKit YAML files against their schemas.
    
    This command validates YAML files in the AlgoKit project to ensure they
    conform to the expected schema structure.
    """
    validator = AlgoKitYAMLValidator()
    
    # Display header
    console.print(Panel.fit(
        "[bold blue]AlgoKit YAML Validator[/bold blue]\n"
        "🔍 Validating YAML files against AlgoKit schemas",
        border_style="blue"
    ))
    
    if file:
        # Validate specific file
        file_path = Path(file)
        if not file_path.exists():
            console.print(f"❌ [red]File {file} not found[/red]")
            raise typer.Exit(1)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Validating file...", total=None)
            
            if file_path.name == "family.yaml":
                is_valid, data = validator.validate_family_yaml(str(file_path), verbose)
                display_validation_result(str(file_path), is_valid, data, "family")
            elif file_path.name.endswith(".yaml") and "algorithms" in str(file_path):
                is_valid, data = validator.validate_algorithm_yaml(str(file_path), verbose)
                display_validation_result(str(file_path), is_valid, data, "algorithm")
            else:
                console.print(f"❌ [red]Unknown file type: {file}[/red]")
                raise typer.Exit(1)
            
            progress.update(task, completed=True)
        
        raise typer.Exit(0 if is_valid else 1)
    
    else:
        # Validate all files
        family_results = []
        algorithm_results = []
        
        # Validate family files
        if not algorithms:
            family_files = validator.find_family_files()
            if family_files:
                console.print("\n[bold cyan]📁 Validating family files...[/bold cyan]")
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Validating family files...", total=len(family_files))
                    
                    for family_file in family_files:
                        is_valid, data = validator.validate_family_yaml(str(family_file), verbose)
                        family_results.append((str(family_file), is_valid, data))
                        
                        if verbose:
                            display_validation_result(str(family_file), is_valid, data, "family")
                            console.print()  # Add spacing
                        
                        progress.advance(task)
        
        # Validate algorithm files
        if not families:
            algorithm_files = validator.find_algorithm_files()
            if algorithm_files:
                console.print("\n[bold cyan]🔧 Validating algorithm files...[/bold cyan]")
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Validating algorithm files...", total=len(algorithm_files))
                    
                    for algorithm_file in algorithm_files:
                        is_valid, data = validator.validate_algorithm_yaml(str(algorithm_file), verbose)
                        algorithm_results.append((str(algorithm_file), is_valid, data))
                        
                        if verbose:
                            display_validation_result(str(algorithm_file), is_valid, data, "algorithm")
                            console.print()  # Add spacing
                        
                        progress.advance(task)
        
        # Display summary
        if summary:
            console.print("\n" + "="*60)
            
            if family_results:
                display_summary_table(family_results, "family")
            
            if algorithm_results:
                display_summary_table(algorithm_results, "algorithm")
        
        # Final result
        all_family_valid = all(result[1] for result in family_results) if family_results else True
        all_algorithm_valid = all(result[1] for result in algorithm_results) if algorithm_results else True
        
        console.print("\n" + "="*60)
        if all_family_valid and all_algorithm_valid:
            console.print("🎉 [bold green]All validations passed![/bold green]")
            raise typer.Exit(0)
        else:
            console.print("⚠️  [bold red]Some validations failed![/bold red]")
            raise typer.Exit(1)


@app.command()
def info() -> None:
    """📊 Show information about the validator and available schemas."""
    console.print(Panel.fit(
        "[bold blue]AlgoKit YAML Validator Information[/bold blue]\n\n"
        "🔍 [bold]Supported File Types:[/bold]\n"
        "  • family.yaml - Algorithm family definitions\n"
        "  • algorithm.yaml - Individual algorithm definitions\n\n"
        "📋 [bold]Schema Coverage:[/bold]\n"
        "  • Required fields validation\n"
        "  • Data type checking\n"
        "  • Nested structure validation\n"
        "  • Optional field handling\n\n"
        "🛠️  [bold]Built with:[/bold]\n"
        "  • Typer - Modern CLI framework\n"
        "  • Rich - Beautiful terminal output\n"
        "  • Cerberus - Schema validation\n"
        "  • PyYAML - YAML parsing",
        border_style="blue"
    ))


if __name__ == "__main__":
    app()