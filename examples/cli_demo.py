#!/usr/bin/env python3
"""Demo script showing how to use the AlgoKit Render CLI."""

import subprocess
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def run_command(cmd: str, description: str) -> None:
    """Run a CLI command and show the result."""
    print(f"\n{'='*60}")
    print(f"🎬 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=project_root)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except Exception as e:
        print(f"Error running command: {e}")

def main():
    """Demonstrate the AlgoKit Render CLI."""
    print("🎬 AlgoKit Render CLI Demo")
    print("This demo shows the available CLI commands")

    # List scenarios
    run_command(
        "uv run python -m algokit.cli.main render scenarios",
        "List Available Scenarios"
    )

    # List scenarios with details
    run_command(
        "uv run python -m algokit.cli.main render scenarios --details",
        "List Scenarios with Details"
    )

    # List algorithms
    run_command(
        "uv run python -m algokit.cli.main render algorithms",
        "List Available Algorithms"
    )

    # Show render help
    run_command(
        "uv run python -m algokit.cli.main render render --help",
        "Render Command Help"
    )

    # Show quick command help
    run_command(
        "uv run python -m algokit.cli.main render quick --help",
        "Quick Render Command Help"
    )

    print(f"\n{'='*60}")
    print("🚀 Ready to render!")
    print(f"{'='*60}")
    print("Try these commands:")
    print("  uv run python -m algokit.cli.main render quick")
    print("  uv run python -m algokit.cli.main render bfs --scenario 'Tiny Test'")
    print("  uv run python -m algokit.cli.main render demo")

if __name__ == "__main__":
    main()
