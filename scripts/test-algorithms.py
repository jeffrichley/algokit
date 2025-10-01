#!/usr/bin/env python3
"""Test script for algorithm-specific coverage and testing.

This script runs tests specifically for algorithm implementations and generates
coverage reports focused only on the algorithm code.
"""

import subprocess
import sys
from pathlib import Path


def run_algorithm_tests():
    """Run tests for algorithm implementations with focused coverage."""
    
    # Get the project root
    project_root = Path(__file__).parent.parent
    
    # Commands to run
    commands = [
        # Run algorithm tests with algorithm-specific coverage
        [
            "uv", "run", "pytest", 
            "tests/pathfinding/", 
            "tests/dynamic_programming/",
            "--cov=src/algokit/pathfinding",
            "--cov=src/algokit/dynamic_programming", 
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov-algorithms",
            "--cov-report=xml:coverage-algorithms.xml",
            "--cov-fail-under=90",
            "-v"
        ],
        
        # Run property-based tests for algorithms
        [
            "uv", "run", "pytest",
            "tests/pathfinding/test_*_properties.py",
            "--cov=src/algokit/pathfinding",
            "--cov-report=term-missing",
            "--cov-fail-under=90",
            "-v"
        ]
    ]
    
    # Run each command
    for i, cmd in enumerate(commands, 1):
        print(f"\n{'='*60}")
        print(f"Running algorithm test suite {i}/{len(commands)}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        try:
            result = subprocess.run(cmd, cwd=project_root, check=True)
            print(f"✅ Test suite {i} passed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Test suite {i} failed with exit code {e.returncode}")
            return e.returncode
    
    print(f"\n{'='*60}")
    print("🎉 All algorithm tests passed!")
    print("📊 Coverage reports generated:")
    print("   - HTML: htmlcov-algorithms/index.html")
    print("   - XML:  coverage-algorithms.xml")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(run_algorithm_tests())
