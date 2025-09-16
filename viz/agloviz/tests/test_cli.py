"""Test AGLoViz CLI functionality."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from agloviz.main import app

runner = CliRunner()


class TestAGLoVizCLI:
    """Test AGLoViz CLI commands."""
    
    def test_version_command(self) -> None:
        """Test version command displays correctly."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "AGLoViz CLI" in result.output
        assert "Version:" in result.output
    
    def test_help_command(self) -> None:
        """Test help command displays correctly."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "AGLoViz - Algorithm Visualization CLI" in result.output
        assert "render" in result.output
        assert "list" in result.output
        assert "validate" in result.output
    
    def test_render_help(self) -> None:
        """Test render command help displays correctly."""
        result = runner.invoke(app, ["render", "--help"])
        assert result.exit_code == 0
        assert "Render algorithm visualizations" in result.output
    
    def test_list_help(self) -> None:
        """Test list command help displays correctly."""
        result = runner.invoke(app, ["list", "--help"])
        assert result.exit_code == 0
        assert "List available algorithms and scenarios" in result.output
    
    def test_validate_help(self) -> None:
        """Test validate command help displays correctly."""
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "Validate scenario files" in result.output


class TestRenderCommands:
    """Test render command functionality."""
    
    @patch('agloviz.commands.render.SceneRegistry')
    @patch('agloviz.commands.render.render_scene')
    @patch('agloviz.commands.render.validate_scenario_file')
    def test_render_bfs_success(self, mock_validate, mock_render, mock_registry) -> None:
        """Test successful BFS rendering."""
        # Mock the dependencies
        mock_validate.return_value = Path("test_scenario.yaml")
        mock_scene_class = MagicMock()
        mock_registry.return_value.get_scene_class.return_value = mock_scene_class
        
        result = runner.invoke(app, [
            "render", "bfs",
            "--scenario", "test_scenario.yaml",
            "--format", "mp4",
            "--quality", "medium"
        ])
        
        assert result.exit_code == 0
        mock_validate.assert_called_once_with("test_scenario.yaml")
        mock_render.assert_called_once()
    
    def test_render_bfs_missing_scenario(self) -> None:
        """Test BFS rendering with missing scenario file."""
        result = runner.invoke(app, [
            "render", "bfs",
            "--scenario", "nonexistent.yaml"
        ])
        
        assert result.exit_code == 1
    
    @patch('agloviz.commands.render.SceneRegistry')
    def test_render_bfs_scene_not_found(self, mock_registry) -> None:
        """Test BFS rendering when scene class not found."""
        mock_registry.return_value.get_scene_class.return_value = None
        
        result = runner.invoke(app, [
            "render", "bfs",
            "--scenario", "test_scenario.yaml"
        ])
        
        assert result.exit_code == 1
        assert "BFS scene class not found" in result.output


class TestListCommands:
    """Test list command functionality."""
    
    @patch('agloviz.commands.list.SceneRegistry')
    def test_list_algorithms_success(self, mock_registry) -> None:
        """Test successful algorithm listing."""
        mock_registry.return_value.list_algorithms.return_value = ["bfs", "dfs"]
        
        result = runner.invoke(app, ["list", "algorithms"])
        
        assert result.exit_code == 0
        assert "Available Algorithms" in result.output
    
    @patch('agloviz.commands.list.get_scenarios_directory')
    def test_list_scenarios_success(self, mock_scenarios_dir) -> None:
        """Test successful scenario listing."""
        # Mock scenarios directory with test files
        test_dir = Path("/tmp/test_scenarios")
        test_dir.mkdir(exist_ok=True)
        
        # Create test scenario file
        test_scenario = test_dir / "test_scenario.yaml"
        test_scenario.write_text("""
name: Test Scenario
description: A test scenario
width: 5
height: 5
start: [0, 0]
goal: [4, 4]
""")
        
        mock_scenarios_dir.return_value = test_dir
        
        result = runner.invoke(app, ["list", "scenarios"])
        
        assert result.exit_code == 0
        assert "Available Scenarios" in result.output
        
        # Cleanup
        test_scenario.unlink()
        test_dir.rmdir()
    
    def test_list_families_success(self) -> None:
        """Test successful family listing."""
        result = runner.invoke(app, ["list", "families"])
        
        assert result.exit_code == 0
        assert "Algorithm Families" in result.output


class TestValidateCommands:
    """Test validate command functionality."""
    
    def test_validate_scenario_success(self) -> None:
        """Test successful scenario validation."""
        # Create temporary valid scenario file
        test_scenario = Path("/tmp/test_scenario.yaml")
        test_scenario.write_text("""
name: Test Scenario
description: A test scenario for validation
width: 5
height: 5
start: [0, 0]
goal: [4, 4]
obstacles:
  - [1, 1]
  - [2, 2]
""")
        
        try:
            result = runner.invoke(app, ["validate", "scenario", str(test_scenario)])
            
            assert result.exit_code == 0
            assert "Validation Successful" in result.output
        finally:
            test_scenario.unlink()
    
    def test_validate_scenario_invalid(self) -> None:
        """Test scenario validation with invalid file."""
        # Create temporary invalid scenario file
        test_scenario = Path("/tmp/invalid_scenario.yaml")
        test_scenario.write_text("""
name: Invalid Scenario
# Missing required fields
""")
        
        try:
            result = runner.invoke(app, ["validate", "scenario", str(test_scenario)])
            
            assert result.exit_code == 1
            assert "Validation Failed" in result.output
        finally:
            test_scenario.unlink()
    
    def test_validate_scenario_not_found(self) -> None:
        """Test scenario validation with non-existent file."""
        result = runner.invoke(app, ["validate", "scenario", "nonexistent.yaml"])
        
        assert result.exit_code == 1
        assert "not found" in result.output


class TestErrorHandling:
    """Test error handling in CLI."""
    
    def test_invalid_command(self) -> None:
        """Test invalid command handling."""
        result = runner.invoke(app, ["invalid_command"])
        
        assert result.exit_code != 0
    
    def test_missing_required_argument(self) -> None:
        """Test missing required argument handling."""
        result = runner.invoke(app, ["render", "bfs"])
        
        assert result.exit_code != 0
        assert "Missing option" in result.output or "Missing argument" in result.output
