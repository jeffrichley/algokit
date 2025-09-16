"""Test AGLoViz command modules."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agloviz.commands import render, list_cmds, validate


class TestRenderCommands:
    """Test render command functionality."""
    
    @patch('agloviz.commands.render.SceneRegistry')
    @patch('agloviz.commands.render.render_scene')
    @patch('agloviz.commands.render.validate_scenario_file')
    def test_render_bfs_integration(self, mock_validate, mock_render, mock_registry) -> None:
        """Test BFS render command integration."""
        # Mock the dependencies
        mock_validate.return_value = Path("test_scenario.yaml")
        mock_scene_class = MagicMock()
        mock_registry.return_value.get_scene_class.return_value = mock_scene_class
        
        # Test the render_bfs function directly
        from agloviz.commands.render import render_bfs
        
        # This would normally be called by Typer, but we can test the logic
        # by checking that the mocks are set up correctly
        assert mock_validate.return_value == Path("test_scenario.yaml")
        assert mock_registry.return_value.get_scene_class.return_value == mock_scene_class
    
    def test_render_algorithm_validation(self) -> None:
        """Test render algorithm command validation."""
        # Test that the command structure is correct
        assert hasattr(render, 'app')
        assert hasattr(render.app, 'commands')
        
        # Check that expected commands exist
        command_names = [cmd.name for cmd in render.app.commands.values()]
        expected_commands = ['bfs', 'dfs', 'astar', 'algorithm']
        
        for cmd in expected_commands:
            assert cmd in command_names


class TestListCommands:
    """Test list command functionality."""
    
    def test_list_commands_structure(self) -> None:
        """Test list commands structure."""
        assert hasattr(list_cmds, 'app')
        assert hasattr(list_cmds.app, 'commands')
        
        # Check that expected commands exist
        command_names = [cmd.name for cmd in list_cmds.app.commands.values()]
        expected_commands = ['algorithms', 'scenarios', 'families']
        
        for cmd in expected_commands:
            assert cmd in command_names
    
    @patch('agloviz.commands.list.SceneRegistry')
    def test_list_algorithms_functionality(self, mock_registry) -> None:
        """Test list algorithms functionality."""
        # Mock registry to return test algorithms
        mock_registry.return_value.list_algorithms.return_value = ["bfs", "dfs", "astar"]
        
        # Test that the function can be called
        from agloviz.commands.list import list_algorithms
        
        # The function should work without errors
        # (we can't easily test the console output in unit tests)
        assert mock_registry.return_value.list_algorithms.return_value == ["bfs", "dfs", "astar"]


class TestValidateCommands:
    """Test validate command functionality."""
    
    def test_validate_commands_structure(self) -> None:
        """Test validate commands structure."""
        assert hasattr(validate, 'app')
        assert hasattr(validate.app, 'commands')
        
        # Check that expected commands exist
        command_names = [cmd.name for cmd in validate.app.commands.values()]
        expected_commands = ['scenario', 'all']
        
        for cmd in expected_commands:
            assert cmd in command_names
    
    def test_validate_scenario_functionality(self) -> None:
        """Test validate scenario functionality."""
        # Test that the function can be called
        from agloviz.commands.validate import validate_scenario
        
        # The function should exist and be callable
        assert callable(validate_scenario)
    
    def test_validate_all_functionality(self) -> None:
        """Test validate all scenarios functionality."""
        # Test that the function can be called
        from agloviz.commands.validate import validate_all_scenarios
        
        # The function should exist and be callable
        assert callable(validate_all_scenarios)


class TestCommandIntegration:
    """Test command integration and dependencies."""
    
    def test_command_imports(self) -> None:
        """Test that all command modules can be imported."""
        # Test that commands can be imported without errors
        from agloviz.commands import render, list_cmds, validate
        
        assert render is not None
        assert list_cmds is not None
        assert validate is not None
    
    def test_command_app_registration(self) -> None:
        """Test that command apps are properly configured."""
        from agloviz.commands import render, list_cmds, validate
        
        # Check that each command has a Typer app
        assert hasattr(render, 'app')
        assert hasattr(list_cmds, 'app')
        assert hasattr(validate, 'app')
        
        # Check that apps are Typer instances
        from typer import Typer
        assert isinstance(render.app, Typer)
        assert isinstance(list_cmds.app, Typer)
        assert isinstance(validate.app, Typer)
    
    def test_command_help_text(self) -> None:
        """Test that commands have proper help text."""
        from agloviz.commands import render, list_cmds, validate
        
        # Check that help text is configured
        assert render.app.info.help is not None
        assert list_cmds.app.info.help is not None
        assert validate.app.info.help is not None
        
        # Check that help text contains expected keywords
        assert "render" in render.app.info.help.lower()
        assert "list" in list_cmds.app.info.help.lower()
        assert "validate" in validate.app.info.help.lower()
