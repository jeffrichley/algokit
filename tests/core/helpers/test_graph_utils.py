"""Tests for graph utilities module."""

import tempfile
from pathlib import Path

import pytest
import yaml

from algokit.core.helpers.graph_utils import (
    HarborNetScenario,
    create_grid_graph,
    load_harbor_scenario,
)


class TestCreateGridGraph:
    """Test create_grid_graph function."""

    @pytest.mark.unit
    def test_create_simple_grid(self) -> None:
        """Test creating a simple grid graph."""
        # Arrange - Set up grid dimensions for a 3x3 grid
        width, height = 3, 3

        # Act - Create the grid graph with specified dimensions
        graph = create_grid_graph(width, height)

        # Assert - Verify correct number of nodes and edges, and key positions exist
        assert graph.number_of_nodes() == 9
        assert graph.number_of_edges() == 12  # 4-connected grid
        assert (0, 0) in graph
        assert (2, 2) in graph

    @pytest.mark.unit
    def test_create_grid_with_diagonal(self) -> None:
        """Test creating a grid with diagonal connections."""
        # Arrange - Set up 2x2 grid with diagonal connections enabled
        width, height = 2, 2

        # Act - Create the grid graph with diagonal connections
        graph = create_grid_graph(width, height, diagonal=True)

        # Assert - Verify correct number of nodes and edges with diagonal connections
        assert graph.number_of_nodes() == 4
        assert graph.number_of_edges() == 6  # 8-connected grid
        assert graph.has_edge((0, 0), (1, 1))  # Diagonal edge

    @pytest.mark.unit
    def test_create_grid_with_blocked_cells(self) -> None:
        """Test creating a grid with blocked cells."""
        # Arrange - Set up 3x3 grid with specific blocked cells
        width, height = 3, 3
        blocked = {(1, 1), (2, 2)}

        # Act - Create the grid graph with blocked cells
        graph = create_grid_graph(width, height, blocked=blocked)

        # Assert - Verify blocked cells are not in the graph
        assert graph.number_of_nodes() == 7  # 9 - 2 blocked
        assert (1, 1) not in graph
        assert (2, 2) not in graph

    @pytest.mark.unit
    def test_create_grid_invalid_dimensions(self) -> None:
        """Test creating a grid with invalid dimensions."""
        # Arrange & Act & Assert - Test invalid width
        with pytest.raises(ValueError, match="Grid dimensions must be positive"):
            create_grid_graph(0, 3)

        # Arrange & Act & Assert - Test invalid height
        with pytest.raises(ValueError, match="Grid dimensions must be positive"):
            create_grid_graph(3, -1)


class TestHarborNetScenario:
    """Test HarborNetScenario class."""

    @pytest.mark.unit
    def test_scenario_creation(self) -> None:
        """Test creating a scenario."""
        # Arrange - Set up scenario parameters
        scenario = HarborNetScenario(
            name="test",
            width=3,
            height=3,
            start=(0, 0),
            goal=(2, 2),
            obstacles={(1, 1)},
            description="Test scenario",
            narrative="A test narrative",
        )

        # Act - No action needed, scenario is created in arrange

        # Assert - Verify all scenario attributes are set correctly
        assert scenario.name == "test"
        assert scenario.width == 3
        assert scenario.height == 3
        assert scenario.start == (0, 0)
        assert scenario.goal == (2, 2)
        assert scenario.obstacles == {(1, 1)}
        assert scenario.description == "Test scenario"
        assert scenario.narrative == "A test narrative"

    @pytest.mark.unit
    def test_scenario_to_graph(self) -> None:
        """Test converting scenario to graph."""
        # Arrange - Create a scenario with obstacles
        scenario = HarborNetScenario(
            name="test",
            width=2,
            height=2,
            start=(0, 0),
            goal=(1, 1),
            obstacles={(0, 1)},
        )

        # Act - Convert scenario to graph
        graph = scenario.to_graph()

        # Assert - Verify graph structure matches scenario
        assert graph.number_of_nodes() == 3  # 4 - 1 obstacle
        assert (0, 1) not in graph
        assert (0, 0) in graph
        assert (1, 1) in graph

    @pytest.mark.unit
    def test_scenario_validation_valid(self) -> None:
        """Test validating a valid scenario."""
        # Arrange - Create a valid scenario
        scenario = HarborNetScenario(
            name="test",
            width=3,
            height=3,
            start=(0, 0),
            goal=(2, 2),
            obstacles={(1, 1)},
        )

        # Act - Validate the scenario
        errors = scenario.validate()

        # Assert - Verify no validation errors
        assert errors == []

    @pytest.mark.unit
    def test_scenario_validation_invalid_dimensions(self) -> None:
        """Test validating scenario with invalid dimensions."""
        # Arrange - Create scenario with invalid dimensions
        scenario = HarborNetScenario(
            name="test",
            width=0,
            height=3,
            start=(0, 0),
            goal=(2, 2),
        )

        # Act - Validate the scenario
        errors = scenario.validate()

        # Assert - Verify dimension error is caught
        assert "Grid dimensions must be positive" in errors

    @pytest.mark.unit
    def test_scenario_validation_invalid_start(self) -> None:
        """Test validating scenario with invalid start position."""
        # Arrange - Create scenario with start outside bounds
        scenario = HarborNetScenario(
            name="test",
            width=3,
            height=3,
            start=(5, 5),  # Outside bounds
            goal=(2, 2),
        )

        # Act - Validate the scenario
        errors = scenario.validate()

        # Assert - Verify start position error is caught
        assert "Start position must be within grid bounds" in errors

    @pytest.mark.unit
    def test_scenario_validation_invalid_goal(self) -> None:
        """Test validating scenario with invalid goal position."""
        # Arrange - Create scenario with goal outside bounds
        scenario = HarborNetScenario(
            name="test",
            width=3,
            height=3,
            start=(0, 0),
            goal=(5, 5),  # Outside bounds
        )

        # Act - Validate the scenario
        errors = scenario.validate()

        # Assert - Verify goal position error is caught
        assert "Goal position must be within grid bounds" in errors

    @pytest.mark.unit
    def test_scenario_validation_obstacle_outside_bounds(self) -> None:
        """Test validating scenario with obstacle outside bounds."""
        # Arrange - Create scenario with obstacle outside bounds
        scenario = HarborNetScenario(
            name="test",
            width=3,
            height=3,
            start=(0, 0),
            goal=(2, 2),
            obstacles={(5, 5)},  # Outside bounds
        )

        # Act - Validate the scenario
        errors = scenario.validate()

        # Assert - Verify obstacle bounds error is caught
        assert any(
            "Obstacle (5, 5) is outside grid bounds" in error for error in errors
        )

    @pytest.mark.unit
    def test_scenario_validation_start_on_obstacle(self) -> None:
        """Test validating scenario with start on obstacle."""
        # Arrange - Create scenario with start position on obstacle
        scenario = HarborNetScenario(
            name="test",
            width=3,
            height=3,
            start=(1, 1),
            goal=(2, 2),
            obstacles={(1, 1)},  # Start is on obstacle
        )

        # Act - Validate the scenario
        errors = scenario.validate()

        # Assert - Verify start on obstacle error is caught
        assert "Start position cannot be on an obstacle" in errors

    @pytest.mark.unit
    def test_scenario_validation_goal_on_obstacle(self) -> None:
        """Test validating scenario with goal on obstacle."""
        # Arrange - Create scenario with goal position on obstacle
        scenario = HarborNetScenario(
            name="test",
            width=3,
            height=3,
            start=(0, 0),
            goal=(1, 1),
            obstacles={(1, 1)},  # Goal is on obstacle
        )

        # Act - Validate the scenario
        errors = scenario.validate()

        # Assert - Verify goal on obstacle error is caught
        assert "Goal position cannot be on an obstacle" in errors


class TestLoadHarborScenario:
    """Test loading HarborNet scenarios."""

    @pytest.mark.unit
    def test_load_scenario_file_not_found(self) -> None:
        """Test loading a scenario from non-existent file."""
        # Arrange & Act & Assert - Test file not found error
        with pytest.raises(FileNotFoundError):
            load_harbor_scenario("nonexistent.yaml")

    @pytest.mark.unit
    def test_load_scenario_missing_required_field(self) -> None:
        """Test loading a scenario with missing required field."""
        # Arrange - Create YAML data missing required fields
        data = {
            "name": "test",
            "width": 3,
            "height": 3,
            # Missing start and goal
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            temp_path = f.name

        try:
            # Act & Assert - Test missing field error
            with pytest.raises(ValueError, match="Missing required field"):
                load_harbor_scenario(temp_path)

        finally:
            Path(temp_path).unlink()

    @pytest.mark.unit
    def test_load_scenario_invalid_data(self) -> None:
        """Test loading a scenario with invalid data."""
        # Arrange - Create YAML data with invalid goal position
        data = {
            "name": "test",
            "width": 3,
            "height": 3,
            "start": [0, 0],
            "goal": [5, 5],  # Outside bounds
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            temp_path = f.name

        try:
            # Act & Assert - Test invalid data error
            with pytest.raises(ValueError, match="Invalid scenario"):
                load_harbor_scenario(temp_path)

        finally:
            Path(temp_path).unlink()

    @pytest.mark.unit
    def test_load_valid_scenario(self) -> None:
        """Test loading a valid scenario."""
        # Arrange - Create valid YAML data
        data = {
            "name": "test_scenario",
            "width": 3,
            "height": 3,
            "start": [0, 0],
            "goal": [2, 2],
            "obstacles": [[1, 1]],
            "description": "Test scenario",
            "narrative": "A test narrative",
            "custom_field": "custom_value",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            temp_path = f.name

        try:
            # Act - Load the scenario from file
            scenario = load_harbor_scenario(temp_path)

            # Assert - Verify all scenario attributes are loaded correctly
            assert scenario.name == "test_scenario"
            assert scenario.width == 3
            assert scenario.height == 3
            assert scenario.start == (0, 0)
            assert scenario.goal == (2, 2)
            assert scenario.obstacles == {(1, 1)}
            assert scenario.description == "Test scenario"
            assert scenario.narrative == "A test narrative"
            assert hasattr(scenario, "custom_field")
            assert scenario.custom_field == "custom_value"

        finally:
            Path(temp_path).unlink()
