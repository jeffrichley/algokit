"""Unit tests for SnakeGrid component."""

import manim as m  # type: ignore[import-untyped]
import numpy as np
import pytest
from agloviz.components.snake_grid import SnakeGrid


@pytest.mark.unit
def test_snake_grid_initialization() -> None:
    """Test that SnakeGrid initializes with correct dimensions."""
    # Arrange - Set up grid parameters for testing
    width, height = 5, 3
    cell_size = 0.6
    
    # Act - Create the SnakeGrid instance
    grid = SnakeGrid(width=width, height=height, cell_size=cell_size)
    
    # Assert - Verify all dimensions and cell counts are correct
    assert grid.grid_width == width
    assert grid.grid_height == height
    assert grid.cell_size == cell_size
    assert len(grid.cells) == width * height
    assert len(grid.centers) == width * height


@pytest.mark.unit
def test_get_cell_returns_correct_cell() -> None:
    """Test that get_cell returns the correct cell for valid coordinates."""
    # Arrange
    grid = SnakeGrid(width=3, height=3)
    
    # Act
    cell = grid.get_cell(1, 1)
    
    # Assert
    assert isinstance(cell, m.Square)
    assert (1, 1) in grid.cells
    assert grid.cells[(1, 1)] is cell


@pytest.mark.unit
def test_get_cell_raises_on_invalid_coordinates() -> None:
    """Test that get_cell raises ValueError for out-of-bounds coordinates."""
    # Arrange
    grid = SnakeGrid(width=3, height=3)
    
    # Act & Assert
    with pytest.raises(ValueError, match="Coordinates \\(-1, 0\\) out of bounds"):
        grid.get_cell(-1, 0)
    
    with pytest.raises(ValueError, match="Coordinates \\(0, 5\\) out of bounds"):
        grid.get_cell(0, 5)


@pytest.mark.unit
def test_get_center_returns_correct_position() -> None:
    """Test that get_center returns correct screen position."""
    # Arrange
    grid = SnakeGrid(width=2, height=2, cell_size=1.0)
    
    # Act
    center = grid.get_center(0, 0)
    
    # Assert
    assert isinstance(center, np.ndarray)
    assert len(center) == 3
    # Bottom-left should be negative x, negative y
    assert center[0] < 0  # x coordinate
    assert center[1] < 0  # y coordinate


@pytest.mark.unit
def test_get_center_raises_on_invalid_coordinates() -> None:
    """Test that get_center raises ValueError for out-of-bounds coordinates."""
    # Arrange
    grid = SnakeGrid(width=2, height=2)
    
    # Act & Assert
    with pytest.raises(ValueError, match="Coordinates \\(3, 3\\) out of bounds"):
        grid.get_center(3, 3)


@pytest.mark.unit
def test_get_dimensions_returns_correct_values() -> None:
    """Test that get_dimensions returns correct width and height."""
    # Arrange
    grid = SnakeGrid(width=7, height=4)
    
    # Act
    width, height = grid.get_dimensions()
    
    # Assert
    assert width == 7
    assert height == 4


@pytest.mark.unit
def test_get_cell_size_returns_correct_value() -> None:
    """Test that get_cell_size returns correct cell size."""
    # Arrange
    cell_size = 0.8
    grid = SnakeGrid(width=3, height=3, cell_size=cell_size)
    
    # Act
    result = grid.get_cell_size()
    
    # Assert
    assert result == cell_size


@pytest.mark.unit
def test_is_empty_returns_false_after_initialization() -> None:
    """Test that grid is not empty after initialization."""
    # Arrange
    grid = SnakeGrid(width=2, height=2)
    
    # Act
    is_empty = grid.is_empty()
    
    # Assert
    assert not is_empty  # Grid should have cells


@pytest.mark.unit
def test_clear_all_fills_sets_opacity_to_zero() -> None:
    """Test that clear_all_fills sets all cell fill opacity to zero."""
    # Arrange
    grid = SnakeGrid(width=2, height=2)
    # Set some cells to have fill
    grid.cells[(0, 0)].set_fill(color=m.RED, opacity=0.5)
    grid.cells[(1, 1)].set_fill(color=m.BLUE, opacity=0.8)
    
    # Act
    grid.clear_all_fills()
    
    # Assert
    for cell in grid.cells.values():
        assert cell.fill_opacity == 0.0


@pytest.mark.unit
def test_get_layer_cells_returns_correct_cells() -> None:
    """Test that get_layer_cells returns cells at correct Manhattan distance."""
    # Arrange
    grid = SnakeGrid(width=5, height=5)
    center = (2, 2)  # Center of 5x5 grid
    
    # Act
    depth_1_cells = grid.get_layer_cells(1, center)
    depth_2_cells = grid.get_layer_cells(2, center)
    
    # Assert
    # Depth 1 should have 4 cells (adjacent to center)
    assert len(depth_1_cells) == 4
    # Depth 2 should have 8 cells (diagonal neighbors)
    assert len(depth_2_cells) == 8


@pytest.mark.unit
def test_animate_cell_fill_returns_animation() -> None:
    """Test that animate_cell_fill returns a valid animation."""
    # Arrange
    grid = SnakeGrid(width=3, height=3)
    
    # Act
    animation = grid.animate_cell_fill(1, 1, m.RED, 0.7)
    
    # Assert
    assert isinstance(animation, m.Animation)


@pytest.mark.unit
def test_highlight_cell_returns_animation() -> None:
    """Test that highlight_cell returns a valid animation."""
    # Arrange
    grid = SnakeGrid(width=3, height=3)
    
    # Act
    animation = grid.highlight_cell(1, 1, m.YELLOW, 1.1)
    
    # Assert
    assert isinstance(animation, m.Animation)


@pytest.mark.unit
def test_wave_shimmer_returns_animation() -> None:
    """Test that wave_shimmer returns a valid animation."""
    # Arrange
    grid = SnakeGrid(width=3, height=3)
    
    # Act
    animation = grid.wave_shimmer()
    
    # Assert
    assert isinstance(animation, m.Animation)


@pytest.mark.unit
def test_add_obstacle_returns_animations() -> None:
    """Test that add_obstacle returns a list of animations."""
    # Arrange
    grid = SnakeGrid(width=3, height=3)
    
    # Act
    animations = grid.add_obstacle(1, 1)
    
    # Assert
    assert isinstance(animations, list)
    assert len(animations) > 0
    for animation in animations:
        assert isinstance(animation, m.Animation)


@pytest.mark.unit
def test_add_ring_overlay_returns_vmobject() -> None:
    """Test that add_ring_overlay returns a valid VMobject."""
    # Arrange
    grid = SnakeGrid(width=5, height=5)
    
    # Act
    ring = grid.add_ring_overlay(2)
    
    # Assert
    assert isinstance(ring, m.VMobject)
