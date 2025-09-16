"""Shared Manim scene infrastructure for algorithm visualizations.

This module provides reusable base classes and utilities for creating
consistent, professional visualizations of algorithms using Manim.
"""

from typing import Any

import manim as m

from algokit.core.helpers import HarborNetScenario


class HarborGridScene(m.Scene):
    """Base scene for HarborNet grid-based algorithm visualizations.
    
    This scene provides a clean grid visualization using Manim primitives.
    """
    
    def __init__(self, scenario: HarborNetScenario, **kwargs: Any) -> None:
        """Initialize the HarborNet grid scene.
        
        Args:
            scenario: HarborNet scenario configuration
            **kwargs: Additional arguments passed to Scene
        """
        super().__init__(**kwargs)
        self.scenario = scenario
        self.grid_width = scenario.width
        self.grid_height = scenario.height
        self.start_pos = scenario.start
        self.goal_pos = scenario.goal
        self.obstacles = scenario.obstacles
        
        # Calculate optimal cell size for 16:9 aspect ratio
        self.cell_size = self._calculate_optimal_cell_size()
        
        # Pre-calculate cell centers for precise positioning
        self.cell_centers = self._calculate_cell_centers()
        
        # Visual configuration
        self.grid_color = m.GRAY_B
        
    def _calculate_optimal_cell_size(self) -> float:
        """Calculate optimal cell size for 16:9 aspect ratio.
        
        Returns:
            Optimal cell size for the grid
        """
        # Base cell size that works well for 16:9 aspect ratio
        base_cell_size = 0.8
        
        # Adjust for very large grids
        max_dimension = max(self.grid_width, self.grid_height)
        if max_dimension > 10:
            # Scale down for larger grids
            scale_factor = 10.0 / max_dimension
            return base_cell_size * scale_factor
        elif max_dimension < 5:
            # Scale up for smaller grids
            scale_factor = 5.0 / max_dimension
            return base_cell_size * scale_factor
        
        return base_cell_size
        
        
    def _calculate_cell_centers(self) -> dict[tuple[int, int], m.np.ndarray]:
        """Pre-calculate the center position of each grid cell.
        
        Returns:
            Dictionary mapping (x, y) grid coordinates to screen positions
        """
        cell_centers = {}
        
        # Calculate grid origin (center of grid)
        grid_origin_x = -(self.grid_width * self.cell_size) / 2
        grid_origin_y = (self.grid_height * self.cell_size) / 2
        
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                # Calculate cell center position
                cell_x = grid_origin_x + (x + 0.5) * self.cell_size
                cell_y = grid_origin_y - (y + 0.5) * self.cell_size
                cell_centers[(x, y)] = m.np.array([cell_x, cell_y, 0])
                
        return cell_centers
        
    def construct(self) -> None:
        """Construct the grid visualization."""
        # Create grid using Manim primitives
        self.create_grid()
        
    def create_grid(self) -> None:
        """Create grid using Manim Square primitives for perfect squares."""
        grid_cells = []
        
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                # Create square cell using Manim primitive
                cell = m.Square(
                    side_length=self.cell_size,
                    color=self.grid_color,
                    stroke_width=2,
                    fill_opacity=0.0  # Transparent fill
                )
                
                # Position using pre-calculated cell centers
                cell.move_to(self.grid_to_screen((x, y)))
                grid_cells.append(cell)
        
        # Group all cells and add to scene
        self.grid_group = m.VGroup(*grid_cells)
        self.add(self.grid_group)
        
    def create_grid_with_growth_animation(self) -> None:
        """Create grid with growth animation - cells start small and grow with rotation."""
        grid_cells = []
        
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                # Create square cell using Manim primitive
                cell = m.Square(
                    side_length=self.cell_size,
                    color=self.grid_color,
                    stroke_width=2,
                    fill_opacity=0.0  # Transparent fill
                )
                
                # Position using pre-calculated cell centers
                cell.move_to(self.grid_to_screen((x, y)))
                
                # Start small and rotated
                cell.scale(0.1)
                cell.rotate(m.PI / 2)  # Half rotation
                
                grid_cells.append(cell)
        
        # Group all cells
        self.grid_group = m.VGroup(*grid_cells)
        self.add(self.grid_group)
        
        # Animate growth with rotation back to normal
        self.play(
            m.AnimationGroup(
                *[cell.animate.scale(10).rotate(-m.PI / 2) for cell in grid_cells],
                lag_ratio=0.05  # Slight stagger for wave effect
            ),
            run_time=1.5
        )
        
        
    def grid_to_screen(self, grid_pos: tuple[int, int]) -> m.np.ndarray:
        """Convert grid coordinates to screen coordinates using pre-calculated centers.
        
        Args:
            grid_pos: (x, y) grid coordinates
            
        Returns:
            Screen position as numpy array
        """
        # Use pre-calculated cell centers for precise positioning
        if grid_pos in self.cell_centers:
            return self.cell_centers[grid_pos]
        else:
            # Fallback for invalid coordinates
            return m.np.array([0, 0, 0])
        
        