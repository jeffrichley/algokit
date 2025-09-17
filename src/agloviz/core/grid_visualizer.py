from manim import *

from agloviz.components.grid_overlay import GridOverlay


class GridVisualizer:
    def __init__(self, grid_overlay: GridOverlay):
        self.grid_overlay = grid_overlay
        self.tokens: dict[tuple[int, int], Mobject] = {}

    def fill_cell(self, x, y, color=BLUE, opacity=1.0):
        """Permanently fill a cell with color"""
        self.grid_overlay.fill_cell(x, y, color, opacity)

    def highlight_cell(self, x, y, color=YELLOW, duration=0.2):
        """Temporarily flash a cell with a color"""
        self.grid_overlay.highlight_cell(x, y, color)

    def place_token(self, token: Mobject, grid_pos: tuple[int, int]) -> None:
        self.grid_overlay.add_to_cell(token, grid_pos)
        self.tokens[grid_pos] = token
        
    def get_cell(self, grid_pos: tuple[int, int]) -> Square:
        """Get the actual cell object at grid position."""
        cell_index = self.grid_overlay._get_cell_index(grid_pos)
        return self.grid_overlay._grid_cells[cell_index]
        
    def get_cell_center(self, grid_pos: tuple[int, int]):
        """Get the center position of a cell."""
        return self.grid_overlay.grid_to_screen(grid_pos)
        
    def set_cell_fill(self, grid_pos: tuple[int, int], color, opacity: float):
        """Set fill color and opacity for a cell while preserving stroke.
        
        Returns:
            The created fill cell for animation purposes
        """
        return self.grid_overlay.fill_cell(grid_pos[0], grid_pos[1], color=color, opacity=opacity)
        
    def fill_cell(self, x: int, y: int, color, opacity: float):
        """Fill a cell with color while preserving stroke (convenience method).
        
        Returns:
            The created fill cell for animation purposes
        """
        return self.grid_overlay.fill_cell(x, y, color=color, opacity=opacity)
        
    def set_cell_stroke(self, grid_pos: tuple[int, int], color, width: float) -> None:
        """Set stroke color and width for a cell."""
        cell = self.get_cell(grid_pos)
        cell.set_stroke(color, width=width)
        
    def set_cell_opacity(self, grid_pos: tuple[int, int], opacity: float) -> None:
        """Set overall opacity for a cell."""
        cell = self.get_cell(grid_pos)
        cell.set_opacity(opacity)
        
    def get_grid_group(self) -> VGroup:
        """Get the grid group for operations that need the full grid."""
        return self.grid_overlay.grid_group
        
    def create_grid(self) -> None:
        """Create the grid using GridOverlay."""
        self.grid_overlay._create_grid()
        
    def animate_grid_growth(self, scene) -> None:
        """Animate grid growth using GridOverlay."""
        self.grid_overlay.animate_grid_growth(scene)
        
    def add_water_cell(self, x: int, y: int) -> tuple:
        """Add water to a cell with wave effects, ensuring proper positioning.
        
        Args:
            x: Grid x coordinate
            y: Grid y coordinate
            
        Returns:
            Tuple of (cell, wave_group) for animation
        """
        return self.grid_overlay.add_water_cell(x, y)
        



