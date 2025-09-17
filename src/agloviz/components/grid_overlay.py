import numpy as np
from manim import (
    BLUE_C,
    BLUE_E,
    DOWN,
    PI,
    UP,
    WHITE,
    AnimationGroup,
    Arc,
    Indicate,
    Mobject,
    Square,
    Text,
    VGroup,
)


class GridOverlay(VGroup):
    """A utility class for overlaying a Manim grid on a scene.
    """

    def __init__(self, grid_width, grid_height, cell_size, grid_color, scenario, **kwargs):
        super().__init__(**kwargs)

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.grid_color = grid_color
        self.scenario = scenario
        self.grid_group = None
        self._grid_cells = []
        self.title = None

        # Create grid cells immediately in __init__ so they're positioned correctly
        self._create_grid()


    def _create_grid(self):
        """Create grid using Manim Square primitives for perfect squares (like original)."""
        grid_cells = []
        
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                # Create square cell using Manim primitive
                cell = Square(
                    side_length=self.cell_size,
                    color=self.grid_color,
                    stroke_width=2,
                    fill_opacity=0.0  # Transparent fill
                )
                
                # Position relative to this VGroup's center
                relative_pos = self.grid_to_screen((x, y))
                cell.move_to(relative_pos)
                grid_cells.append(cell)
        
        # Group all cells and add to self (like original)
        self.grid_group = VGroup(*grid_cells)
        self._grid_cells = grid_cells
        self.add(self.grid_group)

    def animate_grid_growth(self, scene) -> None:
        """Animate the existing grid cells with growth animation."""
        # Set all cells to small and rotated first
        for cell in self._grid_cells:
            cell.scale(0.1)
            cell.rotate(np.pi / 2)  # Half rotation
        
        # Animate growth with rotation back to normal
        scene.play(
            AnimationGroup(
                *[cell.animate.scale(10).rotate(-np.pi / 2) for cell in self._grid_cells],
                lag_ratio=0.05  # Slight stagger for wave effect
            ),
            run_time=1.5
        )

    def grid_to_screen(self, grid_pos: tuple[int, int]) -> np.ndarray:
        """Convert grid coordinates to screen coordinates relative to overlay center.
        
        Args:
            grid_pos: (x, y) grid coordinates
            
        Returns:
            Screen position as numpy array
        """
        x, y = grid_pos
        
        # Calculate position relative to overlay's current center using get_center()
        overlay_center = self.get_center()
        
        # Calculate grid origin relative to overlay center
        grid_origin_x = overlay_center[0] - (self.grid_width * self.cell_size) / 2
        grid_origin_y = overlay_center[1] + (self.grid_height * self.cell_size) / 2
        
        # Calculate cell center position
        cell_x = grid_origin_x + (x + 0.5) * self.cell_size
        cell_y = grid_origin_y - (y + 0.5) * self.cell_size
        
        return np.array([cell_x, cell_y, 0])


    def fill_cell(self, x: int, y: int, color=WHITE, opacity=0.5) -> None:
        """Fill a cell with color."""
        idx = y + x * self.grid_height
        if 0 <= idx < len(self._grid_cells):
            self._grid_cells[idx].set_fill(color=color, opacity=opacity)
        else:
            raise ValueError(f"Cell ({x}, {y}) not found in grid.")

    def highlight_cell(self, x: int, y: int, color=WHITE) -> AnimationGroup:
        """Return an animation that highlights a cell by indicating it."""
        idx = y + x * self.grid_height
        if 0 <= idx < len(self._grid_cells):
            return AnimationGroup(
                Indicate(self._grid_cells[idx], color=color)
            )
        else:
            raise ValueError(f"Cell ({x}, {y}) not found in grid.")

    def _get_cell_index(self, pos: tuple[int, int]) -> int:
        """Get the index of a cell in the grid_group VGroup (like original)."""
        x, y = pos
        return y * self.grid_width + x

    def add_to_cell(self, mobj: Mobject, grid_pos: tuple[int, int]) -> None:
        """Add a Manim object to a specific grid cell, centered within it (like original).
        
        Args:
            mobj: The Mobject to place.
            grid_pos: A tuple (x, y) of the grid cell position.
        """
        # Get the actual cell and use its center (like original)
        cell_index = self._get_cell_index(grid_pos)
        cell = self._grid_cells[cell_index]
        mobj.move_to(cell.get_center())
        self.add(mobj)

    def add_water_cell(self, x: int, y: int) -> tuple[Mobject, Mobject]:
        """Add water to a cell with wave effects (like original BFS scene).
        
        Args:
            x: Grid x coordinate
            y: Grid y coordinate
            
        Returns:
            Tuple of (cell, wave_group) for animation
        """
        # Get the cell
        cell_index = self._get_cell_index((x, y))
        cell = self._grid_cells[cell_index]
        
        # Fill with water color (like original)
        cell.set_fill(BLUE_E, opacity=0.65)
        
        # Create wave overlay (thin Arc pieces like original)
        wave_parts = []
        for i in range(3):  # 3 wave segments
            wave_arc = Arc(
                radius=self.cell_size * 0.15 + i * 0.02,
                angle=PI,
                color=BLUE_C,
                stroke_width=1
            ).move_to(cell.get_center())
            wave_parts.append(wave_arc)
        
        wave_group = VGroup(*wave_parts)
        self.add(wave_group)
        
        return cell, wave_group
