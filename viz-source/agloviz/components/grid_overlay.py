from typing import Any

import numpy as np
from manim import (
    BLUE_C,
    BLUE_E,
    PI,
    WHITE,
    AnimationGroup,
    Arc,
    Indicate,
    Mobject,
    Square,
    VGroup,
)
from manim.typing import Color


class GridOverlay(VGroup):
    """A utility class for overlaying a Manim grid on a scene."""

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        cell_size: float,
        grid_color: Color,
        scenario: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.grid_color = grid_color
        self.scenario = scenario
        self.grid_group = None
        self._grid_cells: list[Any] = []
        self.title = None

        # Create grid cells immediately in __init__ so they're positioned correctly
        self._create_grid()

    def _create_grid(self) -> None:
        """Create grid using Manim Square primitives for perfect squares (like original)."""
        grid_cells = []

        for x in range(self.grid_width):
            for y in range(self.grid_height):
                # Create square cell using Manim primitive
                cell = Square(
                    side_length=self.cell_size,
                    color=self.grid_color,
                    stroke_width=2,
                    fill_opacity=0.0,  # Transparent fill
                )

                # Position relative to this VGroup's center
                relative_pos = self.grid_to_screen((x, y))
                cell.move_to(relative_pos)
                grid_cells.append(cell)

        # Group all cells and add to self (like original)
        self.grid_group = VGroup(*grid_cells)
        self._grid_cells = grid_cells
        self.add(self.grid_group)

    def animate_grid_growth(self, scene: Any) -> None:
        """Animate the existing grid cells with growth animation."""
        # Set all cells to small and rotated first
        for cell in self._grid_cells:
            cell.scale(0.1)
            cell.rotate(np.pi / 2)  # Half rotation

        # Animate growth with rotation back to normal
        scene.play(
            AnimationGroup(
                *[
                    cell.animate.scale(10).rotate(-np.pi / 2)
                    for cell in self._grid_cells
                ],
                lag_ratio=0.05,  # Slight stagger for wave effect
            ),
            run_time=1.5,
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

    def fill_cell(
        self, x: int, y: int, color: Color = WHITE, opacity: float = 0.5
    ) -> Square:
        """Fill a cell with color by creating an overlay fill cell.

        This creates a new Square with fill that overlays the original grid cell,
        preserving the original cell's border completely.

        Returns:
            The created fill cell for animation purposes
        """
        idx = self._get_cell_index((x, y))
        if 0 <= idx < len(self._grid_cells):
            original_cell = self._grid_cells[idx]

            # Create a new fill cell identical to the original but with fill
            fill_cell = Square(
                side_length=self.cell_size,
                color=color,
                stroke_width=0,  # No border on fill cell
                fill_opacity=opacity,
                fill_color=color,
            )

            # Position it exactly on top of the original cell
            fill_cell.move_to(original_cell.get_center())

            # Add it to the overlay so it draws over the original
            self.add(fill_cell)

            return fill_cell
        else:
            raise ValueError(f"Cell ({x}, {y}) not found in grid.")

    def highlight_cell(self, x: int, y: int, color: Color = WHITE) -> AnimationGroup:
        """Return an animation that highlights a cell by indicating it."""
        idx = self._get_cell_index((x, y))
        if 0 <= idx < len(self._grid_cells):
            return AnimationGroup(Indicate(self._grid_cells[idx], color=color))
        else:
            raise ValueError(f"Cell ({x}, {y}) not found in grid.")

    def _get_cell_index(self, pos: tuple[int, int]) -> int:
        """Get the index of a cell in the grid_group VGroup.

        Must match the creation order: for x in range(width): for y in range(height)
        So index = x * height + y
        """
        x, y = pos
        return x * self.grid_height + y

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
        """Add water to a cell with wave effects using overlay approach.

        Creates a water fill cell that overlays the original grid cell,
        preserving the original cell's border completely.

        Args:
            x: Grid x coordinate
            y: Grid y coordinate

        Returns:
            Tuple of (water_cell, wave_group) for animation
        """
        # Get the original cell for positioning reference
        cell_index = self._get_cell_index((x, y))
        original_cell = self._grid_cells[cell_index]

        # Create water fill cell using the fill_cell method
        water_cell = self.fill_cell(x, y, color=BLUE_E, opacity=0.65)

        # Create wave overlay (thin Arc pieces like original)
        wave_parts = []
        for i in range(3):  # 3 wave segments
            wave_arc = Arc(
                radius=self.cell_size * 0.15 + i * 0.02,
                angle=PI,
                color=BLUE_C,
                stroke_width=1,
            ).move_to(original_cell.get_center())
            wave_parts.append(wave_arc)

        wave_group = VGroup(*wave_parts)
        self.add(wave_group)

        return water_cell, wave_group
