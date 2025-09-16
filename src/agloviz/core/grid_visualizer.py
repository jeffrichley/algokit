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



