from manim import BLUE_C, LEFT, ORIGIN, WHITE, Scene

from agloviz.components.grid_overlay import GridOverlay
from agloviz.components.hud_panel import HUDPanel
from agloviz.config.scenario import Scenario


class TestGridHudScene(Scene):
    def construct(self) -> None:
        # 🎯 Create and display the smart grid
        grid = GridOverlay(
            grid_width=10,
            grid_height=6,
            cell_size=0.6,
            grid_color=WHITE,
            scenario=Scenario(name="Test"),
        )
        grid.move_to(ORIGIN)
        self.add(grid)

        # Fill a cell and highlight it
        grid.fill_cell(3, 2, color=BLUE_C, opacity=0.6)
        self.play(grid.highlight_cell(3, 2), run_time=0.5)

        # 🎯 Create and show the HUD panel
        hud = HUDPanel(
            values={"Visited": 0, "Frontier": 1, "Depth": 0, "Queue": 1}, max_lines=2
        )
        hud.to_edge(LEFT, buff=0.4)
        self.add(hud)

        # Animate some stat updates
        self.wait(1)
        hud.update_values({"Visited": 2, "Queue": 3}, scene=self)

        self.wait(1)
        hud.update_values({"Depth": 1, "Frontier": 4}, scene=self)

        self.wait(2)
