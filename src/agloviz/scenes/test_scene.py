from manim import *
from agloviz.components.hud_panel import HUDPanel
from agloviz.config.scenario import Scenario
from agloviz.components.grid_overlay import GridOverlay
from agloviz.components.tokens import StartToken, GoalToken
from agloviz.core.grid_visualizer import GridVisualizer


class TestScene(Scene):
    def construct(self):
        # 🧩 1. Setup scenario (placeholder)
        scenario = Scenario(name="BFS Demo")

        # 🧱 2. Create grid overlay and move to center
        grid = GridOverlay(
            grid_width=8,
            grid_height=6,
            cell_size=0.8,
            grid_color=WHITE,
            scenario=scenario
        )
        grid.move_to(ORIGIN)
        
        # Create grid with growth animation (like original)
        grid.create_grid_with_growth_animation(self)
        
        # Add grid shimmer effect (like original)
        self.play(ApplyWave(grid.grid_group, amplitude=0.05), run_time=0.8)

        # 🎨 3. Create the visualizer to manage placements
        visualizer = GridVisualizer(grid_overlay=grid)

        # 🏁 4. Create tokens (unpositioned)
        start_token = StartToken(cell_size=grid.cell_size)
        goal_token = GoalToken(cell_size=grid.cell_size)

        # 🧲 5. Place tokens using the visualizer
        visualizer.place_token(start_token, (0, 0))
        visualizer.place_token(goal_token, (7, 5))
        
        # Position labels relative to tokens after they're placed (like original)
        start_token.add_label_after_placement()
        goal_token.add_label_after_placement()
        
        # Animate token entrances (like original)
        self.play(GrowFromCenter(start_token), run_time=0.5)
        self.play(Write(start_token.label), run_time=0.3)
        self.play(GrowFromCenter(goal_token), run_time=0.5)
        self.play(Write(goal_token.label), run_time=0.3)

        # 🔷 6. Add water cell with wave effects (like original)
        water_cell, wave_group = grid.add_water_cell(3, 2)
        
        # Animate water with ripple effect (like original)
        self.play(GrowFromCenter(water_cell), run_time=0.5)
        self.play(GrowFromCenter(wave_group), run_time=0.4)

        # 📊 7. HUD display
        hud = HUDPanel(
            values={"Visited": 0, "Frontier": 0, "Depth": 0, "Queue": 1},
            max_lines=2
        )
        hud.to_edge(LEFT, buff=0.4)
        self.add(hud)

        # ⏱️ 8. Update HUD over time
        self.wait(1)
        hud.update_values({"Visited": 2, "Frontier": 3, "Queue": 3}, scene=self)
        self.wait(1)
        hud.update_values({"Depth": 1, "Frontier": 5}, scene=self)

        self.wait(2)
