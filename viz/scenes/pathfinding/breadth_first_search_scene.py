"""BFS-specific Manim visualization scene.

BFS algorithm visualization using HarborGridScene base class.
Follows the video plan from bls_video.md.
"""

import sys
from pathlib import Path
from typing import Any, List, Tuple

import manim as m

# Add the main project to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from algokit.core.helpers import HarborNetScenario, load_harbor_scenario
    from algokit.viz.scenes import HarborGridScene
    from agloviz.components.snaking_queue import SnakingQueue
except ImportError as e:
    print(f"Warning: Could not import Algokit modules: {e}")
    print("BFS scene will run in standalone mode with placeholder data.")
    
    # Create minimal placeholder classes
    class HarborNetScenario:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class HarborGridScene(m.Scene):
        def __init__(self, scenario, **kwargs):
            super().__init__(**kwargs)
            self.scenario = scenario
        
        def construct(self):
            # Placeholder implementation
            text = m.Text("BFS Scene - Standalone Mode", font_size=48)
            self.play(m.Write(text))
            self.wait(2)


class BreadthFirstSearchScene(HarborGridScene):
    """BFS algorithm visualization using HarborGridScene base class.
    
    Implements the BFS algorithm visualization following bls_video.md:
    - Scene 0: Title + Grid establishment
    - Scene 1: Start/Goal/Obstacles placement
    - Scene 2: Queue panel + HUD
    - Scene 3: BFS wavefront exploration
    - Scene 6: Path reconstruction
    """
    
    def __init__(self, scenario=None, **kwargs: Any) -> None:
        """Initialize the BFS scene.
        
        Args:
            scenario: HarborNet scenario configuration or scenario file path
            **kwargs: Additional arguments passed to HarborGridScene
        """
        # Load scenario - prioritize direct parameter, then environment variable
        if scenario is None:
            # Get scenario from environment variable (set by render_scene)
            import os
            scenario_file = os.environ.get("AGLOVIZ_SCENARIO_FILE")
            
            if scenario_file:
                scenario = self._load_scenario_from_file(scenario_file)
            else:
                # Create default scenario if no file provided
                scenario = HarborNetScenario(
                    name="Small Flooded Harbor",
                    description="Default scenario for BFS visualization",
                    width=8, height=6,
                    start=[0, 0], goal=[7, 5],
                    obstacles=[(2, 2), (3, 2), (4, 2), (2, 3), (4, 3), (5, 4), (6, 4)]
                )
        elif isinstance(scenario, str):
            # Load scenario from file path
            scenario = self._load_scenario_from_file(scenario)
        
        super().__init__(scenario, **kwargs)
        
    def _load_scenario_from_file(self, scenario_file: str) -> HarborNetScenario:
        """Load scenario from file with proper error handling.
        
        Args:
            scenario_file: Path to scenario file
            
        Returns:
            Loaded HarborNetScenario object
        """
        try:
            return load_harbor_scenario(scenario_file)
        except ImportError:
            # Fallback: create basic scenario from file
            import yaml
            with open(scenario_file) as f:
                data = yaml.safe_load(f)
            return HarborNetScenario(**data)


class BreadthFirstSearchScene(HarborGridScene):
    """BFS algorithm visualization using HarborGridScene base class.
    
    Implements the BFS algorithm visualization following bls_video.md:
    - Scene 0: Title + Grid establishment
    - Scene 1: Start/Goal/Obstacles placement
    - Scene 2: Queue panel + HUD
    - Scene 3: BFS wavefront exploration
    - Scene 6: Path reconstruction
    """
    
    def __init__(self, scenario=None, **kwargs: Any) -> None:
        """Initialize the BFS scene.
        
        Args:
            scenario: HarborNet scenario configuration or scenario file path
            **kwargs: Additional arguments passed to HarborGridScene
        """
        # Load scenario - prioritize direct parameter, then environment variable
        if scenario is None:
            # Get scenario from environment variable (set by render_scene)
            import os
            scenario_file = os.environ.get("AGLOVIZ_SCENARIO_FILE")
            
            if scenario_file:
                scenario = self._load_scenario_from_file(scenario_file)
            else:
                # Create default scenario if no file provided
                scenario = HarborNetScenario(
                    name="Small Flooded Harbor",
                    description="Default scenario for BFS visualization",
                    width=8, height=6,
                    start=[0, 0], goal=[7, 5],
                    obstacles=[(2, 2), (3, 2), (4, 2), (2, 3), (4, 3), (5, 4), (6, 4)]
                )
        elif isinstance(scenario, str):
            # Load scenario from file path
            scenario = self._load_scenario_from_file(scenario)
        
        super().__init__(scenario, **kwargs)
        
    def _load_scenario_from_file(self, scenario_file: str) -> HarborNetScenario:
        """Load scenario from file with proper error handling.
        
        Args:
            scenario_file: Path to scenario file
            
        Returns:
            Loaded HarborNetScenario object
        """
        try:
            return load_harbor_scenario(scenario_file)
        except ImportError:
            # Fallback: create basic scenario from file
            import yaml
            with open(scenario_file) as f:
                data = yaml.safe_load(f)
            return HarborNetScenario(**data)
        
    def construct(self) -> None:
        """Construct the BFS visualization following bls_video.md."""
        # Scene 0: Title + Grid establishment
        self._establish_scene()
        
        # Scene 1: Cast the actors
        self._cast_actors()
        
        # Scene 2: Meet the Queue
        self._meet_the_queue()
        
        # For now, just show the grid
        self.wait(2)
        
    def _establish_scene(self) -> None:
        """Scene 0: Title + Grid establishment (from bls_video.md)."""
        # 1. Title + grid on
        title = m.Text(
            f"BFS Grid: {self.scenario.name}",
            font_size=36,
            color=m.WHITE
        ).to_edge(m.UP)
        
        # Create grid with growth animation
        self.create_grid_with_growth_animation()
        
        # Play title + grid together (as prescribed)
        self.play(m.Write(title), run_time=1.8)
        
        # Polish: subtle grid shimmer
        self.play(m.ApplyWave(self.grid_group, amplitude=0.05), run_time=0.8)
        
        # 2. Subtitle + legend (1.5s)
        subtitle = m.Text(
            "Shortest paths on unweighted graphs",
            font_size=20,
            color=m.GRAY
        ).next_to(title, m.DOWN, buff=0.2)
        
        # Build legend
        legend = self._build_legend()
        
        # Animate subtitle + legend together with LaggedStart
        self.play(
            m.LaggedStart(
                m.Write(subtitle),
                m.FadeIn(legend, shift=m.DOWN * 0.25),
                lag_ratio=0.2
            ),
            run_time=1.5
        )
        
        # 3. Frame the stage (0.8s)
        frame_rect = m.SurroundingRectangle(
            self.grid_group,
            color=m.GRAY_D,
            stroke_width=2
        )
        
        # Fade in frame, then fade opacity to 30%
        self.play(m.FadeIn(frame_rect), run_time=0.4)
        self.play(frame_rect.animate.set_stroke(opacity=0.3), run_time=0.4)
        
    def _build_legend(self) -> m.VGroup:
        """Build bottom-right legend with visual swatches."""
        # Legend background (aligned under queue)
        legend_bg = m.RoundedRectangle(
            width=2.5,
            height=1.8,
            corner_radius=0.1,
            stroke_color=m.WHITE,
            stroke_width=1,
            fill_color=m.BLACK,
            fill_opacity=0.8
        ).to_edge(m.RIGHT, buff=0.5).shift(m.DOWN * 1.2)
        
        # Legend items
        items = []
        
        # Start (green dot)
        start_dot = m.Dot(radius=0.08, color=m.GREEN_C)
        start_text = m.Text("Start", font_size=16, color=m.WHITE)
        start_group = m.VGroup(start_dot, start_text).arrange(m.RIGHT, buff=0.1)
        
        # Goal (gold star)
        goal_star = m.Star(n=5, outer_radius=0.08, color=m.GOLD)
        goal_text = m.Text("Goal", font_size=16, color=m.WHITE)
        goal_group = m.VGroup(goal_star, goal_text).arrange(m.RIGHT, buff=0.1)
        
        # Water (blue square)
        water_square = m.Square(side_length=0.15, fill_color=m.BLUE_E, fill_opacity=0.65, stroke_width=1)
        water_text = m.Text("Water", font_size=16, color=m.WHITE)
        water_group = m.VGroup(water_square, water_text).arrange(m.RIGHT, buff=0.1)
        
        # Frontier (blue square)
        frontier_square = m.Square(side_length=0.15, fill_color=m.BLUE_C, fill_opacity=0.6, stroke_width=1)
        frontier_text = m.Text("Frontier", font_size=16, color=m.WHITE)
        frontier_group = m.VGroup(frontier_square, frontier_text).arrange(m.RIGHT, buff=0.1)
        
        # Visited (yellow square)
        visited_square = m.Square(side_length=0.15, fill_color=m.YELLOW_E, fill_opacity=0.4, stroke_width=1)
        visited_text = m.Text("Visited", font_size=16, color=m.WHITE)
        visited_group = m.VGroup(visited_square, visited_text).arrange(m.RIGHT, buff=0.1)
        
        # Arrange all items vertically
        all_items = [start_group, goal_group, water_group, frontier_group, visited_group]
        legend_items = m.VGroup(*all_items).arrange(m.DOWN, buff=0.15, aligned_edge=m.LEFT)
        legend_items.move_to(legend_bg.get_center())
        
        return m.VGroup(legend_bg, legend_items)
        
    def _cast_actors(self) -> None:
        """Scene 1: Cast the actors (3–4s) - Start + Goal + Obstacles."""
        # 4. Start + Goal placement
        self._place_start_goal()
        
        # 5. Flooded harbor obstacles
        self._animate_obstacles()
        
    def _place_start_goal(self) -> None:
        """Place start and goal markers with animations."""
        # Start token (green dot)
        start_cell = self.grid_group[self._get_cell_index(self.start_pos)]
        self.start_token = m.Dot(
            radius=self.cell_size * 0.3,
            color=m.GREEN_C
        ).move_to(start_cell.get_center())
        
        # Goal star (gold)
        goal_cell = self.grid_group[self._get_cell_index(self.goal_pos)]
        self.goal_star = m.Star(
            n=5,
            outer_radius=self.cell_size * 0.3,
            color=m.GOLD
        ).move_to(goal_cell.get_center())
        
        # Optional labels
        start_label = m.Text("Start", font_size=14, color=m.WHITE)
        start_label.next_to(self.start_token, m.UP, buff=0.25)
        
        goal_label = m.Text("Goal", font_size=14, color=m.WHITE)
        goal_label.next_to(goal_cell, m.DOWN, buff=0.1)
        
        # Animate start placement
        self.play(m.GrowFromCenter(self.start_token), run_time=0.5)
        self.play(m.Indicate(start_cell), run_time=0.3)
        self.play(m.Write(start_label), run_time=0.3)
        
        # Animate goal placement
        self.play(m.GrowFromCenter(self.goal_star), run_time=0.5)
        self.play(m.Flash(self.goal_star.get_center()), run_time=0.3)
        self.play(m.Write(goal_label), run_time=0.3)
        
    def _animate_obstacles(self) -> None:
        """Animate flooded harbor obstacles with wave effects."""
        obstacle_animations = []
        wave_animations = []
        
        for obstacle_pos in self.obstacles:
            # Get the cell for this obstacle
            cell_index = self._get_cell_index(obstacle_pos)
            cell = self.grid_group[cell_index]
            
            # Fill with water color
            cell.set_fill(m.BLUE_E, opacity=0.65)
            
            # Create wave overlay (thin Arc pieces)
            wave_parts = []
            for i in range(3):  # 3 wave segments
                wave_arc = m.Arc(
                    radius=self.cell_size * 0.15 + i * 0.02,
                    angle=m.PI,
                    color=m.BLUE_C,
                    stroke_width=1
                ).move_to(cell.get_center())
                wave_parts.append(wave_arc)
            
            wave_group = m.VGroup(*wave_parts)
            
            # Add to animation lists
            obstacle_animations.append(m.GrowFromCenter(cell))
            wave_animations.append(m.GrowFromCenter(wave_group))
        
        # Animate obstacles with ripple effect (lag_ratio=0.03 for fast ripple)
        self.play(
            m.LaggedStart(*obstacle_animations, lag_ratio=0.03),
            run_time=1.0
        )
        self.play(
            m.LaggedStart(*wave_animations, lag_ratio=0.03),
            run_time=0.8
        )
        
    def _get_cell_index(self, pos: tuple[int, int]) -> int:
        """Get the index of a cell in the grid_group VGroup."""
        x, y = pos
        return y * self.grid_width + x
        
    def _meet_the_queue(self) -> None:
        """Scene 2: Meet the Queue (2s) - Queue panel + HUD + Enqueue start."""
        # 6. Queue panel enters
        self._build_queue_panel()
        
        # 7. HUD counters
        self._build_hud()
        
        # Enqueue the start token
        self._enqueue_start_token()
        
    def _build_queue_panel(self) -> None:
        """Build queue visualization panel on the right side."""
        # Queue panel background (moved higher to avoid legend)
        self.queue_panel = m.RoundedRectangle(
            width=2.5,
            height=3.0,
            corner_radius=0.2,
            stroke_color=m.WHITE,
            stroke_width=2,
            fill_color=m.BLACK,
            fill_opacity=0.8
        ).to_edge(m.RIGHT, buff=0.5).shift(m.UP * 0.8)
        
        # Queue title
        self.queue_title = m.Text(
            "Queue (FIFO)",
            font_size=18,
            color=m.WHITE
        ).next_to(self.queue_panel, m.UP, buff=0.2)
        
        # Create snaking queue
        self.snaking_queue = SnakingQueue(
            panel_width=2.5,
            panel_height=3.0,
            token_size=0.25
        )
        self.snaking_queue.move_to(self.queue_panel.get_center())
        
        # Animate queue panel sliding in
        self.play(
            m.FadeIn(self.queue_panel, shift=m.RIGHT * 0.5),
            m.Write(self.queue_title),
            run_time=1.0
        )
        
        # Add snaking queue to scene
        self.add(self.snaking_queue)
        
    def _build_hud(self) -> None:
        """Build HUD counters in top-left with 2 lines."""
        # HUD background (taller for 2 lines, positioned halfway down screen)
        self.hud_bg = m.RoundedRectangle(
            width=3.2,
            height=2.0,
            corner_radius=0.1,
            stroke_color=m.WHITE,
            stroke_width=1,
            fill_color=m.BLACK,
            fill_opacity=0.6
        ).to_edge(m.LEFT, buff=0.3).shift(m.DOWN * 2.0)
        
        # HUD text split into 2 lines
        self.hud_line1 = m.Text(
            "Visited: 0  Frontier: 0",
            font_size=14,
            color=m.WHITE
        ).move_to(self.hud_bg.get_center() + m.UP * 0.3)
        
        self.hud_line2 = m.Text(
            "Depth: 0  Queue: 0",
            font_size=14,
            color=m.WHITE
        ).move_to(self.hud_bg.get_center() + m.DOWN * 0.3)
        
        # Animate HUD sliding in
        self.play(
            m.FadeIn(self.hud_bg, shift=m.UP * 0.2),
            m.Write(self.hud_line1),
            m.Write(self.hud_line2),
            run_time=0.8
        )
        
    def _enqueue_start_token(self) -> None:
        """Enqueue the start token into the snaking queue with morphing animation."""
        # Create token at start position (as a circle initially)
        start_token = m.Dot(
            radius=self.cell_size * 0.3,
            color=m.GREEN_C
        ).move_to(self.start_token.get_center())
        
        # Use the snaking queue's enqueue method to get properly positioned token
        # First, temporarily add the token to the queue to get the correct position
        temp_token = self.snaking_queue.enqueue(color=m.GREEN_C, scene=None)
        
        # Create the target token that will be transformed to
        target_token = m.Square(
            side_length=0.25,
            fill_color=m.GREEN_C,
            fill_opacity=0.9,
            stroke_color=m.WHITE,
            stroke_width=2
        ).move_to(temp_token.get_center())
        
        # Remove the temp token and add our properly positioned one
        self.snaking_queue.remove(temp_token)
        self.snaking_queue.tokens.pop()  # Remove from internal storage
        
        # Animate morphing and movement
        self.play(
            m.Transform(start_token, target_token),
            run_time=0.8
        )
        
        # Add the final token to the queue
        self.snaking_queue.tokens.append(target_token)
        self.snaking_queue.add(target_token)
        
        # Update HUD to show queue has 1 item
        self._update_hud(visited=0, frontier=1, depth=0, queue=1)
        
    def _update_hud(self, visited: int, frontier: int, depth: int, queue: int) -> None:
        """Update HUD display with new values."""
        new_line1 = f"Visited: {visited}  Frontier: {frontier}"
        new_line2 = f"Depth: {depth}  Queue: {queue}"
        
        self.play(
            m.Transform(
                self.hud_line1,
                m.Text(new_line1, font_size=14, color=m.WHITE).move_to(self.hud_bg.get_center() + m.UP * 0.3)
            ),
            m.Transform(
                self.hud_line2,
                m.Text(new_line2, font_size=14, color=m.WHITE).move_to(self.hud_bg.get_center() + m.DOWN * 0.3)
            ),
            run_time=0.2
        )
