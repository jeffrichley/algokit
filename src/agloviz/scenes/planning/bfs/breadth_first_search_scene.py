"""BFS-specific Manim visualization scene.

BFS algorithm visualization using HarborGridScene base class.
Follows the video plan from bls_video.md.
"""

import sys
from pathlib import Path
from typing import Any

import manim as m

# Add the main project to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from agloviz.components.grid_overlay import GridOverlay
    from agloviz.components.hud_panel import HUDPanel
    from agloviz.components.legend_panel import create_bfs_legend
    from agloviz.components.snake_queue import SnakeQueue
    from agloviz.components.tokens import GoalToken, StartToken
    from agloviz.core.grid_visualizer import GridVisualizer
    from algokit.core.helpers import (
        HarborNetScenario,
        create_grid_graph,
        load_harbor_scenario,
    )
    from algokit.pathfinding.bfs_with_events import bfs_with_data_collection
    from algokit.viz.adapters import (
        EventType,
        SearchEvent,
        process_events_for_visualization,
    )
    from algokit.viz.scenes import HarborGridScene
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
            self.grid_width = scenario.width
            self.grid_height = scenario.height
            self.start_pos = scenario.start
            self.goal_pos = scenario.goal
            self.obstacles = scenario.obstacles
            self.cell_size = 0.8
            self.grid_group = m.VGroup()
        
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
    - Scene 4: Goal celebration
    - Scene 5: Path reconstruction
    - Scene 6: Complexity display
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
        
        # Override cell size to make squares larger
        self.cell_size = 0.85  # Increased from default 0.8
        
        # Recalculate cell centers with new size and shift down
        self.cell_centers = self._calculate_cell_centers_shifted()
        
        # Initialize GridVisualizer for proper grid management
        self.grid_visualizer = None  # Will be created in _show_grid()
        
        # BFS-specific state
        self.bfs_events: list[SearchEvent] = []
        self.visualization_data: dict[str, Any] = {}
        self.parent_arrows: dict[tuple[int, int], m.Arrow] = {}
        self.cell_states: dict[tuple[int, int], str] = {}  # 'empty', 'frontier', 'visited', 'obstacle'
        self.current_depth = 0
        self.goal_found = False
        self.final_path: list[tuple[int, int]] = []
        
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
        
    def _calculate_cell_centers_shifted(self) -> dict[tuple[int, int], m.np.ndarray]:
        """Calculate cell centers with a downward shift for better positioning.
        
        Returns:
            Dictionary mapping (x, y) grid coordinates to screen positions
        """
        cell_centers = {}
        
        # Calculate grid origin (center of grid) with downward shift
        grid_origin_x = -(self.grid_width * self.cell_size) / 2
        grid_origin_y = (self.grid_height * self.cell_size) / 2 - 0.5  # Shift down by 0.5 units
        
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                # Calculate cell center position
                cell_x = grid_origin_x + (x + 0.5) * self.cell_size
                cell_y = grid_origin_y - (y + 0.5) * self.cell_size
                cell_centers[(x, y)] = m.np.array([cell_x, cell_y, 0])
                
        return cell_centers
        
    def construct(self) -> None:
        """Construct the BFS visualization following the specified order."""
        # 1. Title
        self._show_title()
        
        # 2. Grid
        self._show_grid()
        
        # 3. Legend
        self._show_legend()
        
        # 4. Start token
        self._show_start_token()
        
        # 5. Goal token
        self._show_goal_token()
        
        # 6. Water areas
        self._show_water_areas()
        
        # 7. HUD
        self._show_hud()
        
        # 8. Queue
        self._show_queue()
        
        # Scene 3: BFS wavefront exploration
        # self._run_bfs_algorithm()
        
        # Scene 4: Goal celebration
        # self._celebrate_goal()
        
        # Scene 5: Path reconstruction
        # self._reconstruct_path()
        
        # Scene 6: Complexity display
        #  self._show_complexity()
        
        # Hold final frame
        self.wait(3)
        
    def _show_title(self) -> None:
        """1. Show title - scene controls title/subtitle."""
        # Create title
        self.title = m.Text(
            f"BFS Grid: {self.scenario.name}",
            font_size=36,
            color=m.WHITE
        ).to_edge(m.UP)
        
        self.play(m.Write(self.title), run_time=1.8)
        
        # Create subtitle
        self.subtitle = m.Text(
            "Shortest paths on unweighted graphs",
            font_size=20,
            color=m.GRAY
        ).next_to(self.title, m.DOWN, buff=0.2)
        
        self.play(m.Write(self.subtitle), run_time=1.0)
        
    def _show_grid(self) -> None:
        """2. Show grid - GridVisualizer controls grid only."""
        # Create GridOverlay
        grid_overlay = GridOverlay(
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            cell_size=self.cell_size,
            grid_color=m.GRAY_B,
            scenario=self.scenario
        )
        
        # Initialize GridVisualizer - the ONLY class with GridOverlay reference
        self.grid_visualizer = GridVisualizer(grid_overlay)
        
        # Move the grid to desired position BEFORE adding to scene
        grid_overlay.shift(m.DOWN * 0.5)
        
        # Add grid overlay to scene (already positioned correctly)
        self.add(grid_overlay)
        
        # Animate the existing grid cells with growth animation
        self.grid_visualizer.animate_grid_growth(self)
        
        # Polish: subtle grid shimmer
        grid_group = self.grid_visualizer.get_grid_group()
        self.play(m.ApplyWave(grid_group, amplitude=0.05), run_time=0.8)
        
        # Frame the stage
        frame_rect = m.SurroundingRectangle(
            grid_group,
            color=m.GRAY_D,
            stroke_width=2
        )
        
        # Fade in frame, then fade opacity to 30%
        self.play(m.FadeIn(frame_rect), run_time=0.4)
        self.play(frame_rect.animate.set_stroke(opacity=0.3), run_time=0.4)
        
    def _show_legend(self) -> None:
        """3. Show legend."""
        # Build legend using the LegendPanel component
        self.legend = create_bfs_legend()
        self.legend.to_corner(m.DR, buff=0.5)
        
        self.play(m.FadeIn(self.legend, shift=m.DOWN * 0.25), run_time=1.0)
        
    def _show_start_token(self) -> None:
        """4. Show start token using GridVisualizer."""
        # Start token using StartToken component
        self.start_token = StartToken(cell_size=self.cell_size)
        
        # Place token using GridVisualizer
        self.grid_visualizer.place_token(self.start_token, self.start_pos)
        
        # Add label after placement
        self.start_token.add_label_after_placement()
        
        # Animate start placement using token's built-in animation
        self.start_token.animate_entrance(self)
        start_cell = self.grid_visualizer.get_cell(self.start_pos)
        self.play(m.Indicate(start_cell), run_time=0.3)
        self.play(m.Write(self.start_token.label), run_time=0.3)
        
    def _show_goal_token(self) -> None:
        """5. Show goal token using GridVisualizer."""
        # Goal token using GoalToken component
        self.goal_star = GoalToken(cell_size=self.cell_size)
        
        # Place token using GridVisualizer
        self.grid_visualizer.place_token(self.goal_star, self.goal_pos)
        
        # Add label after placement
        self.goal_star.add_label_after_placement()
        
        # Animate goal placement using token's built-in animation
        self.goal_star.animate_entrance(self)
        self.play(m.Flash(self.goal_star.get_center()), run_time=0.3)
        self.play(m.Write(self.goal_star.label), run_time=0.3)
        
    def _show_water_areas(self) -> None:
        """6. Show water areas (obstacles) using GridVisualizer."""
        obstacle_animations = []
        wave_animations = []
        
        for obstacle_pos in self.obstacles:
            x, y = obstacle_pos
            
            # Use GridVisualizer to fill cell with water color
            self.grid_visualizer.fill_cell(x, y, color=m.BLUE_E, opacity=0.65)
            self.cell_states[obstacle_pos] = "obstacle"
            
            # Get the cell for wave effects using GridVisualizer
            cell = self.grid_visualizer.get_cell(obstacle_pos)
            
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
        
    def _show_hud(self) -> None:
        """7. Show HUD."""
        # Initial HUD values
        hud_values = {
            "Visited": 0,
            "Frontier": 0,
            "Depth": 0,
            "Queue": 0
        }
        
        # Create HUD panel with proper styling
        self.hud_panel = HUDPanel(
            values=hud_values,
            max_lines=2,
            font_size=16,
            corner_radius=0.1,
            stroke_width=2,
            stroke_color=m.WHITE,
            fill_color=m.BLACK,
            fill_opacity=0.1
        )
        
        # Position HUD in top-left area
        self.hud_panel.to_edge(m.LEFT)
        
        # Animate HUD sliding in
        self.play(
            m.FadeIn(self.hud_panel, shift=m.UP * 0.2),
            run_time=0.8
        )
        
    def _show_queue(self) -> None:
        """8. Show queue using only the SnakeQueue component."""
        # Create snaking queue
        self.snaking_queue = SnakeQueue(
            tokens_wide=6,
            tokens_tall=3,
            token_size=0.25
        )
        
        # Position the queue in the top-right area
        self.snaking_queue.to_edge(m.RIGHT, buff=0.5).shift(m.UP * 0.8)
        
        # Add snaking queue to scene with animation
        self.play(
            m.FadeIn(self.snaking_queue, shift=m.RIGHT * 0.5),
            run_time=1.0
        )
        
        # Enqueue the start token
        self._enqueue_start_token()
        
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
        
        # Build legend using the LegendPanel component
        legend = create_bfs_legend()
        legend.to_corner(m.DR, buff=0.5)
        
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
        
        
    def _cast_actors(self) -> None:
        """Scene 1: Cast the actors (3–4s) - Start + Goal + Obstacles."""
        # 4. Start + Goal placement
        self._place_start_goal()
        
        # 5. Flooded harbor obstacles
        self._animate_obstacles()
        
    def _place_start_goal(self) -> None:
        """Place start and goal markers with animations using proper token components."""
        # Start token using StartToken component
        start_cell = self.grid_visualizer.get_cell(self.start_pos)
        self.start_token = StartToken(cell_size=self.cell_size)
        self.start_token.move_to(start_cell.get_center())
        
        # Goal token using GoalToken component
        goal_cell = self.grid_visualizer.get_cell(self.goal_pos)
        self.goal_star = GoalToken(cell_size=self.cell_size)
        self.goal_star.move_to(goal_cell.get_center())
        
        # Add labels after placement
        self.start_token.add_label_after_placement()
        self.goal_star.add_label_after_placement()
        
        # Animate start placement using token's built-in animation
        self.start_token.animate_entrance(self)
        self.play(m.Indicate(start_cell), run_time=0.3)
        self.play(m.Write(self.start_token.label), run_time=0.3)
        
        # Animate goal placement using token's built-in animation
        self.goal_star.animate_entrance(self)
        self.play(m.Flash(self.goal_star.get_center()), run_time=0.3)
        self.play(m.Write(self.goal_star.label), run_time=0.3)
        
    def _animate_obstacles(self) -> None:
        """Animate flooded harbor obstacles with wave effects."""
        obstacle_animations = []
        wave_animations = []
        
        for obstacle_pos in self.obstacles:
            # Get the cell for this obstacle using GridVisualizer
            cell = self.grid_visualizer.get_cell(obstacle_pos)
            
            # Fill with water color
            cell.set_fill(m.BLUE_E, opacity=0.65)
            self.cell_states[obstacle_pos] = "obstacle"
            
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
        self.snaking_queue = SnakeQueue(
            tokens_wide=6,
            tokens_tall=3,
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
        """Build HUD counters using the HUDPanel component."""
        # Initial HUD values
        hud_values = {
            "Visited": 0,
            "Frontier": 0,
            "Depth": 0,
            "Queue": 0
        }
        
        # Create HUD panel with proper styling
        self.hud_panel = HUDPanel(
            values=hud_values,
            max_lines=2,
            font_size=16,
            corner_radius=0.1,
            stroke_width=2,
            stroke_color=m.WHITE,
            fill_color=m.BLACK,
            fill_opacity=0.1
        )
        
        # Position HUD in top-left area
        self.hud_panel.to_edge(m.LEFT) #, buff=0.3)  #.shift(m.DOWN * 2.0)
        
        # Animate HUD sliding in
        self.play(
            m.FadeIn(self.hud_panel, shift=m.UP * 0.2),
            run_time=0.8
        )
        
    def _enqueue_start_token(self) -> None:
        """Enqueue the start token into the snaking queue with morphing animation."""
        # Create token at start position (as a circle initially)
        start_token = m.Dot(
            radius=self.cell_size * 0.3,
            color=m.GREEN_C
        ).move_to(self.start_token.get_center())
        
        # Create the target token for the queue
        target_token = m.Square(
            side_length=0.25,
            fill_color=m.GREEN_C,
            fill_opacity=0.9,
            stroke_color=m.WHITE,
            stroke_width=2
        )
        
        # Enqueue the target token
        self.snaking_queue.enqueue(token=target_token, scene=self)
        
        # Animate morphing and movement
        self.play(
            m.Transform(start_token, target_token),
            run_time=0.8
        )
        
        # Update HUD to show queue has 1 item
        self._update_hud(visited=0, frontier=1, depth=0, queue=1)
        
    def _update_hud(self, visited: int, frontier: int, depth: int, queue: int) -> None:
        """Update HUD display with new values using HUDPanel."""
        new_values = {
            "Visited": visited,
            "Frontier": frontier,
            "Depth": depth,
            "Queue": queue
        }
        
        # Use the HUDPanel's built-in update method
        self.hud_panel.update_values(new_values, scene=self, run_time=0.2)
        
    def _run_bfs_algorithm(self) -> None:
        """Scene 3: Run the BFS algorithm with visualization."""
        # Create graph from scenario
        graph = self.scenario.to_graph()
        
        # Run BFS with event collection
        try:
            path, events = bfs_with_data_collection(graph, self.start_pos, self.goal_pos)
            self.bfs_events = events
            self.final_path = path or []
        except ImportError:
            # Fallback: create mock events for standalone mode
            self.bfs_events = self._create_mock_events()
            self.final_path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5)]
        
        # Process events for visualization
        self.visualization_data = process_events_for_visualization(self.bfs_events)
        
        # Animate the BFS execution
        self._animate_bfs_execution()
        
    def _create_mock_events(self) -> list[SearchEvent]:
        """Create mock events for standalone mode."""
        events = []
        step = 0
        
        # Mock BFS execution
        queue = [self.start_pos]
        visited = set()
        parents = {}
        
        while queue:
            current = queue.pop(0)
            events.append(SearchEvent(EventType.DEQUEUE, current, step=step))
            step += 1
            
            if current not in visited:
                visited.add(current)
                events.append(SearchEvent(EventType.DISCOVER, current, step=step))
                step += 1
                
                if current == self.goal_pos:
                    events.append(SearchEvent(EventType.GOAL_FOUND, current, step=step))
                    step += 1
                    break
                
                # Add neighbors
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = current[0] + dx, current[1] + dy
                    neighbor = (nx, ny)
                    
                    if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height and 
                        neighbor not in visited and neighbor not in self.obstacles):
                        queue.append(neighbor)
                        parents[neighbor] = current
                        events.append(SearchEvent(EventType.ENQUEUE, neighbor, current, step=step))
                        step += 1
        
        return events
        
    def _animate_bfs_execution(self) -> None:
        """Animate the BFS algorithm execution step by step."""
        frontier_count = 0
        visited_count = 0
        
        for event in self.bfs_events:
            if event.type == EventType.DEQUEUE:
                self._animate_dequeue(event.node)
                frontier_count = max(0, frontier_count - 1)
                
            elif event.type == EventType.DISCOVER:
                self._animate_visit(event.node)
                visited_count += 1
                
            elif event.type == EventType.ENQUEUE:
                self._animate_enqueue(event.node, event.parent)
                frontier_count += 1
                
            elif event.type == EventType.GOAL_FOUND:
                self.goal_found = True
                self._animate_goal_discovery(event.node)
                
            # Update HUD
            self._update_hud(visited=visited_count, frontier=frontier_count, 
                           depth=self.current_depth, queue=frontier_count)
            
            # Small pause between events
            self.wait(0.1)
            
    def _animate_dequeue(self, node: tuple[int, int]) -> None:
        """Animate dequeuing a node from the queue."""
        # Remove token from queue
        if not self.snaking_queue.is_empty():
            token = self.snaking_queue.dequeue(scene=self)
            
        # Mark cell as being processed using GridVisualizer
        cell = self.grid_visualizer.get_cell(node)
        
        # Pulse the cell to show it's being processed
        self.play(m.Indicate(cell, scale_factor=1.1, color=m.YELLOW_C), run_time=0.3)
        
    def _animate_visit(self, node: tuple[int, int]) -> None:
        """Animate visiting a node."""
        cell = self.grid_visualizer.get_cell(node)
        
        # Mark as visited
        cell.set_fill(m.YELLOW_E, opacity=0.4)
        self.cell_states[node] = "visited"
        
        # Pulse to show it's visited
        self.play(m.Indicate(cell, scale_factor=1.05, color=m.YELLOW_E), run_time=0.2)
        
    def _animate_enqueue(self, node: tuple[int, int], parent: tuple[int, int] | None) -> None:
        """Animate enqueuing a node."""
        # Mark as frontier using GridVisualizer
        cell = self.grid_visualizer.get_cell(node)
        cell.set_fill(m.BLUE_C, opacity=0.6)
        self.cell_states[node] = "frontier"
        
        # Draw parent arrow if parent exists
        if parent:
            self._draw_parent_arrow(parent, node)
        
        # Add token to queue
        token = m.Square(
            side_length=0.25,
            fill_color=m.BLUE_C,
            fill_opacity=0.9,
            stroke_color=m.WHITE,
            stroke_width=2
        )
        self.snaking_queue.enqueue(token=token, scene=self)
        
        # Show discovery
        self.play(m.Indicate(cell, scale_factor=1.05, color=m.BLUE_C), run_time=0.2)
        
    def _draw_parent_arrow(self, parent: tuple[int, int], child: tuple[int, int]) -> None:
        """Draw an arrow from parent to child."""
        parent_cell = self.grid_visualizer.get_cell(parent)
        child_cell = self.grid_visualizer.get_cell(child)
        
        arrow = m.Arrow(
            parent_cell.get_center(),
            child_cell.get_center(),
            buff=0.1,
            stroke_width=2,
            color=m.WHITE
        )
        
        self.parent_arrows[child] = arrow
        self.play(m.GrowArrow(arrow), run_time=0.3)
        
    def _animate_goal_discovery(self, node: tuple[int, int]) -> None:
        """Animate discovering the goal."""
        cell = self.grid_visualizer.get_cell(node)
        
        # Flash the goal
        self.play(m.Flash(self.goal_star.get_center(), color=m.GOLD), run_time=0.5)
        
    def _celebrate_goal(self) -> None:
        """Scene 4: Goal celebration."""
        if not self.goal_found:
            return
            
        # Freeze motion briefly
        self.wait(0.3)
        
        # Confetti effect
        confetti = []
        for _ in range(20):
            confetti.append(m.Dot(
                radius=0.05,
                color=m.np.random.choice([m.RED, m.GREEN, m.BLUE, m.YELLOW, m.PINK, m.TEAL])
            ).move_to(self.goal_star.get_center()))
        
        confetti_group = m.VGroup(*confetti)
        
        # Animate confetti
        animations = []
        for dot in confetti:
            animations.append(dot.animate.scale(0.6).shift(
                m.np.random.uniform(-2, 2) * m.RIGHT + 
                m.np.random.uniform(-1, 1) * m.UP
            ))
        
        self.play(m.LaggedStart(*animations, lag_ratio=0.1), run_time=1.0)
        
        # Goal reached text
        goal_text = m.Text(
            f"Goal reached at depth {len(self.final_path) - 1 if self.final_path else 0}",
            font_size=24,
            color=m.GOLD
        ).move_to(m.ORIGIN)
        
        self.play(m.GrowFromCenter(goal_text), run_time=0.8)
        self.wait(1.0)
        self.play(m.FadeOut(goal_text), run_time=0.5)
        
        # Dim non-path elements
        grid_group = self.grid_visualizer.get_grid_group()
        self.play(grid_group.animate.set_opacity(0.3), run_time=0.5)
        
    def _reconstruct_path(self) -> None:
        """Scene 5: Path reconstruction."""
        if not self.final_path:
            return
            
        # Create path polyline using GridVisualizer
        path_points = []
        for node in self.final_path:
            cell_center = self.grid_visualizer.get_cell_center(node)
            path_points.append(cell_center)
        
        path_line = m.VMobject()
        path_line.set_points_as_corners(path_points)
        path_line.set_stroke(m.PINK, width=8)
        
        # Animate path creation
        self.play(m.Create(path_line), run_time=2.0)
        
        # Animate tracer moving along path
        tracer = m.Dot(radius=0.1, color=m.WHITE).move_to(path_line.get_start())
        self.add(tracer)
        self.play(m.MoveAlongPath(tracer, path_line), run_time=2.0)
        
        # Brighten path cells using GridVisualizer
        for node in self.final_path:
            self.grid_visualizer.set_cell_stroke(node, m.PINK, width=6)
            self.grid_visualizer.set_cell_opacity(node, 1.0)
        
        # Path caption
        path_caption = m.Text(
            "Shortest path (unweighted BFS)",
            font_size=18,
            color=m.WHITE
        ).to_edge(m.DOWN, buff=0.5)
        
        self.play(m.Write(path_caption), run_time=0.8)
        
    def _show_complexity(self) -> None:
        """Scene 6: Show complexity analysis."""
        # Complexity card
        complexity_card = m.RoundedRectangle(
            width=3.0,
            height=2.0,
            corner_radius=0.2,
            stroke_color=m.WHITE,
            stroke_width=2,
            fill_color=m.BLACK,
            fill_opacity=0.8
        ).to_edge(m.LEFT, buff=0.5).shift(m.UP * 1.0)
        
        # Complexity text
        complexity_text = m.VGroup(
            m.Text("Time: O(V + E)", font_size=16, color=m.WHITE),
            m.Text("Space: O(V)", font_size=16, color=m.WHITE),
            m.Text("(grid ~ O(W·H))", font_size=12, color=m.GRAY)
        ).arrange(m.DOWN, buff=0.2).move_to(complexity_card.get_center())
        
        # Animate complexity card
        self.play(
            m.FadeIn(complexity_card, shift=m.LEFT * 0.3),
            m.Write(complexity_text),
            run_time=1.0
        )
        
        # Show connections to visual elements
        self.wait(1.0)
