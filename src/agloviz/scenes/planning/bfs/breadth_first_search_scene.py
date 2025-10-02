"""BFS-specific Manim visualization scene.

BFS algorithm visualization using HarborGridScene base class.
Follows the video plan from bls_video.md.
"""

import os  # - Used for environment variables
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
    from agloviz.components.tokens import GoalToken, StartToken, Token, WaterToken
    from agloviz.core.grid_visualizer import GridVisualizer
    from agloviz.core.timing_config import get_timing_config
    from agloviz.core.timing_tracker import get_timing_tracker
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
    class HarborNetScenario:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    class HarborGridScene(m.Scene):  # type: ignore[no-redef]
        def __init__(self, scenario: Any, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.scenario = scenario
            self.grid_width = scenario.width
            self.grid_height = scenario.height
            self.start_pos = scenario.start
            self.goal_pos = scenario.goal
            self.obstacles = scenario.obstacles
            self.cell_size = 0.8
            self.grid_group = m.VGroup()

        def construct(self) -> None:
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

    def __init__(self, scenario: Any = None, **kwargs: Any) -> None:
        """Initialize the BFS scene.

        Args:
            scenario: HarborNet scenario configuration or scenario file path
            **kwargs: Additional arguments passed to HarborGridScene
        """
        # Load scenario - prioritize direct parameter, then environment variable
        if scenario is None:
            # Get scenario from environment variable (set by render_scene)
            scenario_file = os.environ.get("AGLOVIZ_SCENARIO_FILE")

            if scenario_file:
                scenario = self._load_scenario_from_file(scenario_file)
            else:
                # Create default scenario if no file provided
                scenario = HarborNetScenario(
                    name="Small Flooded Harbor",
                    description="Default scenario for BFS visualization",
                    width=7,
                    height=5,
                    start=(0, 0),
                    goal=(6, 4),
                    obstacles={(1, 2), (1, 1), (2, 1), (3, 1), (3, 2), (4, 3), (5, 3)},
                    text_below_grid_offset=3.5,
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
        self.grid_visualizer: GridVisualizer | None = (
            None  # Will be created in _show_grid()
        )

        # BFS-specific state
        self.bfs_events: list[SearchEvent] = []
        self.visualization_data: dict[str, Any] = {}
        self.parent_arrows: dict[tuple[int, int], m.Arrow] = {}
        self.cell_states: dict[
            tuple[int, int], str
        ] = {}  # 'empty', 'frontier', 'visited', 'obstacle'
        self.current_depth = 0
        self.goal_found = False
        self.goal_discovered = False  # Goal added to frontier (enqueued)
        self.goal_dequeued = False  # Goal processed (dequeued)
        self.final_path: list[tuple[int, int]] = []
        self.tokens: dict[tuple[int, int], Token] = {}
        self.water_tokens: dict[tuple[int, int], WaterToken] = {}

        # Phase 4: Path reconstruction data
        self.path_validation_passed = False
        self.path_visualization_data: dict[str, Any] = {}
        self.path_segments: list[
            tuple[m.np.ndarray, m.np.ndarray]
        ] = []  # (start_pos, end_pos) pairs
        self.path_dots_data: list[
            tuple[m.np.ndarray, tuple[int, int]]
        ] = []  # (screen_pos, grid_pos) pairs

        # Layer pulse system
        self.depth_rings: dict[int, m.VGroup] = {}  # depth -> ring mobjects
        self.layer_colors: list[
            str
        ] = []  # Will be generated dynamically based on goal depth
        self.completed_depths: set[int] = set()
        self.max_depth: int = 0  # Will be calculated from BFS events

        # Development speed controls (set to 1.0 for cinematic timing)
        # Initialize with cinematic speeds (normal timing)
        self.speed_multipliers = {
            "setup": 1.0,
            "bfs_events": 1.0,
            "path_drawing": 1.0,
            "celebrations": 1.0,
            "educational": 1.0,
            "waits": 1.0,
        }
        self.set_development_mode(False)
        self.max_events_displayed = 999

        # Layout constants - read from scenario configuration
        self.text_below_grid_offset = getattr(
            self.scenario, "text_below_grid_offset", 3.0
        )

        # Initialize timing configuration
        self.timing_config = get_timing_config()

        # Initialize timing tracker and connect it to config
        self.timing_tracker = get_timing_tracker()
        self.timing_tracker.set_timing_config(self.timing_config)

        # Check for timing mode from environment variable (set by CLI)
        self.current_mode = os.environ.get("AGLOVIZ_TIMING_MODE", "cinematic")
        self.timing_config.set_mode(self.current_mode)
        self.max_events_displayed = self.timing_config.get_event_limit()

    def _get_timing(self, stage: str, base_time: float) -> float:
        """Get adjusted timing for development speed control.

        Args:
            stage: Stage category ('setup', 'bfs_events', 'path_drawing', 'celebrations', 'educational', 'waits')
            base_time: Base timing in seconds for cinematic version

        Returns:
            Adjusted timing divided by speed multiplier
        """
        # Use new timing config if available, fallback to old system
        if hasattr(self, "timing_config"):
            # Map stage to timing config category
            if stage == "waits":
                result = self.timing_config.get_wait_time("default", self.current_mode)
            else:
                result = self.timing_config.get_animation_time(
                    "default", self.current_mode
                )

            # Track the legacy timing request
            if hasattr(self, "timing_tracker"):
                return self.timing_tracker.track_legacy_timing(stage, base_time, result)
            return result
        else:
            # Fallback to old system
            multiplier = self.speed_multipliers.get(stage, 1.0)
            result = base_time / multiplier

            # Track the legacy timing request
            if hasattr(self, "timing_tracker"):
                return self.timing_tracker.track_legacy_timing(stage, base_time, result)
            return result

    def get_animation_time(self, animation_name: str) -> float:
        """Get timing for a specific animation from config.

        Args:
            animation_name: Name of the animation in the config

        Returns:
            Animation timing in seconds
        """
        if hasattr(self, "timing_config"):
            result = self.timing_config.get_animation_time(
                animation_name, self.current_mode
            )
            # Track the animation timing request
            if hasattr(self, "timing_tracker"):
                return self.timing_tracker.track_animation_time(animation_name, result)
            return result

        # Default fallback
        fallback = 1.0
        if hasattr(self, "timing_tracker"):
            return self.timing_tracker.track_animation_time(animation_name, fallback)
        return fallback

    def get_wait_time(self, wait_name: str) -> float:
        """Get wait time for a specific wait from config.

        Args:
            wait_name: Name of the wait in the config

        Returns:
            Wait time in seconds
        """
        if hasattr(self, "timing_config"):
            result = self.timing_config.get_wait_time(wait_name, self.current_mode)
            # Track the wait timing request
            if hasattr(self, "timing_tracker"):
                return self.timing_tracker.track_wait_time(wait_name, result)
            return result

        # Default fallback
        fallback = 1.0
        if hasattr(self, "timing_tracker"):
            return self.timing_tracker.track_wait_time(wait_name, fallback)
        return fallback

    def set_development_mode(self, enabled: bool = True) -> None:
        """Toggle between development speed (fast) and cinematic speed (normal).

        Args:
            enabled: If True, use fast development speeds. If False, use cinematic speeds.
        """
        if hasattr(self, "timing_config"):
            # Use new timing config system
            mode = "development" if enabled else "cinematic"
            self.timing_config.set_mode(mode)
            self.current_mode = mode
            print(f"🎬 Timing mode: {mode.upper()} speeds enabled")
        else:
            # Fallback to old system
            if enabled:
                # Fast development speeds
                self.speed_multipliers.update(
                    {
                        "setup": 4.0,  # 4x faster setup
                        "bfs_events": 8.0,  # 8x faster BFS events
                        "path_drawing": 6.0,  # 6x faster path drawing
                        "celebrations": 3.0,  # 3x faster celebrations
                        "educational": 2.0,  # 2x faster educational moments
                        "waits": 10.0,  # 10x faster wait times
                    }
                )
                print("🚀 Development mode: FAST speeds enabled")
            else:
                # Cinematic speeds (normal timing)
                self.speed_multipliers.update(
                    {
                        "setup": 1.0,
                        "bfs_events": 1.0,
                        "path_drawing": 1.0,
                        "celebrations": 1.0,
                        "educational": 1.0,
                        "waits": 1.0,
                    }
                )
                print("🎬 Cinematic mode: NORMAL speeds enabled")

    def set_timing_mode(self, mode: str) -> None:
        """Set timing mode using configuration system.

        Args:
            mode: Timing mode ('cinematic', 'development', 'quick_demo')
        """
        if hasattr(self, "timing_config"):
            self.timing_config.set_mode(mode)
            self.current_mode = mode
            print(f"🎬 Timing mode set to: {mode.upper()}")
        else:
            print("⚠️ Timing config not available, using legacy system")

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
        grid_origin_y = (
            self.grid_height * self.cell_size
        ) / 2 - 0.5  # Shift down by 0.5 units

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

        # 9. Start BFS
        self.start_bfs()

        # 10. Step through BFS events (for demonstration)
        self.step_through_bfs(
            max_events_displayed=self.max_events_displayed
        )  # Process up to 1000 events (likely all of them)

        # 11. Goal celebration will be triggered automatically when goal is dequeued
        # (removed manual call - now event-driven)

        # 12. Debug path data after BFS completion
        self.debug_path_data()

        # 12. Show final educational state with all depth rings
        self._show_all_depth_rings()
        self._highlight_depth_rings()

        # Scene 4: Goal celebration
        # self._celebrate_goal()

        # Scene 5: Path reconstruction
        # self._reconstruct_path()

        # Scene 6: Complexity display
        #  self._show_complexity()

        # Hold final frame with rings visible
        self.wait(self.get_wait_time("initial_setup"))

        # Generate and display timing report
        self.generate_timing_report()

    def _show_title(self) -> None:
        """1. Show title - scene controls title/subtitle."""
        # Create title
        self.title = m.Text(
            f"BFS Grid: {self.scenario.name}", font_size=36, color=m.WHITE
        ).to_edge(m.UP)

        self.play(m.Write(self.title), run_time=self.get_animation_time("title_write"))

        # Create subtitle
        self.subtitle = m.Text(
            "Shortest paths on unweighted graphs", font_size=20, color=m.GRAY
        ).next_to(self.title, m.DOWN, buff=0.2)

        self.play(
            m.Write(self.subtitle), run_time=self.get_animation_time("subtitle_write")
        )

    def _show_grid(self) -> None:
        """2. Show grid - GridVisualizer controls grid only."""
        # Create GridOverlay
        grid_overlay = GridOverlay(
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            cell_size=self.cell_size,
            grid_color=m.GRAY_B,
            scenario=self.scenario,
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
        self.play(
            m.ApplyWave(grid_group, amplitude=0.05),
            run_time=self.get_animation_time("grid_wave"),
        )

        # Frame the stage
        frame_rect = m.SurroundingRectangle(grid_group, color=m.GRAY_D, stroke_width=2)

        # Fade in frame, then fade opacity to 30%
        self.play(
            m.FadeIn(frame_rect), run_time=self.get_animation_time("frame_fade_in")
        )
        self.play(
            frame_rect.animate.set_stroke(opacity=0.3),
            run_time=self.get_animation_time("frame_stroke_adjust"),
        )

    def _show_legend(self) -> None:
        """3. Show legend."""
        # Build legend using the LegendPanel component
        self.legend = create_bfs_legend()
        self.legend.to_corner(m.DR, buff=0.5)

        self.play(
            m.FadeIn(self.legend, shift=m.DOWN * 0.25),
            run_time=self.get_animation_time("legend_fade_in"),
        )

    def _show_start_token(self) -> None:
        """4. Show start token using GridVisualizer."""
        # Start token using StartToken component
        self.start_token = StartToken(cell_size=self.cell_size)

        # Place token using GridVisualizer
        if self.grid_visualizer is not None:
            self.grid_visualizer.place_token(self.start_token, self.start_pos)

        # Add label after placement
        self.start_token.add_label_after_placement()

        # Animate start placement using token's built-in animation
        self.start_token.animate_entrance(self)
        if self.grid_visualizer is not None:
            start_cell = self.grid_visualizer.get_cell(self.start_pos)
            self.play(
                m.Indicate(start_cell),
                run_time=self.get_animation_time("start_indicate"),
            )
        self.play(
            m.Write(self.start_token.label),
            run_time=self.get_animation_time("start_label_write"),
        )

    def _show_goal_token(self) -> None:
        """5. Show goal token using GridVisualizer."""
        # Goal token using GoalToken component
        self.goal_star = GoalToken(cell_size=self.cell_size)

        # Place token using GridVisualizer
        self.grid_visualizer.place_token(self.goal_star, self.goal_pos)  # type: ignore[union-attr]

        # Add label after placement
        self.goal_star.add_label_after_placement()

        # Animate goal placement using token's built-in animation
        self.goal_star.animate_entrance(self)
        self.play(
            m.Flash(self.goal_star.get_center()),
            run_time=self.get_animation_time("goal_flash"),
        )
        self.play(
            m.Write(self.goal_star.label),
            run_time=self.get_animation_time("goal_label_write"),
        )

    def _show_water_areas(self) -> None:
        """6. Show water areas (obstacles) using WaterToken with wave effects."""
        for obstacle_pos in self.obstacles:
            # Create water token with waves
            water_token = WaterToken(cell_size=self.cell_size)

            # Position it at the cell center using GridVisualizer
            self.grid_visualizer.place_token(water_token, obstacle_pos)  # type: ignore[union-attr]

            # Store the token for later reference
            self.water_tokens[obstacle_pos] = water_token
            self.cell_states[obstacle_pos] = "obstacle"

            # Use the token's built-in animation with ripple timing
            water_token.animate_entrance(self)
            # self.wait(0.03)  # Small delay for ripple effect

    def _show_hud(self) -> None:
        """7. Show HUD."""
        # Initial HUD values
        hud_values = {"Visited": 0.0, "Frontier": 0.0, "Depth": 0.0, "Queue": 0.0}

        # Create HUD panel with proper styling
        self.hud_panel = HUDPanel(
            values=hud_values,
            max_lines=2,
            font_size=16,
            corner_radius=0.1,
            stroke_width=2,
            stroke_color=m.WHITE,
            fill_color=m.BLACK,
            fill_opacity=0.1,
        )

        # Position HUD in top-left area
        self.hud_panel.to_edge(m.LEFT)

        # Animate HUD sliding in
        self.play(
            m.FadeIn(self.hud_panel, shift=m.UP * 0.2),
            run_time=self.get_animation_time("hud_fade_in"),
        )

    def _update_hud(
        self,
        visited: int | None = None,
        frontier: int | None = None,
        depth: int | None = None,
        queue: int | None = None,
    ) -> None:
        """Update HUD values and animate the changes.

        Args:
            visited: Number of visited nodes (optional)
            frontier: Number of frontier nodes (optional)
            depth: Current depth level (optional)
            queue: Number of nodes in queue (optional)
        """
        if not hasattr(self, "hud_panel"):
            return

        # Build new values dict with only provided parameters
        new_values = {}
        if visited is not None:
            new_values["Visited"] = float(visited)
        if frontier is not None:
            new_values["Frontier"] = float(frontier)
        if depth is not None:
            new_values["Depth"] = float(depth)
        if queue is not None:
            new_values["Queue"] = float(queue)

        # Update HUD with animation using HUDPanel's update_values method
        if new_values:
            self.hud_panel.update_values(
                new_values, scene=self, run_time=self.get_animation_time("hud_update")
            )

    def _show_queue(self) -> None:
        """8. Show queue using only the SnakeQueue component."""
        # Create snaking queue
        self.snaking_queue = SnakeQueue(tokens_wide=6, tokens_tall=3, token_size=0.25)

        # Position the queue in the top-right area
        self.snaking_queue.to_edge(m.RIGHT, buff=0.5).shift(m.UP * 0.8)

        # Add snaking queue to scene with animation
        self.play(
            m.FadeIn(self.snaking_queue, shift=m.RIGHT * 0.5),
            run_time=self.get_animation_time("queue_fade_in"),
        )

    def start_bfs(self) -> None:
        """9. Start BFS algorithm by getting the plan and enqueuing the start token."""
        # Get the BFS plan by running the algorithm
        self._get_bfs_plan()

        # Enqueue the start token
        # self.enqueue_token(self.start_token)

        # Update HUD to show initial state (start node in queue)
        self._update_hud(visited=0, frontier=0, depth=0, queue=1)

    def _get_bfs_plan(self) -> None:
        """Run BFS algorithm to get the search plan and events."""
        try:
            # Create graph from scenario
            graph = create_grid_graph(
                width=self.grid_width,
                height=self.grid_height,
                blocked=self.obstacles,
                start=self.start_pos,
                goal=self.goal_pos,
            )

            # Run BFS with event collection
            path, events = bfs_with_data_collection(
                graph, self.start_pos, self.goal_pos
            )

            # Store the results
            self.bfs_events = events
            self.final_path = path or []

            # Process events for visualization
            self.visualization_data = process_events_for_visualization(self.bfs_events)

            # Calculate max depth and generate color gradient
            self._calculate_max_depth_and_colors()

            print(
                f"BFS plan generated: {len(events)} events, path length: {len(self.final_path) if self.final_path else 0}"
            )
            print(
                f"Max depth: {self.max_depth}, Generated {len(self.layer_colors)} colors"
            )

            # Debug: Show first 10 events
            print("First 10 events:")
            for i, event in enumerate(events[:10]):
                print(
                    f"  {i}: {event.type.value} at {event.node} (parent: {event.parent})"
                )

        except ImportError as e:
            print(f"ImportError: {e}")
            print("BFS planning requires algokit modules to be properly installed")
            raise

        except Exception as e:
            print(f"Error generating BFS plan: {e}")
            print(f"Graph creation failed with obstacles: {self.obstacles}")
            print(f"Start: {self.start_pos}, Goal: {self.goal_pos}")
            raise

    def _calculate_max_depth_and_colors(self) -> None:
        """Calculate maximum depth from BFS events and generate color gradient."""
        # Find the maximum depth from enqueue events
        max_depth = 0
        for event in self.bfs_events:
            if event.type == EventType.ENQUEUE:
                depth = event.data.get("depth", 0)
                max_depth = max(max_depth, depth)

        # Also check layer complete events
        for event in self.bfs_events:
            if event.type == EventType.LAYER_COMPLETE:
                depth = event.node  # For layer complete, node contains the depth
                max_depth = max(max_depth, depth)

        self.max_depth = max_depth

        # Generate color gradient from start (GREEN) to goal (GOLD)
        self.layer_colors = self._generate_color_gradient(
            start_color=m.GREEN_C,
            end_color=m.GOLD,
            num_colors=max_depth + 2,  # +2 to ensure we have enough colors
        )

        print(f"Calculated max depth: {max_depth}")
        print(f"Generated gradient: {len(self.layer_colors)} colors from GREEN to GOLD")

        # Phase 4A: Validate and prepare path data
        self._validate_and_prepare_path_data()

    def _generate_color_gradient(
        self, start_color: str, end_color: str, num_colors: int
    ) -> list[str]:
        """Generate a smooth color gradient between start and end colors."""
        if num_colors <= 1:
            return [start_color]

        colors = []
        for i in range(num_colors):
            # Calculate interpolation factor (0.0 to 1.0)
            t = i / (num_colors - 1)

            # Interpolate between start and end colors
            interpolated_color = m.interpolate_color(start_color, end_color, t)
            colors.append(interpolated_color)

        return colors

    def _validate_and_prepare_path_data(self) -> None:
        """Phase 4A: Validate path data and prepare visualization structures.

        Validates:
        1. Final path exists and is non-empty
        2. Path starts at start_pos and ends at goal_pos
        3. Path contains only valid grid coordinates
        4. Path doesn't go through obstacles
        5. Path is contiguous (each step is adjacent)

        Prepares:
        1. Path visualization data structure
        2. Screen position mapping for each path node
        3. Path segments for animation
        """
        print("🔍 Phase 4A: Validating and preparing path data...")

        # Step 1: Basic path validation
        if not self.final_path:
            print("❌ ERROR: Final path is empty!")
            self.path_validation_passed = False
            return

        if len(self.final_path) < 2:
            print("❌ ERROR: Final path must have at least 2 nodes (start and goal)")
            self.path_validation_passed = False
            return

        print(f"✅ Path length: {len(self.final_path)} nodes")

        # Step 2: Validate start and goal positions
        if self.final_path[0] != self.start_pos:
            print(
                f"❌ ERROR: Path doesn't start at start_pos {self.start_pos}, starts at {self.final_path[0]}"
            )
            self.path_validation_passed = False
            return

        if self.final_path[-1] != self.goal_pos:
            print(
                f"❌ ERROR: Path doesn't end at goal_pos {self.goal_pos}, ends at {self.final_path[-1]}"
            )
            self.path_validation_passed = False
            return

        print(
            f"✅ Path starts at {self.final_path[0]} and ends at {self.final_path[-1]}"
        )

        # Step 3: Validate each node in path
        for i, node in enumerate(self.final_path):
            x, y = node

            # Check bounds
            if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
                print(
                    f"❌ ERROR: Path node {i} {node} is out of bounds (grid: {self.grid_width}x{self.grid_height})"
                )
                self.path_validation_passed = False
                return

            # Check obstacles
            if node in self.obstacles:
                print(f"❌ ERROR: Path node {i} {node} is an obstacle!")
                self.path_validation_passed = False
                return

        print("✅ All path nodes are within bounds and not obstacles")

        # Step 4: Validate path contiguity (each step is adjacent)
        for i in range(len(self.final_path) - 1):
            current = self.final_path[i]
            next_node = self.final_path[i + 1]

            # Check if nodes are adjacent (Manhattan distance = 1)
            dx = abs(current[0] - next_node[0])
            dy = abs(current[1] - next_node[1])

            if not (dx == 1 and dy == 0) and not (dx == 0 and dy == 1):
                print(
                    f"❌ ERROR: Path step {i}→{i + 1}: {current}→{next_node} is not adjacent (dx={dx}, dy={dy})"
                )
                self.path_validation_passed = False
                return

        print("✅ All path steps are contiguous (adjacent nodes)")

        # Step 5: Prepare visualization data
        self._create_path_visualization_data()

        print(
            "🎯 Phase 4A complete: Path validation passed and visualization data prepared!"
        )
        self.path_validation_passed = True

    def _create_path_visualization_data(self) -> None:
        """Create visualization data structures for path reconstruction animation."""
        print("📊 Creating path visualization data...")

        # Clear previous data
        self.path_visualization_data = {}
        self.path_segments = []
        self.path_dots_data = []

        # Check if grid_visualizer is available (might not be during testing)
        if not hasattr(self, "grid_visualizer") or self.grid_visualizer is None:
            print("⚠️ Grid visualizer not available yet - creating placeholder data")
            # Create placeholder screen positions using cell centers calculation
            for _i, grid_pos in enumerate(self.final_path):
                # Use the same calculation as _calculate_cell_centers_shifted
                x, y = grid_pos
                grid_origin_x = -(self.grid_width * self.cell_size) / 2
                grid_origin_y = (self.grid_height * self.cell_size) / 2 - 0.5
                cell_x = grid_origin_x + (x + 0.5) * self.cell_size
                cell_y = grid_origin_y - (y + 0.5) * self.cell_size
                screen_pos = m.np.array([cell_x, cell_y, 0])
                self.path_dots_data.append((screen_pos, grid_pos))
        else:
            # Convert path coordinates to screen positions using grid visualizer
            for i, grid_pos in enumerate(self.final_path):
                screen_pos = self.grid_visualizer.get_cell_center(grid_pos)
                self.path_dots_data.append((screen_pos, grid_pos))

                print(f"  Node {i}: grid {grid_pos} → screen {screen_pos}")

        # Create path segments (lines between consecutive nodes)
        for i in range(len(self.path_dots_data) - 1):
            start_pos = self.path_dots_data[i][0]  # Screen position
            end_pos = self.path_dots_data[i + 1][0]  # Screen position
            self.path_segments.append((start_pos, end_pos))

            if hasattr(self, "grid_visualizer") and self.grid_visualizer is not None:
                print(f"  Segment {i}: {start_pos} → {end_pos}")

        # Store comprehensive visualization data
        self.path_visualization_data = {
            "total_nodes": len(self.final_path),
            "total_segments": len(self.path_segments),
            "path_length": len(self.final_path) - 1,  # Distance is nodes - 1
            "start_node": self.final_path[0],
            "goal_node": self.final_path[-1],
            "all_nodes": self.final_path.copy(),
            "screen_positions": [pos for pos, _ in self.path_dots_data],
            "segments": self.path_segments.copy(),
            "depth_at_each_node": [
                self._get_node_depth(node) for node in self.final_path
            ],
        }

        print("✅ Created visualization data:")
        print(f"  - {len(self.final_path)} nodes")
        print(f"  - {len(self.path_segments)} segments")
        print(f"  - Path length: {len(self.final_path) - 1}")
        print(f"  - Depths: {self.path_visualization_data['depth_at_each_node']}")

    def _get_node_depth(self, node: tuple[int, int]) -> int:
        """Get the depth at which a node was discovered during BFS."""
        for event in self.bfs_events:
            if event.type == EventType.ENQUEUE and event.node == node:
                return event.data.get("depth", 0)
        return 0  # Fallback

    def get_path_validation_status(self) -> dict[str, Any]:
        """Get comprehensive path validation status and debug information.

        Returns:
            Dictionary containing validation status and debug information
        """
        return {
            "validation_passed": self.path_validation_passed,
            "final_path": self.final_path.copy() if self.final_path else [],
            "path_length": len(self.final_path) if self.final_path else 0,
            "start_pos": self.start_pos,
            "goal_pos": self.goal_pos,
            "grid_dimensions": (self.grid_width, self.grid_height),
            "obstacles": self.obstacles.copy(),
            "visualization_data_ready": bool(self.path_visualization_data),
            "path_segments_count": len(self.path_segments),
            "path_dots_count": len(self.path_dots_data),
            "max_depth": self.max_depth,
            "depth_at_each_node": self.path_visualization_data.get(
                "depth_at_each_node", []
            )
            if self.path_validation_passed
            else [],
        }

    def debug_path_data(self) -> None:
        """Print comprehensive debug information about path data."""
        print("🔍 PATH DATA DEBUG INFORMATION:")
        print("=" * 50)

        status = self.get_path_validation_status()

        print(
            f"Validation Status: {'✅ PASSED' if status['validation_passed'] else '❌ FAILED'}"
        )
        print(f"Path Length: {status['path_length']}")
        print(f"Start Position: {status['start_pos']}")
        print(f"Goal Position: {status['goal_pos']}")
        print(f"Grid Dimensions: {status['grid_dimensions']}")
        print(f"Obstacles: {status['obstacles']}")
        print(f"Max Depth: {status['max_depth']}")

        if status["validation_passed"]:
            print(f"Final Path: {status['final_path']}")
            print(f"Depths at each node: {status['depth_at_each_node']}")
            print(f"Visualization data ready: {status['visualization_data_ready']}")
            print(f"Path segments: {status['path_segments_count']}")
            print(f"Path dots: {status['path_dots_count']}")
        else:
            print("❌ Path validation failed - no detailed data available")

        print("=" * 50)

    def process_next_event(self) -> bool:
        """Process the next BFS event and animate it.

        Returns:
            True if an event was processed, False if no more events
        """
        if not hasattr(self, "event_index"):
            self.event_index = 0

        if self.event_index >= len(self.bfs_events):
            return False

        event = self.bfs_events[self.event_index]
        print(
            f"Processing event {self.event_index}: {event.type.value} at {event.node}"
        )

        # Handle different event types
        if event.type == EventType.ENQUEUE:
            self._animate_enqueue_event(event)
        elif event.type == EventType.DEQUEUE:
            self._animate_dequeue_event(event)
        elif event.type == EventType.LAYER_COMPLETE:
            self._animate_layer_complete_event(event)
        elif event.type == EventType.DISCOVER:
            # Do nothing - token already showing from ENQUEUE
            print(f"  → Skipping {event.type.value} event (token already showing)")
        else:
            # Skip other event types for now
            print(f"  → Skipping {event.type.value} event")

        self.event_index += 1
        return True

    def _animate_discover(self, event: SearchEvent) -> None:
        """Animate discovering a node by placing a token."""
        node = event.node

        # Skip if this is an obstacle
        if node in self.obstacles:
            return

        # Create token
        token = Token(cell_size=self.cell_size)

        # Position it at the cell center using GridVisualizer
        self.grid_visualizer.place_token(token, node)  # type: ignore[union-attr]

        # Store the token for later reference
        self.tokens[node] = token
        self.cell_states[node] = "frontier"

        # Animate the token entrance
        token.animate_entrance(self)

        print(f"  → Discovered {node} - placed token")

    def _animate_enqueue_event(self, event: SearchEvent) -> None:
        """Animate enqueuing a node: place frontier token on grid + copy to snake queue."""
        node = event.node

        # Skip if this is an obstacle
        if node in self.obstacles:
            return

        # Skip if this is the start node (already placed and enqueued)
        if node == self.start_pos:
            print(f"  → Skipping ENQUEUE for start node {node} (already in place)")
            return

        # Skip if we already have a token for this node
        if node in self.tokens:
            print(f"  → Token already exists for {node}, just copying to queue")
            source_token = self.tokens[node]
            token_copy = source_token.copy()
            self.snaking_queue.enqueue(token=token_copy, scene=self)
            return

        # Step 1: Create and place frontier token on grid (special handling for goal)
        if node == self.goal_pos:
            # For goal node, don't create a new token - the goal star is already there
            # Just use the existing goal star as the "token"
            token = self.goal_star
            self.tokens[node] = token
            self.cell_states[node] = "frontier"
            print(f"    Using existing goal star as token for {node}")
        else:
            # For regular nodes, create frontier token
            token = Token(cell_size=self.cell_size)
            self.grid_visualizer.place_token(token, node)  # type: ignore[union-attr]
            self.tokens[node] = token
            self.cell_states[node] = "frontier"
            # Animate the token entrance
            token.animate_entrance(self)

        # Step 2: Update Queue count immediately when enqueueing to snake queue
        token_copy = token.copy()
        self.snaking_queue.enqueue(token=token_copy, scene=self)
        queue_count = (
            len(self.snaking_queue.tokens)
            if hasattr(self.snaking_queue, "tokens")
            else 0
        )
        self._update_hud(queue=queue_count)

        # Step 3: Update Depth immediately after enqueueing (when new depth is found)
        current_depth = event.data.get("depth", 0)
        self._update_hud(depth=current_depth)

        # Step 4: Update Frontier count just after frontier token is added to grid overlay
        frontier_count = len(
            [pos for pos, state in self.cell_states.items() if state == "frontier"]
        )
        self._update_hud(frontier=frontier_count)

        # Step 5: Draw path from start to this enqueued node
        self._draw_path_to_enqueued_node(node, event.parent)

        # Step 6: Check if this is the goal being discovered
        if node == self.goal_pos and not self.goal_discovered:
            self._animate_goal_discovery_ping()
            self.goal_discovered = True

        print(f"  → Enqueued {node}: placed frontier token + copied to queue")

    def _animate_goal_discovery_ping(self) -> None:
        """Animate goal discovery ping when goal is first enqueued (added to frontier)."""
        print("  🎯 Goal discovered! Adding to frontier...")

        # Subtle glow around goal star using ShowPassingFlash
        if hasattr(self, "goal_star"):
            self.play(
                m.ShowPassingFlash(
                    m.SurroundingRectangle(
                        self.goal_star, color=m.GOLD, stroke_width=4
                    ),
                    time_width=0.5,
                ),
                run_time=self.get_animation_time("discovery_ping"),
            )

        # Small "Found in frontier!" text
        discovery_text = (
            m.Text("Found in frontier!", font_size=18, color=m.GOLD).next_to(
                self.goal_star, m.RIGHT, buff=0.3
            )
            if hasattr(self, "goal_star")
            else m.Text("Found in frontier!", font_size=18, color=m.GOLD)
        )

        # Animate text: fade in, hold, fade out
        self.play(
            m.FadeIn(discovery_text, shift=m.DOWN * 0.1),
            run_time=self.get_animation_time("discovery_text_fade_in"),
        )
        self.wait(self.get_wait_time("discovery_ping"))
        self.play(
            m.FadeOut(discovery_text, shift=m.UP * 0.1),
            run_time=self.get_animation_time("discovery_text_fade_out"),
        )

        print("  → Goal discovery ping complete - continuing BFS...")

    def _animate_dequeue_event(self, event: SearchEvent) -> None:
        """Animate dequeuing a node with the specified sequence."""
        node = event.node

        # Skip if this is an obstacle
        if node in self.obstacles:
            return

        # Check if queue is not empty
        if self.snaking_queue.is_empty():
            print(f"  → Warning: Queue is empty, cannot dequeue {node}")
            return

        # Check if we have the original token for this node
        if node not in self.tokens:
            print(f"  → Warning: No original token found for {node}")
            return

        # Step 1: Dequeue token from snake queue (goes back to original position and grows)
        dequeued_token = self.snaking_queue.dequeue(scene=self)

        # Step 2: Update Queue count immediately when dequeueing from snake queue
        queue_count = (
            len(self.snaking_queue.tokens)
            if hasattr(self.snaking_queue, "tokens")
            else 0
        )
        self._update_hud(queue=queue_count)

        self.wait(self.get_wait_time("dequeue_delay"))

        # Step 3: Remove the dequeued token from scene (reveals original)
        self.remove(dequeued_token)

        # Step 4: Change the original token's color to RED_A (except for goal)
        original_token = self.tokens[node]
        if node != self.goal_pos:
            # Regular nodes turn red when visited
            original_token.set_fill(m.RED_A, opacity=0.8)
        else:
            # Goal keeps its gold color - don't change it
            print("    Goal token keeps its original gold color")

        # Update cell state to visited
        self.cell_states[node] = "visited"

        # Step 5: Update Visited count exactly where it is right now (after state change)
        visited_count = len(
            [pos for pos, state in self.cell_states.items() if state == "visited"]
        )
        self._update_hud(visited=visited_count)

        # Step 6: Check if this is the goal being dequeued
        if node == self.goal_pos and not self.goal_dequeued:
            self.goal_dequeued = True
            print("  🎉 Goal dequeued! Triggering celebration...")
            self._celebrate_goal()

        print(f"  → Dequeued {node} - original token now red (visited)")

    def _animate_layer_complete_event(self, event: SearchEvent) -> None:
        """Animate layer completion with hybrid pulse system."""
        completed_depth = (
            event.node
        )  # For layer complete events, node contains the depth
        nodes_count = event.data.get("nodes_count", 0)
        nodes = event.data.get("nodes", [])

        # Always draw the NEXT layer (depth + 1) to avoid off-by-one appearance
        display_depth = completed_depth + 1

        print(
            f"  → Layer {completed_depth} complete, drawing layer {display_depth} with {nodes_count} nodes: {nodes}"
        )

        # Skip if already completed
        if display_depth in self.completed_depths:
            return

        self.completed_depths.add(display_depth)

        # Get color for the display depth from gradient
        if display_depth < len(self.layer_colors):
            color = self.layer_colors[display_depth]
        else:
            # Fallback to modulo if we somehow exceed the gradient
            color = self.layer_colors[display_depth % len(self.layer_colors)]

        # Create rings around all nodes at this depth
        ring_elements = []
        for node in nodes:
            if node in self.obstacles:  # Skip obstacle nodes
                continue
            cell_center = self.grid_visualizer.get_cell_center(node)  # type: ignore[union-attr]
            ring = m.Square(
                side_length=self.cell_size * 1.2,  # Slightly larger than cell
                stroke_color=color,
                stroke_width=3,
                fill_color=color,
                fill_opacity=0.1,  # Subtle fill to make color more visible
            ).move_to(cell_center)
            ring_elements.append(ring)

        if not ring_elements:
            return

        # Create depth ring group
        depth_ring_group = m.VGroup(*ring_elements)
        self.depth_rings[display_depth] = depth_ring_group

        # Step 1: Bright pulse that fades to persistent state
        # Set initial bright appearance for pulse effect
        for ring in ring_elements:
            ring.set_stroke(width=6, opacity=1.0)

        # Add rings to scene immediately so they're visible
        self.add(depth_ring_group)

        # Animate the pulse: bright -> subtle persistent using set_opacity
        self.play(
            m.LaggedStart(
                *[
                    m.AnimationGroup(
                        ring.animate.set_stroke(width=2), ring.animate.set_opacity(0.3)
                    )
                    for ring in ring_elements
                ],
                lag_ratio=0.1,
            ),
            run_time=self.get_animation_time("layer_complete_pulse"),
        )

        # Rings are now persistent at 30% opacity and will stay visible

        # Step 3: Show completion note (positioned below grid)
        completion_text = m.Text(
            f"Depth {display_depth} Complete - {nodes_count} nodes explored",
            font_size=20,
            color=color,
        ).move_to(m.DOWN * self.text_below_grid_offset)

        self.play(
            m.FadeIn(completion_text, shift=m.DOWN * 0.2),
            run_time=self.get_animation_time("layer_complete_pulse") * 0.5,
        )
        self.wait(self.get_wait_time("layer_complete_wait"))
        self.play(
            m.FadeOut(completion_text, shift=m.UP * 0.2),
            run_time=self.get_animation_time("layer_complete_pulse") * 0.5,
        )

        print(f"  → Layer {display_depth} pulse complete - rings now persistent")

    def _show_all_depth_rings(self) -> None:
        """Make all depth rings visible for final educational state."""
        print("Making all depth rings visible for final state")
        for _depth, ring_group in self.depth_rings.items():
            # Ensure all rings are visible and properly styled
            for ring in ring_group:
                ring.set_stroke(
                    opacity=0.3, width=2
                )  # Slightly more visible for final state

    def _highlight_depth_rings(self) -> None:
        """Brighten all depth rings for final educational moment."""
        print("Highlighting all depth rings for educational moment")

        # Create a brief brightening animation for all rings
        all_rings = []
        for ring_group in self.depth_rings.values():
            all_rings.extend(ring_group)

        if all_rings:
            # Brief highlight animation
            self.play(
                m.LaggedStart(
                    *[
                        ring.animate.set_stroke(opacity=0.8, width=3)
                        for ring in all_rings
                    ],
                    lag_ratio=0.05,
                ),
                run_time=self.get_animation_time("depth_ring_highlight"),
            )
            # Return to subtle but visible state
            self.play(
                m.LaggedStart(
                    *[
                        ring.animate.set_stroke(opacity=0.3, width=2)
                        for ring in all_rings
                    ],
                    lag_ratio=0.02,
                ),
                run_time=self.get_animation_time("depth_ring_highlight") * 0.67,
            )

    def _draw_path_to_enqueued_node(
        self, node: tuple[int, int], parent: tuple[int, int] | None
    ) -> None:
        """Draw complete path from start to newly enqueued node with dots and lines."""
        print(f"    Drawing complete path to {node} with parent {parent}")

        if not parent:
            print(f"    No parent for {node}, skipping path")
            return  # No parent, can't draw path

        # Step 1: Start from the newly enqueued node and trace back through all parents
        complete_path = []
        current_node = node

        # Step 2: Trace back through all parents until reaching the start
        while current_node is not None:
            complete_path.append(current_node)
            # Find parent of current node from BFS events
            current_parent = self._find_parent_of_node(current_node)
            if current_parent is not None:
                current_node = current_parent
            else:
                break

        # Reverse to get start -> ... -> node order
        complete_path.reverse()
        print(f"    Complete path: {complete_path}")

        # Step 3: Draw dots and lines for every step in the actual shortest path
        dots = []
        for pos in complete_path:
            cell_center = self.grid_visualizer.get_cell_center(pos)  # type: ignore[union-attr]
            dot = m.Dot(radius=0.05, color=m.YELLOW).move_to(cell_center)
            dots.append(dot)

        print(f"    Created {len(dots)} dots for complete path")

        # Create lines between consecutive dots
        lines = []
        for i in range(len(dots) - 1):
            line = m.Line(
                dots[i].get_center(),
                dots[i + 1].get_center(),
                color=m.YELLOW,
                stroke_width=3,
            )
            lines.append(line)

        print(f"    Created {len(lines)} lines connecting all path nodes")

        # Step 4: Animate the complete path visualization
        # Dots appear in sequence from start to end (this adds them to the scene)
        self.play(
            m.LaggedStart(*[m.GrowFromCenter(dot) for dot in dots], lag_ratio=0.1),
            run_time=self.get_animation_time("path_dot_grow"),
        )
        # Lines draw in sequence showing the path (this adds them to the scene)
        self.play(
            m.LaggedStart(*[m.Create(line) for line in lines], lag_ratio=0.1),
            run_time=self.get_animation_time("path_line_create"),
        )

        # Fade out everything
        all_path_elements = m.VGroup(*dots, *lines)
        self.play(
            m.FadeOut(all_path_elements),
            run_time=self.get_animation_time("path_cleanup") * 0.5,
        )

        print(
            f"    Complete path animation finished for {node} (path length: {len(complete_path)})"
        )

    def _find_parent_of_node(self, node: tuple[int, int]) -> tuple[int, int] | None:
        """Find the parent of a node by looking through BFS events."""
        # Look through events to find when this node was enqueued
        for event in self.bfs_events:
            if event.type == EventType.ENQUEUE and event.node == node:
                return event.parent
        return None

    def step_through_bfs(self, max_events_displayed: int = 9999999) -> None:
        """Step through BFS events up to a maximum number.

        Args:
            max_events_displayed: Maximum number of events to process (default: 1000)
        """
        print(f"Starting to process up to {max_events_displayed} events...")
        print(f"Total events available: {len(self.bfs_events)}")

        events_processed = 0

        while events_processed < max_events_displayed:
            if not self.process_next_event():
                print(
                    f"No more events to process! Finished after {events_processed} total events."
                )
                break

            events_processed += 1
            # Pause between events

        print(f"Finished processing {events_processed} events.")

    def _celebrate_goal(self) -> None:
        """Celebrate reaching the goal with visual effects and depth information."""
        # Get goal depth from final path length
        goal_depth = len(self.final_path) - 1 if self.final_path else self.max_depth

        print(f"🎉 Celebrating goal reached at depth {goal_depth}!")

        # Freeze motion for dramatic effect
        self.wait(self.get_wait_time("celebration_brief"))

        # Flash the goal star
        if hasattr(self, "goal_star"):
            self.play(
                m.Flash(self.goal_star.get_center(), color=m.GOLD),
                run_time=self.get_animation_time("goal_discovery_ping"),
            )

        # Create confetti effect around the goal
        confetti = []
        goal_center = (
            self.goal_star.get_center() if hasattr(self, "goal_star") else m.ORIGIN
        )

        for _ in range(15):
            confetti_dot = m.Dot(
                radius=0.05,
                color=m.np.random.choice(
                    [m.RED, m.GREEN, m.BLUE, m.YELLOW, m.PINK, m.TEAL]
                ),
            ).move_to(goal_center)
            confetti.append(confetti_dot)

        # Animate confetti explosion
        animations = []
        for dot in confetti:
            target_pos = (
                goal_center
                + m.np.random.uniform(-1.5, 1.5) * m.RIGHT
                + m.np.random.uniform(-1, 1) * m.UP
            )
            animations.append(dot.animate.move_to(target_pos).scale(0.3))

        self.play(
            m.LaggedStart(*[m.GrowFromCenter(dot) for dot in confetti], lag_ratio=0.1),
            run_time=self.get_animation_time("confetti_effects") * 0.4,
        )
        self.play(
            m.LaggedStart(*animations, lag_ratio=0.05),
            run_time=self.get_animation_time("confetti_effects") * 0.6,
        )

        # Goal reached text with depth information
        goal_text = m.Text(
            f"Goal reached at depth {goal_depth}!", font_size=32, color=m.GOLD
        ).move_to(m.DOWN * self.text_below_grid_offset)

        self.play(
            m.Write(goal_text), run_time=self.get_animation_time("goal_text_reveal")
        )
        self.wait(self.get_wait_time("celebration_pause"))

        # Fade out celebration effects
        self.play(
            m.FadeOut(goal_text),
            m.FadeOut(m.VGroup(*confetti)),
            run_time=self.get_animation_time("celebration_fade_out"),
        )

        # Phase 4B: Path reconstruction after celebration
        self._animate_path_reconstruction()

    def _animate_path_reconstruction(self) -> None:
        """Phase 4B: Animate dramatic path backtracking showing BFS guarantees shortest path.

        Animation sequence:
        1. Start from goal position
        2. For each step in reverse path:
           - Highlight parent arrow (if exists)
           - Draw path segment with Create animation
           - Brief pause between segments
        3. Build path from goal back to start
        4. Show final path with educational message
        """
        if not self.path_validation_passed:
            print("❌ Path reconstruction skipped - validation failed")
            return

        print("🎬 Phase 4B: Starting dramatic path reconstruction animation...")

        # Freeze motion for dramatic effect
        self.wait(self.get_wait_time("path_reconstruction_brief"))

        # Educational introduction text
        intro_text = m.Text(
            "How did BFS find the shortest path?", font_size=28, color=m.WHITE
        ).move_to(m.DOWN * self.text_below_grid_offset)

        self.play(
            m.Write(intro_text), run_time=self.get_animation_time("educational_text")
        )
        self.wait(self.get_wait_time("educational_pause"))
        self.play(
            m.FadeOut(intro_text),
            run_time=self.get_animation_time("educational_text") * 0.5,
        )

        # Step 1: Create path reconstruction elements
        path_segments = []
        path_dots = []

        # Create path segments and dots for the complete shortest path
        for i in range(len(self.final_path) - 1):
            # Get screen positions
            start_screen_pos = self.path_visualization_data["screen_positions"][i]
            end_screen_pos = self.path_visualization_data["screen_positions"][i + 1]

            # Create path segment (line)
            segment = m.Line(
                start_screen_pos, end_screen_pos, color=m.PINK, stroke_width=8
            )
            path_segments.append(segment)

            # Create path dot for current position
            dot = m.Dot(radius=0.08, color=m.PINK, fill_opacity=0.9).move_to(
                start_screen_pos
            )
            path_dots.append(dot)

        # Add final goal dot
        goal_screen_pos = self.path_visualization_data["screen_positions"][-1]
        goal_dot = m.Dot(radius=0.08, color=m.PINK, fill_opacity=0.9).move_to(
            goal_screen_pos
        )
        path_dots.append(goal_dot)

        print(
            f"Created {len(path_segments)} path segments and {len(path_dots)} path dots"
        )

        # Step 2: Phase 4C - Moving dot tracer animation
        print("🎬 Phase 4C: Starting moving dot tracer animation...")
        # Start with goal dot visible
        self.add(goal_dot)
        self.play(
            m.GrowFromCenter(goal_dot),
            run_time=self.get_animation_time("path_dot_grow"),
        )
        # Create path polyline for smooth tracing
        path_polyline = m.VMobject()
        path_points = []

        # Animate backtracking: draw path segments from goal back to start
        for i in range(len(path_segments) - 1, -1, -1):  # Reverse order
            segment = path_segments[i]
            child_dot = path_dots[i + 1]  # Child is the next position in path
            parent_dot = path_dots[i]  # Parent is the current position

            # Add segment to scene (invisible initially)
            # self.add(segment)

            # Add parent dot
            child_copy = child_dot.copy()
            self.add(child_copy)
            # self.add(parent_dot)
            self.play(
                child_copy.animate.move_to(parent_dot.get_center()),
                run_time=self.get_animation_time("path_line_create"),
            )

            # Animate: show parent dot first, then draw line from child to parent
            self.play(
                m.GrowFromCenter(parent_dot),
                run_time=self.get_animation_time("path_dot_grow"),
            )

        # Build the complete path as a polyline
        for i, _grid_pos in enumerate(self.final_path):
            screen_pos = self.path_visualization_data["screen_positions"][i]
            path_points.append(screen_pos)

        path_polyline.set_points_as_corners(path_points)
        path_polyline.set_stroke(width=0, opacity=0)  # Invisible, just for movement

        print(f"Created path polyline with {len(path_points)} points")

        # Create moving tracer dot
        tracer_dot = m.Dot(radius=0.12, color=m.YELLOW, fill_opacity=1.0).move_to(
            path_points[0]
        )  # Start at first point

        # # Add tracer dot to scene
        # self.add(tracer_dot)

        # Make all path segments invisible initially
        for segment in path_segments:
            segment.set_stroke(opacity=0)
            self.add(segment)

        # Create synchronized animations
        animations = []

        # Synchronized path segment reveals
        for i, segment in enumerate(path_segments):
            # Calculate when this segment should appear (based on path position)
            segment_start_time = i / len(path_segments) * 3.0
            segment_duration = 0.3  # Quick reveal

            segment_animation = m.AnimationGroup(
                segment.animate.set_stroke(opacity=1.0), lag_ratio=0.0
            )
            segment_animation.begin_time = segment_start_time
            segment_animation.run_time = segment_duration
            animations.append(segment_animation)

            # Main tracer movement along path
        tracer_animation = m.MoveAlongPath(
            tracer_dot,
            path_polyline,
            run_time=self.get_animation_time(
                "tracer_movement"
            ),  # 3 second smooth movement
        )
        animations.append(tracer_animation)

        # Execute all animations together
        self.play(*animations)

        # Brief pause to admire the complete path
        self.wait(self.get_wait_time("path_drawing_complete"))

        # Fade out tracer dot
        self.play(
            m.FadeOut(tracer_dot),
            run_time=self.get_animation_time("path_cleanup") * 0.4,
        )

        print("🎯 Phase 4C complete: Moving dot tracer animation finished!")

        # Step 3: Final path highlight and educational message
        self.wait(self.get_wait_time("path_reconstruction_pause"))

        # Highlight the complete path
        all_path_elements = m.VGroup(*path_segments, *path_dots)

        # Pulse effect on the complete path
        self.play(
            all_path_elements.animate.set_stroke(width=10),
            run_time=self.get_animation_time("path_highlight_pulse") * 0.5,
        )

        self.play(
            all_path_elements.animate.set_stroke(width=8),
            run_time=self.get_animation_time("path_highlight_pulse") * 0.5,
        )

        # Educational conclusion
        conclusion_text = m.Text(
            f"Shortest path found: {len(self.final_path) - 1} steps!",
            font_size=26,
            color=m.PINK,
        ).move_to(m.DOWN * self.text_below_grid_offset)

        self.play(
            m.Write(conclusion_text),
            run_time=self.get_animation_time("conclusion_text"),
        )
        self.wait(
            self.get_wait_time("final_study_pause")
        )  # Longer pause to study the path

        # Step 4: Clean up - fade out path reconstruction elements
        self.play(
            m.FadeOut(all_path_elements),
            m.FadeOut(conclusion_text),
            run_time=self.get_animation_time("path_cleanup"),
        )

        print("🎯 Phase 4B complete: Path reconstruction animation finished!")

    def enqueue_token(self, source_token: m.Mobject) -> None:
        """Enqueue any token into the snaking queue by copying it.

        Args:
            source_token: The token to copy and enqueue
        """
        # Create a copy of the source token at its current position
        token_copy = source_token.copy()

        # Enqueue the copy - SnakeQueue will handle sizing
        self.snaking_queue.enqueue(token=token_copy, scene=self)

    def _establish_scene(self) -> None:
        """Scene 0: Title + Grid establishment (from bls_video.md)."""
        # 1. Title + grid on
        title = m.Text(
            f"BFS Grid: {self.scenario.name}", font_size=36, color=m.WHITE
        ).to_edge(m.UP)

        # Create grid with growth animation
        self.create_grid_with_growth_animation()

        # Play title + grid together (as prescribed)
        self.play(m.Write(title), run_time=1.8)

        # Polish: subtle grid shimmer
        self.play(m.ApplyWave(self.grid_group, amplitude=0.05), run_time=0.8)

        # 2. Subtitle + legend (1.5s)
        subtitle = m.Text(
            "Shortest paths on unweighted graphs", font_size=20, color=m.GRAY
        ).next_to(title, m.DOWN, buff=0.2)

        # Build legend using the LegendPanel component
        legend = create_bfs_legend()
        legend.to_corner(m.DR, buff=0.5)

        # Animate subtitle + legend together with LaggedStart
        self.play(
            m.LaggedStart(
                m.Write(subtitle), m.FadeIn(legend, shift=m.DOWN * 0.25), lag_ratio=0.2
            ),
            run_time=1.5,
        )

        # 3. Frame the stage (0.8s)
        frame_rect = m.SurroundingRectangle(
            self.grid_group, color=m.GRAY_D, stroke_width=2
        )

        # Fade in frame, then fade opacity to 30%
        self.play(m.FadeIn(frame_rect), run_time=0.4)
        self.play(frame_rect.animate.set_stroke(opacity=0.3), run_time=0.4)

    def _cast_actors(self) -> None:
        """Scene 1: Cast the actors (3–4s) - Start + Goal + Obstacles."""
        # 4. Start + Goal placement
        self._place_start_goal()

        # 5. Flooded harbor obstacles
        self._show_water_areas()

    def _place_start_goal(self) -> None:
        """Place start and goal markers with animations using proper token components."""
        # Start token using StartToken component
        start_cell = self.grid_visualizer.get_cell(self.start_pos)  # type: ignore[union-attr]
        self.start_token = StartToken(cell_size=self.cell_size)
        self.start_token.move_to(start_cell.get_center())

        # Goal token using GoalToken component
        goal_cell = self.grid_visualizer.get_cell(self.goal_pos)  # type: ignore[union-attr]
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

    def _build_queue_panel(self) -> None:
        """Build queue visualization panel on the right side."""
        # Queue panel background (moved higher to avoid legend)
        self.queue_panel = (
            m.RoundedRectangle(
                width=2.5,
                height=3.0,
                corner_radius=0.2,
                stroke_color=m.WHITE,
                stroke_width=2,
                fill_color=m.BLACK,
                fill_opacity=0.8,
            )
            .to_edge(m.RIGHT, buff=0.5)
            .shift(m.UP * 0.8)
        )

        # Queue title
        self.queue_title = m.Text("Queue (FIFO)", font_size=18, color=m.WHITE).next_to(
            self.queue_panel, m.UP, buff=0.2
        )

        # Create snaking queue
        self.snaking_queue = SnakeQueue(tokens_wide=6, tokens_tall=3, token_size=0.25)
        self.snaking_queue.move_to(self.queue_panel.get_center())

        # Animate queue panel sliding in
        self.play(
            m.FadeIn(self.queue_panel, shift=m.RIGHT * 0.5),
            m.Write(self.queue_title),
            run_time=1.0,
        )

        # Add snaking queue to scene
        self.add(self.snaking_queue)

    def _build_hud(self) -> None:
        """Build HUD counters using the HUDPanel component."""
        # Initial HUD values
        hud_values = {"Visited": 0.0, "Frontier": 0.0, "Depth": 0.0, "Queue": 0.0}

        # Create HUD panel with proper styling
        self.hud_panel = HUDPanel(
            values=hud_values,
            max_lines=2,
            font_size=16,
            corner_radius=0.1,
            stroke_width=2,
            stroke_color=m.WHITE,
            fill_color=m.BLACK,
            fill_opacity=0.1,
        )

        # Position HUD in top-left area
        self.hud_panel.to_edge(m.LEFT)  # , buff=0.3)  #.shift(m.DOWN * 2.0)

        # Animate HUD sliding in
        self.play(m.FadeIn(self.hud_panel, shift=m.UP * 0.2), run_time=0.8)

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
            self._update_hud(
                visited=visited_count,
                frontier=frontier_count,
                depth=self.current_depth,
                queue=frontier_count,
            )

            # Small pause between events
            self.wait(self.get_wait_time("dequeue_delay") * 0.33)  # Very brief pause

    def _animate_dequeue(self, node: tuple[int, int]) -> None:
        """Animate dequeuing a node from the queue."""
        # Remove token from queue
        if not self.snaking_queue.is_empty():
            _token = self.snaking_queue.dequeue(scene=self)

        # Mark cell as being processed using GridVisualizer
        cell = self.grid_visualizer.get_cell(node)  # type: ignore[union-attr]

        # Pulse the cell to show it's being processed
        self.play(m.Indicate(cell, scale_factor=1.1, color=m.YELLOW_C), run_time=0.3)

    def _animate_visit(self, node: tuple[int, int]) -> None:
        """Animate visiting a node."""
        cell = self.grid_visualizer.get_cell(node)  # type: ignore[union-attr]

        # Mark as visited
        cell.set_fill(m.YELLOW_E, opacity=0.4)
        self.cell_states[node] = "visited"

        # Pulse to show it's visited
        self.play(m.Indicate(cell, scale_factor=1.05, color=m.YELLOW_E), run_time=0.2)

    def _animate_enqueue(
        self, node: tuple[int, int], parent: tuple[int, int] | None
    ) -> None:
        """Animate enqueuing a node."""
        # Mark as frontier using GridVisualizer
        cell = self.grid_visualizer.get_cell(node)  # type: ignore[union-attr]
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
            stroke_width=2,
        )
        self.snaking_queue.enqueue(token=token, scene=self)

        # Show discovery
        self.play(m.Indicate(cell, scale_factor=1.05, color=m.BLUE_C), run_time=0.2)

    def _draw_parent_arrow(
        self, parent: tuple[int, int], child: tuple[int, int]
    ) -> None:
        """Draw an arrow from parent to child."""
        parent_cell = self.grid_visualizer.get_cell(parent)  # type: ignore[union-attr]
        child_cell = self.grid_visualizer.get_cell(child)  # type: ignore[union-attr]

        arrow = m.Arrow(
            parent_cell.get_center(),
            child_cell.get_center(),
            buff=0.1,
            stroke_width=2,
            color=m.WHITE,
        )

        self.parent_arrows[child] = arrow
        self.play(m.GrowArrow(arrow), run_time=0.3)

    def _animate_goal_discovery(self, node: tuple[int, int]) -> None:
        """Animate discovering the goal."""
        _cell = self.grid_visualizer.get_cell(node)  # type: ignore[union-attr]

        # Flash the goal
        self.play(m.Flash(self.goal_star.get_center(), color=m.GOLD), run_time=0.5)

    def generate_timing_report(
        self, save_to_file: bool = True, print_to_console: bool = True
    ) -> str:
        """Generate comprehensive timing usage report.

        Args:
            save_to_file: Whether to save report to a file
            print_to_console: Whether to print report to console

        Returns:
            The generated report as a string
        """
        if not hasattr(self, "timing_tracker"):
            print("⚠️ Timing tracker not available - no report generated")
            return ""

        # Generate comprehensive report
        report = self.timing_tracker.generate_comprehensive_report()

        if print_to_console:
            print("\n" + report)

        if save_to_file:
            # Save to output directory with timestamp
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output/bfs_timing_report_{timestamp}.txt"

            # Ensure output directory exists
            from pathlib import Path

            Path("output").mkdir(exist_ok=True)

            self.timing_tracker.save_report_to_file(filename, include_detailed=True)

        return report

    def print_timing_summary(self) -> None:
        """Print a quick summary of timing usage."""
        if not hasattr(self, "timing_tracker"):
            print("⚠️ Timing tracker not available")
            return

        total_requests = (
            self.timing_tracker.total_animation_requests
            + self.timing_tracker.total_wait_requests
            + self.timing_tracker.total_legacy_requests
        )

        print("\n🎬 TIMING SUMMARY:")
        print(f"   Total Requests: {total_requests}")
        print(f"   Animation Requests: {self.timing_tracker.total_animation_requests}")
        print(f"   Wait Requests: {self.timing_tracker.total_wait_requests}")
        print(f"   Legacy Requests: {self.timing_tracker.total_legacy_requests}")
        print(f"   Current Mode: {self.current_mode}")

    def get_timing_stats(self) -> dict[str, Any]:
        """Get timing statistics as a dictionary.

        Returns:
            Dictionary containing timing statistics
        """
        if not hasattr(self, "timing_tracker"):
            return {}

        return {
            "total_animation_requests": self.timing_tracker.total_animation_requests,
            "total_wait_requests": self.timing_tracker.total_wait_requests,
            "total_legacy_requests": self.timing_tracker.total_legacy_requests,
            "animation_requests": dict(self.timing_tracker.animation_requests),
            "wait_requests": dict(self.timing_tracker.wait_requests),
            "legacy_requests": dict(self.timing_tracker.legacy_requests),
            "current_mode": self.current_mode,
        }
