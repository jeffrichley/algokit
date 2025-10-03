"""Render script for the BFS scene."""

import os
import sys
from pathlib import Path

from agloviz.scenes.planning.bfs.breadth_first_search_scene import (
    BreadthFirstSearchScene,
)
from algokit.core.helpers import HarborNetScenario

# Add the main project to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


class RenderBfsScene(BreadthFirstSearchScene):
    """Renderable BFS scene for Manim."""

    def __init__(self, **kwargs: object) -> None:
        # Check for scenario file from environment variable first
        scenario_file = os.environ.get("AGLOVIZ_SCENARIO_FILE")

        if scenario_file:
            print(f"🎯 Using scenario from environment: {scenario_file}")
            super().__init__(
                scenario=None, **kwargs
            )  # Let constructor load from env var
        else:
            print("🏠 Using default scenario")
            # Create default scenario if no environment variable
            # scenario = HarborNetScenario(
            #     name="Small Flooded Harbor",
            #     description="BFS visualization test",
            #     width=8, height=6,
            #     start=(0, 0), goal=(7, 5),
            #     obstacles=[(2, 3), (2, 2), (3, 2), (4, 2), (4, 3), (5, 4), (6, 4)]
            # )
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
            super().__init__(scenario, **kwargs)


if __name__ == "__main__":
    # This would typically be run with: manim -pql render_bfs_scene.py render_bfs_scene
    print("To render this scene, run:")
    print(
        "manim -pql src/agloviz/scenes/planning/bfs/render_bfs_scene.py render_bfs_scene"
    )
    # render_bfs_scene()
