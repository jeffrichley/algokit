"""Render script for the BFS scene."""

import sys
from pathlib import Path

# Add the main project to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from agloviz.scenes.planning.bfs.breadth_first_search_scene import (
    BreadthFirstSearchScene,
)
from algokit.core.helpers import HarborNetScenario

# Create a simple test scenario
scenario = HarborNetScenario(
    name="Small Flooded Harbor",
    description="BFS visualization test",
    width=8, height=6,
    start=(0, 0), goal=(7, 5),
    obstacles={(2, 2), (3, 2), (4, 2), (2, 3), (4, 3), (5, 4), (6, 4)}
)

class RenderBfsScene(BreadthFirstSearchScene):
    """Renderable BFS scene for Manim."""
    
    def __init__(self, **kwargs):
        super().__init__(scenario, **kwargs)


if __name__ == "__main__":
    # This would typically be run with: manim -pql render_bfs_scene.py render_bfs_scene
    print("To render this scene, run:")
    print("manim -pql src/agloviz/scenes/planning/bfs/render_bfs_scene.py render_bfs_scene")
    render_bfs_scene()
