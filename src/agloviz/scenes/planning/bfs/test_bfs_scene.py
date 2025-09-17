"""Test script for the BFS scene."""

import sys
from pathlib import Path

# Add the main project to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from agloviz.scenes.planning.bfs.breadth_first_search_scene import (
    BreadthFirstSearchScene,
)
from algokit.core.helpers import HarborNetScenario


def test_bfs_scene():
    """Test the BFS scene creation."""
    # Create a simple test scenario
    scenario = HarborNetScenario(
        name="Test Harbor",
        description="Test scenario for BFS",
        width=8, height=6,
        start=(0, 0), goal=(7, 5),
        obstacles={(2, 2), (3, 2), (4, 2), (2, 3), (4, 3)}
    )
    
    # Create the scene
    scene = BreadthFirstSearchScene(scenario)
    
    print("✅ BFS scene created successfully!")
    print(f"   Scenario: {scene.scenario.name}")
    print(f"   Grid size: {scene.grid_width}x{scene.grid_height}")
    print(f"   Start: {scene.start_pos}")
    print(f"   Goal: {scene.goal_pos}")
    print(f"   Obstacles: {len(scene.obstacles)}")
    
    return scene


if __name__ == "__main__":
    test_bfs_scene()
