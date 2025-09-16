"""Test scene for grid visualization.

This is a simple test scene that only draws a grid to test our
HarborGridScene implementation.
"""

import sys
from pathlib import Path

import manim as m

# Add the main project to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from algokit.core.helpers import HarborNetScenario
from algokit.viz.scenes import HarborGridScene


class TestGridScene(HarborGridScene):
    """Simple test scene that only draws a grid."""
    
    def __init__(self, **kwargs) -> None:
        """Initialize with a simple test scenario."""
        # Create a simple 5x5 test scenario
        scenario = HarborNetScenario(
            name="Test Grid",
            width=5,
            height=5,
            start=(0, 0),
            goal=(4, 4),
            obstacles=set()
        )
        super().__init__(scenario, **kwargs)
    
    def construct(self) -> None:
        """Construct the test grid visualization."""
        # Add a simple title first
        title = m.Text(
            "Test Grid - 5x5",
            font_size=24,
            color=m.WHITE
        ).to_edge(m.UP)
        self.play(m.Write(title))
        
        # Draw the grid with animation
        super().construct()
        
        # Animate the grid appearing
        self.play(m.Create(self.grid_group))
        
        # Wait a bit to see the result
        self.wait(2)
