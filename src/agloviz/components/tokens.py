from typing import Optional
import manim as m


class StartToken(m.Dot):
    """Start token that extends Manim's Dot class directly."""
    
    def __init__(self, cell_size: float, **kwargs):
        super().__init__(
            radius=cell_size * 0.3,
            color=m.GREEN_C,
            **kwargs
        )
        self.cell_size = cell_size
        self.label = m.Text("Start", font_size=18, color=m.WHITE)

    def add_label_after_placement(self) -> None:
        """Add label positioned relative to token after it's placed in grid (like original)."""
        self.label.next_to(self, m.UP, buff=0.25)

    def animate_entrance(self, scene: m.Scene) -> None:
        """Animate the token entrance."""
        scene.play(m.GrowFromCenter(self), run_time=0.5)


class GoalToken(m.Star):
    """Goal token that extends Manim's Star class directly."""
    
    def __init__(self, cell_size: float, **kwargs):
        super().__init__(
            n=5,
            outer_radius=cell_size * 0.3,
            color=m.GOLD,
            **kwargs
        )
        self.cell_size = cell_size
        self.label = m.Text("Goal", font_size=18, color=m.WHITE)

    def add_label_after_placement(self) -> None:
        """Add label positioned relative to token after it's placed in grid (like original)."""
        self.label.next_to(self, m.DOWN, buff=0.35)

    def animate_entrance(self, scene: m.Scene) -> None:
        """Animate the token entrance."""
        scene.play(m.GrowFromCenter(self), run_time=0.5)


class WaterCellToken(m.Square):
    """Water cell token that extends Manim's Square class directly."""
    
    def __init__(self, cell_size: float, **kwargs):
        super().__init__(
            side_length=cell_size * 0.9,
            fill_color=m.BLUE,
            fill_opacity=0.5,
            stroke_color=m.BLUE_E,
            **kwargs
        )
        self.cell_size = cell_size

    def animate_entrance(self, scene: m.Scene) -> None:
        """Animate the token entrance."""
        scene.play(m.GrowFromCenter(self), run_time=0.5)
