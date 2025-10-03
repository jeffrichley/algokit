from typing import Any

import manim as m


class StartToken(m.Dot):
    """Start token that extends Manim's Dot class directly."""

    def __init__(self, cell_size: float, **kwargs: Any) -> None:
        super().__init__(radius=cell_size * 0.3, color=m.GREEN_C, **kwargs)
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

    def __init__(self, cell_size: float, **kwargs: Any) -> None:
        super().__init__(n=5, outer_radius=cell_size * 0.3, color=m.GOLD, **kwargs)
        self.cell_size = cell_size
        self.label = m.Text("Goal", font_size=18, color=m.WHITE)

    def add_label_after_placement(self) -> None:
        """Add label positioned relative to token after it's placed in grid (like original)."""
        self.label.next_to(self, m.DOWN, buff=0.35)

    def animate_entrance(self, scene: m.Scene) -> None:
        """Animate the token entrance."""
        scene.play(m.GrowFromCenter(self), run_time=0.5)


class Token(m.Square):
    """Token that extends Manim's Square class directly."""

    def __init__(self, cell_size: float, **kwargs: Any) -> None:
        super().__init__(
            side_length=cell_size * 0.6,  # Same size as star (0.3 radius * 2)
            fill_color=m.PURPLE_A,
            fill_opacity=0.8,
            stroke_color=m.WHITE,
            stroke_width=2,
            **kwargs,
        )
        self.cell_size = cell_size

    def animate_entrance(self, scene: m.Scene) -> None:
        """Animate the token entrance."""
        scene.play(m.GrowFromCenter(self), run_time=0.3)


class WaterToken(m.VGroup):
    """Water token with wave effects that extends Manim's VGroup."""

    def __init__(self, cell_size: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cell_size = cell_size

        # Create the water base (square)
        self.water_base = m.Square(
            side_length=cell_size * 0.9,
            fill_color=m.BLUE,
            fill_opacity=0.5,
            stroke_color=m.BLUE_E,
            stroke_width=1,
        )

        # Create wave overlay (thin Arc pieces)
        wave_parts = []
        for i in range(3):  # 3 wave segments
            wave_arc = m.Arc(
                radius=cell_size * 0.15 + i * 0.02,
                angle=m.PI,
                color=m.BLUE_C,
                stroke_width=1,
            )
            wave_parts.append(wave_arc)

        self.wave_group = m.VGroup(*wave_parts)

        # Add both parts to the group
        self.add(self.water_base, self.wave_group)

    def animate_entrance(self, scene: m.Scene) -> None:
        """Animate the token entrance with water and waves."""
        # First animate the water base
        scene.play(m.GrowFromCenter(self.water_base), run_time=0.1)
        # Then animate the waves
        scene.play(m.GrowFromCenter(self.wave_group), run_time=0.1)
