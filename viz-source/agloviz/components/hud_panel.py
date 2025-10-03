from typing import Any

from manim import (
    BLACK,
    DOWN,
    LEFT,
    RIGHT,
    WHITE,
    Integer,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
)
from manim.typing import Color

from agloviz.core.font_utils import get_font


class HUDPanel(VGroup):
    """Self-contained HUD panel with text labels and values.
    Automatically sizes background.
    """

    DEFAULT_CORNER_RADIUS = 0.15
    DEFAULT_STROKE_WIDTH = 2
    DEFAULT_BG_FILL_OPACITY = 0.0
    DEFAULT_STROKE_COLOR = WHITE
    DEFAULT_FILL_COLOR = BLACK

    def __init__(
        self,
        values: dict[str, float],
        max_lines: int = 2,
        font_size: int = 20,
        corner_radius: float | None = None,
        stroke_width: float | None = None,
        stroke_color: Color | None = None,
        fill_color: Color | None = None,
        fill_opacity: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        # Use provided style values or defaults
        self.corner_radius = (
            corner_radius if corner_radius is not None else self.DEFAULT_CORNER_RADIUS
        )
        self.stroke_width = (
            stroke_width if stroke_width is not None else self.DEFAULT_STROKE_WIDTH
        )
        self.stroke_color = (
            stroke_color if stroke_color is not None else self.DEFAULT_STROKE_COLOR
        )
        self.fill_color = (
            fill_color if fill_color is not None else self.DEFAULT_FILL_COLOR
        )
        self.fill_opacity = (
            fill_opacity if fill_opacity is not None else self.DEFAULT_BG_FILL_OPACITY
        )

        self.max_lines = max_lines
        self.font_size = font_size
        self.trackers: dict[str, Integer] = {}
        self.labels: dict[str, Text] = {}
        self.numbers: dict[str, Integer] = {}

        self._build_panel(values)

    def _build_panel(self, values: dict[str, float]) -> None:
        # Create label-number pairs
        rows: list[list[Any]] = [[] for _ in range(self.max_lines)]
        keys = list(values.keys())
        for i, key in enumerate(keys):
            tracker = Integer(
                int(values[key]),
                font_size=self.font_size,
            ).set_font(get_font("hud"))

            label = Text(
                f"{key}:",
                font_size=self.font_size,
            ).set_font(get_font("hud"))

            # Setup updater so `tracker` shows correct value
            label_num = VGroup(label, tracker).arrange(RIGHT, buff=0.1)

            self.trackers[key] = tracker
            self.labels[key] = label
            self.numbers[key] = tracker

            rows[i % self.max_lines].append(label_num)

        # Arrange the rows vertically
        line_groups = [VGroup(*row).arrange(RIGHT, buff=0.4) for row in rows if row]
        self.text_block = VGroup(*line_groups).arrange(
            DOWN, aligned_edge=LEFT, buff=0.25
        )

        # Compute size needed for background
        padding_x = 0.4
        padding_y = 0.4

        bg_width = self.text_block.width + padding_x
        bg_height = self.text_block.height + padding_y

        # After computing bg_width, bg_height:
        background = RoundedRectangle(
            width=bg_width,
            height=bg_height,
            corner_radius=self.corner_radius,
            stroke_width=self.stroke_width,
            stroke_color=self.stroke_color,
            fill_color=self.fill_color,
            fill_opacity=self.fill_opacity,
        )

        self.add(background, self.text_block)
        self.text_block.move_to(background.get_center())

    def update_values(
        self, new_values: dict[str, float], scene: Scene, run_time: float = 0.5
    ) -> None:
        """Animate updates to existing HUD values."""
        animations = []

        for key, new_val in new_values.items():
            if key not in self.trackers:
                continue  # Optional: log warning if desired

            tracker = self.trackers[key]

            # Animate the number change (ensure integer)
            animations.append(tracker.animate.set_value(int(new_val)))

        if animations:
            scene.play(*animations, run_time=run_time)
