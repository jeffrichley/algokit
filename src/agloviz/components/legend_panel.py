"""Legend panel component for Manim visualizations.

A reusable legend component that displays visual swatches with labels,
automatically sizing the background and arranging items vertically.
"""

from typing import Any

import manim as m

from agloviz.core.fonts import get_font


class LegendItem:
    """Represents a single legend item with visual swatch and label."""
    
    def __init__(self, label: str, visual: m.Mobject, description: str = ""):
        self.label = label
        self.visual = visual
        self.description = description


class LegendPanel(m.VGroup):
    """Self-contained legend panel with visual swatches and labels.
    
    Automatically sizes background and arranges items vertically.
    Supports various visual types: dots, stars, squares, etc.
    """
    
    DEFAULT_CORNER_RADIUS = 0.1
    DEFAULT_STROKE_WIDTH = 1
    DEFAULT_BG_FILL_OPACITY = 0.1
    DEFAULT_STROKE_COLOR = m.WHITE
    DEFAULT_FILL_COLOR = m.BLACK
    
    def __init__(
        self,
        items: list[dict[str, Any]],
        font_size: int = 16,
        corner_radius: float | None = None,
        stroke_width: float | None = None,
        stroke_color = None,
        fill_color = None,
        fill_opacity: float | None = None,
        item_buff: float = 0.15,
        swatch_label_buff: float = 0.1,
        **kwargs
    ):
        """Initialize the legend panel.
        
        Args:
            items: List of legend items, each with:
                   - 'label': str - Text label
                   - 'visual_type': str - 'dot', 'star', 'square', 'custom'
                   - 'visual_props': dict - Properties for the visual (color, size, etc.)
                   - 'custom_visual': m.Mobject - Custom visual (if visual_type='custom')
            font_size: Font size for labels
            corner_radius: Corner radius for background
            stroke_width: Stroke width for background
            stroke_color: Stroke color for background
            fill_color: Fill color for background
            fill_opacity: Fill opacity for background
            item_buff: Buffer between legend items
            swatch_label_buff: Buffer between swatch and label
        """
        super().__init__(**kwargs)
        
        # Use provided style values or defaults
        self.corner_radius = corner_radius if corner_radius is not None else self.DEFAULT_CORNER_RADIUS
        self.stroke_width = stroke_width if stroke_width is not None else self.DEFAULT_STROKE_WIDTH
        self.stroke_color = stroke_color if stroke_color is not None else self.DEFAULT_STROKE_COLOR
        self.fill_color = fill_color if fill_color is not None else self.DEFAULT_FILL_COLOR
        self.fill_opacity = fill_opacity if fill_opacity is not None else self.DEFAULT_BG_FILL_OPACITY
        
        self.font_size = font_size
        self.item_buff = item_buff
        self.swatch_label_buff = swatch_label_buff
        
        self._build_legend(items)
    
    def _build_legend(self, items: list[dict[str, Any]]) -> None:
        """Build the legend with items and background."""
        legend_groups = []
        
        for item in items:
            # Create visual swatch
            visual = self._create_visual_swatch(item)
            
            # Create text label
            label = m.Text(
                item["label"],
                font_size=self.font_size,
                color=m.WHITE
            )
            try:
                label.set_font(get_font("legend"))
            except:
                pass  # Fallback to default font if custom font not available
            
            # Arrange swatch and label
            item_group = m.VGroup(visual, label).arrange(
                m.RIGHT, 
                buff=self.swatch_label_buff
            )
            legend_groups.append(item_group)
        
        # Arrange all items vertically
        self.legend_items = m.VGroup(*legend_groups).arrange(
            m.DOWN, 
            buff=self.item_buff, 
            aligned_edge=m.LEFT
        )
        
        # Create background
        self._create_background()
        
        # Add to scene
        self.add(self.background, self.legend_items)
        self.legend_items.move_to(self.background.get_center())
    
    def _create_visual_swatch(self, item: dict[str, Any]) -> m.Mobject:
        """Create visual swatch based on item specification."""
        visual_type = item.get("visual_type", "square")
        props = item.get("visual_props", {})
        
        if visual_type == "custom" and "custom_visual" in item:
            return item["custom_visual"]
        elif visual_type == "dot":
            return m.Dot(
                radius=props.get("radius", 0.08),
                color=props.get("color", m.WHITE)
            )
        elif visual_type == "star":
            return m.Star(
                n=props.get("n", 5),
                outer_radius=props.get("outer_radius", 0.08),
                color=props.get("color", m.GOLD)
            )
        elif visual_type == "square":
            return m.Square(
                side_length=props.get("side_length", 0.15),
                fill_color=props.get("fill_color", m.WHITE),
                fill_opacity=props.get("fill_opacity", 0.6),
                stroke_width=props.get("stroke_width", 1),
                stroke_color=props.get("stroke_color", m.WHITE)
            )
        else:
            # Default to square
            return m.Square(
                side_length=0.15,
                fill_color=m.WHITE,
                fill_opacity=0.6,
                stroke_width=1
            )
    
    def _create_background(self) -> None:
        """Create the background rectangle."""
        # Compute size needed for background
        padding_x = 0.4
        padding_y = 0.4
        
        bg_width = self.legend_items.width + padding_x
        bg_height = self.legend_items.height + padding_y
        
        self.background = m.RoundedRectangle(
            width=bg_width,
            height=bg_height,
            corner_radius=self.corner_radius,
            stroke_width=self.stroke_width,
            stroke_color=self.stroke_color,
            fill_color=self.fill_color,
            fill_opacity=self.fill_opacity
        )


def create_bfs_legend() -> LegendPanel:
    """Create a standard BFS algorithm legend.
    
    Returns:
        LegendPanel configured for BFS visualization
    """
    bfs_items = [
        {
            "label": "Start",
            "visual_type": "dot",
            "visual_props": {"radius": 0.08, "color": m.GREEN_C}
        },
        {
            "label": "Goal",
            "visual_type": "star",
            "visual_props": {"n": 5, "outer_radius": 0.08, "color": m.GOLD}
        },
        {
            "label": "Water",
            "visual_type": "square",
            "visual_props": {
                "side_length": 0.15,
                "fill_color": m.BLUE_E,
                "fill_opacity": 0.65,
                "stroke_width": 1
            }
        },
        {
            "label": "Frontier",
            "visual_type": "square",
            "visual_props": {
                "side_length": 0.15,
                "fill_color": m.BLUE_C,
                "fill_opacity": 0.6,
                "stroke_width": 1
            }
        },
        {
            "label": "Visited",
            "visual_type": "square",
            "visual_props": {
                "side_length": 0.15,
                "fill_color": m.YELLOW_E,
                "fill_opacity": 0.4,
                "stroke_width": 1
            }
        }
    ]
    
    return LegendPanel(items=bfs_items, font_size=16)


def create_custom_legend(items: list[tuple[str, m.Mobject]]) -> LegendPanel:
    """Create a custom legend from a list of (label, visual) tuples.
    
    Args:
        items: List of (label, visual_mobject) tuples
        
    Returns:
        LegendPanel with custom items
    """
    legend_items = []
    for label, visual in items:
        legend_items.append({
            "label": label,
            "visual_type": "custom",
            "custom_visual": visual
        })
    
    return LegendPanel(items=legend_items)
