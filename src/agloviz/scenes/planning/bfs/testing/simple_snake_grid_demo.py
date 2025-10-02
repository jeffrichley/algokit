"""Simple demo of SnakeQueue component key features."""

import manim as m

from agloviz.components.snake_queue import SnakeQueue


class SimpleSnakeQueueDemo(m.Scene):
    """Simple demonstration of SnakeQueue component."""

    def construct(self) -> None:
        """Create a focused demo of SnakeQueue features."""
        # Create title
        title = m.Text("SnakeQueue Component Demo", font_size=36, color=m.WHITE)
        title.to_edge(m.UP)
        self.add(title)

        # Create a queue panel (6 tokens wide, 4 rows tall)
        self.queue = SnakeQueue(tokens_wide=6, tokens_tall=4, token_size=0.25)
        self.add(self.queue)

        # Animate panel entrance
        self.queue.animate_entrance(self)

        # 1. Create tokens down the left side (24 tokens for 4 rows)
        colors = [
            m.GREEN_C,
            m.BLUE_C,
            m.RED,
            m.YELLOW,
            m.PURPLE,
            m.ORANGE,
            m.PINK,
            m.TEAL,
            m.GOLD,
            m.MAROON,
            m.BLUE,
            m.LIGHT_GRAY,
            m.GREEN,
            m.BLUE_A,
            m.RED_A,
            m.YELLOW_A,
            m.PURPLE_A,
            m.ORANGE,
            m.PINK,
            m.TEAL_A,
            m.GOLD,
            m.MAROON_A,
            m.BLUE_B,
            m.GRAY,
        ]
        tokens = []

        # Create tokens positioned down the left side
        for i, color in enumerate(colors):
            token = m.Square(
                side_length=0.25,
                fill_color=color,
                fill_opacity=0.9,
                stroke_color=m.WHITE,
                stroke_width=2,
            )
            # Position down the left side
            token.move_to([-4, 2 - i * 0.4, 0])
            tokens.append(token)
            self.add(token)

        # Create boxes on the right side for dequeued tokens
        self.dequeued_boxes = []
        for i in range(24):
            box = m.RoundedRectangle(
                width=0.3,
                height=0.3,
                corner_radius=0.05,
                fill_color=m.DARK_GRAY,
                fill_opacity=0.3,
                stroke_color=m.WHITE,
                stroke_width=1,
            )
            # Position boxes down the right side
            box.move_to([4, 2 - i * 0.4, 0])
            self.dequeued_boxes.append(box)
            self.add(box)

        # Add label for dequeued tokens area
        dequeued_label = m.Text("Dequeued Tokens", font_size=16, color=m.WHITE)
        dequeued_label.move_to([4, 3, 0])
        self.add(dequeued_label)

        self.wait(0.5)

        # 2. Enqueue tokens one by one (they animate to their grid positions)
        for token in tokens:
            self.queue.enqueue(token=token, scene=self)
            self.wait(0.3)

        self.wait(0.5)

        # 3. Dequeue tokens to show snaking effect and move them to right boxes
        for i in range(6):
            token = self.queue.dequeue(scene=self)
            # Move token to the corresponding right-side box
            target_box = self.dequeued_boxes[i]
            self.play(
                token.animate.move_to(target_box.get_center()).set_ease(
                    m.rate_functions.ease_in_out_cubic
                ),
                run_time=0.5,
            )
            self.wait(0.2)

        self.wait(0.5)

        # 5. Show queue operations
        subtitle = m.Text(
            "Queue Operations: FIFO (First In, First Out)", font_size=20, color=m.GRAY
        )
        subtitle.next_to(title, m.DOWN, buff=0.3)
        self.play(m.Write(subtitle), run_time=1.0)

        self.wait(2.0)
