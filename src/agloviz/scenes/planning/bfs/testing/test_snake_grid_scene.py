"""Test scene for SnakeGrid component demonstration."""

import manim as m

from agloviz.components.snake_queue import SnakeQueue


class TestSnakeQueueScene(m.Scene):
    """Test scene to demonstrate SnakeQueue component functionality."""

    def construct(self) -> None:
        """Create and animate the SnakeQueue component."""
        # Create a queue panel
        self.queue = SnakeQueue(panel_width=2.5, panel_height=3.0, token_size=0.2)

        # Add queue to scene
        self.add(self.queue)

        # Animate panel entrance
        self.queue.animate_entrance(self)

        # Enqueue some tokens
        self.queue.enqueue(color=m.GREEN_C, scene=self)
        self.wait(0.3)

        self.queue.enqueue(color=m.BLUE_C, scene=self)
        self.wait(0.3)

        self.queue.enqueue(color=m.RED, scene=self)
        self.wait(0.3)

        self.queue.enqueue(color=m.YELLOW, scene=self)
        self.wait(0.3)

        self.queue.enqueue(color=m.PURPLE, scene=self)
        self.wait(0.5)

        # Dequeue some tokens to show snaking effect
        self.queue.dequeue(scene=self)
        self.wait(0.3)

        self.queue.dequeue(scene=self)
        self.wait(0.3)

        # Enqueue more tokens
        self.queue.enqueue(color=m.ORANGE, scene=self)
        self.wait(0.3)

        self.queue.enqueue(color=m.PINK, scene=self)
        self.wait(1.0)

        # Clear the queue
        self.play(
            m.AnimationGroup(*[m.FadeOut(token) for token in self.queue.tokens]),
            run_time=1.0,
        )
        self.queue.clear()

        self.wait(2.0)
