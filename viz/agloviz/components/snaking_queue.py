"""Snaking Queue visualization component for Manim.

A custom Manim object that visualizes a queue with tokens arranged in a 
snaking grid pattern. When tokens are dequeued, remaining tokens slide up
to fill the gap, creating a smooth FIFO visualization.
"""

from typing import List, Tuple

import manim as m


class SnakingQueue(m.VGroup):
    """Custom Manim object for a snaking queue visualization.
    
    Tokens are arranged in a grid that snakes left-to-right, top-to-bottom.
    When a token is dequeued, all remaining tokens slide up to fill the gap.
    
    Example:
        queue = SnakingQueue(panel_width=2.5, panel_height=3.0)
        queue.enqueue(color=GREEN_C, scene=self)
        token = queue.dequeue(scene=self)
    """
    
    def __init__(self, panel_width: float = 2.5, panel_height: float = 3.0, 
                 token_size: float = 0.2, **kwargs):
        """Initialize the snaking queue.
        
        Args:
            panel_width: Width of the queue panel
            panel_height: Height of the queue panel  
            token_size: Size of each token
        """
        super().__init__(**kwargs)
        
        # Configuration
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.token_size = token_size
        self.token_spacing = token_size + 0.1  # token + gap
        
        # Calculate grid dimensions
        usable_width = panel_width - 0.6  # account for padding
        usable_height = panel_height - 0.6
        self.tokens_per_row = int(usable_width / self.token_spacing)
        self.max_rows = int(usable_height / self.token_spacing)
        self.max_capacity = self.tokens_per_row * self.max_rows
        
        # Token storage
        self.tokens: List[m.Square] = []
        self.token_positions: List[Tuple[float, float]] = []
        
        # Calculate all possible positions
        self._calculate_positions()
        
    def _calculate_positions(self) -> None:
        """Pre-calculate all possible token positions in the queue."""
        # Starting position (top-left of usable area, relative to queue center)
        start_x = -self.panel_width/2 + 0.3 + self.token_size/2
        start_y = self.panel_height/2 - 0.3 - self.token_size/2
        
        for row in range(self.max_rows):
            for col in range(self.tokens_per_row):
                x = start_x + col * self.token_spacing
                y = start_y - row * self.token_spacing
                self.token_positions.append((x, y))
        
    
    def enqueue(self, color: str = m.BLUE_C, scene: m.Scene = None) -> m.Square:
        """Add a new token to the queue.
        
        Args:
            color: Color of the token
            scene: Scene to animate in (optional)
            
        Returns:
            The created token
            
        Raises:
            ValueError: If queue is at maximum capacity
        """
        if len(self.tokens) >= self.max_capacity:
            raise ValueError(f"Queue is full (max {self.max_capacity} tokens)")
        
        # Create new token
        token = m.Square(
            side_length=self.token_size,
            fill_color=color,
            fill_opacity=0.9,
            stroke_color=m.WHITE,
            stroke_width=2
        )
        
        # Position at calculated location
        pos = self.token_positions[len(self.tokens)]
        token.move_to([pos[0], pos[1], 0])
        
        
        # Add to storage
        self.tokens.append(token)
        self.add(token)
        
        # Animate if scene provided
        if scene:
            scene.play(m.GrowFromCenter(token), run_time=0.2)
        
        return token
    
    def dequeue(self, scene: m.Scene = None) -> m.Square:
        """Remove the first token from the queue.
        
        Args:
            scene: Scene to animate with (optional)
            
        Returns:
            The removed token
            
        Raises:
            ValueError: If queue is empty
        """
        if not self.tokens:
            raise ValueError("Queue is empty")
        
        # Get first token
        token = self.tokens.pop(0)
        
        # Animate removal
        if scene:
            scene.play(m.FadeOut(token), run_time=0.2)
        else:
            self.remove(token)
        
        # Snake remaining tokens up
        self._snake_tokens_up(scene)
        
        return token
    
    def _snake_tokens_up(self, scene: m.Scene = None) -> None:
        """Move all remaining tokens up one position."""
        if not self.tokens:
            return
        
        # Calculate new positions for all remaining tokens
        new_positions = []
        for i, token in enumerate(self.tokens):
            new_pos = self.token_positions[i]
            new_positions.append(new_pos)
        
        # Animate movement
        if scene:
            animations = []
            for token, new_pos in zip(self.tokens, new_positions):
                animations.append(token.animate.move_to([new_pos[0], new_pos[1], 0]))
            scene.play(m.AnimationGroup(*animations), run_time=0.3)
        else:
            for token, new_pos in zip(self.tokens, new_positions):
                token.move_to([new_pos[0], new_pos[1], 0])
    
    def is_empty(self) -> bool:
        """Check if queue is empty.
        
        Returns:
            True if queue has no tokens
        """
        return len(self.tokens) == 0
    
    def size(self) -> int:
        """Get current queue size.
        
        Returns:
            Number of tokens currently in queue
        """
        return len(self.tokens)
    
    def clear(self) -> None:
        """Clear all tokens from queue."""
        for token in self.tokens:
            self.remove(token)
        self.tokens.clear()
    
    def get_capacity(self) -> int:
        """Get maximum queue capacity.
        
        Returns:
            Maximum number of tokens the queue can hold
        """
        return self.max_capacity
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Get queue grid dimensions.
        
        Returns:
            Tuple of (tokens_per_row, max_rows)
        """
        return (self.tokens_per_row, self.max_rows)
