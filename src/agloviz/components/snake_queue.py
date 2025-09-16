"""SnakeQueue visualization component for Manim.

A professional-grade queue component that visualizes tokens in a snaking grid pattern.
When tokens are dequeued, remaining tokens slide up to fill the gap, creating a smooth FIFO visualization.
"""


import manim as m  # type: ignore[import-untyped]


class SnakeQueue(m.VGroup):
    """Custom Manim object for a snaking queue visualization.
    
    Tokens are arranged in a grid that snakes left-to-right, top-to-bottom.
    When a token is dequeued, all remaining tokens slide up to fill the gap.
    Designed specifically for queue visualization with smaller tokens and proper spacing.
    
    Example:
        queue = SnakeQueue(panel_width=2.5, panel_height=3.0)
        queue.enqueue(color=GREEN_C, scene=self)
        token = queue.dequeue(scene=self)
    """
    
    def __init__(
        self,
        tokens_wide: int = 6,
        tokens_tall: int = 3,
        token_size: float = 0.2,
        title: str = "Queue (FIFO)",
        **kwargs: object
    ) -> None:
        """Initialize the snake queue.
        
        Args:
            tokens_wide: Number of tokens per row
            tokens_tall: Number of rows of tokens
            token_size: Size of each token
            title: Title text for the queue panel
            **kwargs: Additional arguments passed to VGroup
        """
        super().__init__(**kwargs)
        
        # Configuration
        self.tokens_wide = tokens_wide
        self.tokens_tall = tokens_tall
        self.token_size = token_size
        self.title = title
        self.token_spacing = token_size + 0.1  # token + gap
        
        # Calculate panel dimensions based on token requirements
        padding = 0.2  # padding from panel edges
        
        # Calculate required panel dimensions
        self.panel_width = (self.tokens_wide * self.token_spacing) + (2 * padding)
        self.panel_height = (self.tokens_tall * self.token_spacing) + (2 * padding)
        
        # Grid dimensions
        self.tokens_per_row = self.tokens_wide
        self.max_rows = self.tokens_tall
        self.max_capacity = self.tokens_wide * self.tokens_tall
        
        # Token storage
        self.tokens: list[m.Square] = []
        self.token_positions: list[tuple[float, float]] = []
        self.original_positions: list[tuple[float, float]] = []
        
        # Create the panel background and title
        self._create_panel()
        
        # Calculate all possible positions
        self._calculate_positions()
        
        # Add updater to track panel transformations
        self.panel_bg.add_updater(self._update_token_positions)
    
    def _create_panel(self) -> None:
        """Create the panel background and title."""
        # Panel background
        self.panel_bg = m.RoundedRectangle(
            width=self.panel_width,
            height=self.panel_height,
            corner_radius=0.2,
            stroke_color=m.WHITE,
            stroke_width=2,
            fill_color=m.BLACK,
            fill_opacity=0.8
        )
        
        # Panel title
        self.panel_title = m.Text(
            self.title,
            font_size=18,
            color=m.WHITE
        ).next_to(self.panel_bg, m.UP, buff=0.2)
        
        # Add panel elements to the group
        self.add(self.panel_bg, self.panel_title)
    
    def _calculate_positions(self) -> None:
        """Pre-calculate all possible token positions in the queue."""
        # Clear existing positions first
        self.token_positions.clear()
        
        # Get the actual bounds of the panel background
        panel_center = self.panel_bg.get_center()
        panel_width = self.panel_bg.width
        panel_height = self.panel_bg.height
        
        # Calculate usable area with padding from panel edges
        # Need to account for token size to ensure tokens fit completely inside
        padding = 0.2  # padding from panel edges
        
        left_bound = panel_center[0] - panel_width/2 + padding + self.token_size/2
        right_bound = panel_center[0] + panel_width/2 - padding - self.token_size/2
        top_bound = panel_center[1] + panel_height/2 - padding - self.token_size/2
        bottom_bound = panel_center[1] - panel_height/2 + padding + self.token_size/2
        
        # Calculate grid dimensions based on configured token layout
        # Use the configured dimensions instead of calculating from space
        self.tokens_per_row = self.tokens_wide
        self.max_rows = self.tokens_tall
        self.max_capacity = self.tokens_wide * self.tokens_tall
        
        # Calculate spacing to center the grid within the usable area
        usable_width = right_bound - left_bound
        usable_height = top_bound - bottom_bound
        
        # Calculate spacing to fit the exact number of tokens
        if self.tokens_per_row > 1:
            horizontal_spacing = usable_width / (self.tokens_per_row - 1)
        else:
            horizontal_spacing = 0
            
        if self.max_rows > 1:
            vertical_spacing = usable_height / (self.max_rows - 1)
        else:
            vertical_spacing = 0
        
        # Start position (top-left of usable area)
        start_x = left_bound
        start_y = top_bound
        
        for row in range(self.max_rows):
            for col in range(self.tokens_per_row):
                x = start_x + col * horizontal_spacing if self.tokens_per_row > 1 else start_x
                y = start_y - row * vertical_spacing if self.max_rows > 1 else start_y
                self.token_positions.append((x, y))
    
    def _update_token_positions(self, mobject: m.Mobject) -> None:
        """Update token positions when panel transforms.
        
        This method is called by the updater whenever the panel moves, rotates, or scales.
        It recalculates positions based on the current panel state and moves existing tokens.
        """
        # Only update if we have tokens
        if not self.tokens:
            return
            
        # Recalculate positions based on current panel state
        self._calculate_positions()
        
        # Move existing tokens to their new positions
        for i, token in enumerate(self.tokens):
            if i < len(self.token_positions):
                new_pos = self.token_positions[i]
                token.move_to([new_pos[0], new_pos[1], 0])
    
    def enqueue(self, token: m.Square = None, color: str = m.BLUE_C, scene: m.Scene = None) -> m.Square:
        """Add a token to the queue.
        
        Args:
            token: Existing token to add (if None, creates a new one)
            color: Color of the token (only used if token is None)
            scene: Scene to animate in (optional)
            
        Returns:
            The token that was added to the queue
            
        Raises:
            ValueError: If queue is at maximum capacity
        """
        if len(self.tokens) >= self.max_capacity:
            raise ValueError(f"Queue is full (max {self.max_capacity} tokens)")
        
        # Create token if not provided
        if token is None:
            raise ValueError("Token is required")
        
        # Store original position before moving
        original_pos = (token.get_center()[0], token.get_center()[1])
        self.original_positions.append(original_pos)
        
        # Add to storage and group
        self.tokens.append(token)
        self.add(token)
        
        # Animate to correct position if scene provided
        if scene:
            # Temporarily disable updater during animation
            self.panel_bg.remove_updater(self._update_token_positions)
            
            # Get the target position for this token
            target_pos = self.token_positions[len(self.tokens) - 1]
            
            
            # Animate from current position to correct grid position
            scene.play(
                token.animate.move_to([target_pos[0], target_pos[1], 0]).set_ease(m.rate_functions.ease_in_out_cubic),
                run_time=0.5
            )
            
            # Re-enable updater after animation
            self.panel_bg.add_updater(self._update_token_positions)
        else:
            raise ValueError("Scene is required")
        
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
        
        # Get first token and its original position
        token = self.tokens.pop(0)
        original_pos = self.original_positions.pop(0)
        
        # Animate token sliding back to original position if scene provided
        if scene:
            # Temporarily disable updater during animation
            self.panel_bg.remove_updater(self._update_token_positions)
            
            # Use the stored original position
            slide_back_pos = [original_pos[0], original_pos[1], 0]
            
            # Animate token sliding back to original position
            scene.play(
                token.animate.move_to(slide_back_pos).set_ease(m.rate_functions.ease_in_out_cubic),
                run_time=0.5
            )
            
            # Remove token from group after animation
            self.remove(token)
            
            # Snake remaining tokens up
            self._snake_tokens_up(scene)
            
            # Re-enable updater after animation
            self.panel_bg.add_updater(self._update_token_positions)
        else:
            # Just remove without animation
            self.remove(token)
        
        return token
    
    def _snake_tokens_up(self, scene: m.Scene = None) -> None:
        """Move all remaining tokens up one position."""
        if not self.tokens:
            return
        
        # Calculate new positions for all remaining tokens
        new_positions = []
        for i, _token in enumerate(self.tokens):
            new_pos = self.token_positions[i]
            new_positions.append(new_pos)
        
        # Animate movement
        if scene:
            animations = []
            for token, new_pos in zip(self.tokens, new_positions, strict=True):
                animations.append(token.animate.move_to([new_pos[0], new_pos[1], 0]))
            scene.play(m.AnimationGroup(*animations), run_time=0.3)
        else:
            for token, new_pos in zip(self.tokens, new_positions, strict=True):
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
    
    def get_dimensions(self) -> tuple[int, int]:
        """Get queue grid dimensions.
        
        Returns:
            Tuple of (tokens_per_row, max_rows)
        """
        return (self.tokens_per_row, self.max_rows)
    
    def animate_entrance(self, scene: m.Scene) -> None:
        """Animate the queue panel sliding in from the right.
        
        Args:
            scene: Scene to animate with
        """
        # Start with panel off-screen
        self.shift(m.RIGHT * 0.5)
        
        # Animate sliding in
        scene.play(
            m.FadeIn(self.panel_bg, shift=m.RIGHT * 0.5),
            m.Write(self.panel_title),
            run_time=1.0
        )
        
        # Recalculate positions after panel has moved
        self._calculate_positions()
