"""Graph utilities for pathfinding algorithms.

This module provides utilities for creating, loading, and manipulating graphs
for the HarborNet: Search & Rescue pathfinding algorithms.
"""

from pathlib import Path
from typing import Any

import networkx as nx
import yaml


class HarborNetScenario:
    """Represents a HarborNet scenario configuration.

    A HarborNet scenario defines a flooded coastal city with:
    - Grid dimensions and obstacles (debris, closed locks)
    - Start position (Command Pier)
    - Goal position (SOS location)
    - Narrative context
    """

    def __init__(
        self,
        name: str,
        width: int,
        height: int,
        start: tuple[int, int],
        goal: tuple[int, int],
        obstacles: set[tuple[int, int]] | None = None,
        description: str | None = None,
        narrative: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a HarborNet scenario.

        Args:
            name: Scenario name
            width: Grid width
            height: Grid height
            start: Start position (Command Pier)
            goal: Goal position (SOS location)
            obstacles: Set of blocked positions
            description: Scenario description
            narrative: HarborNet narrative context
            **kwargs: Additional configuration fields (e.g., text_below_grid_offset)
        """
        self.name = name
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles or set()
        self.description = description
        self.narrative = narrative

        # Store additional configuration fields
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_graph(self) -> nx.Graph:
        """Convert scenario to NetworkX graph.

        Returns:
            NetworkX graph representing the scenario
        """
        return create_grid_graph(
            width=self.width,
            height=self.height,
            blocked=self.obstacles,
            start=self.start,
            goal=self.goal,
        )

    def validate(self) -> list[str]:
        """Validate scenario configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check dimensions
        if self.width <= 0 or self.height <= 0:
            errors.append("Grid dimensions must be positive")

        # Check start position
        if not (0 <= self.start[0] < self.width and 0 <= self.start[1] < self.height):
            errors.append("Start position must be within grid bounds")

        # Check goal position
        if not (0 <= self.goal[0] < self.width and 0 <= self.goal[1] < self.height):
            errors.append("Goal position must be within grid bounds")

        # Check obstacles
        for obstacle in self.obstacles:
            if not (0 <= obstacle[0] < self.width and 0 <= obstacle[1] < self.height):
                errors.append(f"Obstacle {obstacle} is outside grid bounds")

        # Check start/goal not on obstacles
        if self.start in self.obstacles:
            errors.append("Start position cannot be on an obstacle")

        if self.goal in self.obstacles:
            errors.append("Goal position cannot be on an obstacle")

        return errors


def create_grid_graph(
    width: int,
    height: int,
    blocked: set[tuple[int, int]] | None = None,
    start: tuple[int, int] | None = None,
    goal: tuple[int, int] | None = None,
    diagonal: bool = False,
) -> nx.Graph:
    """Create a grid graph for pathfinding algorithms.

    Args:
        width: Grid width
        height: Grid height
        blocked: Set of blocked positions
        start: Start position (optional)
        goal: Goal position (optional)
        diagonal: Whether to include diagonal connections

    Returns:
        NetworkX graph representing the grid

    Raises:
        ValueError: If dimensions are invalid
    """
    if width <= 0 or height <= 0:
        raise ValueError("Grid dimensions must be positive")

    blocked = blocked or set()
    graph = nx.Graph()

    # Add nodes
    for x in range(width):
        for y in range(height):
            if (x, y) not in blocked:
                graph.add_node((x, y))

    # Add edges (4-connected or 8-connected)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diagonal:
        directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])

    for x in range(width):
        for y in range(height):
            if (x, y) in blocked:
                continue

            for dx, dy in directions:
                next_x, next_y = x + dx, y + dy
                if (
                    0 <= next_x < width
                    and 0 <= next_y < height
                    and (next_x, next_y) not in blocked
                ):
                    graph.add_edge((x, y), (next_x, next_y))

    # Add metadata
    graph.graph.update(
        {
            "width": width,
            "height": height,
            "blocked": blocked,
            "start": start,
            "goal": goal,
            "diagonal": diagonal,
        }
    )

    return graph


def load_harbor_scenario(file_path: str | Path) -> HarborNetScenario:
    """Load a HarborNet scenario from YAML file.

    Args:
        file_path: Path to scenario YAML file

    Returns:
        HarborNet scenario object

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file is invalid YAML
        ValueError: If scenario data is invalid
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {file_path}")

    with open(file_path) as f:
        data = yaml.safe_load(f)

    # Validate required fields
    required_fields = ["name", "width", "height", "start", "goal"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Convert obstacles list to set of tuples
    obstacles = set()
    if "obstacles" in data:
        obstacles = {tuple(obs) for obs in data["obstacles"]}

    # Extract standard fields
    standard_fields = {
        "name": data["name"],
        "width": data["width"],
        "height": data["height"],
        "start": tuple(data["start"]),
        "goal": tuple(data["goal"]),
        "obstacles": obstacles,
        "description": data.get("description"),
        "narrative": data.get("narrative"),
    }

    # Extract additional configuration fields (like text_below_grid_offset)
    additional_fields = {
        k: v
        for k, v in data.items()
        if k
        not in [
            "name",
            "width",
            "height",
            "start",
            "goal",
            "obstacles",
            "description",
            "narrative",
        ]
    }

    scenario = HarborNetScenario(**standard_fields, **additional_fields)

    # Validate scenario
    errors = scenario.validate()
    if errors:
        raise ValueError(f"Invalid scenario: {'; '.join(errors)}")

    return scenario
