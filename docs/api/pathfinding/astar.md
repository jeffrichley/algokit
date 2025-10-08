# A* Search API Reference

## Overview

A* (A-star) is a widely-used pathfinding and graph traversal algorithm that finds the shortest path between two points in a weighted graph. It combines Dijkstra's algorithm (guaranteed shortest path) with best-first search (heuristic guidance) for optimal and efficient pathfinding.

**Key Features:**
- Optimal (with admissible heuristic)
- Complete (always finds solution if one exists)
- Heuristic-guided for efficiency
- Supports weighted graphs
- Widely used in games and robotics

For algorithmic details and theory, see the [Pathfinding Overview](../../algorithms/pathfinding/overview.md).

---

## Functions

### Find Shortest Path

::: algokit.algorithms.pathfinding.astar.astar_shortest_path
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      show_source: false

### Find Shortest Distance

::: algokit.algorithms.pathfinding.astar.astar_shortest_distance
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      show_source: false

### Find All Distances

::: algokit.algorithms.pathfinding.astar.astar_all_distances
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      show_source: false

---

## Quick Start Example

```python
from algokit.algorithms.pathfinding.astar import astar_shortest_path
import networkx as nx

# Create a graph
graph = nx.Graph()
graph.add_weighted_edges_from([
    (0, 1, 1.0),
    (0, 2, 4.0),
    (1, 2, 2.0),
    (1, 3, 5.0),
    (2, 3, 1.0),
])

# Define heuristic (straight-line distance)
def euclidean_heuristic(node, goal):
    # Example: node positions
    positions = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}
    x1, y1 = positions[node]
    x2, y2 = positions[goal]
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

# Find shortest path
path = astar_shortest_path(
    graph=graph,
    start=0,
    goal=3,
    heuristic=euclidean_heuristic
)

print(f"Shortest path: {path}")  # [0, 1, 2, 3] or similar
print(f"Path length: {len(path) - 1} steps")
```

---

## Grid-Based Pathfinding

```python
from algokit.algorithms.pathfinding.astar import astar_shortest_path
import networkx as nx

# Create grid graph (10x10)
def create_grid_graph(width, height, obstacles=None):
    """Create a grid graph with optional obstacles."""
    graph = nx.Graph()

    # Add nodes
    for x in range(width):
        for y in range(height):
            if obstacles and (x, y) in obstacles:
                continue
            graph.add_node((x, y))

    # Add edges (4-connected)
    for x in range(width):
        for y in range(height):
            if (x, y) not in graph:
                continue
            # Right
            if x + 1 < width and (x + 1, y) in graph:
                graph.add_edge((x, y), (x + 1, y), weight=1.0)
            # Down
            if y + 1 < height and (x, y + 1) in graph:
                graph.add_edge((x, y), (x, y + 1), weight=1.0)

    return graph

# Manhattan distance heuristic
def manhattan_heuristic(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

# Create graph with obstacles
obstacles = [(3, 3), (3, 4), (3, 5)]
graph = create_grid_graph(10, 10, obstacles=obstacles)

# Find path
path = astar_shortest_path(
    graph=graph,
    start=(0, 0),
    goal=(9, 9),
    heuristic=manhattan_heuristic
)

print(f"Path through grid: {path}")
```

---

## Heuristic Functions

### Manhattan Distance (Grid, 4-connected)

```python
def manhattan_heuristic(node, goal):
    """Best for 4-directional grid movement."""
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])
```

### Euclidean Distance (Continuous/Any-angle)

```python
import math

def euclidean_heuristic(node, goal):
    """Best for straight-line distance."""
    return math.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)
```

### Diagonal Distance (Grid, 8-connected)

```python
def diagonal_heuristic(node, goal):
    """Best for 8-directional grid movement."""
    dx = abs(node[0] - goal[0])
    dy = abs(node[1] - goal[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
```

### Custom Domain Heuristics

```python
def custom_heuristic(node, goal):
    """Design heuristics specific to your domain.

    Requirements:
    - Must be admissible (never overestimate true cost)
    - Should be consistent (satisfy triangle inequality)
    - Trade-off: accuracy vs computation time
    """
    # Your domain-specific logic here
    return estimated_cost_to_goal
```

---

## Performance Tips

### 1. **Choose Good Heuristics**
- Closer to actual cost = fewer nodes expanded
- Must never overestimate (admissibility)
- Balance accuracy vs computation time

### 2. **Use Appropriate Data Structures**
- Priority queue for frontier (heapq)
- Set for visited nodes
- Dict for came_from/g_score

### 3. **Tie-Breaking**
```python
def tie_breaking_heuristic(node, goal):
    """Add small tie-breaker to prefer straighter paths."""
    h = manhattan_heuristic(node, goal)
    # Add small perturbation
    return h * (1.0 + 1e-6)
```

### 4. **Bidirectional Search**
Search from both start and goal, meeting in middle (2x speedup for many cases)

---

## See Also

### Related Algorithms
- [Dijkstra](dijkstra.md) - A* without heuristic
- [BFS](bfs.md) - Unweighted shortest path
- [M*](mstar.md) - Multi-robot A*-based planning

### Documentation
- [Pathfinding Overview](../../algorithms/pathfinding/overview.md)
- [A* Paper (Hart et al., 1968)](https://ieeexplore.ieee.org/document/4082128)
- [Red Blob Games: A* Tutorial](https://www.redblobgames.com/pathfinding/a-star/introduction.html)

### Applications
- Video game pathfinding (NPCs, units)
- Robot motion planning
- GPS and map navigation
- Logistics and routing optimization
