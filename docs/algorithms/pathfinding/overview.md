# Pathfinding & Planning Algorithms Overview

## 🗺️ Introduction

**Pathfinding and Planning Algorithms** focus on finding optimal or near-optimal sequences of actions to achieve specific goals. These algorithms are fundamental to artificial intelligence and are used in robotics, games, autonomous systems, and logistics. They solve the problem of navigating from a start state to a goal state while minimizing cost or maximizing utility.

## 🎯 Core Problem

Given:
- **Start State**: Initial position/configuration
- **Goal State**: Desired position/configuration
- **Action Space**: Possible moves/transitions
- **Cost Function**: Cost of each action/edge

Find:
- **Path**: Sequence of actions from start to goal
- **Optimal**: Minimize total cost (or maximize utility)
- **Complete**: Find solution if one exists
- **Efficient**: Use reasonable time/memory

## 🔑 Key Characteristics

- **Goal-Directed Search**: Navigate toward specific objectives
- **Action Sequencing**: Build plans as sequences of steps
- **Path Planning**: Find routes through state/configuration spaces
- **Constraint Satisfaction**: Respect obstacles and limitations
- **Optimality**: Seek minimum-cost solutions when possible
- **Completeness**: Guarantee finding solution if it exists

## 🎨 Implemented Algorithms

### 1. **Breadth-First Search (BFS)** 🌊
Explores all neighbors level-by-level, guaranteeing shortest path in unweighted graphs.

**Characteristics:**
- **Complete**: Always finds solution if exists
- **Optimal**: Finds shortest path (unweighted)
- **Time**: O(V + E) or O(b^d)
- **Space**: O(V) or O(b^d)
- **Coverage**: 97% 🌟

**Best For:**
- Unweighted graphs
- Finding shortest number of steps
- Level-order traversal
- Social network analysis

**Example Applications:**
- Shortest path in mazes
- Finding connections in social networks
- Web crawling
- Puzzle solving (Rubik's cube)

### 2. **Depth-First Search (DFS)** 🔍
Explores as far as possible along branches before backtracking.

**Characteristics:**
- **Complete**: Not guaranteed (can get stuck in cycles)
- **Optimal**: Not optimal
- **Time**: O(V + E) or O(b^m)
- **Space**: O(V) or O(bm) - less than BFS
- **Coverage**: 98% 🌟

**Best For:**
- Path existence checking
- Topological sorting
- Detecting cycles
- Memory-constrained situations

**Example Applications:**
- Maze solving
- Puzzle solving (Sudoku)
- Cycle detection
- Tree traversals

### 3. **Dijkstra's Algorithm** ⚖️
Finds shortest paths from source to all vertices in weighted graphs with non-negative weights.

**Characteristics:**
- **Complete**: Yes (with non-negative weights)
- **Optimal**: Guarantees shortest path
- **Time**: O((V + E) log V) with priority queue
- **Space**: O(V)
- **Coverage**: 94% 🌟

**Best For:**
- Weighted graphs with non-negative edges
- Finding shortest paths to all nodes
- Road networks
- Network routing

**Example Applications:**
- GPS navigation
- Network routing (OSPF)
- Robotics path planning
- Supply chain optimization

### 4. **A* Search** ⭐
Heuristic-guided search combining path cost and estimated cost to goal.

**Characteristics:**
- **Complete**: Yes (with admissible heuristic)
- **Optimal**: Yes (with admissible & consistent heuristic)
- **Time**: O(b^d) in worst case, much better with good heuristic
- **Space**: O(b^d)
- **Coverage**: 92% 🌟

**Best For:**
- Problems with good heuristics
- Need for optimality
- Game AI
- Robotics navigation

**Example Applications:**
- Video game pathfinding
- Robot motion planning
- Logistics and delivery
- Map navigation apps

### 5. **M* (Multi-Robot Planning)** 🤖🤖
Coordinate paths for multiple robots avoiding collisions using subdimensional expansion.

**Characteristics:**
- **Complete**: Yes for multi-robot coordination
- **Optimal**: Finds optimal joint paths
- **Time**: Exponential in worst case, polynomial with subdimensional expansion
- **Space**: Depends on collision structure
- **Coverage**: 95% 🌟

**Best For:**
- Multi-robot coordination
- Warehouse automation
- Traffic management
- Collaborative robotics

**Example Applications:**
- Warehouse robot coordination
- Drone swarm navigation
- Traffic flow optimization
- Multi-agent games

## 📊 Algorithm Comparison

| Algorithm | Complete | Optimal | Time | Space | Best For |
|-----------|----------|---------|------|-------|----------|
| BFS | ✅ | ✅ (unweighted) | O(b^d) | O(b^d) | Unweighted, shortest hops |
| DFS | ❌ | ❌ | O(b^m) | O(bm) | Memory-limited, existence |
| Dijkstra | ✅ | ✅ | O((V+E)logV) | O(V) | Weighted, all paths |
| A* | ✅ | ✅* | O(b^d) | O(b^d) | Heuristic available |
| M* | ✅ | ✅ | Exponential** | Variable | Multi-robot |

*With admissible heuristic
**Polynomial with subdimensional expansion

## 🧮 Heuristics for A*

### Manhattan Distance (Grid, 4-connected)
```python
h(n) = |goal.x - n.x| + |goal.y - n.y|
```
- Admissible for grid with 4-way movement
- Fast to compute
- Common in grid-based games

### Euclidean Distance (Continuous space)
```python
h(n) = sqrt((goal.x - n.x)² + (goal.y - n.y)²)
```
- Admissible for straight-line distance
- Natural for continuous spaces
- Used in robotics

### Diagonal Distance (Grid, 8-connected)
```python
dx = |goal.x - n.x|
dy = |goal.y - n.y|
h(n) = max(dx, dy) + (√2 - 1) * min(dx, dy)
```
- Admissible for 8-way movement
- More accurate than Manhattan for diagonal moves

### Custom Domain Heuristics
Design heuristics specific to your problem:
- Must be admissible (never overestimate)
- Should be consistent (satisfy triangle inequality)
- Trade-off: accuracy vs computation time

## 🌟 Common Applications

### Robotics & Autonomous Systems
- **Mobile Robots**: Navigate warehouses, homes, outdoors
- **Manipulators**: Plan collision-free arm movements
- **Drones**: 3D path planning with obstacle avoidance
- **Autonomous Vehicles**: Route planning and local planning

### Games & Entertainment
- **Strategy Games**: Unit movement in RTS/turn-based games
- **RPGs**: NPC navigation and pathfinding
- **Puzzles**: Solving optimal solutions (sliding puzzles)
- **Procedural Content**: Generating connected levels

### Logistics & Transportation
- **Delivery Routing**: Optimize package delivery routes
- **Supply Chain**: Warehouse layout and robot coordination
- **Public Transit**: Route planning for buses/trains
- **Traffic Management**: Optimize traffic flow

### Network & Infrastructure
- **Network Routing**: Data packet routing (OSPF, BGP)
- **Circuit Design**: PCB trace routing
- **Pipe/Cable Laying**: Infrastructure planning
- **Emergency Services**: Ambulance/fire dispatch

## 💡 Choosing the Right Algorithm

### Use BFS When:
- ✅ Graph is unweighted
- ✅ Need shortest path (in number of edges)
- ✅ All edges have equal cost
- ✅ State space is relatively small

### Use DFS When:
- ✅ Just need to find any path
- ✅ Memory is very limited
- ✅ Checking for path existence
- ✅ Tree structures (not graphs with cycles)

### Use Dijkstra When:
- ✅ Graph has weighted edges
- ✅ All weights are non-negative
- ✅ Need paths from one source to all vertices
- ✅ No good heuristic available

### Use A* When:
- ✅ Graph has weighted edges
- ✅ Have a good admissible heuristic
- ✅ Need single source-goal path
- ✅ Want faster than Dijkstra

### Use M* When:
- ✅ Multiple agents/robots
- ✅ Need collision-free paths
- ✅ Coordination is required
- ✅ Subdimensional structure exists

## 🔬 Advanced Concepts

### Optimality Conditions

**Admissible Heuristic:**
- Never overestimates true cost to goal
- Guarantees A* finds optimal path
- h(n) ≤ h*(n) for all nodes n

**Consistent Heuristic:**
- Satisfies triangle inequality
- h(n) ≤ c(n,n') + h(n') for all neighbors n'
- Ensures each node expanded at most once

### Tie-Breaking
When f-values are equal in A*:
```python
# Prefer nodes closer to goal
h_tie = h(n) + epsilon * h(n)

# Prefer straighter paths
g_tie = (1 + p) * g(n)  # p << 1
```

### Memory-Bounded Search
For large spaces exceeding memory:
- **IDA*** (Iterative Deepening A*): Depth-first with f-limit
- **RBFS** (Recursive Best-First): Space-efficient A*
- **SMA*** (Simplified Memory-Bounded A*): Use available memory optimally

### Jump Point Search
Optimization for uniform-cost grids:
- Skip intermediate nodes in straight lines
- Dramatically reduces nodes expanded
- Maintains optimality

## 🎓 Problem-Solving Strategy

### 1. **Understand the Problem**
- What is the state space?
- Are edge weights uniform or varied?
- Is a heuristic available?
- Are there multiple agents?

### 2. **Choose Algorithm**
- Unweighted → BFS
- Weighted, no heuristic → Dijkstra
- Weighted, with heuristic → A*
- Multiple agents → M*

### 3. **Design Heuristic** (for A*)
- Domain knowledge
- Ensure admissibility
- Balance accuracy vs speed
- Test on sample problems

### 4. **Implement & Optimize**
- Use priority queue (heap) for efficiency
- Consider bidirectional search
- Implement tie-breaking
- Profile and optimize hot paths

### 5. **Test & Validate**
- Verify correctness
- Check optimality
- Measure performance
- Test edge cases

## 🚀 Getting Started

```python
from algokit.algorithms.pathfinding import (
    BFS,
    DFS,
    Dijkstra,
    AStar,
    MStar
)

# Example: A* with Manhattan heuristic
def manhattan_heuristic(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

# Create grid graph
graph = create_grid_graph(width=10, height=10, obstacles=[(3,3), (3,4)])

# Find path
path = AStar(
    graph=graph,
    start=(0, 0),
    goal=(9, 9),
    heuristic=manhattan_heuristic
)

print(f"Found path of length {len(path)}: {path}")

# Multi-robot planning with M*
robots = [
    {'start': (0,0), 'goal': (9,9)},
    {'start': (9,0), 'goal': (0,9)}
]

paths = MStar(graph=graph, robots=robots)
print(f"Coordinated paths: {paths}")
```

## 📚 Further Reading

### Classic Books
- Russell & Norvig: "Artificial Intelligence: A Modern Approach" - Ch. 3-4
- LaValle: "Planning Algorithms" (Free online)
- Cormen et al. (CLRS): "Introduction to Algorithms" - Graph chapters

### Papers & Resources
- Hart et al.: "A Formal Basis for the Heuristic Determination of Minimum Cost Paths" (A*, 1968)
- Wagner & Choset: "M*: A Complete Multi-Robot Path Planning Algorithm" (2011)
- Harabor & Grastien: "Online Graph Pruning for Pathfinding" (JPS, 2011)

### Online Resources
- [Red Blob Games: Pathfinding](https://www.redblobgames.com/pathfinding/a-star/introduction.html)
- [Stanford CS 161: Algorithms](https://stanford-cs161.github.io/)
- [Moving AI Lab: Benchmarks](https://movingai.com/benchmarks/)

## 🎯 Practice Problems

### Beginner
- Maze solving with BFS/DFS
- Grid pathfinding with A*
- Simple robot navigation
- Shortest path in social graph

### Intermediate
- Weighted road network routing
- 8-puzzle solver with A*
- Multi-robot coordination (2-3 robots)
- Dynamic obstacle avoidance

### Advanced
- Real-time pathfinding in games
- Any-angle pathfinding
- Large-scale multi-robot systems
- 3D path planning for drones

## 🔗 Related Families

- **Graph Algorithms**: Foundation for pathfinding
- **Dynamic Programming**: Some planning problems
- **Reinforcement Learning**: Learning-based planning
- **Control Systems**: Low-level path following
- **Optimization**: Optimal control and trajectory planning

---

**Ready to find paths?** Start with [BFS](bfs.md), explore [A* Search](astar.md), or tackle [Multi-Robot Planning](mstar.md)!
