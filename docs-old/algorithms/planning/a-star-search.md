---
algorithm_key: "a-star-search"
tags: [planning, algorithms, a-star, heuristic-search, pathfinding, graph-search]
title: "A* Search Algorithm"
family: "planning"
---

{{ algorithm_card("a-star-search") }}

## Mathematical Formulation

!!! math "A* Evaluation Function"
    The A* algorithm uses the following evaluation function:
    
    $$f(n) = g(n) + h(n)$$
    
    Where:
    - $f(n)$ is the estimated total cost of the cheapest solution through node $n$
    - $g(n)$ is the actual cost from the start node to node $n$
    - $h(n)$ is the heuristic estimate of the cost from node $n$ to the goal
    
    The algorithm is optimal when $h(n)$ is admissible:
    
    $$h(n) \leq h^*(n)$$
    
    Where $h^*(n)$ is the true cost from node $n$ to the goal.

!!! success "Key Properties"
    - **Optimal**: Finds the shortest path when using an admissible heuristic
    - **Complete**: Will find a solution if one exists
    - **Efficient**: Uses heuristics to guide search toward the goal
    - **Versatile**: Works with any graph structure and cost function

## Implementation Approaches

=== "Basic A* Implementation (Recommended)"
    ```python
    import heapq
    from typing import List, Tuple, Dict, Set, Optional, Callable
    from dataclasses import dataclass
    
    @dataclass
    class Node:
        """Represents a node in the search graph."""
        position: Tuple[int, int]
        g_cost: float = float('inf')
        h_cost: float = 0.0
        f_cost: float = float('inf')
        parent: Optional['Node'] = None
        
        def __lt__(self, other):
            return self.f_cost < other.f_cost
    
    class AStarSearch:
        """
        A* search algorithm implementation.
        
        Args:
            heuristic_func: Function that estimates cost from node to goal
            get_neighbors: Function that returns valid neighbors of a node
            get_cost: Function that returns cost between two adjacent nodes
        """
        
        def __init__(self, 
                     heuristic_func: Callable[[Tuple[int, int], Tuple[int, int]], float],
                     get_neighbors: Callable[[Tuple[int, int]], List[Tuple[int, int]]],
                     get_cost: Callable[[Tuple[int, int], Tuple[int, int]], float] = lambda a, b: 1.0):
            
            self.heuristic_func = heuristic_func
            self.get_neighbors = get_neighbors
            self.get_cost = get_cost
        
        def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
            """
            Find the shortest path from start to goal using A* search.
            
            Args:
                start: Starting position
                goal: Target position
                
            Returns:
                List of positions representing the path, or None if no path exists
            """
            # Initialize open and closed sets
            open_set = []
            closed_set: Set[Tuple[int, int]] = set()
            nodes: Dict[Tuple[int, int], Node] = {}
            
            # Create start node
            start_node = Node(start, g_cost=0.0)
            start_node.h_cost = self.heuristic_func(start, goal)
            start_node.f_cost = start_node.g_cost + start_node.h_cost
            nodes[start] = start_node
            
            # Add start node to open set
            heapq.heappush(open_set, start_node)
            
            while open_set:
                # Get node with lowest f_cost
                current = heapq.heappop(open_set)
                current_pos = current.position
                
                # Check if we've reached the goal
                if current_pos == goal:
                    return self._reconstruct_path(current)
                
                # Add current node to closed set
                closed_set.add(current_pos)
                
                # Explore neighbors
                for neighbor_pos in self.get_neighbors(current_pos):
                    if neighbor_pos in closed_set:
                        continue
                    
                    # Calculate tentative g_cost
                    tentative_g_cost = current.g_cost + self.get_cost(current_pos, neighbor_pos)
                    
                    # Get or create neighbor node
                    if neighbor_pos not in nodes:
                        neighbor_node = Node(neighbor_pos)
                        neighbor_node.h_cost = self.heuristic_func(neighbor_pos, goal)
                        nodes[neighbor_pos] = neighbor_node
                    else:
                        neighbor_node = nodes[neighbor_pos]
                    
                    # Check if this path to neighbor is better
                    if tentative_g_cost < neighbor_node.g_cost:
                        neighbor_node.g_cost = tentative_g_cost
                        neighbor_node.f_cost = neighbor_node.g_cost + neighbor_node.h_cost
                        neighbor_node.parent = current
                        
                        # Add to open set if not already there
                        if neighbor_pos not in [node.position for node in open_set]:
                            heapq.heappush(open_set, neighbor_node)
            
            # No path found
            return None
        
        def _reconstruct_path(self, goal_node: Node) -> List[Tuple[int, int]]:
            """Reconstruct the path from start to goal."""
            path = []
            current = goal_node
            
            while current is not None:
                path.append(current.position)
                current = current.parent
            
            return path[::-1]  # Reverse to get path from start to goal
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/planning/a_star_search.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/planning/a_star_search.py)
    - **Tests**: [`tests/unit/planning/test_a_star_search.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/planning/test_a_star_search.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Standard A*** | $O(b^d)$ | $O(b^d)$ | Optimal with admissible heuristic |
    **Weighted A*** | $O(b^{w \cdot d})$ | $O(b^{w \cdot d})$ | Faster but suboptimal (w = weight) |
    **Grid A*** | $O(|V| \log |V|)$ | $O(|V|)$ | For grid with |V| vertices |

!!! warning "Performance Considerations"
    - **Heuristic quality** significantly affects performance
    - **Memory usage** can be high for large search spaces
    - **Open set management** is critical for efficiency
    - **Tie-breaking** can affect path quality when f-costs are equal

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Game Development"
        - **Pathfinding**: NPC movement and navigation
        - **AI Behavior**: Strategic decision making
        - **Level Design**: Optimal route planning
        - **Real-time Strategy**: Unit movement optimization

    !!! grid-item "Robotics & Navigation"
        - **Autonomous Vehicles**: Route planning and obstacle avoidance
        - **Robot Navigation**: Indoor and outdoor pathfinding
        - **Drone Flight**: 3D path planning with constraints
        - **Warehouse Automation**: Efficient item retrieval

    !!! grid-item "Real-World Applications"
        - **GPS Navigation**: Optimal route calculation
        - **Network Routing**: Data packet path optimization
        - **Resource Allocation**: Optimal task scheduling
        - **Logistics**: Delivery route optimization

    !!! grid-item "Educational Value"
        - **Algorithm Design**: Understanding heuristic search
        - **Graph Theory**: Learning graph traversal techniques
        - **Optimization**: Balancing optimality and efficiency
        - **Problem Solving**: Systematic approach to complex problems

!!! success "Educational Value"
    - **Heuristic Search**: Perfect example of informed search algorithms
    - **Optimality**: Demonstrates conditions for optimal solutions
    - **Efficiency**: Shows how heuristics improve search performance
    - **Problem Modeling**: Illustrates how to model real-world problems as graphs

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Papers"
        1. **Hart, P. E., Nilsson, N. J., & Raphael, B.** (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100-107.
        2. **Pearl, J.** (1984). *Heuristics: Intelligent Search Strategies for Computer Problem Solving*. Addison-Wesley.

    !!! grid-item "Planning Textbooks"
        3. **Russell, S., & Norvig, P.** (2020). *Artificial Intelligence: A Modern Approach*. Pearson.
        4. **LaValle, S. M.** (2006). *Planning Algorithms*. Cambridge University Press.

    !!! grid-item "Online Resources"
        5. [A* Search Algorithm - Wikipedia](https://en.wikipedia.org/wiki/A*_search_algorithm)
        6. [A* Pathfinding Tutorial](https://www.redblobgames.com/pathfinding/a-star/introduction.html)
        7. [Introduction to A* - Stanford CS](https://web.stanford.edu/class/cs97si/04-search.pdf)

    !!! grid-item "Implementation & Practice"
        8. [Python heapq Documentation](https://docs.python.org/3/library/heapq.html)
        9. [Pathfinding.js](https://github.com/qiao/PathFinding.js) - JavaScript A* implementation
        10. [A* Algorithm Visualization](https://qiao.github.io/PathFinding.js/visual/)

!!! tip "Interactive Learning"
    Try implementing A* yourself! Start with a simple grid-based environment to understand the basics. Experiment with different heuristics (Manhattan distance, Euclidean distance) to see how they affect performance. Try weighted A* to understand the trade-off between optimality and speed. Implement obstacle avoidance and diagonal movement to see how the algorithm handles complex environments. This will give you deep insight into the power of heuristic search algorithms.

## Navigation

{{ nav_grid(current_algorithm="a-star-search", current_family="planning", max_related=5) }}