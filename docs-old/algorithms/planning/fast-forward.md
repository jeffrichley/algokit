---
algorithm_key: "fast-forward"
tags: [planning, algorithms, fast-forward, ff, heuristic-search, relaxed-planning]
title: "Fast-Forward (FF) Algorithm"
family: "planning"
---

# Fast-Forward (FF) Algorithm

{{ algorithm_card("fast-forward") }}

!!! abstract "Overview"
    Fast-Forward (FF) is a state-of-the-art classical planning algorithm that combines heuristic search with relaxed planning to efficiently find solutions to planning problems. The algorithm uses a relaxed planning graph to compute heuristic estimates and employs a combination of breadth-first search and hill-climbing to explore the search space.

    FF is particularly effective because it uses the "relaxed plan" heuristic, which provides accurate estimates of the remaining cost to reach the goal. The algorithm has been highly successful in international planning competitions and has influenced many modern planning systems.

## Mathematical Formulation

!!! math "Relaxed Planning Graph"
    The relaxed planning graph is constructed by ignoring delete effects of actions:

    - **Proposition levels** $P_i$: Propositions that can be true at level $i$
    - **Action levels** $A_i$: Actions that can be applied at level $i$

    Construction rules:

    $$P_0 = \text{initial state}$$

    $$A_i = \{a \in A : \text{preconditions}(a) \subseteq P_i\}$$

    $$P_{i+1} = P_i \cup \bigcup_{a \in A_i} \text{add effects}(a)$$

    The relaxed plan heuristic $h^+$ is the cost of the optimal relaxed plan.

!!! success "Key Properties"
    - **Heuristic-guided**: Uses relaxed planning graph for accurate estimates
    - **Efficient**: Combines multiple search strategies
    - **Complete**: Will find a solution if one exists
    - **Competitive**: Performs well in planning competitions

## Implementation Approaches

=== "Basic FF Implementation (Recommended)"
    ```python
    from typing import Set, List, Dict, Tuple, Optional
    from dataclasses import dataclass
    from collections import defaultdict, deque
    import heapq

    @dataclass
    class Proposition:
        """Represents a proposition in the planning domain."""
        name: str
        args: Tuple[str, ...] = ()

        def __hash__(self):
            return hash((self.name, self.args))

        def __eq__(self, other):
            return isinstance(other, Proposition) and self.name == other.name and self.args == other.args

    @dataclass
    class Action:
        """Represents an action in the planning domain."""
        name: str
        preconditions: Set[Proposition]
        add_effects: Set[Proposition]
        delete_effects: Set[Proposition]
        cost: float = 1.0
        args: Tuple[str, ...] = ()

        def __hash__(self):
            return hash((self.name, self.args))

    @dataclass
    class State:
        """Represents a state in the planning problem."""
        propositions: Set[Proposition]
        g_cost: float = 0.0
        h_cost: float = 0.0
        f_cost: float = 0.0
        parent: Optional['State'] = None
        action: Optional[Action] = None

        def __hash__(self):
            return hash(frozenset(self.propositions))

        def __eq__(self, other):
            return isinstance(other, State) and self.propositions == other.propositions

    class RelaxedPlanningGraph:
        """Represents the relaxed planning graph."""

        def __init__(self):
            self.prop_levels: List[Set[Proposition]] = []
            self.action_levels: List[Set[Action]] = []
            self.goal_level: int = -1

        def build(self, initial_state: Set[Proposition], goal: Set[Proposition], actions: List[Action]) -> int:
            """Build the relaxed planning graph and return goal level."""
            self.prop_levels = [initial_state.copy()]
            self.action_levels = []

            level = 0
            while not goal.issubset(self.prop_levels[level]):
                # Find applicable actions
                applicable_actions = set()
                for action in actions:
                    if action.preconditions.issubset(self.prop_levels[level]):
                        applicable_actions.add(action)

                self.action_levels.append(applicable_actions)

                # Compute next proposition level
                next_props = self.prop_levels[level].copy()
                for action in applicable_actions:
                    next_props.update(action.add_effects)

                self.prop_levels.append(next_props)
                level += 1

                # Check for termination
                if level > 1000:  # Prevent infinite loops
                    return -1

            self.goal_level = level
            return level

        def extract_relaxed_plan(self, goal: Set[Proposition]) -> List[Action]:
            """Extract a relaxed plan from the graph."""
            if self.goal_level == -1:
                return []

            relaxed_plan = []
            needed_props = goal.copy()

            # Work backwards from goal level
            for level in range(self.goal_level - 1, -1, -1):
                if not needed_props:
                    break

                # Find actions that achieve needed propositions
                for action in self.action_levels[level]:
                    if action.add_effects & needed_props:
                        relaxed_plan.append(action)
                        needed_props.update(action.preconditions)
                        needed_props -= action.add_effects

            return relaxed_plan

    class FastForward:
        """
        Fast-Forward planning algorithm implementation.

        Args:
            actions: List of available actions
            initial_state: Set of initial propositions
            goal: Set of goal propositions
        """

        def __init__(self, actions: List[Action], initial_state: Set[Proposition], goal: Set[Proposition]):
            self.actions = actions
            self.initial_state = initial_state
            self.goal = goal
            self.rpg = RelaxedPlanningGraph()

        def plan(self) -> Optional[List[Action]]:
            """
            Find a plan using the Fast-Forward algorithm.

            Returns:
                List of actions representing the plan, or None if no plan exists
            """
            # Build relaxed planning graph
            goal_level = self.rpg.build(self.initial_state, self.goal, self.actions)
            if goal_level == -1:
                return None

            # Initialize search
            initial_state = State(self.initial_state)
            initial_state.h_cost = self._compute_heuristic(initial_state)
            initial_state.f_cost = initial_state.g_cost + initial_state.h_cost

            # Use enforced hill-climbing with breadth-first search
            return self._enforced_hill_climbing(initial_state)

        def _compute_heuristic(self, state: State) -> float:
            """Compute the relaxed plan heuristic for a state."""
            # Build RPG from current state
            temp_rpg = RelaxedPlanningGraph()
            goal_level = temp_rpg.build(state.propositions, self.goal, self.actions)

            if goal_level == -1:
                return float('inf')

            # Extract relaxed plan
            relaxed_plan = temp_rpg.extract_relaxed_plan(self.goal)

            # Return cost of relaxed plan
            return sum(action.cost for action in relaxed_plan)

        def _enforced_hill_climbing(self, initial_state: State) -> Optional[List[Action]]:
            """Enforced hill-climbing search."""
            current_state = initial_state
            visited = set()

            while not self._is_goal_state(current_state):
                visited.add(current_state)

                # Find best successor
                best_successor = None
                best_h_cost = float('inf')

                # Breadth-first search for better state
                queue = deque([current_state])
                bfs_visited = set()

                while queue:
                    state = queue.popleft()

                    if state in bfs_visited:
                        continue
                    bfs_visited.add(state)

                    # Check if this state is better
                    if state.h_cost < best_h_cost:
                        best_successor = state
                        best_h_cost = state.h_cost

                    # If we found a better state, use it
                    if state.h_cost < current_state.h_cost:
                        current_state = state
                        break

                    # Add successors to queue
                    for successor in self._get_successors(state):
                        if successor not in visited and successor not in bfs_visited:
                            queue.append(successor)

                # If no better state found, we're stuck
                if best_successor is None or best_successor == current_state:
                    return None

                current_state = best_successor

            # Reconstruct plan
            return self._reconstruct_plan(current_state)

        def _is_goal_state(self, state: State) -> bool:
            """Check if state satisfies the goal."""
            return self.goal.issubset(state.propositions)

        def _get_successors(self, state: State) -> List[State]:
            """Get all successor states from a given state."""
            successors = []

            for action in self.actions:
                if action.preconditions.issubset(state.propositions):
                    # Apply action
                    new_props = state.propositions.copy()
                    new_props -= action.delete_effects
                    new_props.update(action.add_effects)

                    # Create new state
                    successor = State(new_props)
                    successor.g_cost = state.g_cost + action.cost
                    successor.h_cost = self._compute_heuristic(successor)
                    successor.f_cost = successor.g_cost + successor.h_cost
                    successor.parent = state
                    successor.action = action

                    successors.append(successor)

            return successors

        def _reconstruct_plan(self, goal_state: State) -> List[Action]:
            """Reconstruct the plan from the goal state."""
            plan = []
            current = goal_state

            while current.parent is not None:
                plan.append(current.action)
                current = current.parent

            return plan[::-1]  # Reverse to get plan from start to goal
    ```

=== "FF with Multiple Heuristics"
    ```python
    class MultiHeuristicFF(FastForward):
        """
        Fast-Forward with multiple heuristic functions.
        """

        def __init__(self, actions: List[Action], initial_state: Set[Proposition], goal: Set[Proposition]):
            super().__init__(actions, initial_state, goal)
            self.heuristics = [
                self._relaxed_plan_heuristic,
                self._landmark_heuristic,
                self._pattern_heuristic
            ]

        def _compute_heuristic(self, state: State) -> float:
            """Compute heuristic using multiple functions."""
            heuristic_values = []

            for heuristic_func in self.heuristics:
                try:
                    value = heuristic_func(state)
                    if value != float('inf'):
                        heuristic_values.append(value)
                except:
                    continue

            if not heuristic_values:
                return float('inf')

            # Use maximum heuristic value (most informed)
            return max(heuristic_values)

        def _relaxed_plan_heuristic(self, state: State) -> float:
            """Standard relaxed plan heuristic."""
            temp_rpg = RelaxedPlanningGraph()
            goal_level = temp_rpg.build(state.propositions, self.goal, self.actions)

            if goal_level == -1:
                return float('inf')

            relaxed_plan = temp_rpg.extract_relaxed_plan(self.goal)
            return sum(action.cost for action in relaxed_plan)

        def _landmark_heuristic(self, state: State) -> float:
            """Landmark-based heuristic (simplified)."""
            # This is a simplified implementation
            # Real landmark heuristics are more complex
            return len(self.goal - state.propositions)

        def _pattern_heuristic(self, state: State) -> float:
            """Pattern-based heuristic (simplified)."""
            # This is a simplified implementation
            # Real pattern heuristics use database of patterns
            return len(self.goal - state.propositions) * 0.5
    ```

=== "FF with Pruning"
    ```python
    class PrunedFF(FastForward):
        """
        Fast-Forward with advanced pruning techniques.
        """

        def __init__(self, actions: List[Action], initial_state: Set[Proposition], goal: Set[Proposition]):
            super().__init__(actions, initial_state, goal)
            self.pruning_enabled = True

        def _get_successors(self, state: State) -> List[State]:
            """Get successors with pruning."""
            successors = []

            for action in self.actions:
                if action.preconditions.issubset(state.propositions):
                    # Apply action
                    new_props = state.propositions.copy()
                    new_props -= action.delete_effects
                    new_props.update(action.add_effects)

                    # Create new state
                    successor = State(new_props)
                    successor.g_cost = state.g_cost + action.cost
                    successor.h_cost = self._compute_heuristic(successor)
                    successor.f_cost = successor.g_cost + successor.h_cost
                    successor.parent = state
                    successor.action = action

                    # Apply pruning
                    if self._should_prune(successor, state):
                        continue

                    successors.append(successor)

            return successors

        def _should_prune(self, successor: State, parent: State) -> bool:
            """Determine if successor should be pruned."""
            if not self.pruning_enabled:
                return False

            # Prune if heuristic is infinite
            if successor.h_cost == float('inf'):
                return True

            # Prune if no progress made
            if successor.h_cost >= parent.h_cost:
                return True

            # Prune if too expensive
            if successor.f_cost > parent.f_cost * 2:
                return True

            return False
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/planning/fast_forward.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/planning/fast_forward.py)
    - **Tests**: [`tests/unit/planning/test_fast_forward.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/planning/test_fast_forward.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Standard FF** | $O(b^d \cdot n)$ | $O(b^d \cdot n)$ | Heuristic-guided search |
    **Multi-Heuristic FF** | $O(b^d \cdot n \cdot h)$ | $O(b^d \cdot n)$ | Multiple heuristic computation |
    **Pruned FF** | $O(b^d \cdot n)$ | $O(b^d \cdot n)$ | Reduced search space |

!!! warning "Performance Considerations"
    - **Heuristic computation** can be expensive for large domains
    - **Relaxed planning graph** construction scales with problem size
    - **Enforced hill-climbing** may get stuck in local minima
    - **Memory usage** can be high for complex problems

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Classical Planning"
        - **Logistics**: Package delivery and warehouse management
        - **Manufacturing**: Production planning and scheduling
        - **Robotics**: Task planning and execution
        - **Game AI**: Strategic planning and decision making

    !!! grid-item "Heuristic Search"
        - **Route Planning**: Optimal path finding with constraints
        - **Resource Allocation**: Efficient resource distribution
        - **Scheduling**: Task and resource scheduling
        - **Optimization**: Constraint satisfaction problems

    !!! grid-item "Real-World Applications"
        - **Supply Chain**: Inventory and distribution planning
        - **Transportation**: Route and schedule optimization
        - **Healthcare**: Treatment planning and resource allocation
        - **Finance**: Portfolio optimization and risk management

    !!! grid-item "Educational Value"
        - **Planning Theory**: Understanding heuristic-guided planning
        - **Search Algorithms**: Learning enforced hill-climbing
        - **Heuristic Design**: Understanding relaxed planning
        - **Algorithm Design**: Systematic problem decomposition

!!! success "Educational Value"
    - **Heuristic Search**: Perfect example of informed search algorithms
    - **Relaxed Planning**: Demonstrates abstraction in planning
    - **Search Strategies**: Shows combination of different search methods
    - **Algorithm Design**: Illustrates systematic heuristic design

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Papers"
        1. **Hoffmann, J., & Nebel, B.** (2001). The FF planning system: Fast plan generation through heuristic search. *Journal of Artificial Intelligence Research*, 14, 253-302.
        2. **Hoffmann, J.** (2001). FF: The fast-forward planning system. *AI Magazine*, 22(3), 57-62.

    !!! grid-item "Planning Textbooks"
        3. **Ghallab, M., Nau, D., & Traverso, P.** (2016). *Automated Planning and Acting*. Cambridge University Press.
        4. **Russell, S., & Norvig, P.** (2020). *Artificial Intelligence: A Modern Approach*. Pearson.

    !!! grid-item "Online Resources"
        5. [Fast-Forward Planner - Wikipedia](https://en.wikipedia.org/wiki/Fast-forward_planning_system)
        6. [FF Planning System](https://fai.cs.uni-saarland.de/hoffmann/ff.html)
        7. [Heuristic Search in Planning](https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s07/www/slides/heuristic-search-planning.pdf)

    !!! grid-item "Implementation & Practice"
        8. [PDDL - Planning Domain Definition Language](https://planning.wiki/ref/pddl)
        9. [Fast-Forward Source Code](https://github.com/pucrs-automated-planning/ff)
        10. [Planning Competition Results](https://ipc.icaps-conference.org/)

!!! tip "Interactive Learning"
    Try implementing Fast-Forward yourself! Start with simple planning domains to understand the relaxed planning graph construction. Experiment with different heuristic functions to see how they affect search performance. Try implementing enforced hill-climbing to understand how it combines different search strategies. Compare FF with other planning algorithms to see why it's so effective. This will give you deep insight into heuristic-guided planning and relaxed planning techniques.

## Navigation

{{ nav_grid(current_algorithm="fast-forward", current_family="planning", max_related=5) }}
