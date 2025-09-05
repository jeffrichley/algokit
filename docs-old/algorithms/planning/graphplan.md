---
algorithm_key: "graphplan"
tags: [planning, algorithms, graphplan, classical-planning, sat-solving, planning-graphs]
title: "Graphplan Algorithm"
family: "planning"
---

# Graphplan Algorithm

{{ algorithm_card("graphplan") }}

!!! abstract "Overview"
    Graphplan is a classical planning algorithm that constructs a planning graph to find valid plans for achieving goals from initial states. Unlike traditional search-based planners, Graphplan uses a graph structure to represent the planning problem and employs constraint satisfaction techniques to extract valid plans.

    The algorithm works in two phases: graph expansion (building the planning graph level by level) and solution extraction (finding a valid plan by solving constraints). Graphplan is particularly effective for problems with many interacting subgoals and has influenced many modern planning algorithms.

## Mathematical Formulation

!!! math "Planning Graph Structure"
    A planning graph consists of alternating levels of propositions and actions:
    
    - **Proposition levels** $P_i$: Set of propositions that can be true at time step $i$
    - **Action levels** $A_i$: Set of actions that can be executed at time step $i$
    
    The graph is constructed using:
    
    $$P_{i+1} = P_i \cup \bigcup_{a \in A_i} \text{effects}(a)$$
    
    $$A_{i+1} = \{a \in A : \text{preconditions}(a) \subseteq P_i \text{ and } \neg \text{conflicts}(a, A_i)\}$$
    
    Where conflicts are determined by mutual exclusion (mutex) relationships.

!!! success "Key Properties"
    - **Systematic**: Explores all possible action sequences up to a given depth
    - **Constraint-based**: Uses mutex relationships to prune invalid combinations
    - **Complete**: Will find a solution if one exists within the graph depth
    - **Efficient**: Avoids redundant search through graph structure

## Implementation Approaches

=== "Basic Graphplan Implementation (Recommended)"
    ```python
    from typing import Set, List, Dict, Tuple, Optional
    from dataclasses import dataclass
    from collections import defaultdict
    import itertools
    
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
        effects: Set[Proposition]
        args: Tuple[str, ...] = ()
        
        def __hash__(self):
            return hash((self.name, self.args))
    
    class PlanningGraph:
        """Represents the planning graph structure."""
        
        def __init__(self):
            self.prop_levels: List[Set[Proposition]] = []
            self.action_levels: List[Set[Action]] = []
            self.prop_mutex: List[Set[Tuple[Proposition, Proposition]]] = []
            self.action_mutex: List[Set[Tuple[Action, Action]]] = []
    
    class Graphplan:
        """
        Graphplan algorithm implementation.
        
        Args:
            actions: List of available actions
            initial_state: Set of initial propositions
            goal: Set of goal propositions
        """
        
        def __init__(self, actions: List[Action], initial_state: Set[Proposition], goal: Set[Proposition]):
            self.actions = actions
            self.initial_state = initial_state
            self.goal = goal
            self.graph = PlanningGraph()
        
        def plan(self) -> Optional[List[Set[Action]]]:
            """
            Find a plan using the Graphplan algorithm.
            
            Returns:
                List of action sets representing the plan, or None if no plan exists
            """
            # Build planning graph until goal is reachable
            level = 0
            while not self._goal_reachable(level):
                if not self._extend_graph():
                    return None  # No solution exists
                level += 1
            
            # Extract solution
            return self._extract_solution(level)
        
        def _extend_graph(self) -> bool:
            """Extend the planning graph by one level."""
            if not self.graph.prop_levels:
                # Initialize with initial state
                self.graph.prop_levels.append(self.initial_state.copy())
                self.graph.prop_mutex.append(set())
            
            # Create action level
            current_props = self.graph.prop_levels[-1]
            current_prop_mutex = self.graph.prop_mutex[-1]
            
            # Find applicable actions
            applicable_actions = set()
            for action in self.actions:
                if action.preconditions.issubset(current_props):
                    # Check for action mutex
                    is_mutex = False
                    for other_action in applicable_actions:
                        if self._actions_mutex(action, other_action, current_prop_mutex):
                            is_mutex = True
                            break
                    
                    if not is_mutex:
                        applicable_actions.add(action)
            
            self.graph.action_levels.append(applicable_actions)
            
            # Create next proposition level
            next_props = current_props.copy()
            for action in applicable_actions:
                next_props.update(action.effects)
            
            self.graph.prop_levels.append(next_props)
            
            # Compute proposition mutex
            next_prop_mutex = set()
            for prop1, prop2 in itertools.combinations(next_props, 2):
                if self._propositions_mutex(prop1, prop2, applicable_actions, current_prop_mutex):
                    next_prop_mutex.add((prop1, prop2))
            
            self.graph.prop_mutex.append(next_prop_mutex)
            
            # Check if graph has stabilized
            return len(next_props) > len(current_props) or len(applicable_actions) > 0
        
        def _actions_mutex(self, action1: Action, action2: Action, prop_mutex: Set[Tuple[Proposition, Proposition]]) -> bool:
            """Check if two actions are mutually exclusive."""
            # Inconsistent effects
            if action1.effects & action2.effects:
                return True
            
            # Interference
            if (action1.effects & action2.preconditions) or (action2.effects & action1.preconditions):
                return True
            
            # Competing needs
            for pre1 in action1.preconditions:
                for pre2 in action2.preconditions:
                    if (pre1, pre2) in prop_mutex or (pre2, pre1) in prop_mutex:
                        return True
            
            return False
        
        def _propositions_mutex(self, prop1: Proposition, prop2: Proposition, 
                              actions: Set[Action], prop_mutex: Set[Tuple[Proposition, Proposition]]) -> bool:
            """Check if two propositions are mutually exclusive."""
            # All actions that achieve prop1 are mutex with all actions that achieve prop2
            achievers1 = {a for a in actions if prop1 in a.effects}
            achievers2 = {a for a in actions if prop2 in a.effects}
            
            if not achievers1 or not achievers2:
                return False
            
            for a1 in achievers1:
                for a2 in achievers2:
                    if not self._actions_mutex(a1, a2, prop_mutex):
                        return False
            
            return True
        
        def _goal_reachable(self, level: int) -> bool:
            """Check if goal is reachable at the given level."""
            if level >= len(self.graph.prop_levels):
                return False
            
            current_props = self.graph.prop_levels[level]
            current_mutex = self.graph.prop_mutex[level]
            
            # Check if all goal propositions are present and not mutex
            if not self.goal.issubset(current_props):
                return False
            
            # Check for mutex relationships in goal
            for prop1, prop2 in itertools.combinations(self.goal, 2):
                if (prop1, prop2) in current_mutex or (prop2, prop1) in current_mutex:
                    return False
            
            return True
        
        def _extract_solution(self, level: int) -> List[Set[Action]]:
            """Extract a solution from the planning graph."""
            solution = []
            remaining_goals = self.goal.copy()
            
            # Work backwards from the goal level
            for i in range(level, 0, -1):
                # Find actions that achieve remaining goals
                level_actions = set()
                for action in self.graph.action_levels[i-1]:
                    if action.effects & remaining_goals:
                        level_actions.add(action)
                
                solution.insert(0, level_actions)
                
                # Update remaining goals
                new_goals = set()
                for action in level_actions:
                    new_goals.update(action.preconditions)
                
                remaining_goals = new_goals
            
            return solution
    ```

=== "Graphplan with SAT Solving"
    ```python
    class SATGraphplan(Graphplan):
        """
        Graphplan implementation using SAT solving for solution extraction.
        """
        
        def __init__(self, actions: List[Action], initial_state: Set[Proposition], goal: Set[Proposition]):
            super().__init__(actions, initial_state, goal)
            self.sat_solver = None  # Would integrate with actual SAT solver
        
        def _extract_solution_sat(self, level: int) -> Optional[List[Set[Action]]]:
            """Extract solution using SAT solving approach."""
            # Convert planning graph to SAT formula
            cnf_formula = self._graph_to_cnf(level)
            
            # Solve SAT formula (simplified - would use actual SAT solver)
            solution = self._solve_sat(cnf_formula)
            
            if solution:
                return self._sat_solution_to_plan(solution, level)
            
            return None
        
        def _graph_to_cnf(self, level: int) -> List[List[int]]:
            """Convert planning graph to CNF formula."""
            # This is a simplified version - actual implementation would be more complex
            cnf = []
            
            # Add constraints for goal achievement
            for goal_prop in self.goal:
                # Goal must be achieved by some action
                goal_clause = []
                for i in range(level):
                    for action in self.graph.action_levels[i]:
                        if goal_prop in action.effects:
                            goal_clause.append(self._action_to_var(action, i))
                cnf.append(goal_clause)
            
            # Add mutex constraints
            for i in range(level):
                for action1, action2 in self.graph.action_mutex[i]:
                    cnf.append([-self._action_to_var(action1, i), -self._action_to_var(action2, i)])
            
            return cnf
        
        def _action_to_var(self, action: Action, level: int) -> int:
            """Convert action and level to SAT variable."""
            # Simplified variable encoding
            return hash((action, level)) % 1000000
        
        def _solve_sat(self, cnf: List[List[int]]) -> Optional[Dict[int, bool]]:
            """Solve SAT formula (simplified implementation)."""
            # This would integrate with an actual SAT solver like MiniSat or Z3
            # For now, return a dummy solution
            return {i: True for i in range(100)}
        
        def _sat_solution_to_plan(self, solution: Dict[int, bool], level: int) -> List[Set[Action]]:
            """Convert SAT solution to action plan."""
            plan = []
            for i in range(level):
                level_actions = set()
                for action in self.graph.action_levels[i]:
                    var = self._action_to_var(action, i)
                    if solution.get(var, False):
                        level_actions.add(action)
                plan.append(level_actions)
            return plan
    ```

=== "Parallel Graphplan"
    ```python
    class ParallelGraphplan(Graphplan):
        """
        Graphplan implementation that allows parallel action execution.
        """
        
        def __init__(self, actions: List[Action], initial_state: Set[Proposition], goal: Set[Proposition]):
            super().__init__(actions, initial_state, goal)
            self.parallel_execution = True
        
        def _actions_mutex(self, action1: Action, action2: Action, prop_mutex: Set[Tuple[Proposition, Proposition]]) -> bool:
            """Check mutex with parallel execution considerations."""
            # Inconsistent effects (cannot be parallel)
            if action1.effects & action2.effects:
                return True
            
            # Resource conflicts (simplified)
            if hasattr(action1, 'resources') and hasattr(action2, 'resources'):
                if action1.resources & action2.resources:
                    return True
            
            # Interference (cannot be parallel)
            if (action1.effects & action2.preconditions) or (action2.effects & action1.preconditions):
                return True
            
            return False
        
        def _extract_parallel_solution(self, level: int) -> List[Set[Action]]:
            """Extract solution with parallel action execution."""
            solution = []
            remaining_goals = self.goal.copy()
            
            for i in range(level, 0, -1):
                # Find all actions that can be executed in parallel
                level_actions = set()
                for action in self.graph.action_levels[i-1]:
                    if action.effects & remaining_goals:
                        # Check if action can be added to parallel set
                        can_add = True
                        for existing_action in level_actions:
                            if self._actions_mutex(action, existing_action, self.graph.prop_mutex[i-1]):
                                can_add = False
                                break
                        
                        if can_add:
                            level_actions.add(action)
                
                solution.insert(0, level_actions)
                
                # Update remaining goals
                new_goals = set()
                for action in level_actions:
                    new_goals.update(action.preconditions)
                
                remaining_goals = new_goals
            
            return solution
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/planning/graphplan.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/planning/graphplan.py)
    - **Tests**: [`tests/unit/planning/test_graphplan.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/planning/test_graphplan.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Standard Graphplan** | $O(b^d \cdot n)$ | $O(b^d \cdot n)$ | Graph construction and solution extraction |
    **SAT Graphplan** | $O(2^n)$ | $O(n)$ | SAT solving complexity |
    **Parallel Graphplan** | $O(b^d \cdot n)$ | $O(b^d \cdot n)$ | Reduced plan length |

!!! warning "Performance Considerations"
    - **Graph size** grows exponentially with problem complexity
    - **Mutex computation** can be expensive for large action sets
    - **Solution extraction** may require backtracking
    - **Memory usage** can be high for complex domains

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Classical Planning"
        - **Logistics**: Package delivery and warehouse management
        - **Manufacturing**: Production planning and scheduling
        - **Robotics**: Task planning and execution
        - **Game AI**: Strategic planning and decision making

    !!! grid-item "Automated Reasoning"
        - **Theorem Proving**: Automated proof generation
        - **Constraint Satisfaction**: Complex constraint solving
        - **Resource Allocation**: Optimal resource distribution
        - **Scheduling**: Task and resource scheduling

    !!! grid-item "Real-World Applications"
        - **Supply Chain**: Inventory and distribution planning
        - **Transportation**: Route and schedule optimization
        - **Healthcare**: Treatment planning and resource allocation
        - **Finance**: Portfolio optimization and risk management

    !!! grid-item "Educational Value"
        - **Planning Theory**: Understanding classical planning approaches
        - **Graph Algorithms**: Learning graph-based problem solving
        - **Constraint Satisfaction**: Understanding mutex relationships
        - **Algorithm Design**: Systematic problem decomposition

!!! success "Educational Value"
    - **Classical Planning**: Perfect example of systematic planning approaches
    - **Graph Theory**: Demonstrates graph-based problem representation
    - **Constraint Satisfaction**: Shows how to handle complex constraints
    - **Algorithm Design**: Illustrates systematic problem decomposition

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Papers"
        1. **Blum, A. L., & Furst, M. L.** (1997). Fast planning through planning graph analysis. *Artificial Intelligence*, 90(1-2), 281-300.
        2. **Kambhampati, S.** (2000). Planning graph as a (dynamic) CSP: Exploiting EBL, DDB and other CSP search techniques in Graphplan. *Journal of Artificial Intelligence Research*, 12, 1-34.

    !!! grid-item "Planning Textbooks"
        3. **Ghallab, M., Nau, D., & Traverso, P.** (2016). *Automated Planning and Acting*. Cambridge University Press.
        4. **Russell, S., & Norvig, P.** (2020). *Artificial Intelligence: A Modern Approach*. Pearson.

    !!! grid-item "Online Resources"
        5. [Graphplan Algorithm - Wikipedia](https://en.wikipedia.org/wiki/Graphplan)
        6. [Planning Graph Tutorial](https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s07/www/slides/planning-graphs.pdf)
        7. [Classical Planning Overview](https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s07/www/slides/classical-planning.pdf)

    !!! grid-item "Implementation & Practice"
        8. [PDDL - Planning Domain Definition Language](https://planning.wiki/ref/pddl)
        9. [Fast-Forward Planner](https://fai.cs.uni-saarland.de/hoffmann/ff.html)
        10. [Graphplan Implementation](https://github.com/pucrs-automated-planning/graphplan)

!!! tip "Interactive Learning"
    Try implementing Graphplan yourself! Start with simple planning domains like the blocks world or logistics problems. Experiment with different mutex relationships to understand how they affect the planning graph. Try implementing SAT-based solution extraction to see how constraint satisfaction can be used for planning. Compare Graphplan with other planning algorithms to understand the trade-offs between different approaches. This will give you deep insight into classical planning and graph-based algorithms.

## Navigation

{{ nav_grid(current_algorithm="graphplan", current_family="planning", max_related=5) }}
